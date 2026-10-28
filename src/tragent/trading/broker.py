"""The broker process: the only holder of each user's Gateway connection.

Run exactly one with: ``python -m tragent.trading.broker``

It does four things on one event loop:

* every 10 seconds, connects whoever is not connected, saves each user's balances and positions,
  and reconciles their orders with what the Gateway knows;
* places and cancels orders as commands arrive from Kafka;
* records each order status change and fill as the Gateway reports it, publishing the changed
  order back to Kafka for the API to push to the browser;
* publishes each user's balances and positions, at most once a second, whenever IBKR updates
  them: every position's value and P&L about once a second, account values every few minutes and
  right after a trade.

Every order sent to IBKR carries its own ID as the order reference, so reconciliation can match
the Gateway's trades to orders after any restart.

One user's failure must not stop the others: each user's sync, each resent order, each published
snapshot, and each command fails alone and is logged. ib_async already does the same for its
event handlers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from math import isnan

from confluent_kafka import Consumer, Producer
from ib_async import (
    IB,
    AccountValue,
    LimitOrder,
    MarketOrder,
    PnLSingle,
    PortfolioItem,
    Stock,
    Trade,
)
from redis import Redis

from tragent import redis_client
from tragent.config import load_settings
from tragent.db import DatabasePool, build_connection_pool
from tragent.trading import messaging, orders, store
from tragent.trading.containers import gateway_address

logger = logging.getLogger(__name__)

_ROUND_SECONDS = 10
_PUBLISH_SECONDS = 1
# Each round renews a user's status, so it disappears soon after this process stops.
_STATUS_SECONDS = 30
# An order still unsent after this long lost its command, so reconciliation sends it again...
_RESEND_AFTER_SECONDS = 30
# ...but an order this old is rejected instead, since the user expected it filled long ago.
_SEND_WITHIN_SECONDS = 60
_ORDER_REF = "tragent:"
_BALANCE_TAGS = {
    "NetLiquidation": "net_liquidation",
    "TotalCashValue": "cash",
    "BuyingPower": "buying_power",
    "AvailableFunds": "available_funds",
}
# Fills decide "filled"; pending statuses change nothing.
_REPORTED_STATUS: dict[str, orders.Status] = {
    "PreSubmitted": "submitted",
    "Submitted": "submitted",
    "Cancelled": "cancelled",
    "ApiCancelled": "cancelled",
    "Inactive": "rejected",
}


def status_key(user_id: int) -> str:
    return f"tragent:broker:{user_id}"


class Broker:
    def __init__(
        self,
        pool: DatabasePool,
        redis: Redis,
        producer: Producer,
        new_connection: Callable[[], IB] = IB,
    ) -> None:
        self._pool = pool
        self._redis = redis
        self._producer = producer
        self._new_connection = new_connection
        self.connections: dict[int, IB] = {}
        # Users whose balances or positions IBKR updated since the last publish.
        self._changed: set[int] = set()

    async def run(self, commands: Consumer) -> None:
        await asyncio.gather(self._sync_forever(), self._consume(commands), self._publish_forever())

    async def sync_round(self) -> None:
        """Connect whoever is not connected, then save and reconcile every connected user."""

        user_ids = store.account_user_ids(self._pool)
        for user_id in self.connections.keys() - set(user_ids):
            self.connections.pop(user_id).disconnect()
        for user_id in user_ids:
            try:
                await self._sync_user(user_id)
            except Exception:
                logger.exception("Syncing the account of user %s failed", user_id)
                self._write_status(user_id, "Syncing this account failed")
        for order in orders.unsent_orders(self._pool, _RESEND_AFTER_SECONDS):
            try:
                self.submit(order)
            except Exception:
                logger.exception("Resending order %s failed", order.id)

    async def _sync_user(self, user_id: int) -> None:
        ib = self.connections.get(user_id) or self._connect(user_id)
        if not ib.isConnected():
            try:
                host, port = gateway_address(user_id)
                await ib.connectAsync(host, port, clientId=1, timeout=5)
            except (OSError, TimeoutError) as error:
                # The Gateway may still be starting or logging in; the next round tries again.
                self._write_status(user_id, f"Gateway not ready ({type(error).__name__})")
                return
            logger.info("Connected to the Gateway of user %s", user_id)
        _follow_positions(ib)
        current = _current_snapshot(ib)
        if current is not None:
            store.save_snapshot(self._pool, user_id, current)
        self._write_status(user_id)
        for trade in ib.trades():
            self.record(trade)

    def handle(self, command: dict) -> None:
        order = orders.get_order(self._pool, command["order_id"])
        if command["type"] == "place":
            self.submit(order)
        else:
            self.cancel(order)

    def submit(self, order: orders.Order) -> None:
        # A repeated command finds its order already sent.
        if order.status != "pending_submit" or order.ib_order_id is not None:
            return
        ib = self.connections.get(order.user_id)
        if (datetime.now(UTC) - order.created_at).total_seconds() > _SEND_WITHIN_SECONDS:
            self._update(order.id, reported="rejected", error="The order was not sent in time")
        elif ib is None or not ib.isConnected():
            self._update(order.id, reported="rejected", error="The IBKR Gateway is not connected")
        else:
            action = order.side.upper()
            ib_order = (
                MarketOrder(action, order.quantity)
                if order.order_type == "market"
                else LimitOrder(action, order.quantity, order.limit_price)
            )
            # Without an explicit time in force, IBKR applies its preset and warns, which
            # ib_async reports as a cancellation even though the order goes on to fill.
            ib_order.tif = "DAY"
            ib_order.orderRef = f"{_ORDER_REF}{order.id}"
            self.record(ib.placeOrder(Stock(order.symbol, "SMART", "USD"), ib_order))

    def cancel(self, order: orders.Order) -> None:
        if order.status in orders.FINAL_STATUSES:
            return
        if order.ib_order_id is None:
            self._update(order.id, reported="cancelled")
            return
        ib = self.connections.get(order.user_id)
        if ib is not None and ib.isConnected():
            trade = next((t for t in ib.trades() if t.order.orderId == order.ib_order_id), None)
            if trade is not None:
                ib.cancelOrder(trade.order)

    def record(self, trade: Trade) -> None:
        """Apply what the Gateway knows about one of our orders: its status and every fill."""

        if not trade.order.orderRef.startswith(_ORDER_REF):
            return
        reported = _REPORTED_STATUS.get(trade.orderStatus.status)
        error = None
        if reported in ("cancelled", "rejected"):
            error = next((entry.message for entry in reversed(trade.log) if entry.errorCode), None)
        self._update(
            int(trade.order.orderRef.removeprefix(_ORDER_REF)),
            reported=reported,
            fills=[
                orders.Fill(
                    exec_id=fill.execution.execId,
                    shares=fill.execution.shares,
                    price=fill.execution.price,
                    # The commission arrives in a later report than the execution.
                    commission=fill.commissionReport.commission
                    if fill.commissionReport.execId
                    else None,
                    executed_at=fill.execution.time,
                )
                for fill in trade.fills
            ],
            ib_order_id=trade.order.orderId or None,
            error=error,
        )

    def publish_changed(self) -> None:
        """Publish the current balances and positions of every user IBKR updated."""

        changed, self._changed = self._changed, set()
        for user_id in changed:
            ib = self.connections.get(user_id)
            try:
                current = _current_snapshot(ib) if ib is not None and ib.isConnected() else None
                if current is not None:
                    messaging.publish_account(self._producer, user_id, current)
            except Exception:
                logger.exception("Publishing the account of user %s failed", user_id)

    def _update(self, order_id: int, **report) -> None:
        changed = orders.apply_report(self._pool, order_id, **report)
        if changed is not None:
            logger.info("Order %s is %s", changed.id, changed.status)
            messaging.publish_event(self._producer, changed)

    def _connect(self, user_id: int) -> IB:
        ib = self._new_connection()
        # Each event carries the whole trade, so any of them brings its order up to date.
        ib.orderStatusEvent += self.record
        ib.execDetailsEvent += lambda trade, _fill: self.record(trade)
        ib.commissionReportEvent += lambda trade, _fill, _report: self.record(trade)
        # Bursts of updates collapse into one publish.
        ib.accountValueEvent += lambda _value: self._changed.add(user_id)
        ib.updatePortfolioEvent += lambda _item: self._changed.add(user_id)
        ib.pnlSingleEvent += lambda _pnl: self._changed.add(user_id)
        self.connections[user_id] = ib
        return ib

    async def _sync_forever(self) -> None:
        while True:
            await self.sync_round()
            await asyncio.sleep(_ROUND_SECONDS)

    async def _publish_forever(self) -> None:
        while True:
            await asyncio.sleep(_PUBLISH_SECONDS)
            self.publish_changed()

    async def _consume(self, commands: Consumer) -> None:
        while True:
            # Polling blocks, so it waits on a thread while the Gateways' events keep flowing.
            message = await asyncio.to_thread(commands.poll, 1.0)
            if message is None:
                continue
            if message.error():
                logger.warning("Kafka: %s", message.error())
                continue
            try:
                self.handle(json.loads(message.value()))
            except Exception:
                # Committed anyway: redelivering a command that fails would block every later
                # command in its partition. Reconciliation still settles a placed order unsent.
                logger.exception("Handling the command %s failed", message.value())
            # Committed only once handled, so a crash redelivers the command; handling twice is
            # harmless because an order is sent only while it is still unsent.
            commands.commit(message=message, asynchronous=False)

    def _write_status(self, user_id: int, error: str | None = None) -> None:
        status = {"connected": error is None, "error": error}
        self._redis.set(status_key(user_id), json.dumps(status), ex=_STATUS_SECONDS)


def _follow_positions(ib: IB) -> None:
    """Keep a P&L subscription, which IBKR updates about every second, for each held position.

    The portfolio's own prices update only every few minutes. A dropped connection ends every
    subscription, so the round after reconnecting makes them again.
    """

    account_id = ib.managedAccounts()[0]
    held = {item.contract.conId for item in ib.portfolio(account_id) if item.position}
    followed = {pnl.conId for pnl in ib.pnlSingle(account_id)}
    for con_id in held - followed:
        ib.reqPnLSingle(account_id, "", con_id)
    for con_id in followed - held:
        ib.cancelPnLSingle(account_id, "", con_id)


def _current_snapshot(ib: IB) -> store.Snapshot | None:
    account_id = ib.managedAccounts()[0]
    return snapshot(
        account_id,
        ib.accountValues(account_id),
        ib.portfolio(account_id),
        ib.pnlSingle(account_id),
    )


def snapshot(
    account_id: str,
    values: Iterable[AccountValue],
    portfolio: Iterable[PortfolioItem],
    pnl: Iterable[PnLSingle],
) -> store.Snapshot | None:
    """Balances in the account's base currency, which net liquidation is reported in, and positions.

    None until IBKR has sent the account's values, which a just-logged-in Gateway takes a moment
    to do. A position takes its value and P&L from its P&L subscription once that has reported for
    the position's current size; until then, from the portfolio.
    """

    values = list(values)
    currency = next((value.currency for value in values if value.tag == "NetLiquidation"), None)
    if currency is None:
        return None
    balances = {
        _BALANCE_TAGS[value.tag]: float(value.value)
        for value in values
        if value.tag in _BALANCE_TAGS and value.currency == currency
    }
    live = {entry.conId: entry for entry in pnl}
    return store.Snapshot(
        store.Balances(
            account_id=account_id,
            currency=currency,
            net_liquidation=balances.get("net_liquidation"),
            cash=balances.get("cash"),
            buying_power=balances.get("buying_power"),
            available_funds=balances.get("available_funds"),
        ),
        tuple(
            _position(item, live.get(item.contract.conId)) for item in portfolio if item.position
        ),
    )


def _position(item: PortfolioItem, pnl: PnLSingle | None) -> store.Position:
    if pnl is None or isnan(pnl.value) or pnl.position != item.position:
        return store.Position(
            symbol=item.contract.symbol,
            quantity=item.position,
            average_cost=item.averageCost,
            market_price=item.marketPrice,
            market_value=item.marketValue,
            unrealized_pnl=item.unrealizedPNL,
        )
    return store.Position(
        symbol=item.contract.symbol,
        quantity=item.position,
        average_cost=item.averageCost,
        market_price=pnl.value / pnl.position,
        market_value=pnl.value,
        unrealized_pnl=pnl.unrealizedPnL,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    pool = build_connection_pool(settings)
    pool.open(wait=True, timeout=settings.postgres_pool_timeout_seconds)
    messaging.ensure_topics(settings)
    broker = Broker(pool, redis_client.connect(settings), messaging.producer(settings))
    commands = messaging.consumer(settings, "tragent-broker", [messaging.COMMANDS])
    asyncio.run(broker.run(commands))


if __name__ == "__main__":
    main()
