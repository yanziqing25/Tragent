"""Orders: the status machine, the broker placing and recording them, and the order API."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import fakeredis
import pytest
from fastapi.testclient import TestClient
from ib_async import CommissionReport, Execution, Fill, OrderStatus, Stock, Trade, TradeLogEntry
from ib_async import Order as IBOrder
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

import tragent.api.app as app_module
from tests.conftest import FakeConsumer, FakeMessage
from tragent import redis_client
from tragent.api.auth import create_access_token
from tragent.config import Settings
from tragent.trading import broker, messaging, orders


def _order(**changes: object) -> orders.Order:
    now = datetime.now(UTC)
    order = orders.Order(
        id=12,
        user_id=5,
        client_order_id=uuid4(),
        symbol="AAPL",
        side="buy",
        order_type="limit",
        quantity=10,
        limit_price=150.0,
        status="pending_submit",
        filled_quantity=0,
        avg_fill_price=None,
        ib_order_id=None,
        error=None,
        deployment_id=None,
        created_at=now,
        updated_at=now,
    )
    return replace(order, **changes)


@pytest.mark.parametrize(
    ("current", "reported", "filled", "expected"),
    [
        ("pending_submit", None, 0, "pending_submit"),
        ("pending_submit", "submitted", 0, "submitted"),
        ("submitted", "submitted", 4, "partially_filled"),
        ("partially_filled", "cancelled", 4, "cancelled"),
        ("submitted", "rejected", 0, "rejected"),
        # A cancellation claim cannot outlast fills that complete the order.
        ("cancelled", None, 10, "filled"),
        # An end status holds against later reports.
        ("cancelled", "submitted", 4, "cancelled"),
        ("filled", "cancelled", 10, "filled"),
    ],
)
def test_next_status(current, reported, filled, expected) -> None:
    assert orders.next_status(current, reported, filled, 10) == expected


def test_order_requests_need_a_price_only_for_limit_orders() -> None:
    def request(**fields: object) -> orders.OrderRequest:
        return orders.OrderRequest(
            client_order_id=uuid4(), side="buy", quantity=1, **{"symbol": "brk.b", **fields}
        )

    assert request(order_type="market").symbol == "BRK.B"
    assert request(order_type="limit", limit_price=10).limit_price == 10
    for invalid in (
        {"order_type": "market", "limit_price": 10},
        {"order_type": "limit"},
        {"order_type": "market", "symbol": "AAPL;"},
    ):
        with pytest.raises(ValidationError):
            request(**invalid)


class _FakeGateway:
    def __init__(self) -> None:
        self.placed: list[Trade] = []
        self.cancelled: list[IBOrder] = []

    def isConnected(self) -> bool:
        return True

    def placeOrder(self, contract: Stock, order: IBOrder) -> Trade:
        order.orderId = 900 + len(self.placed)
        trade = Trade(contract, order, OrderStatus(status="PendingSubmit"))
        self.placed.append(trade)
        return trade

    def trades(self) -> list[Trade]:
        return self.placed

    def cancelOrder(self, order: IBOrder) -> None:
        self.cancelled.append(order)


@pytest.fixture
def reports(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Every report the broker applies; each one changes its order."""

    applied: list[dict] = []

    def apply_report(_pool: object, order_id: int, **report: object) -> orders.Order:
        applied.append({"order_id": order_id, **report})
        return _order(id=order_id)

    monkeypatch.setattr(broker.orders, "apply_report", apply_report)
    return applied


def test_the_broker_sends_an_order_once_with_its_reference(reports, kafka) -> None:
    gateway = _FakeGateway()
    runner = broker.Broker(None, None, kafka.producer)
    runner.connections[5] = gateway

    runner.submit(_order())
    runner.submit(_order(ib_order_id=900))

    (trade,) = gateway.placed
    assert (trade.order.action, trade.order.orderType, trade.order.lmtPrice) == ("BUY", "LMT", 150)
    assert (trade.order.tif, trade.order.orderRef) == ("DAY", "tragent:12")
    assert reports == [
        {"order_id": 12, "reported": None, "fills": [], "ib_order_id": 900, "error": None}
    ]
    assert [message[:2] for message in kafka.producer.messages] == [(messaging.ORDER_EVENTS, "5")]


def test_the_broker_rejects_late_or_unreachable_orders(reports, kafka) -> None:
    runner = broker.Broker(None, None, kafka.producer)

    runner.submit(_order(id=1, created_at=datetime.now(UTC) - timedelta(minutes=5)))
    runner.submit(_order(id=2))

    assert [(r["order_id"], r["reported"], r["error"]) for r in reports] == [
        (1, "rejected", "The order was not sent in time"),
        (2, "rejected", "The IBKR Gateway is not connected"),
    ]


def test_cancelling_reaches_the_gateway_only_for_sent_orders(reports, kafka) -> None:
    gateway = _FakeGateway()
    runner = broker.Broker(None, None, kafka.producer)
    runner.connections[5] = gateway
    runner.submit(_order())
    reports.clear()

    runner.cancel(_order(ib_order_id=900, status="submitted"))
    runner.cancel(_order(id=13))
    runner.cancel(_order(id=14, status="filled"))

    assert [order.orderId for order in gateway.cancelled] == [900]
    assert reports == [{"order_id": 13, "reported": "cancelled"}]


def test_recording_a_trade_maps_its_status_fills_and_error(reports, kafka) -> None:
    runner = broker.Broker(None, None, kafka.producer)
    executed_at = datetime(2026, 10, 23, 15, tzinfo=UTC)
    trade = Trade(
        Stock("AAPL"),
        IBOrder(orderId=901, orderRef="tragent:12"),
        OrderStatus(status="Cancelled"),
        fills=[
            Fill(
                Stock("AAPL"),
                Execution(execId="e1", time=executed_at, shares=4, price=150.5),
                CommissionReport(execId="e1", commission=1.0),
                executed_at,
            ),
            Fill(
                Stock("AAPL"),
                Execution(execId="e2", time=executed_at, shares=6, price=150.0),
                CommissionReport(),
                executed_at,
            ),
        ],
        log=[TradeLogEntry(executed_at, "Cancelled", "Order canceled by user", 202)],
    )

    runner.record(trade)
    runner.record(Trade(Stock("AAPL"), IBOrder(orderId=3, orderRef="manual"), OrderStatus()))

    (report,) = reports
    assert report["reported"] == "cancelled"
    assert report["error"] == "Order canceled by user"
    assert report["fills"] == [
        orders.Fill("e1", 4, 150.5, 1.0, executed_at),
        orders.Fill("e2", 6, 150.0, None, executed_at),
    ]


def test_handled_commands_are_committed(monkeypatch: pytest.MonkeyPatch, kafka) -> None:
    handled: list[dict] = []
    runner = broker.Broker(None, None, kafka.producer)
    monkeypatch.setattr(runner, "handle", handled.append)
    commands = FakeConsumer()
    message = FakeMessage("5", json.dumps({"type": "place", "order_id": 12}).encode())
    commands.messages.append(message)

    async def consume_until_committed() -> None:
        task = asyncio.create_task(runner._consume(commands))
        while not commands.committed:
            await asyncio.sleep(0.01)
        task.cancel()

    asyncio.run(consume_until_committed())

    assert handled == [{"type": "place", "order_id": 12}]
    assert commands.committed == [message]


def test_a_failing_command_is_logged_and_skipped(monkeypatch: pytest.MonkeyPatch, kafka) -> None:
    handled: list[int] = []

    def handle(command: dict) -> None:
        if command["order_id"] == 1:
            raise ValueError("unexpected order")
        handled.append(command["order_id"])

    runner = broker.Broker(None, None, kafka.producer)
    monkeypatch.setattr(runner, "handle", handle)
    commands = FakeConsumer()
    for order_id in (1, 2):
        commands.messages.append(
            FakeMessage("5", json.dumps({"type": "place", "order_id": order_id}).encode())
        )

    async def consume_until_both_committed() -> None:
        task = asyncio.create_task(runner._consume(commands))
        while len(commands.committed) < 2:
            await asyncio.sleep(0.01)
        task.cancel()

    asyncio.run(consume_until_both_committed())

    assert handled == [2]


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, settings: Settings, kafka):
    server = fakeredis.FakeServer()
    monkeypatch.setattr(
        redis_client,
        "connect",
        lambda _s: fakeredis.FakeRedis(server=server, decode_responses=True),
    )
    monkeypatch.setattr(
        redis_client, "connect_async", lambda _s: fakeredis.FakeAsyncRedis(server=server)
    )
    pool = SimpleNamespace(open=lambda **_kwargs: None, close=lambda: None)
    monkeypatch.setattr(app_module, "build_connection_pool", lambda _settings: pool)
    monkeypatch.setattr(app_module, "setup_persistence", lambda *_args: None)
    headers = {"Authorization": f"Bearer {create_access_token(5, settings)}"}
    with TestClient(app_module.create_app(settings), headers=headers) as client:
        yield client


def test_placing_an_order_is_idempotent_and_needs_a_linked_account(
    monkeypatch: pytest.MonkeyPatch, api, kafka, settings: Settings
) -> None:
    saved: dict[object, orders.Order] = {}

    def create_order(_pool, user_id, request):
        if user_id != 5:
            raise orders.NoAccountError
        if request.client_order_id in saved:
            return saved[request.client_order_id], False
        saved[request.client_order_id] = _order(client_order_id=request.client_order_id)
        return saved[request.client_order_id], True

    monkeypatch.setattr(orders, "create_order", create_order)
    body = {
        "client_order_id": str(uuid4()),
        "symbol": "aapl",
        "side": "buy",
        "order_type": "limit",
        "quantity": 10,
        "limit_price": 150,
    }

    first = api.post("/api/trading/orders", json=body)
    repeated = api.post("/api/trading/orders", json=body)
    api.headers["Authorization"] = f"Bearer {create_access_token(6, settings)}"
    unlinked = api.post("/api/trading/orders", json={**body, "client_order_id": str(uuid4())})

    assert (first.status_code, repeated.status_code, unlinked.status_code) == (201, 200, 409)
    assert first.json()["id"] == repeated.json()["id"] == 12
    assert kafka.producer.messages == [
        (messaging.COMMANDS, "5", json.dumps({"type": "place", "order_id": 12}))
    ]


def test_only_open_orders_of_ones_own_can_be_cancelled(
    monkeypatch: pytest.MonkeyPatch, api, kafka
) -> None:
    known = {
        12: _order(status="submitted"),
        13: _order(id=13, status="filled"),
        14: _order(id=14, user_id=6),
    }
    monkeypatch.setattr(orders, "get_order", lambda _pool, order_id: known.get(order_id))

    responses = [
        api.delete(f"/api/trading/orders/{order_id}").status_code for order_id in (12, 13, 14, 99)
    ]

    assert responses == [202, 409, 404, 404]
    assert kafka.producer.messages == [
        (messaging.COMMANDS, "5", json.dumps({"type": "cancel", "order_id": 12}))
    ]


def test_the_websocket_needs_a_token_and_relays_the_users_messages_by_type(
    api, kafka, settings: Settings
) -> None:
    with pytest.raises(WebSocketDisconnect):
        with api.websocket_connect("/api/trading/ws?token=bad"):
            pass

    token = create_access_token(5, settings)
    with api.websocket_connect(f"/api/trading/ws?token={token}") as socket:
        (events,) = kafka.consumers
        events.messages.append(FakeMessage("6", orders.order_json(_order(id=1, user_id=6))))
        events.messages.append(FakeMessage("5", orders.order_json(_order(id=2))))
        snapshot = b'{"balances": null, "positions": [], "synced_at": "2026-10-23T17:00:00Z"}'
        events.messages.append(FakeMessage("5", snapshot, messaging.ACCOUNT_SNAPSHOTS))

        order = json.loads(socket.receive_text())
        account = json.loads(socket.receive_text())

    assert (order["type"], order["data"]["id"]) == ("order", 2)
    assert account == {"type": "account", "data": json.loads(snapshot)}
