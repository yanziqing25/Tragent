"""Research strategies deployed to paper trading.

A deployment starts flat and acts only on signals that arrive after it. Each trading morning it
reads its strategy's signal on the latest completed daily bar, moves its own tranche count by
that signal, and orders the difference between its target and the shares its own fills hold, so
a user's other holdings of the symbol are never touched. A market order placed before the open
fills at the open, as the backtest assumes. One run per session, and an idempotency key derived
from the session, mean a repeated run that day finds what it already did.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from math import floor
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
from confluent_kafka import Producer
from psycopg.types.json import Jsonb

from tragent.db import DatabasePool
from tragent.quant.market_data import MarketDataProvider, latest_completed_us_market_session
from tragent.quant.models import StrategyName, build_strategy_config
from tragent.quant.strategies import generate_trade_signals, required_warmup_bars
from tragent.research.reporting import ResearchReportView
from tragent.trading import messaging, orders, store

logger = logging.getLogger(__name__)

Status = Literal["active", "paused", "stopped"]
RUN_TIME = (9, 0)
EASTERN = ZoneInfo("America/New_York")
_CALENDAR = xcals.get_calendar("XNYS")

_COLUMNS = """
    d.id, d.user_id, d.thread_id, d.symbol, d.strategy_name, d.parameters, d.position_tranches,
    d.capital::float8, d.tranches, d.status, d.created_at
"""
# Signed shares and net cash of a deployment's fills.
_FILLS = """
    LEFT JOIN LATERAL (
        SELECT
            coalesce(sum(CASE o.side WHEN 'buy' THEN e.shares ELSE -e.shares END), 0)::float8
                AS held_shares,
            coalesce(sum(
                CASE o.side WHEN 'buy' THEN -e.shares * e.price ELSE e.shares * e.price END
                - coalesce(e.commission, 0)
            ), 0)::float8 AS net_cash
        FROM executions e JOIN orders o ON o.id = e.order_id
        WHERE o.deployment_id = d.id
    ) fills ON true
"""


class NotDeployableError(Exception):
    """The research task or request cannot become a deployment; the message says why."""


@dataclass(frozen=True, slots=True)
class Deployment:
    id: int
    user_id: int
    thread_id: str
    symbol: str
    strategy_name: StrategyName
    parameters: dict[str, int | float]
    position_tranches: int
    capital: float
    tranches: int
    status: Status
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DeploymentView(Deployment):
    held_shares: float
    # The fills' cash in and out plus the held shares at the last synced price.
    pnl: float | None


@dataclass(frozen=True, slots=True)
class Run:
    session: date
    signal: int | None
    tranches: int
    target_shares: int | None
    held_shares: float
    order_id: int | None
    note: str | None
    ran_at: datetime


def deploy(
    pool: DatabasePool,
    user_id: int,
    thread_id: str,
    report: ResearchReportView,
    position_tranches: int,
    capital: float,
) -> Deployment:
    """Deploy the strategy a research task selected and its Critic accepted."""

    if report.critic_review_status != "accepted" or report.selected_candidate_id is None:
        raise NotDeployableError("Only a strategy the Critic accepted can be deployed")
    if report.bar_interval != "1d":
        raise NotDeployableError("Only daily strategies can be deployed")
    candidate = next(c for c in report.candidates if c.id == report.selected_candidate_id)
    available = _available_funds(pool, user_id)
    if capital > available:
        raise NotDeployableError(f"The capital exceeds the available funds of {available:,.2f}")
    with pool.connection() as connection:
        row = connection.execute(
            f"""
            INSERT INTO deployments AS d (
                user_id, thread_id, symbol, strategy_name, parameters, position_tranches, capital
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING {_COLUMNS}
            """,
            (
                user_id,
                thread_id,
                report.symbol,
                candidate.strategy_name,
                Jsonb(candidate.parameters),
                position_tranches,
                capital,
            ),
        ).fetchone()
    return Deployment(**row)


def list_deployments(pool: DatabasePool, user_id: int) -> list[DeploymentView]:
    with pool.connection() as connection:
        rows = connection.execute(
            f"""
            SELECT {_COLUMNS}, fills.held_shares,
                fills.net_cash + CASE WHEN fills.held_shares = 0 THEN 0
                    ELSE fills.held_shares * p.market_price::float8 END AS pnl
            FROM deployments d
            {_FILLS}
            LEFT JOIN positions p ON p.user_id = d.user_id AND p.symbol = d.symbol
            WHERE d.user_id = %s ORDER BY d.created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [DeploymentView(**row) for row in rows]


def get_deployment(pool: DatabasePool, deployment_id: int) -> Deployment | None:
    with pool.connection() as connection:
        row = connection.execute(
            f"SELECT {_COLUMNS} FROM deployments d WHERE d.id = %s", (deployment_id,)
        ).fetchone()
    return Deployment(**row) if row else None


def list_runs(pool: DatabasePool, deployment_id: int) -> list[Run]:
    with pool.connection() as connection:
        rows = connection.execute(
            """
            SELECT session, signal, tranches, target_shares, held_shares::float8, order_id, note,
                ran_at
            FROM deployment_runs WHERE deployment_id = %s ORDER BY session DESC
            """,
            (deployment_id,),
        ).fetchall()
    return [Run(**row) for row in rows]


def set_status(pool: DatabasePool, deployment_id: int, status: Status) -> None:
    with pool.connection() as connection:
        connection.execute(
            "UPDATE deployments SET status = %s WHERE id = %s", (status, deployment_id)
        )


def stop(
    pool: DatabasePool, producer: Producer, deployment: Deployment, *, close_position: bool
) -> orders.Order | None:
    """Stop the deployment for good, selling the shares its fills hold if asked."""

    set_status(pool, deployment.id, "stopped")
    held = _held_shares(pool, deployment.id)
    if not close_position or held <= 0:
        return None
    return _place(pool, producer, deployment, "sell", held, uuid4())


def run_all(
    pool: DatabasePool, producer: Producer, provider: MarketDataProvider, session: date
) -> None:
    """Run every active deployment for this session, if the market opens that day."""

    if not _CALENDAR.is_session(session):
        return
    with pool.connection() as connection:
        rows = connection.execute(
            f"SELECT {_COLUMNS} FROM deployments d WHERE d.status = 'active' ORDER BY d.id"
        ).fetchall()
    for row in rows:
        deployment = Deployment(**row)
        try:
            run(pool, producer, provider, deployment, session)
        except Exception:
            # One deployment's failure, such as a missing download, must not stop the others.
            logger.exception("Running deployment %s failed", deployment.id)


def run(
    pool: DatabasePool,
    producer: Producer,
    provider: MarketDataProvider,
    deployment: Deployment,
    session: date,
) -> None:
    """Act on the latest completed bar's signal once for this session."""

    held = _held_shares(pool, deployment.id)
    latest = latest_completed_us_market_session()
    config = build_strategy_config(deployment.strategy_name, deployment.parameters)
    # Calendar days that comfortably hold the strategy's warm-up in trading days.
    start = latest - timedelta(days=2 * required_warmup_bars(config) + 30)
    bars = provider.get_daily_bars(deployment.symbol, start, latest).bars
    if bars.index[-1].date() != latest:
        _record(
            pool,
            deployment,
            session,
            tranches=deployment.tranches,
            held=held,
            note=f"No daily bar for {latest} yet",
        )
        return
    signal = int(generate_trade_signals(bars["close"], config).iloc[-1])
    tranches = min(deployment.position_tranches, max(0, deployment.tranches + signal))
    price = float(bars["close"].iloc[-1])
    target = floor(deployment.capital * tranches / deployment.position_tranches / price)
    change = target - held
    order, note = None, None
    if change > 0 and change * price > _available_funds(pool, deployment.user_id):
        note = "Not enough available funds"
    elif change:
        key = uuid5(NAMESPACE_URL, f"tragent:deployment:{deployment.id}:{session}")
        order = _place(
            pool, producer, deployment, "buy" if change > 0 else "sell", abs(change), key
        )
    _record(
        pool,
        deployment,
        session,
        signal=signal,
        tranches=tranches,
        target=target,
        held=held,
        order_id=order.id if order else None,
        note=note,
    )


def next_run(now: datetime) -> datetime:
    """The next morning run at or after ``now``."""

    local = now.astimezone(EASTERN)
    day = local.date()
    if local.time() > time(*RUN_TIME):
        day += timedelta(days=1)
    session = _CALENDAR.date_to_session(day, direction="next")
    return datetime(session.year, session.month, session.day, *RUN_TIME, tzinfo=EASTERN)


def _place(
    pool: DatabasePool,
    producer: Producer,
    deployment: Deployment,
    side: Literal["buy", "sell"],
    quantity: float,
    key: UUID,
) -> orders.Order:
    request = orders.OrderRequest(
        client_order_id=key,
        symbol=deployment.symbol,
        side=side,
        order_type="market",
        quantity=int(quantity),
    )
    order, created = orders.create_order(pool, deployment.user_id, request, deployment.id)
    if created:
        messaging.send_command(producer, order, "place")
    return order


def _record(
    pool: DatabasePool,
    deployment: Deployment,
    session: date,
    *,
    tranches: int,
    held: float,
    signal: int | None = None,
    target: int | None = None,
    order_id: int | None = None,
    note: str | None = None,
) -> None:
    """Save the session's run and the tranches it moved to, unless the session already ran."""

    with pool.connection() as connection, connection.transaction():
        inserted = connection.execute(
            """
            INSERT INTO deployment_runs (
                deployment_id, session, signal, tranches, target_shares, held_shares, order_id,
                note
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (deployment_id, session) DO NOTHING
            RETURNING session
            """,
            (deployment.id, session, signal, tranches, target, held, order_id, note),
        ).fetchone()
        if inserted is not None:
            connection.execute(
                "UPDATE deployments SET tranches = %s WHERE id = %s", (tranches, deployment.id)
            )


def _held_shares(pool: DatabasePool, deployment_id: int) -> float:
    with pool.connection() as connection:
        row = connection.execute(
            f"SELECT fills.held_shares FROM deployments d {_FILLS} WHERE d.id = %s",
            (deployment_id,),
        ).fetchone()
    return row["held_shares"]


def _available_funds(pool: DatabasePool, user_id: int) -> float:
    overview = store.account_overview(pool, user_id)
    if overview is None or overview.snapshot is None:
        return 0
    return overview.snapshot.balances.available_funds or 0


__all__ = [
    "EASTERN",
    "RUN_TIME",
    "Deployment",
    "DeploymentView",
    "NotDeployableError",
    "Run",
    "deploy",
    "get_deployment",
    "list_deployments",
    "list_runs",
    "next_run",
    "run",
    "run_all",
    "set_status",
    "stop",
]
