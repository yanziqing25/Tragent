"""Orders: idempotent creation, the status machine, and the executions that fill them.

A fill is a fact and a broker status report only a claim. So an order is filled once its
executions add up to its quantity, whatever was reported, while a reported end status (cancelled
or rejected) holds against any later report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, model_validator

from tragent.db import DatabasePool

Status = Literal[
    "pending_submit", "submitted", "partially_filled", "filled", "cancelled", "rejected"
]
FINAL_STATUSES = frozenset({"filled", "cancelled", "rejected"})

_COLUMNS = """
    id, user_id, client_order_id, symbol, side, order_type, quantity::float8, limit_price::float8,
    status, filled_quantity::float8, avg_fill_price::float8, ib_order_id, error, deployment_id,
    created_at, updated_at
"""


class NoAccountError(Exception):
    """The user has no linked broker account to send orders to."""


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Chosen by the client once per order, so a retried request finds the order it already made.
    client_order_id: UUID
    symbol: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, to_upper=True, pattern=r"^[A-Za-z][A-Za-z.]{0,9}$"
        ),
    ]
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"]
    quantity: int = Field(gt=0)
    limit_price: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def only_limit_orders_have_a_price(self) -> Self:
        if (self.order_type == "limit") != (self.limit_price is not None):
            raise ValueError("a limit order needs a limit price, and a market order takes none")
        return self


@dataclass(frozen=True, slots=True)
class Order:
    id: int
    user_id: int
    client_order_id: UUID
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"]
    quantity: float
    limit_price: float | None
    status: Status
    filled_quantity: float
    avg_fill_price: float | None
    ib_order_id: int | None
    error: str | None
    # The strategy deployment that placed it, or None for a manual order.
    deployment_id: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Fill:
    exec_id: str
    shares: float
    price: float
    commission: float | None
    executed_at: datetime


_ORDER_JSON = TypeAdapter(Order)


def order_json(order: Order) -> bytes:
    """The order as the REST API returns it, for pushing to the browser."""

    return _ORDER_JSON.dump_json(order)


def next_status(current: Status, reported: Status | None, filled: float, quantity: float) -> Status:
    if filled >= quantity:
        return "filled"
    if current in FINAL_STATUSES:
        return current
    if reported in ("cancelled", "rejected"):
        return reported
    if filled > 0:
        return "partially_filled"
    if reported == "submitted":
        return "submitted"
    return current


def create_order(
    pool: DatabasePool, user_id: int, request: OrderRequest, deployment_id: int | None = None
) -> tuple[Order, bool]:
    """Save a new order, or return the one this idempotency key already made, and whether it is new."""

    with pool.connection() as connection, connection.transaction():
        linked = connection.execute(
            "SELECT 1 FROM broker_accounts WHERE user_id = %s", (user_id,)
        ).fetchone()
        if linked is None:
            raise NoAccountError
        row = connection.execute(
            f"""
            INSERT INTO orders (
                user_id, client_order_id, symbol, side, order_type, quantity, limit_price,
                deployment_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, client_order_id) DO NOTHING
            RETURNING {_COLUMNS}
            """,
            (
                user_id,
                request.client_order_id,
                request.symbol,
                request.side,
                request.order_type,
                request.quantity,
                request.limit_price,
                deployment_id,
            ),
        ).fetchone()
        if row is not None:
            return Order(**row), True
        row = connection.execute(
            f"SELECT {_COLUMNS} FROM orders WHERE user_id = %s AND client_order_id = %s",
            (user_id, request.client_order_id),
        ).fetchone()
    return Order(**row), False


def get_order(pool: DatabasePool, order_id: int) -> Order | None:
    with pool.connection() as connection:
        row = connection.execute(
            f"SELECT {_COLUMNS} FROM orders WHERE id = %s", (order_id,)
        ).fetchone()
    return Order(**row) if row else None


def list_orders(pool: DatabasePool, user_id: int, limit: int) -> list[Order]:
    with pool.connection() as connection:
        rows = connection.execute(
            f"SELECT {_COLUMNS} FROM orders WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        ).fetchall()
    return [Order(**row) for row in rows]


def unsent_orders(pool: DatabasePool, older_than_seconds: float) -> list[Order]:
    """Orders saved this long ago that never reached a Gateway."""

    with pool.connection() as connection:
        rows = connection.execute(
            f"""
            SELECT {_COLUMNS} FROM orders
            WHERE status = 'pending_submit' AND ib_order_id IS NULL
                AND created_at < now() - make_interval(secs => %s)
            ORDER BY id
            """,
            (older_than_seconds,),
        ).fetchall()
    return [Order(**row) for row in rows]


def apply_report(
    pool: DatabasePool,
    order_id: int,
    *,
    reported: Status | None = None,
    fills: Sequence[Fill] = (),
    ib_order_id: int | None = None,
    error: str | None = None,
) -> Order | None:
    """Record what the broker reported about an order; return the order if that changed it.

    Reports arrive more than once, from live events and from reconciliation, so executions are
    keyed by the broker's execution ID and the status moves only as ``next_status`` allows.
    """

    with pool.connection() as connection, connection.transaction():
        order = connection.execute(
            f"SELECT {_COLUMNS} FROM orders WHERE id = %s FOR UPDATE", (order_id,)
        ).fetchone()
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO executions (exec_id, order_id, shares, price, commission, executed_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (exec_id) DO UPDATE SET commission = EXCLUDED.commission
                """,
                [
                    (
                        fill.exec_id,
                        order_id,
                        fill.shares,
                        fill.price,
                        fill.commission,
                        fill.executed_at,
                    )
                    for fill in fills
                ],
            )
        totals = connection.execute(
            """
            SELECT coalesce(sum(shares), 0)::float8 AS filled,
                (sum(shares * price) / nullif(sum(shares), 0))::float8 AS average
            FROM executions WHERE order_id = %s
            """,
            (order_id,),
        ).fetchone()
        status = next_status(order["status"], reported, totals["filled"], order["quantity"])
        row = connection.execute(
            f"""
            UPDATE orders SET
                status = %(status)s, filled_quantity = %(filled)s, avg_fill_price = %(average)s,
                ib_order_id = coalesce(ib_order_id, %(ib_order_id)s),
                error = coalesce(%(error)s, error), updated_at = now()
            WHERE id = %(id)s AND (status, filled_quantity, avg_fill_price, ib_order_id, error)
                IS DISTINCT FROM (
                    %(status)s, %(filled)s::numeric, %(average)s::numeric,
                    coalesce(ib_order_id, %(ib_order_id)s), coalesce(%(error)s, error)
                )
            RETURNING {_COLUMNS}
            """,
            {
                "id": order_id,
                "status": status,
                "filled": totals["filled"],
                "average": totals["average"],
                "ib_order_id": ib_order_id,
                "error": error,
            },
        ).fetchone()
    return Order(**row) if row else None


__all__ = [
    "FINAL_STATUSES",
    "Fill",
    "NoAccountError",
    "Order",
    "OrderRequest",
    "Status",
    "apply_report",
    "create_order",
    "get_order",
    "list_orders",
    "next_status",
    "order_json",
    "unsent_orders",
]
