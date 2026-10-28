"""Broker accounts and the balances and positions last synced from them."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from tragent.db import DatabasePool


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    quantity: float
    average_cost: float
    market_price: float | None
    market_value: float | None
    unrealized_pnl: float | None


@dataclass(frozen=True, slots=True)
class Balances:
    account_id: str
    currency: str
    net_liquidation: float | None
    cash: float | None
    buying_power: float | None
    available_funds: float | None


@dataclass(frozen=True, slots=True)
class Snapshot:
    balances: Balances
    positions: tuple[Position, ...]


@dataclass(frozen=True, slots=True)
class Overview:
    username: str
    snapshot: Snapshot | None
    synced_at: datetime | None


def save_account(pool: DatabasePool, user_id: int, username: str, password: str) -> None:
    """Store new credentials and drop the old snapshot, which belongs to the old login."""

    with pool.connection() as connection, connection.transaction():
        connection.execute(
            """
            INSERT INTO broker_accounts (user_id, ibkr_username, ibkr_password)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                ibkr_username = EXCLUDED.ibkr_username,
                ibkr_password = EXCLUDED.ibkr_password,
                account_id = NULL, currency = NULL, net_liquidation = NULL, cash = NULL,
                buying_power = NULL, available_funds = NULL, synced_at = NULL
            """,
            (user_id, username, password),
        )
        connection.execute("DELETE FROM positions WHERE user_id = %s", (user_id,))


def delete_account(pool: DatabasePool, user_id: int) -> bool:
    with pool.connection() as connection:
        deleted = connection.execute(
            "DELETE FROM broker_accounts WHERE user_id = %s", (user_id,)
        ).rowcount
    return deleted > 0


def account_credentials(pool: DatabasePool, user_id: int) -> tuple[str, str] | None:
    with pool.connection() as connection:
        row = connection.execute(
            "SELECT ibkr_username, ibkr_password FROM broker_accounts WHERE user_id = %s",
            (user_id,),
        ).fetchone()
    return (row["ibkr_username"], row["ibkr_password"]) if row else None


def account_user_ids(pool: DatabasePool) -> list[int]:
    with pool.connection() as connection:
        rows = connection.execute("SELECT user_id FROM broker_accounts ORDER BY user_id")
        return [row["user_id"] for row in rows]


def save_snapshot(pool: DatabasePool, user_id: int, snapshot: Snapshot) -> None:
    """Replace the user's balances and positions with the broker's current state at once."""

    balances = snapshot.balances
    with pool.connection() as connection, connection.transaction():
        connection.execute(
            """
            UPDATE broker_accounts SET
                account_id = %s, currency = %s, net_liquidation = %s, cash = %s,
                buying_power = %s, available_funds = %s, synced_at = now()
            WHERE user_id = %s
            """,
            (
                balances.account_id,
                balances.currency,
                balances.net_liquidation,
                balances.cash,
                balances.buying_power,
                balances.available_funds,
                user_id,
            ),
        )
        connection.execute("DELETE FROM positions WHERE user_id = %s", (user_id,))
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO positions (
                    user_id, symbol, quantity, average_cost, market_price, market_value,
                    unrealized_pnl
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        user_id,
                        position.symbol,
                        position.quantity,
                        position.average_cost,
                        position.market_price,
                        position.market_value,
                        position.unrealized_pnl,
                    )
                    for position in snapshot.positions
                ],
            )


def account_overview(pool: DatabasePool, user_id: int) -> Overview | None:
    with pool.connection() as connection:
        account = connection.execute(
            """
            SELECT ibkr_username, account_id, currency, synced_at,
                net_liquidation::float8, cash::float8, buying_power::float8,
                available_funds::float8
            FROM broker_accounts WHERE user_id = %s
            """,
            (user_id,),
        ).fetchone()
        if account is None:
            return None
        rows = connection.execute(
            """
            SELECT symbol, quantity::float8, average_cost::float8, market_price::float8,
                market_value::float8, unrealized_pnl::float8
            FROM positions WHERE user_id = %s ORDER BY symbol
            """,
            (user_id,),
        ).fetchall()
    snapshot = None
    if account["synced_at"] is not None:
        snapshot = Snapshot(
            Balances(
                account_id=account["account_id"],
                currency=account["currency"],
                net_liquidation=account["net_liquidation"],
                cash=account["cash"],
                buying_power=account["buying_power"],
                available_funds=account["available_funds"],
            ),
            tuple(Position(**row) for row in rows),
        )
    return Overview(account["ibkr_username"], snapshot, account["synced_at"])


__all__ = [
    "Balances",
    "Overview",
    "Position",
    "Snapshot",
    "account_credentials",
    "account_overview",
    "account_user_ids",
    "delete_account",
    "save_account",
    "save_snapshot",
]
