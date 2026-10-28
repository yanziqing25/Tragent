"""Trade history: IBKR statements users upload, and CSV exports of every trade they hold here.

An Activity Statement CSV holds many sections, each led by its own header row; only the stock
executions of its Trades section are imported. The original file stays in the object store, and
each trade is imported once however many overlapping statements arrive.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from tragent import storage
from tragent.db import DatabasePool

MAX_STATEMENT_BYTES = 10 * 1024 * 1024
# Statements state US stock execution times in exchange time.
_EASTERN = ZoneInfo("America/New_York")
# IBKR writes statements in the language of the user's Client Portal; these are English and Chinese.
_TRADES = {"Trades", "交易"}
_STOCKS = {"Stocks", "股票"}
_COLUMNS = {
    "category": ("Asset Category", "资产分类"),
    "currency": ("Currency", "货币"),
    "symbol": ("Symbol", "代码"),
    "time": ("Date/Time", "日期/时间"),
    "quantity": ("Quantity", "数量"),
    "price": ("T. Price", "交易价格"),
    "commission": ("Comm/Fee", "佣金/税"),
}
_EXPORT_COLUMNS = ("traded_at", "source", "symbol", "side", "quantity", "price", "commission")


class StatementError(Exception):
    """The file is not an IBKR statement with stock trades; the message says why."""


@dataclass(frozen=True, slots=True)
class StatementTrade:
    symbol: str
    currency: str
    traded_at: datetime
    # Negative for a sale.
    quantity: float
    price: float
    commission: float


@dataclass(frozen=True, slots=True)
class Statement:
    id: int
    filename: str
    imported: int
    skipped: int
    uploaded_at: datetime


@dataclass(frozen=True, slots=True)
class HistoryTrade:
    traded_at: datetime
    source: Literal["platform", "statement"]
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    commission: float | None


def parse_trades(text: str) -> list[StatementTrade]:
    trades: list[StatementTrade] = []
    columns: dict[str, int] | None = None
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 3 or row[0] not in _TRADES:
            continue
        if row[1] == "Header":
            columns = _column_indexes(row)
            continue
        # Order rows are executions; closed lots and subtotals repeat them.
        if row[1] != "Data" or row[2] != "Order" or columns is None:
            continue
        field = {name: row[index] for name, index in columns.items()}
        if field["category"] not in _STOCKS:
            continue
        trades.append(
            StatementTrade(
                symbol=field["symbol"],
                currency=field["currency"],
                traded_at=datetime.strptime(field["time"], "%Y-%m-%d, %H:%M:%S")
                .replace(tzinfo=_EASTERN)
                .astimezone(UTC),
                quantity=_number(field["quantity"]),
                price=_number(field["price"]),
                # Statements show fees as negative amounts.
                commission=-_number(field["commission"]),
            )
        )
    if not trades:
        raise StatementError("The file has no stock trades from an IBKR Activity Statement")
    return trades


def _column_indexes(header: list[str]) -> dict[str, int] | None:
    """Where each field is, by its first column of either name, since a Chinese header repeats
    one; None for a header, such as that of forex trades, without every stock trade field."""

    indexes = {}
    for name, labels in _COLUMNS.items():
        index = next((i for i, label in enumerate(header) if label in labels), None)
        if index is None:
            return None
        indexes[name] = index
    return indexes


def import_statement(pool: DatabasePool, user_id: int, filename: str, data: bytes) -> Statement:
    """Keep the original file, and import each of its trades not already imported."""

    try:
        trades = parse_trades(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        # Not text, or a Trades section without the expected columns or numbers.
        raise StatementError("The file is not an IBKR Activity Statement CSV") from None
    key = f"statements/{user_id}/{uuid4()}.csv"
    storage.put(key, data, "text/csv", download_name=filename)
    with pool.connection() as connection, connection.transaction():
        statement_id = connection.execute(
            """
            INSERT INTO statements (user_id, filename, object_key, imported, skipped)
            VALUES (%s, %s, %s, 0, 0) RETURNING id
            """,
            (user_id, filename, key),
        ).fetchone()["id"]
        imported = 0
        for trade in trades:
            imported += connection.execute(
                """
                INSERT INTO statement_trades (
                    user_id, statement_id, symbol, currency, traded_at, quantity, price,
                    commission
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, symbol, traded_at, quantity, price) DO NOTHING
                """,
                (
                    user_id,
                    statement_id,
                    trade.symbol,
                    trade.currency,
                    trade.traded_at,
                    trade.quantity,
                    trade.price,
                    trade.commission,
                ),
            ).rowcount
        row = connection.execute(
            """
            UPDATE statements SET imported = %s, skipped = %s WHERE id = %s
            RETURNING id, filename, imported, skipped, uploaded_at
            """,
            (imported, len(trades) - imported, statement_id),
        ).fetchone()
    return Statement(**row)


def list_statements(pool: DatabasePool, user_id: int) -> list[Statement]:
    with pool.connection() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, imported, skipped, uploaded_at FROM statements
            WHERE user_id = %s ORDER BY uploaded_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [Statement(**row) for row in rows]


def statement_download_url(pool: DatabasePool, user_id: int, statement_id: int) -> str | None:
    with pool.connection() as connection:
        row = connection.execute(
            "SELECT object_key FROM statements WHERE id = %s AND user_id = %s",
            (statement_id, user_id),
        ).fetchone()
    return storage.download_url(row["object_key"]) if row else None


def trade_history(pool: DatabasePool, user_id: int) -> list[HistoryTrade]:
    """Every trade the user holds here, newest first: platform fills and imported trades."""

    with pool.connection() as connection:
        rows = connection.execute(
            """
            SELECT e.executed_at AS traded_at, 'platform' AS source, o.symbol, o.side,
                e.shares::float8 AS quantity, e.price::float8 AS price,
                e.commission::float8 AS commission
            FROM executions e JOIN orders o ON o.id = e.order_id WHERE o.user_id = %(user)s
            UNION ALL
            SELECT traded_at, 'statement', symbol,
                CASE WHEN quantity > 0 THEN 'buy' ELSE 'sell' END,
                abs(quantity)::float8, price::float8, commission::float8
            FROM statement_trades WHERE user_id = %(user)s
            ORDER BY traded_at DESC
            """,
            {"user": user_id},
        ).fetchall()
    return [HistoryTrade(**row) for row in rows]


def export_history(pool: DatabasePool, user_id: int) -> str:
    """Store the trade history as a CSV and return a short-lived link to download it."""

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_EXPORT_COLUMNS)
    for trade in trade_history(pool, user_id):
        writer.writerow(
            [
                trade.traded_at.isoformat(),
                trade.source,
                trade.symbol,
                trade.side,
                trade.quantity,
                trade.price,
                "" if trade.commission is None else trade.commission,
            ]
        )
    now = datetime.now(UTC)
    key = f"exports/{user_id}/{now:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}.csv"
    storage.put(
        key, output.getvalue().encode(), "text/csv", download_name=f"trades-{now:%Y-%m-%d}.csv"
    )
    return storage.download_url(key)


def _number(text: str) -> float:
    return float(text.replace(",", ""))


__all__ = [
    "MAX_STATEMENT_BYTES",
    "HistoryTrade",
    "Statement",
    "StatementError",
    "StatementTrade",
    "export_history",
    "import_statement",
    "list_statements",
    "parse_trades",
    "statement_download_url",
    "trade_history",
]
