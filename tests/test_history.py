"""Importing IBKR statements and exporting trade history."""

from datetime import UTC, datetime
from types import SimpleNamespace

import fakeredis
import pytest
from fastapi.testclient import TestClient

import tragent.api.app as app_module
from tragent import redis_client
from tragent.api.auth import create_access_token
from tragent.config import Settings
from tragent.trading import history

STATEMENT = """\
Statement,Header,Field Name,Field Value
Statement,Data,BrokerName,Interactive Brokers LLC
Trades,Header,DataDiscriminator,Asset Category,Currency,Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code
Trades,Data,Order,Stocks,USD,AAPL,"2026-10-23, 11:00:03",10,347.82,347.73,-3478.2,-1.00003,3479.20003,0,-0.9,O
Trades,Data,Order,Stocks,USD,AAPL,"2026-10-23, 12:48:16","-1,000",347.18,347.73,347180,-1.007155,-347.89,-0.71,-0.55,C
Trades,SubTotal,,Stocks,USD,AAPL,,9,,,-3131.02,-2.007185,,,,
Trades,Total,,Stocks,USD,,,,,,-3131.02,-2.007185,,,,
Trades,Header,DataDiscriminator,Asset Category,Currency,Symbol,Date/Time,Quantity,T. Price,Proceeds,Comm in USD,MTM in USD,Code
Trades,Data,Order,Forex,USD,EUR.USD,"2026-10-22, 10:00:00",1000,1.08,-1080,-2,0,
"""


def test_only_stock_executions_are_read_with_their_times_in_utc() -> None:
    bought, sold = history.parse_trades(STATEMENT)

    assert bought == history.StatementTrade(
        "AAPL", "USD", datetime(2026, 10, 23, 15, 0, 3, tzinfo=UTC), 10, 347.82, 1.00003
    )
    assert (sold.quantity, sold.commission) == (-1000, 1.007155)


# A statement downloaded from the Chinese Client Portal: its Trades header names the symbol and the
# trade code both "代码".
CHINESE_STATEMENT = """\
\ufeffStatement,Header,Field Name,Field Value
交易,Header,DataDiscriminator,资产分类,货币,代码,日期/时间,数量,交易价格,收盘价格,收益,佣金/税,基础,已实现的损益,按市值计算的损益,代码
交易,Data,Order,股票,USD,MSFT,"2026-05-06, 10:39:35",3,285.23,287.51,-855.69,-0.34696625,856.03696625,0,6.84,O
交易,Data,Order,股票和指数期权,USD,AAPL 260619C00300000,"2026-06-01, 10:00:00",1,2.5,2.4,-250,-0.7,250.7,0,-10,O
交易,SubTotal,,股票,USD,MSFT,,3,,,-855.69,-0.34696625,,0,6.84,
"""


def test_chinese_statements_read_the_first_column_of_a_repeated_name() -> None:
    (trade,) = history.parse_trades(CHINESE_STATEMENT.removeprefix("\ufeff"))

    assert trade == history.StatementTrade(
        "MSFT", "USD", datetime(2026, 5, 6, 14, 39, 35, tzinfo=UTC), 3, 285.23, 0.34696625
    )


@pytest.mark.parametrize(
    "data",
    [
        b"Statement,Header,Field Name,Field Value\n",
        b"Trades,Header,Symbol\nTrades,Data,AAPL\n",
        b"\xff\xfe\x00",
    ],
)
def test_files_without_stock_trades_are_refused(data: bytes) -> None:
    with pytest.raises(history.StatementError):
        history.import_statement(None, 5, "statement.csv", data)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, settings: Settings):
    monkeypatch.setattr(redis_client, "connect", lambda _s: fakeredis.FakeRedis())
    monkeypatch.setattr(redis_client, "connect_async", lambda _s: fakeredis.FakeAsyncRedis())
    pool = SimpleNamespace(open=lambda **_kwargs: None, close=lambda: None)
    monkeypatch.setattr(app_module, "build_connection_pool", lambda _settings: pool)
    monkeypatch.setattr(app_module, "setup_persistence", lambda *_args: None)
    headers = {"Authorization": f"Bearer {create_access_token(5, settings)}"}
    with TestClient(app_module.create_app(settings), headers=headers) as client:
        yield client


def test_uploads_are_size_limited_and_report_what_they_imported(
    monkeypatch: pytest.MonkeyPatch, api
) -> None:
    imported: list[tuple] = []

    def import_statement(_pool, user_id, filename, data):
        imported.append((user_id, filename, data))
        return history.Statement(1, filename, 2, 0, datetime.now(UTC))

    monkeypatch.setattr(history, "import_statement", import_statement)
    monkeypatch.setattr(history, "MAX_STATEMENT_BYTES", 10)

    accepted = api.post("/api/trading/statements", files={"file": ("u.csv", b"0123456789")})
    too_large = api.post("/api/trading/statements", files={"file": ("big.csv", b"01234567890")})

    assert accepted.status_code == 201
    assert accepted.json()["imported"] == 2
    assert imported == [(5, "u.csv", b"0123456789")]
    assert too_large.status_code == 413


def test_a_file_that_is_not_a_statement_is_unprocessable(api) -> None:
    response = api.post("/api/trading/statements", files={"file": ("notes.csv", b"a,b\n1,2\n")})

    assert response.status_code == 422
    assert "no stock trades" in response.json()["detail"]


def test_exports_are_stored_and_downloaded_from_the_store(
    monkeypatch: pytest.MonkeyPatch, api, object_store: dict[str, bytes]
) -> None:
    monkeypatch.setattr(
        history,
        "trade_history",
        lambda _pool, _user: [
            history.HistoryTrade(
                datetime(2026, 10, 23, 15, tzinfo=UTC), "platform", "AAPL", "buy", 1, 347.5, None
            )
        ],
    )

    link = api.post("/api/trading/exports").json()["url"]

    ((key, csv),) = object_store.items()
    assert key.startswith("exports/5/") and link == f"https://store/{key}"
    assert csv.decode().splitlines() == [
        "traded_at,source,symbol,side,quantity,price,commission",
        "2026-10-23T15:00:00+00:00,platform,AAPL,buy,1,347.5,",
    ]
