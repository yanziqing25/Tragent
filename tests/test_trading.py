"""Linking an IBKR paper account, its Gateway container, and the broker's sync round."""

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import fakeredis
import pytest
from docker.errors import NotFound
from eventkit import Event
from fastapi.testclient import TestClient
from ib_async import AccountValue, Contract, PnLSingle, PortfolioItem

import tragent.api.app as app_module
from tragent import redis_client
from tragent.api.auth import create_access_token
from tragent.config import Settings
from tragent.trading import api as trading_api
from tragent.trading import broker, containers, messaging, store

_ACCOUNT = "DU1234567"


def _value(tag: str, value: str, currency: str = "USD") -> AccountValue:
    return AccountValue(_ACCOUNT, tag, value, currency, "")


def _item(symbol: str, position: float, con_id: int = 0) -> PortfolioItem:
    return PortfolioItem(
        Contract(symbol=symbol, conId=con_id), position, 101.0, 1010.0, 95.0, 60.0, 0.0, _ACCOUNT
    )


_SNAPSHOT = store.Snapshot(
    store.Balances(_ACCOUNT, "USD", 1_000_000.0, 900_000.0, 4_000_000.0, 950_000.0),
    (store.Position("AAPL", 10.0, 95.0, 101.0, 1010.0, 60.0),),
)


def test_snapshot_reads_base_currency_balances_and_open_positions() -> None:
    values = [
        _value("NetLiquidation", "1000000"),
        _value("TotalCashValue", "900000"),
        _value("TotalCashValue", "-50", "BASE"),
        _value("BuyingPower", "4000000"),
        _value("AvailableFunds", "950000"),
        _value("GrossPositionValue", "1010"),
    ]

    result = broker.snapshot(_ACCOUNT, values, [_item("AAPL", 10), _item("MSFT", 0)], [])

    assert result == _SNAPSHOT


def test_there_is_no_snapshot_until_ibkr_sends_the_account_values() -> None:
    assert broker.snapshot(_ACCOUNT, [], [_item("AAPL", 10)], []) is None


def test_positions_take_value_and_pnl_from_their_subscription_once_it_matches() -> None:
    values = [_value("NetLiquidation", "1000000")]
    portfolio = [_item("AAPL", 10, con_id=1), _item("MSFT", 5, con_id=2), _item("TSLA", 3, 3)]
    pnl = [
        PnLSingle(conId=1, position=10, value=1050.0, unrealizedPnL=100.0),
        # Not reported yet, or still at the size before a fill: the portfolio's values stand.
        PnLSingle(conId=2, position=5),
        PnLSingle(conId=3, position=1, value=200.0, unrealizedPnL=5.0),
    ]

    aapl, msft, tsla = broker.snapshot(_ACCOUNT, values, portfolio, pnl).positions

    assert (aapl.market_price, aapl.market_value, aapl.unrealized_pnl) == (105.0, 1050.0, 100.0)
    assert (msft.market_price, msft.market_value, msft.unrealized_pnl) == (101.0, 1010.0, 60.0)
    assert (tsla.market_price, tsla.market_value, tsla.unrealized_pnl) == (101.0, 1010.0, 60.0)


class _FakeIB:
    def __init__(self, *, reachable: bool = True, items: list[PortfolioItem] = ()) -> None:
        self.reachable = reachable
        self.connected = False
        self.items = list(items)
        self.subscriptions: list[PnLSingle] = []
        self.orderStatusEvent = Event()
        self.execDetailsEvent = Event()
        self.commissionReportEvent = Event()
        self.accountValueEvent = Event()
        self.updatePortfolioEvent = Event()
        self.pnlSingleEvent = Event()

    def isConnected(self) -> bool:
        return self.connected

    async def connectAsync(self, host: str, port: int, **_kwargs: object) -> None:
        if not self.reachable:
            raise ConnectionRefusedError
        self.port = port
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def managedAccounts(self) -> list[str]:
        return [_ACCOUNT]

    def accountValues(self, _account: str) -> list[AccountValue]:
        return [_value("NetLiquidation", "1000000")]

    def portfolio(self, _account: str) -> list[PortfolioItem]:
        return self.items

    def pnlSingle(self, _account: str) -> list[PnLSingle]:
        return self.subscriptions

    def reqPnLSingle(self, account: str, model_code: str, con_id: int) -> None:
        self.subscriptions.append(PnLSingle(account=account, conId=con_id))

    def cancelPnLSingle(self, _account: str, _model_code: str, con_id: int) -> None:
        self.subscriptions = [pnl for pnl in self.subscriptions if pnl.conId != con_id]

    def trades(self) -> list:
        return []


def test_a_sync_round_connects_saves_and_drops_unlinked_accounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = fakeredis.FakeRedis(decode_responses=True)
    saved: list[int] = []
    monkeypatch.setattr(store, "account_user_ids", lambda _pool: [1, 2])
    monkeypatch.setattr(store, "save_snapshot", lambda _pool, user_id, _snap: saved.append(user_id))
    monkeypatch.setattr(broker.orders, "unsent_orders", lambda _pool, _seconds: [])
    unlinked = _FakeIB(reachable=True)
    unlinked.connected = True
    starting = iter([_FakeIB(reachable=True), _FakeIB(reachable=False)])
    runner = broker.Broker(None, redis, None, lambda: next(starting))
    runner.connections[9] = unlinked
    connections = runner.connections

    asyncio.run(runner.sync_round())

    assert set(connections) == {1, 2} and not unlinked.connected
    assert connections[1].port == containers.gateway_address(1)[1] == 4101
    assert saved == [1]
    assert json.loads(redis.get(broker.status_key(1))) == {"connected": True, "error": None}
    assert json.loads(redis.get(broker.status_key(2)))["connected"] is False
    assert 0 < redis.ttl(broker.status_key(1)) <= 30


def test_each_held_position_and_no_other_keeps_a_pnl_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(store, "account_user_ids", lambda _pool: [1])
    monkeypatch.setattr(store, "save_snapshot", lambda *_args: None)
    monkeypatch.setattr(broker.orders, "unsent_orders", lambda _pool, _seconds: [])
    ib = _FakeIB(items=[_item("AAPL", 10, con_id=1), _item("MSFT", 0, con_id=2)])
    ib.subscriptions = [PnLSingle(conId=2), PnLSingle(conId=3)]
    runner = broker.Broker(None, redis, None, lambda: ib)

    asyncio.run(runner.sync_round())

    assert [pnl.conId for pnl in ib.subscriptions] == [1]


def test_updates_publish_one_snapshot_per_user_per_tick(kafka) -> None:
    ib = _FakeIB(items=[_item("AAPL", 10, con_id=1)])
    ib.connected = True
    runner = broker.Broker(None, None, kafka.producer, lambda: ib)
    runner._connect(5)

    ib.pnlSingleEvent.emit(PnLSingle(conId=1))
    ib.accountValueEvent.emit(_value("NetLiquidation", "1000000"))
    runner.publish_changed()
    runner.publish_changed()

    ((topic, key, value),) = kafka.producer.messages
    published = json.loads(value)
    assert (topic, key) == (messaging.ACCOUNT_SNAPSHOTS, "5")
    assert published["balances"]["net_liquidation"] == 1_000_000.0
    assert published["positions"][0]["symbol"] == "AAPL"
    assert "synced_at" in published


def _docker(
    monkeypatch: pytest.MonkeyPatch, existing: set[str], logs: bytes = b""
) -> SimpleNamespace:
    """A Docker client whose new containers print these logs; records runs and removals."""

    calls = SimpleNamespace(runs=[], removed=[])

    def get(name: str) -> SimpleNamespace:
        if name not in existing:
            raise NotFound(name)
        return SimpleNamespace(remove=lambda force: calls.removed.append(name))

    def run(_image: str, **kwargs: object) -> SimpleNamespace:
        calls.runs.append(kwargs)
        return SimpleNamespace(
            logs=lambda: logs, remove=lambda force: calls.removed.append(kwargs["name"])
        )

    client = SimpleNamespace(containers=SimpleNamespace(get=get, run=run))
    monkeypatch.setattr(containers.docker, "from_env", lambda: client)
    return calls


class _BrokenIB(_FakeIB):
    """A connected Gateway whose account data cannot be read."""

    def managedAccounts(self) -> list[str]:
        raise ValueError("unexpected account data")


def test_one_users_failing_sync_or_publish_spares_the_others(
    monkeypatch: pytest.MonkeyPatch, kafka
) -> None:
    redis = fakeredis.FakeRedis(decode_responses=True)
    saved: list[int] = []
    monkeypatch.setattr(store, "account_user_ids", lambda _pool: [1, 2])
    monkeypatch.setattr(store, "save_snapshot", lambda _pool, user_id, _snap: saved.append(user_id))
    monkeypatch.setattr(broker.orders, "unsent_orders", lambda _pool, _seconds: [])
    gateways = iter([_BrokenIB(), _FakeIB()])
    runner = broker.Broker(None, redis, kafka.producer, lambda: next(gateways))

    asyncio.run(runner.sync_round())
    runner._changed = {1, 2}
    runner.publish_changed()

    assert saved == [2]
    assert json.loads(redis.get(broker.status_key(1))) == {
        "connected": False,
        "error": "Syncing this account failed",
    }
    assert json.loads(redis.get(broker.status_key(2)))["connected"] is True
    assert [key for _topic, key, _value in kafka.producer.messages] == ["2"]


def test_gateway_is_replaced_and_published_on_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _docker(monkeypatch, existing={"tragent-gateway-3"})

    containers.run_gateway(3, "paperuser", "secret")
    containers.stop_gateway(4)

    assert calls.removed == ["tragent-gateway-3"]
    (run,) = calls.runs
    assert run["name"] == "tragent-gateway-3"
    assert run["ports"] == {"4004/tcp": ("127.0.0.1", 4103)}
    assert run["network"] is None
    assert run["environment"]["TRADING_MODE"] == "paper"
    assert run["environment"]["TWS_USERID"] == "paperuser"
    assert run["restart_policy"] == {"Name": "unless-stopped"}


@pytest.mark.parametrize(
    ("logs", "accepted"),
    [
        (b"IBC: Click button: Paper Log In\nIBC: Login has completed\n", True),
        (
            b"IBC: Click button: Paper Log In\nIBC: detected dialog entitled: Gateway; event=Opened\n",
            False,
        ),
    ],
)
def test_a_gateway_is_kept_only_if_ibkr_accepts_the_login(
    monkeypatch: pytest.MonkeyPatch, logs: bytes, accepted: bool
) -> None:
    calls = _docker(monkeypatch, existing=set(), logs=logs)

    assert containers.log_in(3, "paperuser", "secret") is accepted
    assert calls.removed == ([] if accepted else ["tragent-gateway-3"])


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, settings: Settings):
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
    with TestClient(app_module.create_app(settings), headers=headers) as test_client:
        yield test_client, fakeredis.FakeRedis(server=server, decode_responses=True)


def test_linking_logs_a_gateway_in_and_the_overview_shows_its_status(
    monkeypatch: pytest.MonkeyPatch,
    client,
) -> None:
    http, redis = client
    accounts: dict[int, str] = {}
    gateways: list[tuple] = []
    monkeypatch.setattr(
        store,
        "save_account",
        lambda _pool, user_id, username, _pw: accounts.update({user_id: username}),
    )
    monkeypatch.setattr(
        store, "delete_account", lambda _pool, user_id: bool(accounts.pop(user_id, None))
    )
    monkeypatch.setattr(
        store,
        "account_overview",
        lambda _pool, user_id: (
            store.Overview(accounts[user_id], _SNAPSHOT, datetime.now(UTC))
            if user_id in accounts
            else None
        ),
    )
    monkeypatch.setattr(
        trading_api.containers,
        "log_in",
        lambda *args: gateways.append(("log in", *args)) or True,
    )
    monkeypatch.setattr(
        trading_api.containers, "stop_gateway", lambda *args: gateways.append(("stop", *args))
    )

    assert http.get("/api/trading/account").status_code == 404
    assert (
        http.put(
            "/api/trading/account", json={"username": "paperuser", "password": "pw"}
        ).status_code
        == 204
    )
    waiting = http.get("/api/trading/account").json()
    redis.set(broker.status_key(5), json.dumps({"connected": True, "error": None}))
    live = http.get("/api/trading/account").json()
    unlinked = http.delete("/api/trading/account")

    assert gateways == [("log in", 5, "paperuser", "pw"), ("stop", 5)]
    assert waiting["connection"] == {
        "connected": False,
        "error": "The broker process is not running",
    }
    assert "password" not in waiting
    assert live["connection"]["connected"] is True
    assert live["balances"]["net_liquidation"] == 1_000_000.0
    assert live["positions"][0]["symbol"] == "AAPL"
    assert unlinked.status_code == 204
    assert http.delete("/api/trading/account").status_code == 404


def test_a_rejected_login_is_not_saved_and_restores_the_saved_one(
    monkeypatch: pytest.MonkeyPatch,
    client,
) -> None:
    http, _redis = client
    saved: dict[int, tuple[str, str]] = {}
    restored: list[tuple] = []
    monkeypatch.setattr(trading_api.containers, "log_in", lambda *_args: False)
    monkeypatch.setattr(trading_api.containers, "run_gateway", lambda *args: restored.append(args))
    monkeypatch.setattr(store, "save_account", lambda *args: saved.update({args[1]: args[2:]}))
    monkeypatch.setattr(store, "account_credentials", lambda _pool, user_id: saved.get(user_id))
    body = {"username": "paperuser", "password": "wrong"}

    first = http.put("/api/trading/account", json=body)
    saved[5] = ("paperuser", "right")
    change = http.put("/api/trading/account", json=body)

    assert first.status_code == change.status_code == 400
    assert "did not accept this login" in first.json()["detail"]
    assert saved == {5: ("paperuser", "right")}
    assert restored == [(5, "paperuser", "right")]


def test_on_a_network_the_broker_reaches_a_gateway_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(containers, "_network", lambda: "tragent_default")

    assert containers.gateway_address(3) == ("tragent-gateway-3", 4004)
