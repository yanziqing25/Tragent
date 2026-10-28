"""Deploying research strategies and each trading morning's run."""

from dataclasses import replace
from datetime import UTC, date, datetime
from types import SimpleNamespace

import fakeredis
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import tragent.api.app as app_module
from tragent import redis_client
from tragent.api.auth import create_access_token
from tragent.config import Settings
from tragent.research.reporting import ResearchReportView
from tragent.trading import api as trading_api
from tragent.trading import deployments, store

_LATEST = date(2026, 10, 23)


def _report(**changes: object) -> ResearchReportView:
    candidate = SimpleNamespace(
        id="c1",
        strategy_name="rsi",
        parameters={"rsi_window": 14, "entry_threshold": 30.0, "exit_threshold": 70.0},
    )
    report = SimpleNamespace(
        symbol="AAPL",
        bar_interval="1d",
        critic_review_status="accepted",
        selected_candidate_id="c1",
        candidates=(candidate,),
    )
    return SimpleNamespace(**{**vars(report), **changes})


def _deployment(**changes: object) -> deployments.Deployment:
    deployment = deployments.Deployment(
        id=7,
        user_id=5,
        thread_id="t",
        symbol="AAPL",
        strategy_name="rsi",
        parameters={"rsi_window": 14, "entry_threshold": 30.0, "exit_threshold": 70.0},
        position_tranches=5,
        capital=10_000.0,
        tranches=0,
        status="active",
        created_at=datetime.now(UTC),
    )
    return replace(deployment, **changes)


@pytest.mark.parametrize(
    ("report", "available", "reason"),
    [
        (_report(critic_review_status="rejected"), 50_000, "Critic accepted"),
        (_report(selected_candidate_id=None), 50_000, "Critic accepted"),
        (_report(bar_interval="1m"), 50_000, "daily strategies"),
        (_report(), 5_000, "exceeds the available funds"),
        (_report(), None, "exceeds the available funds of 0.00"),
    ],
)
def test_only_accepted_daily_strategies_within_the_funds_deploy(
    monkeypatch: pytest.MonkeyPatch, report, available, reason
) -> None:
    overview = None
    if available is not None:
        balances = store.Balances("DU1", "USD", available, available, available, available)
        overview = store.Overview("paper", store.Snapshot(balances, ()), datetime.now(UTC))
    monkeypatch.setattr(store, "account_overview", lambda _pool, _user_id: overview)

    with pytest.raises(deployments.NotDeployableError, match=reason):
        deployments.deploy(None, 5, "t", report, 5, 10_000)


@pytest.fixture
def morning(monkeypatch: pytest.MonkeyPatch):
    """A run's surroundings: held shares, funds, the last bar, and what it placed and saved."""

    state = SimpleNamespace(
        held=0.0, funds=50_000.0, last_bar=_LATEST, signal=0, placed=[], runs=[]
    )
    monkeypatch.setattr(deployments, "_held_shares", lambda _pool, _id: state.held)
    monkeypatch.setattr(deployments, "_available_funds", lambda _pool, _id: state.funds)
    monkeypatch.setattr(deployments, "latest_completed_us_market_session", lambda: _LATEST)
    monkeypatch.setattr(
        deployments,
        "generate_trade_signals",
        lambda close, _config: pd.Series(
            [0] * (len(close) - 1) + [state.signal], index=close.index
        ),
    )

    def place(_pool, _producer, deployment, side, quantity, key):
        state.placed.append((side, quantity, key))
        return SimpleNamespace(id=100 + len(state.placed))

    monkeypatch.setattr(deployments, "_place", place)
    monkeypatch.setattr(
        deployments, "_record", lambda _pool, _deployment, _session, **run: state.runs.append(run)
    )

    def get_daily_bars(_symbol, _start, end):
        index = pd.DatetimeIndex(pd.bdate_range(end=state.last_bar, periods=60))
        return SimpleNamespace(bars=pd.DataFrame({"close": 200.0}, index=index))

    state.provider = SimpleNamespace(get_daily_bars=get_daily_bars)
    return state


def test_a_buy_signal_buys_one_tranche_once_per_session(morning) -> None:
    morning.signal = 1

    deployments.run(None, None, morning.provider, _deployment(), date(2026, 10, 26))
    deployments.run(None, None, morning.provider, _deployment(), date(2026, 10, 26))
    deployments.run(None, None, morning.provider, _deployment(), date(2026, 10, 27))

    # 10,000 × 1/5 at 200 is 10 shares; a session's orders share one idempotency key.
    (side, quantity, key), (_, _, repeated), (_, _, next_day) = morning.placed
    assert (side, quantity) == ("buy", 10)
    assert key == repeated != next_day
    assert morning.runs[0] == {
        "signal": 1,
        "tranches": 1,
        "target": 10,
        "held": 0.0,
        "order_id": 101,
        "note": None,
    }


def test_a_sell_signal_sells_down_to_the_lower_tranche(morning) -> None:
    morning.signal, morning.held = -1, 20.0

    deployments.run(None, None, morning.provider, _deployment(tranches=2), date(2026, 10, 26))

    assert [placed[:2] for placed in morning.placed] == [("sell", 10)]
    assert {k: morning.runs[0][k] for k in ("signal", "tranches", "target", "held")} == {
        "signal": -1,
        "tranches": 1,
        "target": 10,
        "held": 20.0,
    }


@pytest.mark.parametrize(
    ("setup", "note"),
    [
        ({"signal": 1, "funds": 1_000.0}, "Not enough available funds"),
        ({"last_bar": date(2026, 10, 22)}, "No daily bar for 2026-10-23 yet"),
        ({"signal": 0}, None),
    ],
)
def test_runs_without_an_order_say_why(morning, setup: dict, note: str | None) -> None:
    for name, value in setup.items():
        setattr(morning, name, value)

    deployments.run(None, None, morning.provider, _deployment(), date(2026, 10, 26))

    assert morning.placed == []
    assert morning.runs[0].get("note") == note


def test_the_morning_run_skips_holidays_and_one_failing_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[int] = []

    def run(_pool, _producer, _provider, deployment, _session):
        if deployment.id == 1:
            raise ValueError("download failed")
        ran.append(deployment.id)

    rows = [_row(1), _row(2)]
    connection = SimpleNamespace(execute=lambda *_args: SimpleNamespace(fetchall=lambda: rows))
    pool = SimpleNamespace(connection=lambda: _Context(connection))
    monkeypatch.setattr(deployments, "run", run)

    deployments.run_all(pool, None, None, date(2026, 11, 26))  # Thanksgiving
    deployments.run_all(pool, None, None, date(2026, 11, 27))

    assert ran == [2]


def _row(deployment_id: int) -> dict:
    deployment = _deployment(id=deployment_id)
    return {field: getattr(deployment, field) for field in deployment.__dataclass_fields__}


class _Context:
    def __init__(self, value: object) -> None:
        self.value = value

    def __enter__(self) -> object:
        return self.value

    def __exit__(self, *_args: object) -> None:
        return None


def test_the_next_run_is_the_next_trading_morning() -> None:
    eastern = deployments.EASTERN
    friday_noon = datetime(2026, 11, 27, 12, tzinfo=eastern)
    wednesday_dawn = datetime(2026, 11, 25, 6, tzinfo=eastern)

    assert deployments.next_run(friday_noon) == datetime(2026, 11, 30, 9, tzinfo=eastern)
    # Thanksgiving is skipped.
    assert deployments.next_run(datetime(2026, 11, 25, 10, tzinfo=eastern)) == datetime(
        2026, 11, 27, 9, tzinfo=eastern
    )
    assert deployments.next_run(wednesday_dawn) == datetime(2026, 11, 25, 9, tzinfo=eastern)


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


def test_deployments_move_only_between_allowed_statuses(
    monkeypatch: pytest.MonkeyPatch, api
) -> None:
    known = {
        1: _deployment(id=1),
        2: _deployment(id=2, status="paused"),
        3: _deployment(id=3, status="stopped"),
        4: _deployment(id=4, user_id=6),
    }
    changed: list[tuple] = []
    monkeypatch.setattr(deployments, "get_deployment", lambda _pool, id_: known.get(id_))
    monkeypatch.setattr(
        deployments, "set_status", lambda _pool, id_, status: changed.append((id_, status))
    )
    monkeypatch.setattr(
        deployments,
        "stop",
        lambda _pool, _producer, d, close_position: changed.append(
            (d.id, "stopped", close_position)
        ),
    )

    def post(id_: int, action: str) -> int:
        body = {"close_position": True} if action == "stop" else None
        return api.post(f"/api/trading/deployments/{id_}/{action}", json=body).status_code

    codes = [
        post(1, "pause"),
        post(2, "pause"),
        post(2, "resume"),
        post(1, "resume"),
        post(1, "stop"),
        post(3, "stop"),
        post(4, "pause"),
    ]

    assert codes == [204, 409, 204, 409, 204, 409, 404]
    assert changed == [(1, "paused"), (2, "active"), (1, "stopped", True)]


def test_a_report_that_cannot_be_deployed_is_a_conflict(
    monkeypatch: pytest.MonkeyPatch, api
) -> None:
    monkeypatch.setattr(trading_api, "get_research_report_view", lambda *_a, **_k: _report())
    monkeypatch.setattr(
        trading_api,
        "get_research_status",
        lambda *_a, **_k: SimpleNamespace(goal={"position_tranches": 5}),
    )

    def deploy(*_args):
        raise deployments.NotDeployableError("Only daily strategies can be deployed")

    monkeypatch.setattr(deployments, "deploy", deploy)
    response = api.post(
        "/api/trading/deployments",
        json={"thread_id": "a141160f-62fa-47f8-84fe-f49ab0b2f319", "capital": 1000},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Only daily strategies can be deployed"
