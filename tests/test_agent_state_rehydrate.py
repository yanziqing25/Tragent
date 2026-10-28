"""Durable workspace checkpoints rehydrate deterministically without provider access."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from unittest.mock import MagicMock
from uuid import uuid4

import pandas as pd

from tragent.quant import DataQualityReport, MarketData, MarketDataDownloadError
from tragent.research.agent.market_snapshot import MARKET_DATA_FILENAME
from tragent.research.agent.models import (
    AgentBudget,
    MarketContextEvidence,
    ResearchGoal,
    ToolObservation,
)
from tragent.research.agent.runtime import (
    calculate_asset_statistics,
    get_market_data,
    run_backtest,
)
from tragent.research.agent.state import GraphState, validate_graph_state
from tragent.research.agent.workspace import (
    ResearchGraphContext,
    ResearchWorkspace,
    record_tool_call,
)

_BAR_COLUMNS = ("open", "high", "low", "close", "volume")


class FakeProvider:
    def __init__(self, outcomes: Mapping[str, object]) -> None:
        self.outcomes = dict(outcomes)
        self.calls: list[tuple[str, date, date]] = []

    def get_daily_bars(self, symbol: str, start_date: date, end_date: date) -> MarketData:
        self.calls.append((symbol, start_date, end_date))
        outcome = self.outcomes[symbol]
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, MarketData)
        return outcome


class NeverProvider:
    def __init__(self) -> None:
        self.calls = 0

    def get_daily_bars(self, *_args: object, **_kwargs: object) -> MarketData:
        self.calls += 1
        raise AssertionError("workspace rehydration must not access the market-data provider")


def _goal() -> ResearchGoal:
    return ResearchGoal(
        objective="Compare two transparent long-only candidates.",
        symbol="AAPL",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 20),
        allowed_strategies=("moving_average",),
        max_drawdown_limit=0.99,
    )


def _market_data(symbol: str, *, scale: float) -> MarketData:
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    closes = [scale * (100.0 + offset * 0.35 + (offset % 5) * 0.6) for offset in range(40)]
    opens = [close * (1.004 if offset % 2 == 0 else 0.996) for offset, close in enumerate(closes)]
    volume = [1_000.0 + offset * 10.0 for offset in range(40)]
    volume[7] = float("nan")
    bars = pd.DataFrame(
        {
            "open": opens,
            "high": [max(open_price, close) + scale for open_price, close in zip(opens, closes)],
            "low": [min(open_price, close) - scale for open_price, close in zip(opens, closes)],
            "close": closes,
            "volume": volume,
        },
        index=index,
        dtype="float64",
    )
    assert tuple(bars.columns) == _BAR_COLUMNS
    return MarketData(
        symbol=symbol,
        bars=bars,
        quality=DataQualityReport(
            symbol=symbol,
            valid_rows=len(bars),
            start_date=index[0],
            end_date=index[-1],
            missing_value_counts={
                "open": 0,
                "high": 0,
                "low": 0,
                "close": 0,
                "volume": 1,
            },
            warnings=(f"{symbol}: volume contains one missing value.",),
        ),
    )


def bound(workspace: ResearchWorkspace, provider: object, artifacts: str) -> ResearchWorkspace:
    return workspace.bind(
        ResearchGraphContext(model=MagicMock(), provider=provider, artifacts=artifacts)
    )


def _record(
    workspace: ResearchWorkspace,
    observation: ToolObservation,
    *,
    arguments: dict[str, object] | None = None,
) -> None:
    workspace.agent_steps += 1
    workspace.tool_calls += 1
    record_tool_call(
        workspace.journal,
        step=workspace.agent_steps,
        tool_name=observation.tool_name,
        arguments=arguments or {},
        observation=observation,
    )


def _graph_state(workspace: ResearchWorkspace) -> GraphState:
    goal = workspace.goal
    return validate_graph_state(
        {
            "thread_id": str(uuid4()),
            "user_id": 1,
            "request": goal.objective,
            "reference_date": "2026-07-13",
            "status": "running",
            "cost_defaults": {
                "transaction_cost_bps": goal.costs.transaction_cost_bps,
                "slippage_bps": goal.costs.slippage_bps,
            },
            "position_tranches_default": goal.position_tranches,
            "budget": workspace.budget.model_dump(mode="json"),
            "workspace": workspace.model_dump(mode="json"),
        }
    )


def _restored(state: GraphState, provider: object, artifacts: str) -> ResearchWorkspace:
    durable = validate_graph_state(state.model_dump(mode="json")).workspace
    assert durable is not None
    return bound(durable, provider, artifacts)


def _workspace_with_complete_final_batch(
    artifacts: str,
) -> tuple[ResearchWorkspace, tuple[str, str]]:
    provider = FakeProvider(
        {
            "AAPL": _market_data("AAPL", scale=1.0),
            "SPY": _market_data("SPY", scale=2.0),
        }
    )
    workspace = bound(ResearchWorkspace(goal=_goal(), budget=AgentBudget()), provider, artifacts)
    for operation in (get_market_data, calculate_asset_statistics):
        observation = operation(workspace)
        assert observation.status == "success"
        _record(workspace, observation)

    first, second = (
        run_backtest(
            workspace,
            stage="training",
            strategy_name="moving_average",
            parameters=parameters,
        )
        for parameters in (
            {"short_window": 2, "long_window": 3},
            {"short_window": 3, "long_window": 5},
        )
    )
    assert first.status == second.status == "success"
    fingerprints = (first.identifiers["fingerprint"], second.identifiers["fingerprint"])
    assert run_backtest(workspace, stage="final", selected_fingerprint=fingerprints[0]).status == (
        "success"
    )
    workspace.runtime_warnings.append("A deterministic checkpoint warning.")
    return workspace, fingerprints


def test_workspace_round_trip_restores_complete_quant_state_without_side_effects(
    object_store: dict[str, bytes],
) -> None:
    artifacts = f"research/{uuid4()}"
    source, fingerprints = _workspace_with_complete_final_batch(artifacts)
    evidence = MarketContextEvidence(
        id="sec-source-1",
        source_type="official",
        source="SEC",
        title="10-Q",
        excerpt="Bounded filing evidence.",
        published_at="2024-02-01",
        url="https://www.sec.gov/example",
    )
    _record(
        source,
        ToolObservation(tool_name="search_market_context", status="success", evidence=(evidence,)),
        arguments={"query": "risks"},
    )
    provider = NeverProvider()

    restored = _restored(_graph_state(source), provider, artifacts)

    assert provider.calls == 0
    assert list(object_store) == [f"{artifacts}/{MARKET_DATA_FILENAME}"]
    assert restored.model_dump(mode="json") == source.model_dump(mode="json")
    assert restored.phase == "locked"
    assert restored.locked_fingerprint == fingerprints[0]
    assert restored.final_batch_completed is True
    assert [candidate.fingerprint for candidate in restored.candidates] == list(fingerprints)
    assert all(candidate.test is not None for candidate in restored.candidates)
    assert restored.retrieved_evidence == {evidence.id: evidence}

    assert restored.aligned_data is not None and source.aligned_data is not None
    for name in ("asset", "benchmark"):
        pd.testing.assert_frame_equal(
            getattr(restored.aligned_data, name).bars,
            getattr(source.aligned_data, name).bars,
            check_freq=False,
        )
        assert getattr(restored.aligned_data, name).quality == (
            getattr(source.aligned_data, name).quality
        )
    assert restored.aligned_data.warnings == source.aligned_data.warnings


def test_rehydrated_success_cache_keeps_idempotent_data_tools_side_effect_free() -> None:
    artifacts = f"research/{uuid4()}"
    source, _ = _workspace_with_complete_final_batch(artifacts)
    provider = NeverProvider()
    restored = _restored(_graph_state(source), provider, artifacts)
    before = restored.model_dump(mode="json")

    assert get_market_data(restored) == source.latest_success("get_market_data")
    assert calculate_asset_statistics(restored) == source.latest_success(
        "calculate_asset_statistics"
    )
    assert provider.calls == 0
    assert restored.model_dump(mode="json") == before


def test_error_and_rejected_observations_are_not_cached_and_download_retry_is_restored() -> None:
    artifacts = f"research/{uuid4()}"
    first_provider = FakeProvider(
        {
            "AAPL": MarketDataDownloadError("first transient download failure"),
            "SPY": _market_data("SPY", scale=2.0),
        }
    )
    source = bound(ResearchWorkspace(goal=_goal(), budget=AgentBudget()), first_provider, artifacts)

    first_download = get_market_data(source)
    rejected_statistics = calculate_asset_statistics(source)
    assert first_download.status == "error"
    assert rejected_statistics.status == "rejected"
    for observation in (first_download, rejected_statistics):
        _record(source, observation)
    assert source.download_failures == 1
    assert source.aligned_data is None

    retry_provider = FakeProvider(
        {
            "AAPL": MarketDataDownloadError("second transient download failure"),
            "SPY": _market_data("SPY", scale=2.0),
        }
    )
    restored = _restored(_graph_state(source), retry_provider, artifacts)

    assert restored.download_failures == 1
    assert calculate_asset_statistics(restored).status == "rejected"
    second_download = get_market_data(restored)

    assert second_download.status == "error"
    assert restored.download_failures == 2
    assert restored.fatal_reason == "Market data download failed twice."
    assert [call[0] for call in retry_provider.calls] == ["AAPL"]
    assert restored.aligned_data is None
