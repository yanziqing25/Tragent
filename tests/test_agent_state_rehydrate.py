"""Durable workspace snapshot and deterministic rehydration behavior tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from trading.agent.models import AgentBudget, ResearchGoal, ToolObservation
from trading.agent.runtime import (
    calculate_asset_statistics,
    get_market_data,
    run_backtest,
)
from trading.agent.state import (
    GraphState,
    restore_workspace_from_state,
    validate_graph_state,
    workspace_to_state,
)
from trading.agent.workspace import ResearchWorkspace, record_event
from trading.quant import DataQualityReport, MarketData, MarketDataDownloadError

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


def _record_no_args(workspace: ResearchWorkspace, observation: ToolObservation) -> None:
    _record(workspace, observation, arguments={})


def _record(
    workspace: ResearchWorkspace,
    observation: ToolObservation,
    *,
    arguments: dict[str, object],
) -> None:
    workspace.agent_steps += 1
    workspace.tool_calls += 1
    record_event(
        workspace,
        tool_name=observation.tool_name,
        normalized_arguments=arguments,
        observation=observation,
        tool_call_id=f"call-{workspace.agent_steps}",
        replayable=True,
    )


def _graph_state(workspace: ResearchWorkspace) -> GraphState:
    goal = workspace.goal
    state: dict[str, object] = {
        "thread_id": str(uuid4()),
        "user_id": 1,
        "request": goal.objective,
        "reference_date": "2026-07-13",
        "status": "running",
        "failure_stage": None,
        "confirmation_error": None,
        "failure_summary": None,
        "cost_defaults": {
            "transaction_cost_bps": goal.costs.transaction_cost_bps,
            "slippage_bps": goal.costs.slippage_bps,
        },
        "position_tranches_default": goal.position_tranches,
        "draft_goal": None,
        "confirmed_goal": goal.model_dump(mode="json"),
        "missing_goal_fields": [],
        "defaulted_fields": [],
        "budget": workspace.budget.model_dump(mode="json"),
    }
    state.update(workspace_to_state(workspace))
    return validate_graph_state(state)


def _workspace_with_complete_final_batch(
    report_dir: Path,
) -> tuple[ResearchWorkspace, tuple[str, str]]:
    provider = FakeProvider(
        {
            "AAPL": _market_data("AAPL", scale=1.0),
            "SPY": _market_data("SPY", scale=2.0),
        }
    )
    workspace = ResearchWorkspace(
        goal=_goal(),
        budget=AgentBudget(),
        provider=provider,
        output_dir=report_dir,
    )

    for operation in (
        get_market_data,
        calculate_asset_statistics,
    ):
        observation = operation(workspace)
        assert observation.status == "success"
        _record_no_args(workspace, observation)

    first = run_backtest(
        workspace,
        stage="training",
        strategy_name="moving_average",
        parameters={"short_window": 2, "long_window": 3},
    )
    second = run_backtest(
        workspace,
        stage="training",
        strategy_name="moving_average",
        parameters={"short_window": 3, "long_window": 5},
    )
    assert first.status == second.status == "success"
    first_fingerprint = first.identifiers["fingerprint"]
    second_fingerprint = second.identifiers["fingerprint"]

    final = run_backtest(
        workspace,
        stage="final",
        selected_fingerprint=first_fingerprint,
    )
    assert final.status == "success"
    workspace.runtime_warnings.append("A deterministic checkpoint warning.")
    return workspace, (first_fingerprint, second_fingerprint)


def test_workspace_round_trip_restores_complete_quant_state_without_side_effects(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / str(uuid4())
    source, fingerprints = _workspace_with_complete_final_batch(report_dir)
    state = _graph_state(source)
    provider = NeverProvider()

    _, restored = restore_workspace_from_state(state, provider=provider, output_dir=report_dir)

    assert provider.calls == 0
    assert not report_dir.exists()
    assert restored.goal == source.goal
    assert restored.budget == source.budget
    assert restored.phase == "locked"
    assert restored.locked_fingerprint == fingerprints[0]
    assert restored.final_batch_completed is True
    assert restored.current_candidate == source.current_candidate
    assert restored.validated_configs == source.validated_configs
    assert restored.agent_steps == source.agent_steps
    assert restored.tool_calls == source.tool_calls
    assert restored.training_backtests == source.training_backtests
    assert restored.download_failures == source.download_failures
    assert restored.runtime_warnings == source.runtime_warnings
    assert restored.unresolved_issues == source.unresolved_issues
    assert restored.fatal_reason is None
    assert restored.events == source.events
    assert restored.trace == source.trace
    assert restored.result is None

    assert restored.aligned_data is not None
    assert source.aligned_data is not None
    pd.testing.assert_frame_equal(
        restored.aligned_data.asset.bars,
        source.aligned_data.asset.bars,
        check_freq=False,
    )
    pd.testing.assert_frame_equal(
        restored.aligned_data.benchmark.bars,
        source.aligned_data.benchmark.bars,
        check_freq=False,
    )
    assert restored.aligned_data.asset.quality == source.aligned_data.asset.quality
    assert restored.aligned_data.benchmark.quality == source.aligned_data.benchmark.quality
    assert restored.aligned_data.warnings == source.aligned_data.warnings

    assert set(restored.training_evaluations) == set(fingerprints)
    assert set(restored.final_evaluations) == set(fingerprints)
    for fingerprint in fingerprints:
        restored_training = restored.training_evaluations[fingerprint]
        source_training = source.training_evaluations[fingerprint]
        assert restored_training.split_index == source_training.split_index
        assert restored_training.split_date == source_training.split_date
        assert restored_training.training.metrics == source_training.training.metrics
        restored_final = restored.final_evaluations[fingerprint]
        source_final = source.final_evaluations[fingerprint]
        assert restored_final.training.metrics == source_final.training.metrics
        assert restored_final.test.metrics == source_final.test.metrics


def test_rehydrated_success_cache_keeps_idempotent_data_tools_side_effect_free(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / str(uuid4())
    source, _ = _workspace_with_complete_final_batch(report_dir)
    state = _graph_state(source)
    provider = NeverProvider()
    _, restored = restore_workspace_from_state(state, provider=provider, output_dir=report_dir)
    expected = dict(source.cached_observations)
    counters_before = (
        restored.agent_steps,
        restored.tool_calls,
        restored.training_backtests,
    )
    events_before = list(restored.events)
    trace_before = list(restored.trace)

    assert restored.asset_statistics is None
    actual = {
        "get_market_data": get_market_data(restored),
        "calculate_asset_statistics": calculate_asset_statistics(restored),
    }

    assert actual == expected
    assert set(restored.cached_observations) == set(actual)
    assert restored.asset_statistics is None
    assert provider.calls == 0
    assert (
        restored.agent_steps,
        restored.tool_calls,
        restored.training_backtests,
    ) == counters_before
    assert restored.events == events_before
    assert restored.trace == trace_before
    assert not report_dir.exists()


def test_error_and_rejected_observations_are_not_cached_and_download_retry_is_restored(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / str(uuid4())
    first_provider = FakeProvider(
        {
            "AAPL": MarketDataDownloadError("first transient download failure"),
            "SPY": _market_data("SPY", scale=2.0),
        }
    )
    source = ResearchWorkspace(
        goal=_goal(),
        budget=AgentBudget(),
        provider=first_provider,
        output_dir=report_dir,
    )

    first_download = get_market_data(source)
    rejected_statistics = calculate_asset_statistics(source)
    assert first_download.status == "error"
    assert rejected_statistics.status == "rejected"
    for observation in (first_download, rejected_statistics):
        _record_no_args(source, observation)
    assert source.download_failures == 1
    assert source.cached_observations == {}

    state = _graph_state(source)
    retry_provider = FakeProvider(
        {
            "AAPL": MarketDataDownloadError("second transient download failure"),
            "SPY": _market_data("SPY", scale=2.0),
        }
    )
    _, restored = restore_workspace_from_state(
        state,
        provider=retry_provider,
        output_dir=report_dir,
    )

    assert restored.cached_observations == {}
    assert restored.download_failures == 1
    assert calculate_asset_statistics(restored).status == "rejected"
    second_download = get_market_data(restored)

    assert second_download.status == "error"
    assert restored.download_failures == 2
    assert restored.fatal_reason == "Market data download failed twice."
    assert restored.unresolved_issues == ["Market data download failed twice."]
    assert [call[0] for call in retry_provider.calls] == ["AAPL"]
    assert restored.cached_observations == {}
