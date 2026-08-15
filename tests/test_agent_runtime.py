"""Deterministic runtime tests without LangChain or external services."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from trading.agent import finalization, runtime
from trading.agent.finalization import (
    finalize_critic_review,
    finish_budget_exhausted,
    finish_failed,
)
from trading.agent.models import (
    AgentBudget,
    CriticReview,
    ResearchGoal,
    ToolObservation,
    performance_metrics_dict,
)
from trading.agent.runtime import (
    analyze_drawdown,
    backtest_fingerprint,
    calculate_asset_statistics,
    compare_with_benchmark,
    get_market_data,
    run_backtest,
    submit_research_draft,
)
from trading.agent.workspace import (
    ResearchWorkspace,
    consume_agent_step,
    consume_tool_call,
    lock_selection,
    record_event,
)
from trading.quant import (
    BacktestCosts,
    DataQualityReport,
    MarketData,
    MarketDataDownloadError,
    MarketDataError,
    MovingAverageConfig,
    RsiConfig,
    build_strategy_config,
)
from trading.reporting import load_report_view


def research_goal(**changes: object) -> ResearchGoal:
    values: dict[str, object] = {
        "objective": "Compare transparent long-only strategies.",
        "symbol": "AAPL",
        "start_date": date(2024, 1, 1),
        "end_date": date(2024, 1, 20),
        "allowed_strategies": ("moving_average",),
        "max_drawdown_limit": 0.99,
    }
    values.update(changes)
    return ResearchGoal(**values)


def market_data(symbol: str, *, scale: float = 1.0) -> MarketData:
    index = pd.date_range("2024-01-01", periods=20, freq="D")
    closes = [
        100,
        101,
        102,
        101,
        103,
        104,
        103,
        105,
        106,
        105,
        107,
        108,
        107,
        109,
        110,
        109,
        111,
        112,
        111,
        113,
    ]
    bars = pd.DataFrame(
        {
            "open": [value * scale for value in closes],
            "high": [(value + 1) * scale for value in closes],
            "low": [(value - 1) * scale for value in closes],
            "close": [value * scale for value in closes],
            "volume": [1_000.0] * len(closes),
        },
        index=index,
    )
    return MarketData(
        symbol=symbol,
        bars=bars,
        quality=DataQualityReport(
            symbol=symbol,
            valid_rows=len(bars),
            start_date=index[0],
            end_date=index[-1],
        ),
    )


class FakeProvider:
    def __init__(self, outcomes: Mapping[str, object] | None = None) -> None:
        self.outcomes = dict(
            outcomes
            or {
                "AAPL": market_data("AAPL"),
                "SPY": market_data("SPY", scale=2.0),
            }
        )
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


def workspace(
    tmp_path: Path,
    *,
    goal: ResearchGoal | None = None,
    provider: FakeProvider | None = None,
    budget: AgentBudget | None = None,
) -> ResearchWorkspace:
    return ResearchWorkspace(
        goal=goal or research_goal(),
        budget=budget or AgentBudget(),
        provider=provider or FakeProvider(),
        output_dir=tmp_path,
    )


def prepare_training(
    research: ResearchWorkspace,
    *,
    short_window: int = 2,
    long_window: int = 3,
) -> str:
    assert get_market_data(research).status == "success"
    observation = run_backtest(
        research,
        stage="training",
        strategy_name="moving_average",
        parameters={"short_window": short_window, "long_window": long_window},
    )
    assert observation.status == "success"
    return observation.identifiers["fingerprint"]


def finalize_latest_review(
    research: ResearchWorkspace,
    *,
    verdict: str = "accept",
) -> object:
    research.critic_rounds[-1].review = CriticReview(
        verdict=verdict,
        summary="The deterministic evidence supports this review.",
    )
    return finalize_critic_review(research)


def test_strategy_construction_is_exact_and_fingerprint_is_canonical() -> None:
    moving_average = build_strategy_config("moving_average", {"long_window": 20, "short_window": 5})
    rsi_ints = build_strategy_config(
        "rsi", {"rsi_window": 14, "entry_threshold": 30, "exit_threshold": 70}
    )
    rsi_floats = build_strategy_config(
        "rsi", {"exit_threshold": 70.0, "entry_threshold": 30.0, "rsi_window": 14}
    )

    assert moving_average == MovingAverageConfig(5, 20)
    assert rsi_ints == rsi_floats == RsiConfig(14, 30, 70)
    assert backtest_fingerprint(research_goal(), rsi_ints) == backtest_fingerprint(
        research_goal(), rsi_floats
    )
    with pytest.raises(ValueError, match="exactly"):
        build_strategy_config("moving_average", {"short_window": 5, "long_window": 20, "extra": 1})
    with pytest.raises(ValueError, match="unsupported"):
        build_strategy_config("bollinger", {"window": 20})


def test_fingerprint_contains_goal_strategy_parameters_and_costs() -> None:
    base_goal = research_goal()
    base_config = MovingAverageConfig(2, 3)
    baseline = backtest_fingerprint(base_goal, base_config)
    variants = [
        (research_goal(symbol="MSFT"), base_config),
        (research_goal(start_date=date(2024, 1, 2)), base_config),
        (research_goal(end_date=date(2024, 1, 19)), base_config),
        (research_goal(costs=BacktestCosts(4, 5)), base_config),
        (research_goal(costs=BacktestCosts(5, 4)), base_config),
        (base_goal, MovingAverageConfig(2, 4)),
        (base_goal, RsiConfig(2, 30, 70)),
    ]

    assert len(baseline) == 64
    assert all(backtest_fingerprint(goal, config) != baseline for goal, config in variants)


def test_runner_facing_counters_lock_and_event_shape(tmp_path: Path) -> None:
    research = workspace(
        tmp_path,
        budget=AgentBudget(
            max_agent_steps=2,
            max_tool_calls=2,
            max_training_backtests=1,
        ),
    )

    assert consume_agent_step(research) is True
    assert consume_agent_step(research) is True
    assert consume_agent_step(research) is False
    assert consume_tool_call(research) is True
    assert consume_tool_call(research) is True
    assert consume_tool_call(research) is False
    assert lock_selection(research, None) is True
    assert research.phase == "locked"
    assert research.locked_fingerprint is None
    assert lock_selection(research, "a" * 64) is False

    observation = ToolObservation(
        tool_name="get_market_data", status="rejected", summary="Market data was unavailable."
    )
    event = record_event(
        research,
        tool_name="get_market_data",
        normalized_arguments={},
        observation=observation,
        tool_call_id=None,
        replayable=False,
    )
    assert event.observation == observation
    assert event.replayable is False
    assert research.events == [event]


def test_market_data_uses_only_goal_inputs_and_success_is_cached(tmp_path: Path) -> None:
    provider = FakeProvider()
    research = workspace(tmp_path, provider=provider)

    first = get_market_data(research)
    second = get_market_data(research)

    expected = [
        ("AAPL", date(2024, 1, 1), date(2024, 1, 20)),
        ("SPY", date(2024, 1, 1), date(2024, 1, 20)),
    ]
    assert provider.calls == expected
    assert first == second
    assert research.aligned_data is not None
    assert first.metrics["common_rows"] == 20


def test_download_retries_once_but_deterministic_data_error_is_immediately_fatal(
    tmp_path: Path,
) -> None:
    download_provider = FakeProvider(
        {
            "AAPL": [
                MarketDataDownloadError("unsafe-secret-one"),
                MarketDataDownloadError("unsafe-secret-two"),
            ],
            "SPY": market_data("SPY"),
        }
    )
    retrying = workspace(tmp_path / "retry", provider=download_provider)

    first = get_market_data(retrying)
    second = get_market_data(retrying)

    assert first.status == second.status == "error"
    assert retrying.fatal_reason is not None
    assert "unsafe-secret" not in first.to_compact_json() + second.to_compact_json()
    assert download_provider.calls == [
        ("AAPL", date(2024, 1, 1), date(2024, 1, 20)),
        ("AAPL", date(2024, 1, 1), date(2024, 1, 20)),
    ]

    deterministic_provider = FakeProvider(
        {"AAPL": MarketDataError("Bad normalized data."), "SPY": market_data("SPY")}
    )
    deterministic = workspace(tmp_path / "fatal", provider=deterministic_provider)
    observation = get_market_data(deterministic)
    assert observation.status == "error"
    assert deterministic.fatal_reason is not None
    assert deterministic_provider.calls == [("AAPL", date(2024, 1, 1), date(2024, 1, 20))]


def test_data_strategy_operations_are_compact_and_do_not_require_a_fixed_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = workspace(tmp_path)
    statistics_calls = 0
    real_statistics = runtime.quant_calculate_asset_statistics

    def spy_statistics(data: MarketData):
        nonlocal statistics_calls
        statistics_calls += 1
        return real_statistics(data)

    monkeypatch.setattr(runtime, "quant_calculate_asset_statistics", spy_statistics)

    assert calculate_asset_statistics(research).status == "rejected"
    assert statistics_calls == 0

    market_data = get_market_data(research)
    first_statistics = calculate_asset_statistics(research)
    second_statistics = calculate_asset_statistics(research)

    assert market_data.status == "success"
    assert market_data.metrics["asset_valid_rows"] == 20
    assert market_data.metrics["benchmark_valid_rows"] == 20
    assert first_statistics == second_statistics
    assert statistics_calls == 1


def test_symbol_spy_warning_flows_from_data_into_runtime_state(tmp_path: Path) -> None:
    spy = market_data("SPY")
    provider = FakeProvider({"SPY": spy})
    research = workspace(tmp_path, goal=research_goal(symbol="SPY"), provider=provider)

    observation = get_market_data(research)

    assert observation.status == "success"
    assert any("same" in warning.lower() for warning in observation.warnings)
    assert any("same" in warning.lower() for warning in research.runtime_warnings)
    assert provider.calls == [("SPY", date(2024, 1, 1), date(2024, 1, 20))]


def test_training_rechecks_config_counts_only_success_and_never_exposes_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = workspace(
        tmp_path,
        budget=AgentBudget(max_training_backtests=1),
    )
    calls: list[MovingAverageConfig] = []
    real_run = runtime.quant_run_training_evaluation

    def spy_run(data, config, *, costs, position_tranches):
        calls.append(config)
        return real_run(
            data,
            config,
            costs=costs,
            position_tranches=position_tranches,
        )

    monkeypatch.setattr(runtime, "quant_run_training_evaluation", spy_run)

    no_data = run_backtest(
        research,
        stage="training",
        strategy_name="moving_average",
        parameters={"short_window": 2, "long_window": 3},
    )
    assert no_data.status == "rejected"
    assert calls == []

    assert get_market_data(research).status == "success"
    invalid = run_backtest(
        research,
        stage="training",
        strategy_name="moving_average",
        parameters={"short_window": 5, "long_window": 2},
    )
    assert invalid.status == "rejected"
    assert calls == []

    success = run_backtest(
        research,
        stage="training",
        strategy_name="moving_average",
        parameters={"short_window": 2, "long_window": 3},
    )
    duplicate = run_backtest(
        research,
        stage="training",
        strategy_name="moving_average",
        parameters={"long_window": 3, "short_window": 2},
    )
    over_budget = run_backtest(
        research,
        stage="training",
        strategy_name="moving_average",
        parameters={"short_window": 2, "long_window": 4},
    )

    assert success.status == "success"
    assert duplicate.status == over_budget.status == "rejected"
    assert "exact fingerprint" in duplicate.summary
    assert "budget" in over_budget.summary
    assert calls == [MovingAverageConfig(2, 3)]
    assert research.training_backtests == 1
    encoded = success.to_compact_json().lower()
    assert '"segment":"test"' not in encoded
    assert '"test_metrics"' not in encoded


def test_final_locks_before_atomic_batch_and_exposes_only_selected_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = workspace(
        tmp_path,
        goal=research_goal(allowed_strategies=("moving_average", "rsi")),
    )
    selected = prepare_training(research, short_window=2, long_window=3)
    other = run_backtest(
        research,
        stage="training",
        strategy_name="moving_average",
        parameters={"short_window": 2, "long_window": 4},
    ).identifiers["fingerprint"]
    missing_family = run_backtest(research, stage="final", selected_fingerprint=selected)
    assert missing_family.status == "rejected"
    assert missing_family.identifiers["strategy_name"] == "rsi"
    assert research.phase == "researching"

    first_rsi = run_backtest(
        research,
        stage="training",
        strategy_name="rsi",
        parameters={"rsi_window": 2, "entry_threshold": 40, "exit_threshold": 60},
    ).identifiers["fingerprint"]
    anomalous_family = run_backtest(research, stage="final", selected_fingerprint=selected)
    assert anomalous_family.status == "rejected"
    assert "non-positive" in anomalous_family.summary
    second_rsi = run_backtest(
        research,
        stage="training",
        strategy_name="rsi",
        parameters={"rsi_window": 3, "entry_threshold": 40, "exit_threshold": 60},
    ).identifiers["fingerprint"]

    final_calls: list[object] = []
    real_final = runtime.quant_run_final_evaluation

    def spy_final(data, config, *, costs, position_tranches):
        assert research.phase == "locked"
        final_calls.append(config)
        return real_final(
            data,
            config,
            costs=costs,
            position_tranches=position_tranches,
        )

    monkeypatch.setattr(runtime, "quant_run_final_evaluation", spy_final)

    observation = run_backtest(research, stage="final", selected_fingerprint=selected)

    assert observation.status == "success"
    assert observation.identifiers == {
        "fingerprint": selected,
        "strategy_name": "moving_average",
        "segment": "test",
    }
    assert research.phase == "locked"
    assert research.locked_fingerprint == selected
    assert research.final_batch_completed is True
    assert set(research.final_evaluations) == {selected, other, first_rsi, second_rsi}
    assert len(final_calls) == 4
    assert observation.metrics == performance_metrics_dict(
        research.final_evaluations[selected].test.metrics
    )
    assert (
        str(research.final_evaluations[other].test.metrics.total_return) not in observation.summary
    )

    monkeypatch.setattr(runtime, "quant_run_final_evaluation", real_final)
    protected_reserve = workspace(
        tmp_path / "protected-reserve",
        goal=research_goal(allowed_strategies=("moving_average", "rsi")),
    )
    reserve_selected = prepare_training(protected_reserve)
    protected_reserve.tool_calls = protected_reserve.budget.max_tool_calls - 5
    reserve_final = run_backtest(
        protected_reserve,
        stage="final",
        selected_fingerprint=reserve_selected,
    )
    assert reserve_final.status == "success"

    training_calls = 0

    def must_not_train(*_args, **_kwargs):
        nonlocal training_calls
        training_calls += 1
        raise AssertionError("training must not run after final lock")

    monkeypatch.setattr(runtime, "quant_run_training_evaluation", must_not_train)
    rejected = run_backtest(
        research,
        stage="training",
        strategy_name="moving_average",
        parameters={"short_window": 3, "long_window": 5},
    )
    assert rejected.status == "rejected"
    assert training_calls == 0


def test_final_batch_failure_is_atomic_fatal_and_keeps_locked_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = workspace(tmp_path)
    selected = prepare_training(research, short_window=2, long_window=3)
    run_backtest(
        research,
        stage="training",
        strategy_name="moving_average",
        parameters={"short_window": 2, "long_window": 4},
    )
    real_final = runtime.quant_run_final_evaluation
    calls = 0

    def fail_second(data, config, *, costs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("unsafe-internal-detail")
        return real_final(data, config, costs=costs)

    monkeypatch.setattr(runtime, "quant_run_final_evaluation", fail_second)

    observation = run_backtest(research, stage="final", selected_fingerprint=selected)

    assert observation.status == "error"
    assert "unsafe-internal-detail" not in observation.to_compact_json()
    assert research.phase == "locked"
    assert research.locked_fingerprint == selected
    assert research.final_batch_completed is False
    assert research.final_evaluations == {}
    assert research.fatal_reason is not None
    assert research.training_backtests == 2


def test_analysis_and_comparison_enforce_test_access_and_use_stored_results(tmp_path: Path) -> None:
    research = workspace(tmp_path)
    selected = prepare_training(research, short_window=2, long_window=3)
    other = run_backtest(
        research,
        stage="training",
        strategy_name="moving_average",
        parameters={"short_window": 2, "long_window": 4},
    ).identifiers["fingerprint"]

    assert analyze_drawdown(research, fingerprint=other, segment="training").status == "success"
    early_comparison = compare_with_benchmark(research, fingerprint=selected, segment="test")
    assert early_comparison.status == "rejected"
    assert early_comparison.tool_name == "compare_with_benchmark"

    assert run_backtest(research, stage="final", selected_fingerprint=selected).status == "success"
    assert analyze_drawdown(research, fingerprint=other, segment="test").status == "rejected"
    drawdown = analyze_drawdown(research, fingerprint=selected, segment="test")
    comparison = compare_with_benchmark(research, fingerprint=selected, segment="test")
    metrics = research.final_evaluations[selected].test.metrics

    assert drawdown.metrics["max_drawdown"] == metrics.max_drawdown
    assert comparison.metrics == {
        "strategy_total_return": metrics.total_return,
        "asset_buy_and_hold_return": metrics.asset_buy_and_hold_return,
        "excess_vs_asset_buy_and_hold": metrics.total_return - metrics.asset_buy_and_hold_return,
        "benchmark_return": metrics.benchmark_return,
        "excess_vs_benchmark": metrics.total_return - metrics.benchmark_return,
    }


def test_submit_selected_is_reviewed_before_trusted_result_and_report(tmp_path: Path) -> None:
    research = workspace(tmp_path)
    selected = prepare_training(research)
    assert run_backtest(research, stage="final", selected_fingerprint=selected).status == "success"

    observation = submit_research_draft(
        research,
        outcome="selected",
        selected_fingerprint=selected,
        conclusion="The frozen candidate met the drawdown constraint.",
        limitations=("Daily adjusted data only.",),
    )

    assert observation.status == "success"
    assert research.phase == "locked"
    assert research.active_agent == "critic"
    assert research.result is None

    finalize_latest_review(research)

    assert research.result is not None
    assert research.result.status == "selected"
    assert research.result.selected_fingerprint == selected
    assert research.result.training_metrics == performance_metrics_dict(
        research.final_evaluations[selected].training.metrics
    )
    assert research.result.test_metrics == performance_metrics_dict(
        research.final_evaluations[selected].test.metrics
    )
    assert (research.output_dir / "report.json").is_file()


def test_submit_gate_enforces_only_an_explicit_drawdown_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = workspace(
        tmp_path,
        goal=research_goal(max_drawdown_limit=0.0),
        budget=AgentBudget(max_training_backtests=1),
    )
    selected = prepare_training(research)
    assert run_backtest(research, stage="final", selected_fingerprint=selected).status == "success"
    report_calls = 0

    def must_not_report(*_args, **_kwargs):
        nonlocal report_calls
        report_calls += 1
        raise AssertionError("report must not run for a rejected submission")

    monkeypatch.setattr(finalization, "generate_report", must_not_report)
    observation = submit_research_draft(
        research,
        outcome="selected",
        selected_fingerprint=selected,
        conclusion="Select it.",
        limitations=(),
    )

    assert observation.status == "rejected"
    assert report_calls == 0
    assert research.phase == "locked"

    unlimited = workspace(
        tmp_path / "unlimited",
        goal=research_goal(max_drawdown_limit=None),
        budget=AgentBudget(max_training_backtests=1),
    )
    unlimited_selected = prepare_training(unlimited)
    assert (
        run_backtest(
            unlimited,
            stage="final",
            selected_fingerprint=unlimited_selected,
        ).status
        == "success"
    )
    assert (
        submit_research_draft(
            unlimited,
            outcome="selected",
            selected_fingerprint=unlimited_selected,
            conclusion="Select it without a drawdown constraint.",
            limitations=(),
        ).status
        == "success"
    )


def test_locked_candidate_can_submit_no_strategy(tmp_path: Path) -> None:
    research = workspace(tmp_path)
    selected = prepare_training(research)
    assert run_backtest(research, stage="final", selected_fingerprint=selected).status == "success"

    observation = submit_research_draft(
        research,
        outcome="no_strategy",
        selected_fingerprint=selected,
        conclusion="No training candidate met the research objective.",
        limitations=(),
    )

    assert observation.status == "success"
    assert research.result is None

    finalize_latest_review(research)

    assert research.result is not None
    assert research.result.status == "no_strategy"
    assert research.result.evaluated_fingerprint == selected
    assert research.result.training_metrics
    assert research.result.test_metrics

    rejected = workspace(tmp_path / "critic-rejected")
    rejected_fingerprint = prepare_training(rejected)
    assert (
        run_backtest(
            rejected,
            stage="final",
            selected_fingerprint=rejected_fingerprint,
        ).status
        == "success"
    )
    assert (
        submit_research_draft(
            rejected,
            outcome="selected",
            selected_fingerprint=rejected_fingerprint,
            conclusion="The Research Agent proposed the frozen candidate.",
            limitations=(),
        ).status
        == "success"
    )
    finalize_latest_review(rejected, verdict="reject")
    assert rejected.result is not None
    assert rejected.result.status == "critic_rejected"
    assert rejected.result.selected_fingerprint is None
    assert rejected.result.test_metrics
    report = load_report_view(rejected.output_dir / "report.json")
    assert report.critic_review_status == "rejected"
    assert report.selected_candidate_id is None


def test_report_failure_is_sanitized_and_runner_terminal_helpers_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = workspace(tmp_path / "submit")
    selected = prepare_training(research)
    assert run_backtest(research, stage="final", selected_fingerprint=selected).status == "success"

    def fail_report(*_args, **_kwargs):
        raise OSError("unsafe-report-secret")

    monkeypatch.setattr(finalization, "generate_report", fail_report)
    observation = submit_research_draft(
        research,
        outcome="selected",
        selected_fingerprint=selected,
        conclusion="Use it.",
        limitations=(),
    )
    assert observation.status == "success"
    result = finalize_latest_review(research)
    assert research.result is not None
    assert research.result.status == "failed"
    assert not (research.output_dir / "report.json").exists()
    assert research.phase == "completed"
    assert "unsafe-report-secret" not in (
        observation.to_compact_json() + research.result.model_dump_json()
    )
    assert getattr(result, "status") == "failed"

    failed = workspace(tmp_path / "failed")
    failed_result = finish_failed(failed, "Market data was invalid.")
    assert failed.phase == "completed"
    assert failed_result.status == "failed"
    assert not (failed.output_dir / "report.json").exists()

    exhausted = workspace(tmp_path / "budget")
    exhausted_result = finish_budget_exhausted(exhausted, "Tool call budget was exhausted.")
    assert exhausted.phase == "completed"
    assert exhausted_result.status == "budget_exhausted"
    assert not (exhausted.output_dir / "report.json").exists()


def test_budget_terminal_reports_training_only_or_preserves_locked_final_metrics(
    tmp_path: Path,
) -> None:
    training_only = workspace(tmp_path / "training-only")
    prepare_training(training_only)
    training_result = finish_budget_exhausted(
        training_only,
        "The Tool call budget was exhausted.",
    )

    assert training_only.final_evaluations == {}
    assert training_result.training_metrics == {}
    assert training_result.test_metrics == {}
    report = load_report_view(training_only.output_dir / "report.json")
    assert report.research_incomplete is True
    assert len(report.candidates) >= 1
    assert all(len(candidate.segments) == 1 for candidate in report.candidates)

    research = workspace(tmp_path / "after-final")
    selected = prepare_training(research)
    assert run_backtest(research, stage="final", selected_fingerprint=selected).status == "success"

    result = finish_budget_exhausted(research, "The Tool call budget was exhausted.")

    final = research.final_evaluations[selected]
    assert result.status == "budget_exhausted"
    assert result.evaluated_fingerprint == selected
    assert result.training_metrics == performance_metrics_dict(final.training.metrics)
    assert result.test_metrics == performance_metrics_dict(final.test.metrics)
