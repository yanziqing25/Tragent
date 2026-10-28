"""Deterministic runtime tests without LangChain or external services."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from hashlib import sha256
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tragent import storage
from tragent.quant import (
    BacktestCosts,
    DataQualityReport,
    MarketData,
    MarketDataDownloadError,
    MarketDataError,
    MovingAverageConfig,
    RsiConfig,
    build_strategy_config,
)
from tragent.research.agent import finalization, runtime
from tragent.research.agent.critic import critic_input_messages
from tragent.research.agent.finalization import (
    finalize_critic_review,
    finish_budget_exhausted,
    finish_failed,
)
from tragent.research.agent.models import (
    AgentBudget,
    CriticReview,
    MarketContextEvidence,
    ResearchGoal,
    ToolObservation,
    performance_metrics_dict,
)
from tragent.research.agent.retrieval import CONTEXT_SOURCES
from tragent.research.agent.runtime import (
    analyze_drawdown,
    backtest_fingerprint,
    calculate_asset_statistics,
    compare_with_benchmark,
    get_market_data,
    inspect_retrieved_context,
    run_backtest,
    search_market_context,
    submit_research_draft,
)
from tragent.research.agent.workspace import (
    ResearchGraphContext,
    ResearchWorkspace,
    consume_agent_step,
    record_tool_call,
)
from tragent.research.reporting import load_report_view


def _artifact_names(prefix: str) -> set[str]:
    """The names of the task artifacts stored under a prefix."""

    return {
        name for name in ("market_data.json", "report.json") if storage.exists(f"{prefix}/{name}")
    }


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
    retriever: object | None = None,
) -> ResearchWorkspace:
    return ResearchWorkspace(goal=goal or research_goal(), budget=budget or AgentBudget()).bind(
        ResearchGraphContext(
            model=MagicMock(),
            provider=provider or FakeProvider(),
            artifacts="research/t",
            retriever=retriever,  # type: ignore[arg-type]
        )
    )


def act(research: ResearchWorkspace, operation, **arguments: object) -> ToolObservation:
    """Run one Tool operation and journal it, as the Agent harness does."""

    research.agent_steps += 1
    observation = operation(research, **arguments)
    record_tool_call(
        research.journal,
        step=research.agent_steps,
        tool_name=observation.tool_name,
        arguments=arguments,
        observation=observation,
    )
    return observation


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
    with pytest.raises(ValueError, match="Unexpected keyword argument"):
        build_strategy_config("moving_average", {"short_window": 5, "long_window": 20, "extra": 1})
    with pytest.raises(ValueError, match="Field required"):
        build_strategy_config("moving_average", {"short_window": 5})
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


def test_runner_facing_counters_and_journal_shape(tmp_path: Path) -> None:
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

    observation = ToolObservation(
        tool_name="get_market_data", status="rejected", summary="Market data was unavailable."
    )
    record_tool_call(
        research.journal,
        step=research.agent_steps,
        tool_name="get_market_data",
        arguments={},
        observation=observation,
    )
    assert [(record.step, record.observation) for record in research.journal] == [(2, observation)]
    assert research.latest_success("get_market_data") is None


def test_market_data_uses_only_goal_inputs_and_success_is_cached(tmp_path: Path) -> None:
    provider = FakeProvider()
    research = workspace(tmp_path, provider=provider)

    first = act(research, get_market_data)
    second = act(research, get_market_data)

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

    first = act(retrying, get_market_data)
    second = act(retrying, get_market_data)

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

    assert act(research, calculate_asset_statistics).status == "rejected"
    assert statistics_calls == 0

    market_data = act(research, get_market_data)
    first_statistics = act(research, calculate_asset_statistics)
    second_statistics = act(research, calculate_asset_statistics)

    assert market_data.status == "success"
    assert market_data.metrics["asset_valid_rows"] == 20
    assert market_data.metrics["benchmark_valid_rows"] == 20
    assert first_statistics == second_statistics
    assert statistics_calls == 1


def test_asset_statistics_never_see_sample_out_bars(tmp_path: Path) -> None:
    changed = market_data("AAPL")
    changed.bars.loc[changed.bars.index[14:], ["open", "high", "low", "close"]] *= 0.5

    observations = []
    for name, asset in (("original", market_data("AAPL")), ("changed", changed)):
        provider = FakeProvider({"AAPL": asset, "SPY": market_data("SPY", scale=2.0)})
        research = workspace(tmp_path / name, provider=provider)
        act(research, get_market_data)
        observations.append(act(research, calculate_asset_statistics))

    assert observations[0] == observations[1]
    assert observations[0].identifiers["segment"] == "training"
    assert observations[0].metrics["observations"] == 14


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
    first_rsi = run_backtest(
        research,
        stage="training",
        strategy_name="rsi",
        parameters={"rsi_window": 2, "entry_threshold": 40, "exit_threshold": 60},
    ).identifiers["fingerprint"]
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
    assert [item.fingerprint for item in research.candidates if item.test] == [
        selected,
        other,
        first_rsi,
        second_rsi,
    ]
    assert len(final_calls) == 4
    assert observation.metrics == {
        **performance_metrics_dict(research.candidate(selected).test.metrics),
        "mean_exposure": research.candidate(selected).test.mean_exposure,
    }
    assert str(research.candidate(other).test.metrics.total_return) not in observation.summary

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
    assert all(candidate.test is None for candidate in research.candidates)
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
    metrics = research.candidate(selected).test.metrics

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
        research.candidate(selected).training.metrics
    )
    assert research.result.test_metrics == performance_metrics_dict(
        research.candidate(selected).test.metrics
    )
    assert "report.json" in _artifact_names(research.artifacts)


def test_critic_packet_carries_sample_in_asset_statistics(tmp_path: Path) -> None:
    research = workspace(tmp_path)
    selected = prepare_training(research)
    statistics = act(research, calculate_asset_statistics)
    assert run_backtest(research, stage="final", selected_fingerprint=selected).status == "success"
    assert (
        submit_research_draft(
            research,
            outcome="selected",
            selected_fingerprint=selected,
            conclusion="The frozen candidate met the drawdown constraint.",
            limitations=(),
        ).status
        == "success"
    )

    packet = json.loads(critic_input_messages(research)[0].content)["critic_review_packet"]

    assert packet["sample_in_asset_statistics"] == dict(statistics.metrics)


def test_retrieved_context_is_post_lock_cited_and_reviewable(tmp_path: Path) -> None:
    evidence_id = {source: sha256(source.encode()).hexdigest()[:24] for source in CONTEXT_SOURCES}
    barrier = Barrier(len(CONTEXT_SOURCES))
    searched_sources: list[str] = []

    class FakeRetriever:
        def search(self, **kwargs: object) -> tuple[MarketContextEvidence, ...]:
            source = str(kwargs["source"])
            barrier.wait(timeout=2)
            searched_sources.append(source)
            if source == "reddit":
                raise OSError("source unavailable")
            source_type = (
                "official"
                if source == "sec"
                else "news"
                if source == "yahoo_finance"
                else "community"
            )
            return (
                MarketContextEvidence(
                    id=evidence_id[source],
                    source_type=source_type,
                    source=source,
                    title=f"{source} evidence",
                    excerpt=f"Relevant bounded context from {source}.",
                    published_at="2024-01-15",
                    url=f"https://example.com/{source}",
                ),
            )

    research = workspace(tmp_path, retriever=FakeRetriever())
    queries = dict.fromkeys(CONTEXT_SOURCES, "material risks")
    assert search_market_context(research, queries=queries).status == "rejected"
    selected = prepare_training(research)
    assert run_backtest(research, stage="final", selected_fingerprint=selected).status == "success"

    observation = search_market_context(research, queries=queries)
    assert set(searched_sources) == set(CONTEXT_SOURCES)
    assert observation.evidence[0].id == evidence_id["sec"]
    assert observation.metrics["reddit"] is None
    assert "reddit retrieval failed." in observation.warnings
    record_tool_call(
        research.journal,
        step=research.agent_steps,
        tool_name="search_market_context",
        arguments={"queries": queries},
        observation=observation,
    )
    assert (
        submit_research_draft(
            research,
            outcome="selected",
            selected_fingerprint=selected,
            conclusion="Select the frozen candidate.",
            limitations=(),
            market_context_summary=f"The latest filing highlights uncertainty ({'f' * 24}).",
        ).status
        == "rejected"
    )
    evidence_ids = tuple(evidence_id[source] for source in CONTEXT_SOURCES[:-1])
    assert (
        submit_research_draft(
            research,
            outcome="selected",
            selected_fingerprint=selected,
            conclusion="Select the frozen candidate.",
            limitations=(),
            market_context_summary=(
                f"The official, news, and community sources were assessed ({', '.join(evidence_ids)})."
            ),
        ).status
        == "success"
    )
    inspected = inspect_retrieved_context(research)
    assert {item.id for item in inspected.evidence} == set(evidence_ids)
    assert inspected.metrics["reddit"] is None

    finalize_latest_review(research)
    report = load_report_view(f"{research.artifacts}/report.json")
    assert {item.id for item in report.market_context_sources} == set(evidence_ids)
    assert report.market_context_summary is not None


def test_context_search_covers_every_source_then_allows_one_reasoned_revision(
    tmp_path: Path,
) -> None:
    class FakeRetriever:
        def search(self, **kwargs: object) -> tuple[MarketContextEvidence, ...]:
            source, query = str(kwargs["source"]), str(kwargs["query"])
            return (
                MarketContextEvidence(
                    id=sha256(f"{source}:{query}".encode()).hexdigest()[:24],
                    source_type="official" if source == "sec" else "community",
                    source=source,
                    title=f"{source} result for {query}",
                    excerpt=f"Context about {query}.",
                    published_at="2024-01-15",
                    url=f"https://example.com/{source}/{query}",
                ),
            )

    research = workspace(tmp_path, retriever=FakeRetriever())
    selected = prepare_training(research)
    assert run_backtest(research, stage="final", selected_fingerprint=selected).status == "success"
    first = dict.fromkeys(CONTEXT_SOURCES, "Apple")
    revised = {"reddit": '"Apple Inc"'}
    reason = "Reddit results discussed other companies."

    assert act(research, search_market_context, queries={"sec": "Apple"}).status == "rejected"
    assert act(research, search_market_context, queries=first).status == "success"
    assert act(research, search_market_context, queries=revised).status == "rejected"
    revision = act(research, search_market_context, queries=revised, revision_reason=reason)
    assert revision.status == "success"
    assert act(research, search_market_context, queries=revised, revision_reason=reason).status == (
        "rejected"
    )

    assert (
        submit_research_draft(
            research,
            outcome="selected",
            selected_fingerprint=selected,
            conclusion="Select the frozen candidate.",
            limitations=(),
            market_context_summary=f"Community view ({revision.evidence[0].id}).",
        ).status
        == "success"
    )
    inspected = inspect_retrieved_context(research)
    assert inspected.metrics == dict.fromkeys(CONTEXT_SOURCES, 1)
    assert f"Revised reddit: {reason}" in inspected.summary


def test_draft_cites_exactly_the_retrieved_evidence_its_summary_names(tmp_path: Path) -> None:
    filing_ids = {"a" * 24: "First filing excerpt", "b" * 24: "Second filing excerpt"}

    class FakeRetriever:
        def search(self, **kwargs: object) -> tuple[MarketContextEvidence, ...]:
            if kwargs["source"] != "sec":
                return ()
            return tuple(
                MarketContextEvidence(
                    id=evidence_id,
                    source_type="official",
                    source="SEC",
                    title=title,
                    excerpt=f"{title}.",
                    published_at="2024-01-15",
                    url=f"https://example.com/{evidence_id}",
                )
                for evidence_id, title in filing_ids.items()
            )

    research = workspace(tmp_path, retriever=FakeRetriever())
    selected = prepare_training(research)
    assert run_backtest(research, stage="final", selected_fingerprint=selected).status == "success"
    act(research, search_market_context, queries=dict.fromkeys(CONTEXT_SOURCES, "risks"))

    def submit(summary: str) -> ToolObservation:
        return submit_research_draft(
            research,
            outcome="selected",
            selected_fingerprint=selected,
            conclusion="Select the frozen candidate.",
            limitations=(),
            market_context_summary=summary,
        )

    rejected = submit(f"Both filings note risks ({'a' * 24}, {'c' * 24}).")

    assert rejected.status == "rejected"
    assert "c" * 24 in rejected.summary
    assert submit(f"The first filing notes risks ({'a' * 24}, again {'a' * 24}).").status == (
        "success"
    )
    assert research.critic_rounds[-1].draft.cited_evidence_ids == ("a" * 24,)


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
    report = load_report_view(f"{rejected.artifacts}/report.json")
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
    assert "report.json" not in _artifact_names(research.artifacts)
    assert research.phase == "completed"
    assert "unsafe-report-secret" not in (
        observation.to_compact_json() + research.result.model_dump_json()
    )
    assert getattr(result, "status") == "failed"

    failed = workspace(tmp_path / "failed")
    failed_result = finish_failed(failed, "Market data was invalid.")
    assert failed.phase == "completed"
    assert failed_result.status == "failed"
    assert "report.json" not in _artifact_names(failed.artifacts)

    exhausted = workspace(tmp_path / "budget")
    exhausted_result = finish_budget_exhausted(exhausted, "Tool call budget was exhausted.")
    assert exhausted.phase == "completed"
    assert exhausted_result.status == "budget_exhausted"
    assert "report.json" not in _artifact_names(exhausted.artifacts)


def test_budget_terminal_reports_training_only_or_preserves_locked_final_metrics(
    tmp_path: Path,
) -> None:
    training_only = workspace(tmp_path / "training-only")
    prepare_training(training_only)
    training_result = finish_budget_exhausted(
        training_only,
        "The Tool call budget was exhausted.",
    )

    assert training_only.final_batch_completed is False
    assert training_result.training_metrics == {}
    assert training_result.test_metrics == {}
    report = load_report_view(f"{training_only.artifacts}/report.json")
    assert report.research_incomplete is True
    assert len(report.candidates) >= 1
    assert all(len(candidate.segments) == 1 for candidate in report.candidates)

    research = workspace(tmp_path / "after-final")
    selected = prepare_training(research)
    assert run_backtest(research, stage="final", selected_fingerprint=selected).status == "success"

    result = finish_budget_exhausted(research, "The Tool call budget was exhausted.")

    final = research.candidate(selected)
    assert final is not None and final.test is not None
    assert result.status == "budget_exhausted"
    assert result.evaluated_fingerprint == selected
    assert result.training_metrics == performance_metrics_dict(final.training.metrics)
    assert result.test_metrics == performance_metrics_dict(final.test.metrics)
