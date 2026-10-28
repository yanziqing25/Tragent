"""Deterministic Tool operations used by the Research Agent."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Literal

from tragent.quant import (
    AlignedMarketData,
    FinalEvaluation,
    MarketDataDownloadError,
    MarketDataError,
    StrategyConfig,
    align_market_data,
    build_strategy_config,
    strategy_name,
    strategy_parameters,
    training_asset,
)
from tragent.quant import (
    calculate_asset_statistics as quant_calculate_asset_statistics,
)
from tragent.quant import (
    run_final_evaluation as quant_run_final_evaluation,
)
from tragent.quant import (
    run_training_evaluation as quant_run_training_evaluation,
)
from tragent.quant.models import INTRADAY_RSI_BASELINE, INTRADAY_RSI_VARIANTS
from tragent.research.agent.models import (
    MarketContextEvidence,
    ResearchDraft,
    ResearchGoal,
    ToolObservation,
    evaluation_warnings,
    performance_metrics_dict,
    safe_warnings,
)
from tragent.research.agent.retrieval import CONTEXT_SOURCES, ContextSource
from tragent.research.agent.workspace import (
    CandidateRecord,
    CriticRound,
    ResearchWorkspace,
    SegmentRecord,
    ToolCallRecord,
    add_runtime_warning,
    canonical_json,
    set_fatal,
)

# Retrieved evidence IDs are the first 24 hex digits of a SHA-256 digest.
_EVIDENCE_ID = re.compile(r"\b[0-9a-f]{24}\b")


def backtest_fingerprint(goal: ResearchGoal, config: StrategyConfig) -> str:
    """Hash the complete canonical identity of one deterministic backtest."""

    payload = _backtest_payload(goal, config)
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def get_market_data(workspace: ResearchWorkspace) -> ToolObservation:
    """Load the confirmed daily or minute market-data bundle."""

    cached = workspace.latest_success("get_market_data")
    if cached is not None:
        return cached
    if workspace.fatal_reason is not None:
        return _error("get_market_data", "Research cannot continue after a fatal data error.")

    goal = workspace.goal
    try:
        if goal.bar_interval == "1m":
            assert goal.end_date is not None
            assert goal.training_sessions is not None and goal.test_sessions is not None
            asset = workspace.provider.get_minute_bars(
                goal.symbol,
                goal.end_date,
                goal.training_sessions + goal.test_sessions,
            )
            aligned = AlignedMarketData(
                asset=asset,
                benchmark=None,
                bar_interval="1m",
                session_labels=asset.session_labels,
                training_sessions=goal.training_sessions,
                test_sessions=goal.test_sessions,
            )
        else:
            assert goal.start_date is not None and goal.end_date is not None
            assert goal.benchmark_symbol is not None
            asset = workspace.provider.get_daily_bars(
                goal.symbol,
                goal.start_date,
                goal.end_date,
            )
            if goal.symbol == goal.benchmark_symbol:
                benchmark = asset
            else:
                benchmark = workspace.provider.get_daily_bars(
                    goal.benchmark_symbol,
                    goal.start_date,
                    goal.end_date,
                )
            aligned = align_market_data(asset, benchmark)
    except MarketDataDownloadError:
        workspace.download_failures += 1
        if workspace.download_failures >= 2:
            set_fatal(workspace, "Market data download failed twice.")
            return _error(
                "get_market_data",
                "Market data download failed again; the research run must stop.",
            )
        return _error(
            "get_market_data",
            "Market data download failed; the same confirmed request may be retried once.",
        )
    except MarketDataError:
        set_fatal(workspace, "Market data was invalid or insufficient.")
        return _error(
            "get_market_data",
            "Market data was invalid or insufficient; the research run must stop.",
        )

    workspace.store_market_data(aligned)
    warnings = list(aligned.warnings)
    warnings.extend(aligned.asset.quality.warnings)
    if aligned.benchmark is not None:
        warnings.extend(aligned.benchmark.quality.warnings)
    if goal.symbol == goal.benchmark_symbol:
        warning = (
            f"The asset and benchmark are both {goal.symbol}, so their buy-and-hold paths "
            "are the same."
        )
        add_runtime_warning(workspace, warning)
        warnings.append(warning)
    if goal.bar_interval == "1m":
        observation = _success(
            "get_market_data",
            (
                f"Loaded {len(aligned.asset.bars)} regular-session minute bars across "
                f"{goal.training_sessions + goal.test_sessions} completed XNYS sessions."
            ),
            identifiers={"asset": goal.symbol, "interval": "1m"},
            metrics={
                "bars": len(aligned.asset.bars),
                "training_sessions": goal.training_sessions,
                "test_sessions": goal.test_sessions,
                "asset_valid_rows": aligned.asset.quality.valid_rows,
            },
            warnings=warnings,
        )
    else:
        assert aligned.benchmark is not None and goal.benchmark_symbol is not None
        observation = _success(
            "get_market_data",
            (
                f"Loaded {len(aligned.asset.bars)} common daily rows from "
                f"{aligned.asset.bars.index[0].date()} to {aligned.asset.bars.index[-1].date()}."
            ),
            identifiers={"asset": goal.symbol, "benchmark": goal.benchmark_symbol},
            metrics={
                "common_rows": len(aligned.asset.bars),
                "asset_valid_rows": aligned.asset.quality.valid_rows,
                "benchmark_valid_rows": aligned.benchmark.quality.valid_rows,
                "asset_duplicate_rows_removed": aligned.asset.quality.duplicate_rows_removed,
                "benchmark_duplicate_rows_removed": aligned.benchmark.quality.duplicate_rows_removed,
                "asset_invalid_price_rows_removed": aligned.asset.quality.invalid_price_rows_removed,
                "benchmark_invalid_price_rows_removed": (
                    aligned.benchmark.quality.invalid_price_rows_removed
                ),
            },
            warnings=warnings,
        )
    return observation


def calculate_asset_statistics(workspace: ResearchWorkspace) -> ToolObservation:
    """Calculate and cache descriptive statistics for the asset's sample-in bars only."""

    cached = workspace.latest_success("calculate_asset_statistics")
    if cached is not None:
        return cached
    data = workspace.aligned_data
    if data is None:
        return _rejected("calculate_asset_statistics", "Market data must be loaded first.")
    try:
        statistics = quant_calculate_asset_statistics(training_asset(data))
    except ValueError:
        set_fatal(workspace, "Asset statistics could not be calculated from valid data.")
        return _error(
            "calculate_asset_statistics",
            "Asset statistics could not be calculated; the run must stop.",
        )
    return _success(
        "calculate_asset_statistics",
        f"Calculated deterministic sample-in statistics for {statistics.symbol}.",
        identifiers={"asset": statistics.symbol, "segment": "training"},
        metrics={
            "observations": statistics.observations,
            "total_return": statistics.total_return,
            "annualized_return": statistics.annualized_return,
            "annualized_volatility": statistics.annualized_volatility,
            "max_drawdown": statistics.max_drawdown,
        },
    )


def run_backtest(
    workspace: ResearchWorkspace,
    *,
    stage: Literal["training", "final"],
    strategy_name: str | None = None,
    parameters: Mapping[str, object] | None = None,
    selected_fingerprint: str | None = None,
) -> ToolObservation:
    """Run a training evaluation or irreversibly lock the atomic final batch."""

    if stage == "training":
        if strategy_name is None or parameters is None or selected_fingerprint is not None:
            return _rejected("run_backtest", "Training requires only strategy and parameters.")
        return _run_training(workspace, strategy_name=strategy_name, parameters=parameters)
    if stage == "final":
        if strategy_name is not None or parameters is not None:
            return _rejected("run_backtest", "Final evaluation accepts only a frozen fingerprint.")
        return _run_final(workspace, selected_fingerprint=selected_fingerprint)
    return _rejected("run_backtest", "stage must be training or final.")


def analyze_drawdown(
    workspace: ResearchWorkspace,
    *,
    fingerprint: str,
    segment: str,
) -> ToolObservation:
    """Read the existing deterministic drawdown analysis for an allowed segment."""

    selected = _segment_result(
        workspace,
        tool_name="analyze_drawdown",
        fingerprint=fingerprint,
        segment=segment,
    )
    if isinstance(selected, ToolObservation):
        return selected
    drawdown = selected.drawdown
    recovery = (
        "not recovered" if drawdown.recovery_date is None else str(drawdown.recovery_date.date())
    )
    return _success(
        "analyze_drawdown",
        (
            f"Peak {drawdown.peak_date.date()}, trough {drawdown.trough_date.date()}, "
            f"recovery {recovery}."
        ),
        identifiers={"fingerprint": fingerprint, "segment": segment},
        metrics={
            "max_drawdown": drawdown.max_drawdown,
            "underwater_bars": drawdown.underwater_bars,
        },
        warnings=selected.warnings,
    )


def compare_with_benchmark(
    workspace: ResearchWorkspace,
    *,
    fingerprint: str,
    segment: str,
) -> ToolObservation:
    """Compare stored strategy return with asset and configured benchmark buy-and-hold."""

    selected = _segment_result(
        workspace,
        tool_name="compare_with_benchmark",
        fingerprint=fingerprint,
        segment=segment,
    )
    if isinstance(selected, ToolObservation):
        return selected
    metrics = selected.metrics
    if metrics.benchmark_return is None:
        return _success(
            "compare_with_benchmark",
            "Compared the stored strategy return with Session Buy and Hold.",
            identifiers={"fingerprint": fingerprint, "segment": segment},
            metrics={
                "strategy_total_return": metrics.total_return,
                "asset_buy_and_hold_return": metrics.asset_buy_and_hold_return,
                "excess_vs_asset_buy_and_hold": (
                    metrics.total_return - metrics.asset_buy_and_hold_return
                ),
            },
            warnings=selected.warnings,
        )
    return _success(
        "compare_with_benchmark",
        "Compared the stored strategy return with both deterministic buy-and-hold paths.",
        identifiers={"fingerprint": fingerprint, "segment": segment},
        metrics={
            "strategy_total_return": metrics.total_return,
            "asset_buy_and_hold_return": metrics.asset_buy_and_hold_return,
            "excess_vs_asset_buy_and_hold": (
                metrics.total_return - metrics.asset_buy_and_hold_return
            ),
            "benchmark_return": metrics.benchmark_return,
            "excess_vs_benchmark": metrics.total_return - metrics.benchmark_return,
        },
        warnings=selected.warnings,
    )


def search_market_context(
    workspace: ResearchWorkspace,
    *,
    queries: Mapping[ContextSource, str],
    revision_reason: str | None = None,
) -> ToolObservation:
    """Search the queried sources concurrently: every source first, then at most one revision."""

    tool_name = "search_market_context"
    if workspace.phase != "locked" or not workspace.final_batch_completed:
        return _rejected(tool_name, "Market context is available only after final evaluation.")
    if workspace.retriever is None:
        return _error(tool_name, "Market-context retrieval is not configured.")
    searches = len(_context_searches(workspace))
    if searches == 0 and (queries.keys() != set(CONTEXT_SOURCES) or revision_reason):
        return _rejected(tool_name, "The first search queries every source, without a revision.")
    if searches == 1 and not revision_reason:
        return _rejected(tool_name, "A revision search states what was off-target.")
    if searches == 2:
        return _rejected(tool_name, "Market context allows one revision search only.")
    assert workspace.goal.end_date is not None
    sources = [source for source in CONTEXT_SOURCES if source in queries]
    with ThreadPoolExecutor(max_workers=len(sources)) as executor:
        futures = {
            source: executor.submit(
                workspace.retriever.search,
                symbol=workspace.goal.symbol,
                as_of=workspace.goal.end_date,
                query=queries[source],
                source=source,
            )
            for source in sources
        }

        results: dict[ContextSource, tuple[MarketContextEvidence, ...]] = {}
        counts: dict[str, int | None] = {}
        warnings: list[str] = []
        for source in sources:
            try:
                results[source] = futures[source].result()
                counts[source] = len(results[source])
                if not results[source]:
                    warnings.append(f"{source} returned no evidence.")
            except Exception:
                results[source] = ()
                counts[source] = None
                warnings.append(f"{source} retrieval failed.")

    evidence = _bounded_context_evidence(results)
    return _success(
        tool_name,
        (
            f"Searched {len(sources)} source(s) concurrently and surfaced "
            f"{len(evidence)} bounded evidence item(s)."
        ),
        identifiers={
            "symbol": workspace.goal.symbol,
            "sources_attempted": str(len(sources)),
        },
        metrics=counts,
        warnings=warnings,
        evidence=evidence,
    )


def _bounded_context_evidence(
    results: Mapping[ContextSource, tuple[MarketContextEvidence, ...]],
) -> tuple[MarketContextEvidence, ...]:
    """Surface every non-empty source before adding extra results, within the Tool limit."""

    selected: list[MarketContextEvidence] = []
    for rank in range(3):
        for source, found in results.items():
            if rank < len(found):
                selected.append(found[rank])
                if len(selected) == 8:
                    return tuple(selected)
    return tuple(selected)


def inspect_retrieved_context(workspace: ResearchWorkspace) -> ToolObservation:
    """Return only the external evidence cited by the active frozen draft."""

    tool_name = "inspect_retrieved_context"
    if not workspace.critic_rounds:
        return _rejected(tool_name, "No frozen Research draft is available.")
    draft = workspace.critic_rounds[-1].draft
    retrieved = workspace.retrieved_evidence
    evidence = tuple(
        retrieved[evidence_id].model_copy(update={"excerpt": retrieved[evidence_id].excerpt[:240]})
        for evidence_id in draft.cited_evidence_ids
        if evidence_id in retrieved
    )
    searches = _context_searches(workspace)
    # A source's coverage is what its latest search found.
    coverage = {
        source: found for search in searches for source, found in search.observation.metrics.items()
    }
    described = ", ".join(
        f"{source}={'failed' if found is None else found}" for source, found in coverage.items()
    )
    summary = f"Inspected {len(evidence)} cited source(s). Search coverage: {described or 'none'}."
    if len(searches) == 2:
        revision = searches[1].arguments
        summary += f" Revised {', '.join(revision['queries'])}: {revision['revision_reason']}"
    return _success(tool_name, summary, metrics=coverage, evidence=evidence)


def submit_research_draft(
    workspace: ResearchWorkspace,
    *,
    outcome: str,
    selected_fingerprint: str | None,
    conclusion: str,
    limitations: Sequence[str],
    market_context_summary: str | None = None,
) -> ToolObservation:
    """Submit one frozen Research draft for independent Critic review."""

    tool_name = "submit_research_draft"
    if workspace.active_agent != "research":
        return _rejected(tool_name, "Only the active Research Agent may submit a draft.")
    if len(workspace.critic_rounds) >= 2:
        return _rejected(tool_name, "The two allowed Critic review rounds are already used.")
    if workspace.phase != "locked" or not workspace.final_batch_completed:
        return _rejected(
            tool_name,
            "A complete frozen final evaluation is required before submission.",
        )
    if outcome not in {"selected", "no_strategy"}:
        return _rejected(tool_name, "outcome must be selected or no_strategy.")

    locked = workspace.locked_fingerprint
    if outcome == "selected":
        if locked is None or selected_fingerprint != locked:
            return _rejected(
                tool_name,
                "The selected fingerprint must match the irreversible final lock.",
            )
        final = workspace.candidate(locked)
        assert final is not None and final.test is not None
        drawdown_limit = workspace.goal.max_drawdown_limit
        if drawdown_limit is not None and (
            final.training.metrics.max_drawdown < -drawdown_limit
            or final.test.metrics.max_drawdown < -drawdown_limit
        ):
            return _rejected(
                tool_name,
                "The locked candidate does not satisfy the training and test drawdown constraint.",
            )
    elif (locked is None and selected_fingerprint is not None) or (
        locked is not None and selected_fingerprint != locked
    ):
        return _rejected(
            tool_name,
            "The no-strategy fingerprint must match the irreversible final lock semantics.",
        )

    if workspace.retriever is not None:
        if not _context_searches(workspace):
            return _rejected(
                tool_name,
                "Search the configured market-context sources before submitting.",
            )
        if not market_context_summary:
            return _rejected(
                tool_name,
                "The draft requires a cross-source market-context assessment.",
            )

    # The draft cites exactly the evidence its summary names, so the Critic reviews what it reads.
    cited_evidence_ids = tuple(dict.fromkeys(_EVIDENCE_ID.findall(market_context_summary or "")))
    unknown_evidence = set(cited_evidence_ids) - workspace.retrieved_evidence.keys()
    if unknown_evidence:
        return _rejected(
            tool_name,
            "The market-context summary names evidence that was not retrieved: "
            + ", ".join(sorted(unknown_evidence)),
        )
    if workspace.retriever is None and bool(market_context_summary) != bool(cited_evidence_ids):
        return _rejected(
            tool_name,
            "A market-context summary must name the evidence IDs it relies on.",
        )

    try:
        draft = ResearchDraft(
            outcome=outcome,
            selected_fingerprint=selected_fingerprint,
            conclusion=conclusion,
            limitations=tuple(limitations),
            market_context_summary=market_context_summary,
            cited_evidence_ids=cited_evidence_ids,
        )
    except (TypeError, ValueError):
        return _rejected(tool_name, "The Research draft text or fingerprint is invalid.")

    workspace.critic_rounds.append(CriticRound(draft=draft))
    workspace.active_agent = "critic"
    return _success(
        tool_name,
        "The frozen Research draft was submitted for independent Critic review.",
        identifiers={
            "outcome": outcome,
            "fingerprint": draft.selected_fingerprint,
            "review_round": str(len(workspace.critic_rounds)),
        },
    )


def _context_searches(workspace: ResearchWorkspace) -> list[ToolCallRecord]:
    return [
        record
        for record in workspace.journal
        if record.tool_name == "search_market_context" and record.observation.status == "success"
    ]


def _run_training(
    workspace: ResearchWorkspace,
    *,
    strategy_name: str,
    parameters: Mapping[str, object],
) -> ToolObservation:
    if workspace.fatal_reason is not None:
        return _error("run_backtest", "Research cannot continue after a fatal error.")
    if workspace.phase != "researching":
        return _rejected("run_backtest", "New training is forbidden after final lock.")
    data = workspace.aligned_data
    if data is None:
        return _rejected("run_backtest", "Market data must be loaded before training.")
    if strategy_name not in workspace.goal.allowed_strategies:
        return _rejected("run_backtest", "The strategy is not allowed by the confirmed goal.")
    if strategy_name in INTRADAY_RSI_VARIANTS and not any(
        item.strategy_name == INTRADAY_RSI_BASELINE for item in workspace.candidates
    ):
        return _rejected(
            "run_backtest", "Train intraday_rsi with parameters={} first for a matched comparison."
        )
    try:
        config = build_strategy_config(strategy_name, parameters)
        fingerprint = backtest_fingerprint(workspace.goal, config)
    except (TypeError, ValueError):
        return _rejected("run_backtest", "The strategy parameters are invalid.")
    if workspace.candidate(fingerprint) is not None:
        return _rejected(
            "run_backtest",
            "This exact fingerprint already has a successful training evaluation.",
        )
    if workspace.training_backtests >= workspace.budget.max_training_backtests:
        return _rejected("run_backtest", "The unique training backtest budget is exhausted.")
    try:
        evaluation = quant_run_training_evaluation(
            data,
            config,
            costs=workspace.goal.costs,
            position_tranches=workspace.goal.position_tranches,
        )
    except ValueError:
        return _rejected(
            "run_backtest",
            "The strategy cannot be evaluated on the available training rows.",
        )

    warnings = evaluation_warnings(evaluation, include_test=False)
    workspace.candidates.append(
        CandidateRecord(
            fingerprint=fingerprint,
            strategy_name=evaluation.strategy_name,
            parameters=strategy_parameters(config),
            training=SegmentRecord.from_result(evaluation.training, warnings),
        )
    )
    return _success(
        "run_backtest",
        "Completed a training-only evaluation; no test metrics were accessed.",
        identifiers={
            "fingerprint": fingerprint,
            "strategy_name": evaluation.strategy_name,
            "segment": "training",
        },
        metrics={
            **performance_metrics_dict(evaluation.training.metrics),
            "mean_exposure": evaluation.training.mean_exposure,
        },
        warnings=warnings,
    )


def _run_final(
    workspace: ResearchWorkspace,
    *,
    selected_fingerprint: str | None,
) -> ToolObservation:
    if workspace.fatal_reason is not None:
        return _error("run_backtest", "Research cannot continue after a fatal error.")
    if workspace.phase != "researching":
        return _rejected("run_backtest", "The final selection is already locked.")
    if not workspace.candidates:
        return _rejected("run_backtest", "At least one successful training evaluation is required.")
    if selected_fingerprint is None:
        return _rejected(
            "run_backtest", "Final evaluation requires a tested candidate fingerprint."
        )
    if workspace.candidate(selected_fingerprint) is None:
        return _rejected("run_backtest", "The selected fingerprint has no training evaluation.")
    workspace.locked_fingerprint = selected_fingerprint
    workspace.phase = "locked"

    data = workspace.aligned_data
    assert data is not None
    try:
        finals: dict[str, FinalEvaluation] = {
            candidate.fingerprint: quant_run_final_evaluation(
                data,
                candidate.config,
                costs=workspace.goal.costs,
                position_tranches=workspace.goal.position_tranches,
            )
            for candidate in workspace.candidates
        }
    except Exception:
        set_fatal(workspace, "The frozen final evaluation batch failed.")
        return _error(
            "run_backtest",
            "The frozen final evaluation batch failed atomically; no partial result was kept.",
        )

    for candidate in workspace.candidates:
        final = finals[candidate.fingerprint]
        candidate.test = SegmentRecord.from_result(
            final.test, evaluation_warnings(final, include_test=True)
        )
    selected = finals[selected_fingerprint]
    return _success(
        "run_backtest",
        "Final evaluation completed for the irreversibly locked candidate.",
        identifiers={
            "fingerprint": selected_fingerprint,
            "strategy_name": selected.strategy_name,
            "segment": "test",
        },
        metrics={
            **performance_metrics_dict(selected.test.metrics),
            "mean_exposure": selected.test.mean_exposure,
        },
        warnings=evaluation_warnings(selected, include_test=True),
    )


def _segment_result(
    workspace: ResearchWorkspace,
    *,
    tool_name: str,
    fingerprint: str,
    segment: str,
) -> SegmentRecord | ToolObservation:
    candidate = workspace.candidate(fingerprint)
    if segment == "training":
        if candidate is None:
            return _rejected(tool_name, "The fingerprint has no training evaluation.")
        return candidate.training
    if segment != "test":
        return _rejected(tool_name, "segment must be training or test.")
    if not workspace.final_batch_completed or fingerprint != workspace.locked_fingerprint:
        return _rejected(
            tool_name,
            "Test analysis is restricted to the locked non-empty fingerprint.",
        )
    assert candidate is not None and candidate.test is not None
    return candidate.test


def _backtest_payload(goal: ResearchGoal, config: StrategyConfig) -> dict[str, object]:
    return {
        "symbol": goal.symbol,
        "bar_interval": goal.bar_interval,
        "benchmark_symbol": goal.benchmark_symbol,
        "start_date": goal.start_date.isoformat() if goal.start_date else None,
        "end_date": goal.end_date.isoformat() if goal.end_date else None,
        "training_sessions": goal.training_sessions,
        "test_sessions": goal.test_sessions,
        "strategy_name": strategy_name(config),
        "normalized_parameters": strategy_parameters(config),
        "position_tranches": goal.position_tranches,
        "transaction_cost_bps": goal.costs.transaction_cost_bps,
        "slippage_bps": goal.costs.slippage_bps,
    }


def _success(
    tool_name: str,
    summary: str,
    *,
    identifiers: Mapping[str, str] | None = None,
    metrics: Mapping[str, int | float | None] | None = None,
    warnings: Sequence[str] = (),
    evidence: Sequence[MarketContextEvidence] = (),
) -> ToolObservation:
    return ToolObservation(
        tool_name=tool_name,
        status="success",
        summary=summary,
        identifiers=dict(identifiers or {}),
        metrics=dict(metrics or {}),
        warnings=safe_warnings(warnings),
        evidence=tuple(evidence),
    )


def _rejected(tool_name: str, summary: str) -> ToolObservation:
    return ToolObservation(tool_name=tool_name, status="rejected", summary=summary)


def _error(
    tool_name: str,
    summary: str,
    *,
    warnings: Sequence[str] = (),
) -> ToolObservation:
    return ToolObservation(
        tool_name=tool_name,
        status="error",
        summary=summary,
        warnings=safe_warnings(warnings),
    )


__all__ = [
    "analyze_drawdown",
    "backtest_fingerprint",
    "calculate_asset_statistics",
    "compare_with_benchmark",
    "get_market_data",
    "run_backtest",
    "search_market_context",
    "inspect_retrieved_context",
    "submit_research_draft",
]
