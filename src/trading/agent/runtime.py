"""Deterministic Tool operations used by the Research Agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256

from trading.agent.models import (
    ResearchDraft,
    ResearchGoal,
    ToolObservation,
    evaluation_warnings,
    performance_metrics_dict,
    safe_warnings,
)
from trading.agent.workspace import (
    CriticRoundRuntime,
    ResearchWorkspace,
    add_runtime_warning,
    canonical_json,
    lock_selection,
    set_fatal,
)
from trading.quant import (
    AlignedMarketData,
    FinalEvaluation,
    MarketDataDownloadError,
    MarketDataError,
    StrategyConfig,
    TrainingEvaluation,
    align_market_data,
    build_strategy_config,
    strategy_name,
    strategy_parameters,
)
from trading.quant import (
    calculate_asset_statistics as quant_calculate_asset_statistics,
)
from trading.quant import (
    run_final_evaluation as quant_run_final_evaluation,
)
from trading.quant import (
    run_training_evaluation as quant_run_training_evaluation,
)

_SAME_BUY_AND_HOLD_WARNING = (
    "The asset and benchmark are both SPY, so their two buy-and-hold paths are the same."
)


def backtest_fingerprint(goal: ResearchGoal, config: StrategyConfig) -> str:
    """Hash the complete canonical identity of one deterministic backtest."""

    payload = _backtest_payload(goal, config)
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def get_market_data(workspace: ResearchWorkspace) -> ToolObservation:
    """Load the confirmed daily or minute market-data bundle."""

    cached = workspace.cached_observations.get("get_market_data")
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
    except Exception:
        set_fatal(workspace, "An unexpected market data error occurred.")
        return _error("get_market_data", "An unexpected market data error stopped the run.")

    workspace.aligned_data = aligned
    warnings = list(aligned.warnings)
    warnings.extend(aligned.asset.quality.warnings)
    if aligned.benchmark is not None:
        warnings.extend(aligned.benchmark.quality.warnings)
    if goal.symbol == goal.benchmark_symbol:
        add_runtime_warning(workspace, _SAME_BUY_AND_HOLD_WARNING)
        warnings.append(_SAME_BUY_AND_HOLD_WARNING)
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
    workspace.cached_observations["get_market_data"] = observation
    return observation


def calculate_asset_statistics(workspace: ResearchWorkspace) -> ToolObservation:
    """Calculate and cache M1 descriptive statistics for the confirmed asset."""

    cached = workspace.cached_observations.get("calculate_asset_statistics")
    if cached is not None:
        return cached
    if workspace.aligned_data is None:
        return _rejected("calculate_asset_statistics", "Market data must be loaded first.")
    try:
        statistics = quant_calculate_asset_statistics(workspace.aligned_data.asset)
    except ValueError:
        set_fatal(workspace, "Asset statistics could not be calculated from valid data.")
        return _error(
            "calculate_asset_statistics",
            "Asset statistics could not be calculated; the run must stop.",
        )
    except Exception:
        set_fatal(workspace, "An unexpected asset statistics error occurred.")
        return _error(
            "calculate_asset_statistics",
            "An unexpected asset statistics error stopped the run.",
        )
    workspace.asset_statistics = statistics
    observation = _success(
        "calculate_asset_statistics",
        f"Calculated deterministic statistics for {statistics.symbol}.",
        identifiers={"asset": statistics.symbol},
        metrics={
            "observations": statistics.observations,
            "total_return": statistics.total_return,
            "annualized_return": statistics.annualized_return,
            "annualized_volatility": statistics.annualized_volatility,
            "max_drawdown": statistics.max_drawdown,
        },
    )
    workspace.cached_observations["calculate_asset_statistics"] = observation
    return observation


def run_backtest(
    workspace: ResearchWorkspace,
    *,
    stage: str,
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
    """Compare stored strategy return with asset and SPY buy-and-hold."""

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


def submit_research_draft(
    workspace: ResearchWorkspace,
    *,
    outcome: str,
    selected_fingerprint: str | None,
    conclusion: str,
    limitations: Sequence[str],
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
        final = workspace.final_evaluations[locked]
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

    try:
        draft = ResearchDraft(
            outcome=outcome,
            selected_fingerprint=selected_fingerprint,
            conclusion=conclusion,
            limitations=tuple(limitations),
        )
    except (TypeError, ValueError):
        return _rejected(tool_name, "The Research draft text or fingerprint is invalid.")

    workspace.critic_rounds.append(CriticRoundRuntime(draft=draft))
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
    if workspace.aligned_data is None:
        return _rejected("run_backtest", "Market data must be loaded before training.")
    if strategy_name not in workspace.goal.allowed_strategies:
        return _rejected("run_backtest", "The strategy is not allowed by the confirmed goal.")
    try:
        config = build_strategy_config(strategy_name, parameters)
        fingerprint = backtest_fingerprint(workspace.goal, config)
    except (TypeError, ValueError):
        return _rejected("run_backtest", "The strategy parameters are invalid.")
    if fingerprint in workspace.training_evaluations:
        return _rejected(
            "run_backtest",
            "This exact fingerprint already has a successful training evaluation.",
        )
    if workspace.training_backtests >= workspace.budget.max_training_backtests:
        return _rejected("run_backtest", "The unique training backtest budget is exhausted.")
    try:
        evaluation = quant_run_training_evaluation(
            workspace.aligned_data,
            config,
            costs=workspace.goal.costs,
            position_tranches=workspace.goal.position_tranches,
        )
    except ValueError:
        return _rejected(
            "run_backtest",
            "The strategy cannot be evaluated on the available training rows.",
        )
    except Exception:
        set_fatal(workspace, "An unexpected training backtest error occurred.")
        return _error("run_backtest", "An unexpected training error stopped the run.")

    workspace.current_candidate = config
    workspace.validated_configs[fingerprint] = config
    workspace.fingerprint_payloads[fingerprint] = _backtest_payload(workspace.goal, config)
    workspace.training_evaluations[fingerprint] = evaluation
    workspace.training_backtests += 1
    return _success(
        "run_backtest",
        "Completed a training-only evaluation; no test metrics were accessed.",
        identifiers={
            "fingerprint": fingerprint,
            "strategy_name": evaluation.strategy_name,
            "segment": "training",
        },
        metrics=performance_metrics_dict(evaluation.training.metrics),
        warnings=evaluation_warnings(evaluation, include_test=False),
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
    if not workspace.training_evaluations:
        return _rejected("run_backtest", "At least one successful training evaluation is required.")
    if selected_fingerprint is None:
        return _rejected(
            "run_backtest", "Final evaluation requires a tested candidate fingerprint."
        )
    if selected_fingerprint not in workspace.training_evaluations:
        return _rejected("run_backtest", "The selected fingerprint has no training evaluation.")
    readiness_issue = _final_readiness_issue(workspace)
    if readiness_issue is not None:
        strategy_name, reason = readiness_issue
        return ToolObservation(
            tool_name="run_backtest",
            status="rejected",
            summary=(
                f"Sample-in evidence is not ready for final selection: {strategy_name} {reason}. "
                f"Run one materially different training candidate for {strategy_name}, then "
                "reconsider final."
            ),
            identifiers={
                "error_category": "insufficient_sample_in_evidence",
                "strategy_name": strategy_name,
                "required_action": "run_backtest_training",
            },
        )
    if not lock_selection(workspace, selected_fingerprint):
        return _rejected("run_backtest", "The final selection could not be locked.")

    temporary: dict[str, FinalEvaluation] = {}
    try:
        assert workspace.aligned_data is not None
        for fingerprint, training in workspace.training_evaluations.items():
            temporary[fingerprint] = quant_run_final_evaluation(
                workspace.aligned_data,
                training.config,
                costs=workspace.goal.costs,
                position_tranches=workspace.goal.position_tranches,
            )
    except Exception:
        temporary.clear()
        workspace.final_evaluations.clear()
        workspace.final_batch_completed = False
        set_fatal(workspace, "The frozen final evaluation batch failed.")
        return _error(
            "run_backtest",
            "The frozen final evaluation batch failed atomically; no partial result was kept.",
        )

    workspace.final_evaluations = temporary
    workspace.final_batch_completed = True
    selected = temporary[selected_fingerprint]
    return _success(
        "run_backtest",
        "Final evaluation completed for the irreversibly locked candidate.",
        identifiers={
            "fingerprint": selected_fingerprint,
            "strategy_name": selected.strategy_name,
            "segment": "test",
        },
        metrics=performance_metrics_dict(selected.test.metrics),
        warnings=evaluation_warnings(selected, include_test=True),
    )


def _final_readiness_issue(workspace: ResearchWorkspace) -> tuple[str, str] | None:
    if workspace.training_backtests >= workspace.budget.max_training_backtests:
        return None
    remaining_tool_calls = workspace.budget.max_tool_calls - workspace.tool_calls
    if remaining_tool_calls <= 5:
        return None

    by_strategy: dict[str, list[TrainingEvaluation]] = {
        family_name: [] for family_name in workspace.goal.allowed_strategies
    }
    for evaluation in workspace.training_evaluations.values():
        by_strategy[evaluation.strategy_name].append(evaluation)

    for family_name in workspace.goal.allowed_strategies:
        evaluations = by_strategy[family_name]
        if not evaluations:
            return family_name, "has no successful training candidate"
        if len(evaluations) != 1:
            continue
        metrics = evaluations[0].training.metrics
        if metrics.total_return <= 0:
            return family_name, "has only one candidate with non-positive total return"
        drawdown_limit = workspace.goal.max_drawdown_limit
        if drawdown_limit is not None and metrics.max_drawdown < -drawdown_limit:
            return family_name, "has only one candidate that breaches the drawdown constraint"
    return None


def _segment_result(
    workspace: ResearchWorkspace,
    *,
    tool_name: str,
    fingerprint: str,
    segment: str,
):
    if segment == "training":
        evaluation = workspace.training_evaluations.get(fingerprint)
        if evaluation is None:
            return _rejected(tool_name, "The fingerprint has no training evaluation.")
        return evaluation.training
    if segment != "test":
        return _rejected(tool_name, "segment must be training or test.")
    if (
        workspace.phase == "researching"
        or not workspace.final_batch_completed
        or workspace.locked_fingerprint is None
        or fingerprint != workspace.locked_fingerprint
    ):
        return _rejected(
            tool_name,
            "Test analysis is restricted to the locked non-empty fingerprint.",
        )
    evaluation = workspace.final_evaluations.get(fingerprint)
    if evaluation is None:
        return _rejected(tool_name, "The locked fingerprint has no final evaluation.")
    return evaluation.test


def _backtest_payload(goal: ResearchGoal, config: StrategyConfig) -> dict[str, object]:
    return {
        "symbol": goal.symbol,
        "bar_interval": goal.bar_interval,
        "benchmark_symbol": goal.benchmark_symbol,
        "start_date": None if goal.start_date is None else goal.start_date.isoformat(),
        "end_date": None if goal.end_date is None else goal.end_date.isoformat(),
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
) -> ToolObservation:
    return ToolObservation(
        tool_name=tool_name,
        status="success",
        summary=summary,
        identifiers=dict(identifiers or {}),
        metrics=dict(metrics or {}),
        warnings=safe_warnings(warnings),
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
    "submit_research_draft",
]
