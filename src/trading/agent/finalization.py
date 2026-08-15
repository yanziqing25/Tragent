"""Terminal Research Agent results and report generation."""

from __future__ import annotations

from typing import Literal

from trading.agent.models import (
    BudgetUsage,
    ResearchResult,
    evaluation_warnings,
    performance_metrics_dict,
    safe_warnings,
)
from trading.agent.workspace import (
    ResearchWorkspace,
    add_runtime_warning,
    set_fatal,
)
from trading.reporting import (
    CriticReviewReport,
    ReportInput,
    generate_report,
)


def finalize_critic_review(workspace: ResearchWorkspace) -> ResearchResult:
    """Generate the formal terminal result after a final Critic verdict."""

    if (
        workspace.active_agent != "critic"
        or not workspace.critic_rounds
        or workspace.critic_rounds[-1].review is None
    ):
        raise ValueError("formal finalization requires the active Critic verdict")
    review_round = workspace.critic_rounds[-1]
    review = review_round.review
    assert review is not None
    if review.verdict == "revise":
        raise ValueError("a revise verdict is not terminal")

    draft = review_round.draft
    locked = workspace.locked_fingerprint
    if locked is None or draft.selected_fingerprint != locked:
        raise ValueError("the reviewed draft does not match the frozen fingerprint")
    status = draft.outcome if review.verdict == "accept" else "critic_rejected"
    selected_fingerprint = locked if status == "selected" else None
    stop_reason = (
        None
        if status == "selected"
        else (
            "No strategy was selected after independent Critic review."
            if status == "no_strategy"
            else "The independent Critic rejected the submitted research conclusion."
        )
    )
    try:
        _write_report(
            workspace,
            selected_fingerprint=selected_fingerprint,
            stop_reason=stop_reason,
            limitations=draft.limitations,
        )
    except OSError:
        add_runtime_warning(workspace, "Report artifacts could not be written.")
        set_fatal(workspace, "Report generation failed.")
        return _finish_terminal(
            workspace,
            status="failed",
            stop_reason="Report generation failed.",
            attempt_report=False,
        )

    final = workspace.final_evaluations[locked]
    result = ResearchResult(
        status=status,  # type: ignore[arg-type]
        goal=workspace.goal,
        selected_fingerprint=selected_fingerprint,
        evaluated_fingerprint=locked,
        training_metrics=performance_metrics_dict(final.training.metrics),
        test_metrics=performance_metrics_dict(final.test.metrics),
        conclusion=draft.conclusion,
        stop_reason=stop_reason,
        warnings=_collect_result_warnings(workspace),
        limitations=draft.limitations,
        budget_usage=_budget_usage(workspace),
        trace=tuple(workspace.trace),
    )

    workspace.result = result
    workspace.phase = "completed"
    workspace.active_agent = None
    return result


def finish_failed(workspace: ResearchWorkspace, stop_reason: str) -> ResearchResult:
    """Finish a run that encountered a fatal error."""

    return _finish_terminal(
        workspace,
        status="failed",
        stop_reason=stop_reason,
        attempt_report=True,
    )


def finish_budget_exhausted(workspace: ResearchWorkspace, stop_reason: str) -> ResearchResult:
    """Finish a run that exhausted its Agent budget."""

    return _finish_terminal(
        workspace,
        status="budget_exhausted",
        stop_reason=stop_reason,
        attempt_report=True,
    )


def _finish_terminal(
    workspace: ResearchWorkspace,
    *,
    status: Literal["failed", "budget_exhausted"],
    stop_reason: str,
    attempt_report: bool,
) -> ResearchResult:
    if workspace.phase == "completed" and workspace.result is not None:
        return workspace.result
    safe_reason = (stop_reason.strip() or "The research run stopped safely.")[:2_000]
    if attempt_report:
        try:
            _write_report(
                workspace,
                selected_fingerprint=None,
                stop_reason=safe_reason,
                limitations=(),
            )
        except OSError:
            add_runtime_warning(workspace, "Report artifacts could not be written.")
    evaluated_fingerprint = (
        workspace.locked_fingerprint if workspace.phase != "researching" else None
    )
    training_metrics: dict[str, int | float | None] = {}
    test_metrics: dict[str, int | float | None] = {}
    if evaluated_fingerprint is not None and workspace.final_batch_completed:
        final = workspace.final_evaluations.get(evaluated_fingerprint)
        if final is not None:
            training_metrics = performance_metrics_dict(final.training.metrics)
            test_metrics = performance_metrics_dict(final.test.metrics)
    result = ResearchResult(
        status=status,
        goal=workspace.goal,
        evaluated_fingerprint=evaluated_fingerprint,
        training_metrics=training_metrics,
        test_metrics=test_metrics,
        conclusion="The research run stopped safely without selecting a strategy.",
        stop_reason=safe_reason,
        warnings=_collect_result_warnings(workspace),
        budget_usage=_budget_usage(workspace),
        trace=tuple(workspace.trace),
    )
    workspace.result = result
    workspace.phase = "completed"
    workspace.active_agent = None
    return result


def _write_report(
    workspace: ResearchWorkspace,
    *,
    selected_fingerprint: str | None,
    stop_reason: str | None,
    limitations: tuple[str, ...],
) -> None:
    generate_report(
        _report_input(
            workspace,
            selected_fingerprint=selected_fingerprint,
            stop_reason=stop_reason,
            limitations=limitations,
        )
    )


def _report_input(
    workspace: ResearchWorkspace,
    *,
    selected_fingerprint: str | None,
    stop_reason: str | None,
    limitations: tuple[str, ...],
) -> ReportInput:
    return ReportInput(
        research_objective=workspace.goal.objective,
        symbol=workspace.goal.symbol,
        benchmark_symbol=workspace.goal.benchmark_symbol,
        bar_interval=workspace.goal.bar_interval,
        final_evaluations=workspace.final_evaluations,
        training_evaluations=workspace.training_evaluations,
        output_dir=workspace.output_dir,
        selected_fingerprint=selected_fingerprint,
        stop_reason=stop_reason,
        limitations=limitations,
        runtime_warnings=tuple(workspace.runtime_warnings),
        critic_reviews=tuple(
            CriticReviewReport(
                verdict=review.verdict,
                summary=review.summary,
                issues=review.issues,
                required_revision=review.required_revision,
            )
            for review_round in workspace.critic_rounds
            if (review := review_round.review) is not None
        ),
        critic_review_status=_critic_report_status(workspace),
    )


def _critic_report_status(
    workspace: ResearchWorkspace,
) -> Literal["not_started", "incomplete", "revision_requested", "accepted", "rejected"]:
    if not workspace.critic_rounds:
        return "not_started"
    review = workspace.critic_rounds[-1].review
    if review is None:
        return "incomplete"
    return {
        "accept": "accepted",
        "revise": "revision_requested",
        "reject": "rejected",
    }[review.verdict]


def _budget_usage(workspace: ResearchWorkspace) -> BudgetUsage:
    return BudgetUsage(
        agent_steps=workspace.agent_steps,
        tool_calls=workspace.tool_calls,
        training_backtests=workspace.training_backtests,
    )


def _collect_result_warnings(workspace: ResearchWorkspace) -> tuple[str, ...]:
    collected = list(workspace.runtime_warnings)
    for evaluation in workspace.final_evaluations.values():
        collected.extend(evaluation_warnings(evaluation, include_test=True))
    return safe_warnings(collected)


__all__ = ["finalize_critic_review", "finish_budget_exhausted", "finish_failed"]
