"""Prompt and domain operations for the independent Critic Agent."""

from __future__ import annotations

from dataclasses import asdict

from langchain_core.messages import BaseMessage, HumanMessage

from tragent.research.agent.models import CriticReview, ToolObservation
from tragent.research.agent.workspace import CriticRound, ResearchWorkspace, canonical_json

MAX_CRITIC_AGENT_STEPS = 10
MAX_CRITIC_TOOL_CALLS = 6

CRITIC_SYSTEM_PROMPT = """You are the independent Critic Agent for a frozen US-equity study.
Choose exactly one tool call per turn and base each decision on the available observations.
Financial metrics are authoritative tool outputs. The candidate, parameters, and sample-out
evaluation are frozen.

For 1m, periods are minute bars, positions close each session, comparison uses Session Buy and Hold,
and annualized metrics are N/A. A short window supports a limited historical conclusion; reject
claims of long-term effectiveness, not the study merely for its duration.

Before submitting any verdict, call inspect_retrieved_context and use its Observation to verify the
draft's citations and assessment of official evidence, third-party news, and unverified community
views, including source failures, gaps, conflicts, and confidence, and that news or community
items dated after the study end are presented as current background. Use drawdown or benchmark
tools when they help resolve a review question.

Judge whether the conclusion is supported by the metrics, goal constraints, and exploration shown
in sample_in_candidates. A candidate's success or failure alone does not establish a family-wide
claim. Reject a family comparison when a requested family's only candidate is clearly anomalous
(very low trade count, a constraint breach, non-positive return, or materially weak peer-relative
evidence) and lacks a materially different follow-up. Disclosing this gap does not resolve it.
This follow-up requirement applies to tunable families. The fixed 1m intraday_rsi,
rsi_price_exit, and rsi_vwap_exit policies take parameters={}; compare the variants with the
intraday_rsi baseline on returns, costs, and average exposure rather than requesting tuning.
A normal informative result need not be retested; a well-supported no_strategy conclusion is valid.

Finish through submit_critic_review: accept a supported draft, revise a correctable evidence check,
conclusion, or limitation, and reject a draft whose correction requires another backtest or a changed
candidate. Revisions keep the frozen parameters and fingerprint; round two must accept or reject.
Reserve one decision and tool call for submission, and express unresolved uncertainty in the verdict."""


def critic_system_prompt(workspace: ResearchWorkspace) -> str:
    """Stable review rules plus the active round's budget, rebuilt for every call."""

    review_round = _active_round(workspace)
    state = {
        "message_type": "critic_current_state",
        "budget": {
            "agent_steps": {
                "used": review_round.agent_steps,
                "remaining": MAX_CRITIC_AGENT_STEPS - review_round.agent_steps,
            },
            "tool_calls": {
                "used": review_round.tool_calls,
                "remaining": MAX_CRITIC_TOOL_CALLS - review_round.tool_calls,
            },
        },
    }
    return f"{CRITIC_SYSTEM_PROMPT}\n\n" + canonical_json({"critic_current_state": state})


def critic_input_messages(workspace: ResearchWorkspace) -> list[BaseMessage]:
    """Open one review with the frozen evidence packet; no Research messages or raw data."""

    packet = _review_packet(workspace, _active_round(workspace))
    return [HumanMessage(canonical_json({"critic_review_packet": packet}))]


def submit_critic_review(
    workspace: ResearchWorkspace,
    review_round: CriticRound,
    *,
    verdict: str,
    summary: str,
    issues: tuple[str, ...],
    required_revision: str | None,
) -> ToolObservation:
    """Validate and store one Critic verdict without finalizing reports inside the Tool."""

    tool_name = "submit_critic_review"
    try:
        _require_active_round(workspace, review_round)
    except ValueError:
        return ToolObservation(
            tool_name=tool_name,
            status="rejected",
            summary="Only the active Critic round may submit a review.",
        )
    if review_round.review is not None:
        return ToolObservation(
            tool_name=tool_name,
            status="rejected",
            summary="This Critic round already has a submitted review.",
        )
    if len(workspace.critic_rounds) == 2 and verdict == "revise":
        return ToolObservation(
            tool_name=tool_name,
            status="rejected",
            summary="Round two cannot revise; submit accept or reject.",
            identifiers={
                "error_category": "field_combination",
                "required_fields": "verdict=accept|reject",
            },
        )
    try:
        review = CriticReview(
            verdict=verdict,
            summary=summary,
            issues=issues,
            required_revision=required_revision,
        )
    except (TypeError, ValueError):
        return ToolObservation(
            tool_name=tool_name,
            status="rejected",
            summary="The Critic review fields are invalid or inconsistent.",
        )

    review_round.review = review
    return ToolObservation(
        tool_name=tool_name,
        status="success",
        summary=f"Critic review round {len(workspace.critic_rounds)} submitted: {review.verdict}.",
        identifiers={
            "verdict": review.verdict,
            "review_round": str(len(workspace.critic_rounds)),
        },
    )


def _review_packet(
    workspace: ResearchWorkspace,
    review_round: CriticRound,
) -> dict[str, object]:
    selected = workspace.candidate(workspace.locked_fingerprint)
    if selected is None or selected.test is None:
        raise ValueError("Critic review requires one frozen evaluated fingerprint")
    prior_review = (
        workspace.critic_rounds[-2].review.model_dump(mode="json")
        if len(workspace.critic_rounds) == 2 and workspace.critic_rounds[-2].review is not None
        else None
    )
    asset_statistics = workspace.latest_success("calculate_asset_statistics")
    if workspace.goal.bar_interval == "1m":
        split = (
            f"{workspace.goal.training_sessions} complete XNYS sample-in session(s) / "
            f"{workspace.goal.test_sessions} complete XNYS sample-out session(s)"
        )
        signal_execution = (
            "A close signal executes no earlier than the next minute open; positions are "
            "liquidated at each session's final minute open."
        )
    else:
        split = "70/30 chronological sample-in/sample-out"
        signal_execution = "A close signal executes no earlier than the next bar."
    return {
        "message_type": "critic_review_packet",
        "review_round": len(workspace.critic_rounds),
        "goal": workspace.goal.model_dump(mode="json"),
        "research_draft": review_round.draft.model_dump(mode="json"),
        "locked_candidate": {
            "fingerprint": selected.fingerprint,
            "strategy_name": selected.strategy_name,
            "parameters": selected.parameters,
            "in_sample_metrics": asdict(selected.training.metrics),
            "out_of_sample_metrics": asdict(selected.test.metrics),
            "in_sample_mean_exposure": selected.training.mean_exposure,
            "out_of_sample_mean_exposure": selected.test.mean_exposure,
        },
        "sample_in_candidates": [
            {
                "fingerprint": candidate.fingerprint,
                "strategy_name": candidate.strategy_name,
                "parameters": candidate.parameters,
                "metrics": asdict(candidate.training.metrics),
                "mean_exposure": candidate.training.mean_exposure,
            }
            for candidate in workspace.candidates[:8]
        ],
        "sample_in_asset_statistics": None
        if asset_statistics is None
        else dict(asset_statistics.metrics),
        "method": {
            "split": split,
            "signal_execution": signal_execution,
            "transaction_cost_bps": workspace.goal.costs.transaction_cost_bps,
            "slippage_bps": workspace.goal.costs.slippage_bps,
        },
        "warnings": list(selected.test.warnings),
        "previous_critic_review": prior_review,
    }


def _require_active_round(
    workspace: ResearchWorkspace,
    review_round: CriticRound,
) -> None:
    if _active_round(workspace) is not review_round:
        raise ValueError("Critic operation requires the active review round")


def _active_round(workspace: ResearchWorkspace) -> CriticRound:
    if workspace.active_agent != "critic" or not workspace.critic_rounds:
        raise ValueError("Critic operation requires the active review round")
    return workspace.critic_rounds[-1]


__all__ = [
    "CRITIC_SYSTEM_PROMPT",
    "MAX_CRITIC_AGENT_STEPS",
    "MAX_CRITIC_TOOL_CALLS",
    "critic_input_messages",
    "critic_system_prompt",
    "submit_critic_review",
]
