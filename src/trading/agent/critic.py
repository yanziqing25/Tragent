"""Prompt and domain operations for the independent Critic Agent."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from langchain_core.messages import BaseMessage, convert_to_messages

from trading.agent.models import CriticReview, ToolObservation
from trading.agent.workspace import (
    CriticRoundRuntime,
    ResearchWorkspace,
    RuntimeEvent,
)
from trading.quant import strategy_parameters

MAX_CRITIC_AGENT_STEPS = 10
MAX_CRITIC_TOOL_CALLS = 6

CRITIC_SYSTEM_PROMPT = """You are the independent Critic Agent for a frozen US-equity study.
Use only the bound read-only tools and submit exactly one Tool call per turn.
If evidence is needed, call one evidence Tool now and wait for its Observation before submitting;
never request evidence and submit a verdict in the same response.
Never calculate, invent, correct, or override financial metrics yourself.
For 1m studies, periods mean minute bars, actual positions close every session, comparison uses
Session Buy and Hold without SPY, and annualized metrics are intentionally unavailable. Treat a
short session window as narrow evidence; do not reject it merely for being short, but reject any
conclusion that generalizes it into long-term effectiveness.
The selected fingerprint, strategy, parameters, and sample-out evaluation are irreversible.
You may inspect deterministic drawdown or benchmark evidence when it materially helps the review.
Then finish through submit_critic_review with an evidence-grounded accept, revise, or reject verdict.
A verdict evaluates whether the Research draft is supported by sufficient evidence, not whether you
personally favor the locked candidate. One failed default parameterization does not eliminate its
entire family, and one passing parameterization does not by itself prove that a claimed family
comparison was adequately explored. Use the supplied sample-in candidate summaries to assess
parameter sensitivity. If missing exploration would require another backtest, reject rather than
requesting an impossible revision. A failed candidate may still support a well-evidenced
no_strategy draft. Do not accept merely because the draft discloses shallow evidence: if a requested
family has one clearly anomalous candidate—such as very low trade count, a constraint breach,
non-positive return, or materially weak peer-relative evidence—and no materially different
follow-up candidate, reject the family comparison. Do not require two candidates for every family
when the first result is already normal and informative.
A revise verdict may request only a clearer evidence check, conclusion, or limitation; it may not
request another backtest, different parameters, or a different fingerprint. Round two cannot revise.
Reserve one model decision and one ToolNode attempt for submit_critic_review; budgets are ceilings,
not targets, and uncertainty may be expressed through revise or reject instead of more Tool calls.
Do not claim to have recomputed or mathematically proved the backtest. Do not reveal hidden reasoning.
Tool order remains your dynamic decision based on the observations supplied in this context."""


def build_critic_messages(
    workspace: ResearchWorkspace,
    review_round: CriticRoundRuntime,
) -> list[BaseMessage]:
    """Build one bounded context without Research messages or raw market data."""

    _require_active_round(workspace, review_round)
    recent_events = review_round.events[-4:]
    context = {
        "critic_review_packet": _review_packet(workspace, review_round),
        "critic_current_state": {
            "message_type": "critic_current_state",
            "recent_actions": [
                {
                    "tool_name": event.tool_name,
                    "arguments": event.normalized_arguments,
                    "observation": event.observation.model_dump(mode="json"),
                }
                for event in recent_events
            ],
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
        },
    }
    return convert_to_messages(
        [
            ("developer", CRITIC_SYSTEM_PROMPT),
            ("user", _compact_json(context)),
        ]
    )


def submit_critic_review(
    workspace: ResearchWorkspace,
    review_round: CriticRoundRuntime,
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
    review_round: CriticRoundRuntime,
) -> dict[str, object]:
    fingerprint = workspace.locked_fingerprint
    if fingerprint is None or fingerprint not in workspace.final_evaluations:
        raise ValueError("Critic review requires one frozen evaluated fingerprint")
    selected = workspace.final_evaluations[fingerprint]
    prior_review = (
        workspace.critic_rounds[-2].review.model_dump(mode="json")
        if len(workspace.critic_rounds) == 2 and workspace.critic_rounds[-2].review is not None
        else None
    )
    if workspace.goal.bar_interval == "1m":
        split = (
            f"{workspace.goal.training_sessions} complete XNYS sample-in session(s) / "
            f"{workspace.goal.test_sessions} complete XNYS sample-out session(s)"
        )
        signal_execution = (
            "A close signal executes no earlier than the next minute open; positions are "
            "liquidated at each session close."
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
            "fingerprint": fingerprint,
            "strategy_name": selected.strategy_name,
            "parameters": strategy_parameters(selected.config),
            "in_sample_metrics": asdict(selected.training.metrics),
            "out_of_sample_metrics": asdict(selected.test.metrics),
        },
        "sample_in_candidates": [
            {
                "fingerprint": candidate_fingerprint,
                "strategy_name": evaluation.strategy_name,
                "parameters": strategy_parameters(evaluation.config),
                "metrics": asdict(evaluation.training.metrics),
            }
            for candidate_fingerprint, evaluation in list(workspace.training_evaluations.items())[
                :8
            ]
        ],
        "method": {
            "split": split,
            "signal_execution": signal_execution,
            "transaction_cost_bps": workspace.goal.costs.transaction_cost_bps,
            "slippage_bps": workspace.goal.costs.slippage_bps,
        },
        "warnings": _warnings(selected),
        "previous_critic_review": prior_review,
    }


def _warnings(evaluation: Any) -> list[str]:
    quality_warnings = list(evaluation.asset_quality.warnings)
    if evaluation.benchmark_quality is not None:
        quality_warnings.extend(evaluation.benchmark_quality.warnings)
    values = [
        *quality_warnings,
        *evaluation.warnings,
        *evaluation.training.warnings,
        *evaluation.test.warnings,
    ]
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value not in result:
            result.append(value[:300])
        if len(result) == 10:
            break
    return result


def _require_active_round(
    workspace: ResearchWorkspace,
    review_round: CriticRoundRuntime,
) -> None:
    if (
        workspace.active_agent != "critic"
        or not workspace.critic_rounds
        or workspace.critic_rounds[-1] is not review_round
    ):
        raise ValueError("Critic operation requires the active review round")


def record_critic_event(
    review_round: CriticRoundRuntime,
    *,
    tool_name: str,
    arguments: Mapping[str, object],
    observation: ToolObservation,
    tool_call_id: str | None,
    replayable: bool,
) -> None:
    normalized = json.loads(_compact_json(dict(arguments)))
    assert isinstance(normalized, dict)
    review_round.events.append(
        RuntimeEvent(
            tool_name=tool_name,
            normalized_arguments=normalized,
            observation=observation,
            tool_call_id=tool_call_id,
            replayable=bool(replayable and tool_call_id),
        )
    )
    review_round.event_steps.append(review_round.agent_steps)


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "CRITIC_SYSTEM_PROMPT",
    "MAX_CRITIC_AGENT_STEPS",
    "MAX_CRITIC_TOOL_CALLS",
    "build_critic_messages",
    "record_critic_event",
    "submit_critic_review",
]
