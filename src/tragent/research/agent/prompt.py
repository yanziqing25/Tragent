"""Stable rules and compact context construction for the Research Agent."""

from __future__ import annotations

from dataclasses import asdict

from langchain_core.messages import BaseMessage, HumanMessage

from tragent.research.agent.models import ResearchGoal
from tragent.research.agent.workspace import (
    CandidateRecord,
    ResearchWorkspace,
    ToolCallRecord,
    canonical_json,
)

SYSTEM_PROMPT = """You are the Research Agent in a two-Agent long-only US-equity study.
Choose exactly one tool call per turn, using the confirmed goal, your earlier observations, and the
authoritative_current_state refreshed below before every turn to decide the next action and
parameters. Obtain factual results through tools; financial metrics are
authoritative tool outputs.

Periods are bars: days for 1d, minutes for 1m. Minute studies use completed XNYS regular sessions,
independent sample-in/sample-out accounts, session-end liquidation, and Session Buy and Hold.
Annualized return, volatility, and Sharpe are N/A for 1m. Limit conclusions to the observed window.

The fixed 1m policies intraday_rsi, rsi_price_exit and rsi_vwap_exit take parameters={} and have
no parameter search. They reset RSI(14,30,70) and VWAP daily, enter one configured capital tranche
after the first 60 bars, stop entries in the last 30 minutes and cool down five minutes.
intraday_rsi exits on RSI70; rsi_price_exit locks a VWAP exit for a below-VWAP entry, otherwise
RSI70; rsi_vwap_exit admits only below-VWAP entries and exits at VWAP. Train the intraday_rsi
baseline before either variant, then compare returns and exposure.

Reserve calls for final evaluation, context search, draft submission, and a possible Critic revision.
Stop exploring when evidence is sufficient. Reuse earlier observations and check tested_candidates
before proposing new parameters; avoid repeating completed work. In 1m studies, train one candidate
from every allowed family before refining parameters or selecting the final candidate, as the
remaining budget permits. For other studies, prefer untested requested families unless
observations justify refinement.
A candidate's result supports that parameterization;
family-wide conclusions require adequate exploration.
When a requested family's only candidate is clearly anomalous (very low trade count, a constraint
breach, non-positive return, or materially weak peer-relative evidence), try materially different
parameters before final selection if the family is tunable and the remaining budget covers both
that trial and reserved work.
Choose follow-up parameters from the observations; a normal informative result need not be retested.

Once training evidence is sufficient, select a tested candidate for final evaluation, obtain market
context, and finish through submit_research_draft. Both selected and no_strategy conclusions are
valid when supported by evidence. In market_context_summary, distinguish official disclosures,
third-party news, and unverified community views. Cite only evidence about this company, and
assess its relevance, agreement, conflicts, gaps, and confidence; name the supporting evidence IDs
in it, which become the draft's citations. News and community views provide context, not proof of
a fact. They may postdate the study end; present those as current background, not as information
available during the study.

Before submitting, check the draft as the Critic will: every number and comparison, including
"only" or "best" claims, matches an Observation; external views appear only through evidence IDs
named in the summary; and one candidate's result is not stated as a conclusion about its whole
family.

When critic_revision is present, address it using the frozen evidence and resubmit the draft."""


def research_system_prompt(workspace: ResearchWorkspace) -> str:
    """Stable rules plus the authoritative state, rebuilt from durable state for every call."""

    return f"{SYSTEM_PROMPT}\n\n" + canonical_json(
        {"authoritative_current_state": _state_payload(workspace)}
    )


def research_input_messages(workspace: ResearchWorkspace) -> list[BaseMessage]:
    """Open one Research run with the confirmed goal, or with the Critic's revision request."""

    payload: dict[str, object] = {"confirmed_research_goal": _goal_payload(workspace.goal)}
    revision = _critic_revision(workspace)
    if revision is not None:
        payload["critic_revision"] = revision
        payload["submitted_draft"] = workspace.critic_rounds[-1].draft.model_dump(mode="json")
        payload["retrieved_context"] = _retrieved_context(workspace.journal)
    return [HumanMessage(canonical_json(payload))]


def _goal_payload(goal: ResearchGoal) -> dict[str, object]:
    payload = goal.model_dump(mode="json")
    payload["message_type"] = "confirmed_research_goal"
    return payload


def _state_payload(workspace: ResearchWorkspace) -> dict[str, object]:
    return {
        "message_type": "authoritative_current_state",
        "phase": workspace.phase,
        "locked_candidate": _locked_candidate_payload(workspace),
        "tested_candidates": [
            _candidate_payload(candidate) for candidate in workspace.candidates[:8]
        ],
        "unresolved_issues": [workspace.fatal_reason] if workspace.fatal_reason else [],
        "final_batch_completed": workspace.final_batch_completed,
        "budget": _budget_payload(workspace),
    }


def _critic_revision(workspace: ResearchWorkspace) -> dict[str, object] | None:
    if not workspace.critic_rounds:
        return None
    review = workspace.critic_rounds[-1].review
    if review is None or review.verdict != "revise":
        return None
    return review.model_dump(mode="json")


def _retrieved_context(journal: list[ToolCallRecord]) -> list[dict[str, object]]:
    """Every successful context search, carried into a revision run's fresh conversation."""

    return [
        {
            "queries": search.arguments["queries"],
            "revision_reason": search.arguments.get("revision_reason"),
            "summary": search.observation.summary,
            "source_counts": dict(search.observation.metrics),
            "warnings": list(search.observation.warnings),
            "evidence": [
                evidence.model_dump(mode="json") for evidence in search.observation.evidence
            ],
        }
        for search in journal
        if search.tool_name == "search_market_context" and search.observation.status == "success"
    ]


def _candidate_payload(candidate: CandidateRecord) -> dict[str, object]:
    return {
        "fingerprint": candidate.fingerprint,
        "strategy_name": candidate.strategy_name,
        "parameters": candidate.parameters,
        "training_metrics": asdict(candidate.training.metrics),
        "training_mean_exposure": candidate.training.mean_exposure,
    }


def _locked_candidate_payload(workspace: ResearchWorkspace) -> dict[str, object] | None:
    locked = workspace.candidate(workspace.locked_fingerprint)
    if locked is None:
        return None
    payload = _candidate_payload(locked)
    if locked.test is not None:
        payload["test_metrics"] = asdict(locked.test.metrics)
        payload["test_mean_exposure"] = locked.test.mean_exposure
    return payload


def _budget_payload(workspace: ResearchWorkspace) -> dict[str, object]:
    budget = workspace.budget
    return {
        "agent_steps": {
            "used": workspace.agent_steps,
            "remaining": max(0, budget.max_agent_steps - workspace.agent_steps),
        },
        "tool_calls": {
            "used": workspace.tool_calls,
            "remaining": max(0, budget.max_tool_calls - workspace.tool_calls),
        },
        "training_backtests": {
            "used": workspace.training_backtests,
            "remaining": max(
                0,
                budget.max_training_backtests - workspace.training_backtests,
            ),
        },
    }
