"""Tests for the Research Agent's deliberately compact model context."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from tragent.quant import BacktestCosts, PerformanceMetrics
from tragent.research.agent.models import (
    AgentBudget,
    CriticReview,
    MarketContextEvidence,
    ResearchDraft,
    ResearchGoal,
    ToolObservation,
)
from tragent.research.agent.prompt import (
    SYSTEM_PROMPT,
    research_input_messages,
    research_system_prompt,
)
from tragent.research.agent.workspace import (
    CandidateRecord,
    CriticRound,
    DrawdownRecord,
    ResearchWorkspace,
    SegmentRecord,
    ToolCallRecord,
)

LOCKED_FINGERPRINT = "a" * 64
OTHER_FINGERPRINT = "b" * 64
MOVING_AVERAGE = {"short_window": 20, "long_window": 100}
RSI = {"rsi_window": 14, "entry_threshold": 30, "exit_threshold": 70}


def goal() -> ResearchGoal:
    return ResearchGoal(
        objective="Compare long-only strategies without calculating metrics in the model.",
        symbol="AAPL",
        start_date="2020-01-01",
        end_date="2024-12-31",
        allowed_strategies=("moving_average", "rsi", "macd"),
        max_drawdown_limit=0.20,
        costs=BacktestCosts(5.0, 5.0),
    )


def segment(total_return: float, max_drawdown: float, exposure: float) -> SegmentRecord:
    return SegmentRecord(
        metrics=PerformanceMetrics(
            total_return=total_return,
            annualized_return=total_return / 2,
            annualized_volatility=0.15,
            sharpe_ratio=0.5,
            max_drawdown=max_drawdown,
            trade_count=4,
            turnover=7.0,
            asset_buy_and_hold_return=0.09,
            benchmark_return=0.07,
        ),
        drawdown=DrawdownRecord(
            max_drawdown=max_drawdown,
            peak_date=datetime(2021, 1, 4),
            trough_date=datetime(2021, 3, 1),
            recovery_date=None,
            underwater_bars=40,
        ),
        mean_exposure=exposure,
        warnings=(),
    )


def candidate(
    fingerprint: str,
    parameters: dict[str, int],
    *,
    total_return: float,
    max_drawdown: float = -0.10,
    test_return: float | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        fingerprint=fingerprint,
        strategy_name="moving_average" if "short_window" in parameters else "rsi",
        parameters=parameters,
        training=segment(total_return, max_drawdown, 0.2),
        test=None if test_return is None else segment(test_return, -0.12, 0.15),
    )


def workspace(**changes: object) -> ResearchWorkspace:
    return ResearchWorkspace(goal=goal(), budget=AgentBudget(), **changes)


def event(step: int, *, success: bool = True, tool_name: str | None = None) -> ToolCallRecord:
    selected_tool = tool_name or ("run_backtest" if success else "get_market_data")
    return ToolCallRecord(
        step=step,
        tool_name=selected_tool,
        arguments={"stage": "training", "sequence": step},
        observation=ToolObservation(
            tool_name=selected_tool,
            status="success" if success else "rejected",
            summary=f"observation-{step}",
            identifiers={"segment": "training"} if success else {},
            metrics={"total_return": step / 100} if success else {},
        ),
    )


def json_content(message: HumanMessage) -> dict[str, object]:
    assert isinstance(message.content, str)
    parsed = json.loads(message.content)
    assert isinstance(parsed, dict)
    return parsed


def system_state(research: ResearchWorkspace) -> dict[str, object]:
    rules, state = research_system_prompt(research).rsplit("\n\n", 1)
    assert rules == SYSTEM_PROMPT
    return json.loads(state)["authoritative_current_state"]


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        (["inspect_retrieved_context", "submit_critic_review"], True),
        (["submit_critic_review"], False),
        (["submit_critic_review", "inspect_retrieved_context"], False),
    ],
)
def test_critic_evaluation_requires_evidence_before_verdict(
    monkeypatch: pytest.MonkeyPatch, actions: list[str], expected: bool
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_agent_evals import _policy_evaluator

    scores = _policy_evaluator(
        {},
        {"actions": [{"tool_name": name} for name in actions], "verdict": "reject"},
        {"allowed_verdicts": ["reject"]},
    )

    assert next(score["score"] for score in scores if score["key"] == "evidence_review") is expected


def test_a_research_run_opens_with_the_confirmed_goal() -> None:
    messages = research_input_messages(workspace())

    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    context = json_content(messages[0])
    assert set(context) == {"confirmed_research_goal"}
    goal_payload = context["confirmed_research_goal"]
    assert goal_payload["message_type"] == "confirmed_research_goal"
    assert goal_payload["symbol"] == "AAPL"
    assert goal_payload["benchmark_symbol"] == "SPY"
    assert goal_payload["costs"] == {
        "transaction_cost_bps": 5.0,
        "slippage_bps": 5.0,
    }
    assert system_state(workspace())["message_type"] == "authoritative_current_state"


def test_system_state_leaves_observations_to_the_conversation() -> None:
    journal = [event(1, tool_name="get_market_data"), *[event(i) for i in range(2, 8)]]

    prompt = research_system_prompt(workspace(journal=journal))

    assert "observation-" not in prompt
    assert {"recent_actions", "completed_context"}.isdisjoint(system_state(workspace()))


def test_revision_run_opens_with_the_review_draft_and_every_source_result() -> None:
    sources = ("sec", "yahoo_finance", "stocktwits", "reddit")
    evidence = []
    for index, source in enumerate(sources):
        if source != "reddit":
            evidence.append(
                MarketContextEvidence(
                    id=f"evidence-{index}",
                    source_type=(
                        "official"
                        if source == "sec"
                        else "news"
                        if source == "yahoo_finance"
                        else "community"
                    ),
                    source=source,
                    title=f"{source} title",
                    excerpt=f"{source} excerpt",
                    published_at="2024-01-15",
                    url=f"https://example.com/{source}",
                )
            )
    search = ToolCallRecord(
        step=1,
        tool_name="search_market_context",
        arguments={"queries": dict.fromkeys(sources, "Apple"), "revision_reason": None},
        observation=ToolObservation(
            tool_name="search_market_context",
            status="success",
            summary="Searched all configured sources.",
            metrics={source: 0 if source == "reddit" else 1 for source in sources},
            warnings=("reddit returned no evidence.",),
            evidence=tuple(evidence),
        ),
    )

    draft = ResearchDraft(
        outcome="selected",
        selected_fingerprint=LOCKED_FINGERPRINT,
        conclusion="Select the frozen candidate.",
        market_context_summary="All sources were assessed.",
        cited_evidence_ids=("evidence-0",),
    )
    review = CriticReview(
        verdict="revise",
        summary="The context summary omits a failed source.",
        required_revision="Disclose the reddit gap.",
    )
    revising = workspace(
        phase="locked",
        journal=[search],
        critic_rounds=[CriticRound(draft=draft, review=review)],
    )

    opening = json_content(research_input_messages(revising)[0])

    assert opening["critic_revision"]["required_revision"] == "Disclose the reddit gap."
    assert opening["submitted_draft"]["cited_evidence_ids"] == ["evidence-0"]
    (context,) = opening["retrieved_context"]
    assert context["queries"] == dict.fromkeys(sources, "Apple")
    assert context["source_counts"] == {
        source: 0 if source == "reddit" else 1 for source in sources
    }
    assert len(context["evidence"]) == 3
    assert context["warnings"] == ["reddit returned no evidence."]


def test_researching_context_has_training_metrics_and_budget_only() -> None:
    researching = workspace(
        candidates=[candidate(LOCKED_FINGERPRINT, MOVING_AVERAGE, total_return=0.12)],
        agent_steps=4,
        tool_calls=3,
        fatal_reason="Market data was invalid or insufficient.",
    )

    state = system_state(researching)

    assert state["phase"] == "researching"
    assert state["tested_candidates"][0]["training_metrics"]["total_return"] == 0.12
    assert state["tested_candidates"][0]["training_mean_exposure"] == 0.2
    assert state["locked_candidate"] is None
    assert "test_metrics" not in json.dumps(state)
    assert state["unresolved_issues"] == ["Market data was invalid or insufficient."]
    assert state["budget"]["tool_calls"] == {"used": 3, "remaining": 27}
    assert state["budget"]["training_backtests"]["used"] == 1


def test_locked_context_pins_only_the_selected_test_metrics() -> None:
    locked = workspace(
        phase="locked",
        candidates=[
            candidate(LOCKED_FINGERPRINT, MOVING_AVERAGE, total_return=0.12, test_return=0.06),
            candidate(
                OTHER_FINGERPRINT, RSI, total_return=0.20, max_drawdown=-0.25, test_return=9876.5
            ),
        ],
        locked_fingerprint=LOCKED_FINGERPRINT,
        journal=[event(index) for index in range(1, 7)],
    )

    state = system_state(locked)
    pinned = state["locked_candidate"]
    serialized = json.dumps(state)

    assert state["final_batch_completed"] is True
    assert pinned["fingerprint"] == LOCKED_FINGERPRINT
    assert pinned["training_metrics"]["total_return"] == 0.12
    assert pinned["test_metrics"]["total_return"] == 0.06
    assert pinned["test_mean_exposure"] == 0.15
    assert "9876.5" not in serialized
    assert serialized.count('"test_metrics"') == 1


def test_tested_candidates_are_capped_at_eight() -> None:
    many = workspace(
        candidates=[
            candidate(f"{index:064x}", MOVING_AVERAGE, total_return=index / 100)
            for index in range(9)
        ]
    )

    state = system_state(many)

    assert len(state["tested_candidates"]) == 8
