"""Tests for the Research Agent's deliberately compact model context."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_openai_messages

from trading.agent.critic import CRITIC_SYSTEM_PROMPT
from trading.agent.models import (
    AgentBudget,
    ResearchGoal,
    ResearchTraceRecord,
    ToolObservation,
)
from trading.agent.prompt import SYSTEM_PROMPT, build_research_messages
from trading.quant import BacktestCosts, MovingAverageConfig, RsiConfig

LOCKED_FINGERPRINT = "a" * 64
OTHER_FINGERPRINT = "b" * 64


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


def metrics(total_return: float, max_drawdown: float) -> SimpleNamespace:
    return SimpleNamespace(
        total_return=total_return,
        annualized_return=total_return / 2,
        annualized_volatility=0.15,
        sharpe_ratio=0.5,
        max_drawdown=max_drawdown,
        trade_count=4,
        turnover=7.0,
        asset_buy_and_hold_return=0.09,
        benchmark_return=0.07,
    )


def training_evaluation(
    config: MovingAverageConfig | RsiConfig,
    *,
    total_return: float,
    max_drawdown: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        strategy_name=("moving_average" if isinstance(config, MovingAverageConfig) else "rsi"),
        config=config,
        training=SimpleNamespace(metrics=metrics(total_return, max_drawdown)),
    )


def final_evaluation(
    config: MovingAverageConfig | RsiConfig,
    *,
    training_return: float,
    test_return: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        strategy_name=("moving_average" if isinstance(config, MovingAverageConfig) else "rsi"),
        config=config,
        training=SimpleNamespace(metrics=metrics(training_return, -0.10)),
        test=SimpleNamespace(metrics=metrics(test_return, -0.12)),
    )


@dataclass(frozen=True, slots=True)
class Event:
    tool_name: str
    normalized_arguments: dict[str, object]
    observation: ToolObservation
    tool_call_id: str | None = None
    replayable: bool = False


@dataclass(slots=True)
class Workspace:
    goal: ResearchGoal = field(default_factory=goal)
    budget: AgentBudget = field(default_factory=AgentBudget)
    phase: str = "researching"
    current_candidate: MovingAverageConfig | RsiConfig | None = None
    training_evaluations: dict[str, object] = field(default_factory=dict)
    final_evaluations: dict[str, object] = field(default_factory=dict)
    locked_fingerprint: str | None = None
    final_batch_completed: bool = False
    agent_steps: int = 0
    tool_calls: int = 0
    training_backtests: int = 0
    trace: list[ResearchTraceRecord] = field(default_factory=list)
    unresolved_issues: list[str] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    aligned_data: object = field(
        default_factory=lambda: pd.DataFrame({"SECRET_PRICE_COLUMN": [123.45]})
    )
    complete_report: str = "FULL_REPORT_MUST_NOT_APPEAR"


def event(
    step: int,
    *,
    replayable: bool = True,
    tool_name: str | None = None,
) -> Event:
    selected_tool = tool_name or ("run_backtest" if replayable else "get_market_data")
    observation = ToolObservation(
        tool_name=selected_tool,
        status="success" if replayable else "rejected",
        summary=f"observation-{step}",
        identifiers={"segment": "training"} if replayable else {},
        metrics={"total_return": step / 100} if replayable else {},
    )
    return Event(
        tool_name=selected_tool,
        normalized_arguments={"stage": "training", "sequence": step},
        observation=observation,
        tool_call_id=f"call-{step}" if replayable else None,
        replayable=replayable,
    )


def json_content(message: HumanMessage) -> dict[str, object]:
    assert isinstance(message.content, str)
    parsed = json.loads(message.content)
    assert isinstance(parsed, dict)
    return parsed


def test_system_prompt_sets_agent_rules_without_a_fixed_workflow() -> None:
    lower = SYSTEM_PROMPT.lower()
    critic_lower = CRITIC_SYSTEM_PROMPT.lower()

    assert "tool" in lower
    assert "one" in lower
    assert "financial metrics" in lower
    assert "submit_research_draft" in SYSTEM_PROMPT
    assert "run_backtest validates" in lower
    assert "budgets are ceilings, not targets" in lower
    assert "untested allowed strategy family" in lower
    assert "not its entire strategy family" in lower
    assert "never repeat an exact listed candidate" in lower
    assert "only one candidate" in lower
    assert "must try materially different parameters" in lower
    assert "very low" in lower
    assert "trade count" in lower
    assert "fixed grid" in lower
    assert "equal per-family quota" in lower
    assert "automatically require a second candidate" in lower
    assert "never call tools in parallel" in lower
    assert "completed_context" in lower
    assert "selected_fingerprint" in SYSTEM_PROMPT
    assert "tested_candidates" in SYSTEM_PROMPT
    assert "reserve five tool calls" in lower
    assert "chain-of-thought" not in lower
    assert "parameter grid" not in lower
    assert "first get_market_data" not in lower
    assert "supported by sufficient evidence" in critic_lower
    assert "one failed default parameterization" in critic_lower
    assert "sample-in candidate summaries" in critic_lower
    assert "reject rather than" in critic_lower
    assert "failed candidate may still support" in critic_lower
    assert "do not accept merely because" in critic_lower
    assert "no materially different" in critic_lower
    assert "do not require two candidates for every family" in critic_lower


def test_messages_have_fixed_outer_order_and_confirmed_goal() -> None:
    workspace = Workspace()

    messages = build_research_messages(workspace)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert convert_to_openai_messages([messages[0]])[0]["role"] == "developer"
    assert isinstance(messages[1], HumanMessage)
    context = json_content(messages[1])
    goal_payload = context["confirmed_research_goal"]
    state_payload = context["authoritative_current_state"]
    assert goal_payload["message_type"] == "confirmed_research_goal"
    assert goal_payload["symbol"] == "AAPL"
    assert goal_payload["benchmark_symbol"] == "SPY"
    assert goal_payload["costs"] == {
        "transaction_cost_bps": 5.0,
        "slippage_bps": 5.0,
    }
    assert state_payload["message_type"] == "authoritative_current_state"


def test_only_last_five_actions_are_kept_as_structured_context() -> None:
    workspace = Workspace(
        events=[event(1, tool_name="get_market_data"), *[event(index) for index in range(2, 8)]]
    )

    messages = build_research_messages(workspace)
    assert len(messages) == 2
    assert "observation-2" not in "".join(str(message.content) for message in messages)
    assert "observation-3" in "".join(str(message.content) for message in messages)
    state = json_content(messages[-1])["authoritative_current_state"]
    assert state["completed_context"][0]["tool_name"] == "get_market_data"
    assert state["completed_context"][0]["summary"] == "observation-1"
    assert [item["arguments"]["sequence"] for item in state["recent_actions"]] == [
        3,
        4,
        5,
        6,
        7,
    ]
    assert all(item["tool_name"] == "run_backtest" for item in state["recent_actions"])


def test_non_replayable_actions_share_the_structured_action_context() -> None:
    workspace = Workspace(
        events=[event(1), event(2), event(3, replayable=False), event(4), event(5), event(6)]
    )

    messages = build_research_messages(workspace)
    state = json_content(messages[-1])["authoritative_current_state"]

    correction = state["recent_actions"][1]
    assert correction["tool_name"] == "get_market_data"
    assert correction["observation"]["summary"] == "observation-3"


def test_researching_context_has_training_only_and_excludes_large_objects() -> None:
    config = MovingAverageConfig(short_window=20, long_window=100)
    workspace = Workspace(
        current_candidate=config,
        training_evaluations={
            LOCKED_FINGERPRINT: training_evaluation(
                config,
                total_return=0.12,
                max_drawdown=-0.10,
            )
        },
        # A defensive sentinel: final data must not leak while phase is researching.
        final_evaluations={
            LOCKED_FINGERPRINT: final_evaluation(
                config,
                training_return=0.12,
                test_return=9876.54321,
            )
        },
        agent_steps=4,
        tool_calls=3,
        training_backtests=1,
        unresolved_issues=["Need a candidate within the drawdown limit."],
    )

    messages = build_research_messages(workspace)
    state = json_content(messages[-1])["authoritative_current_state"]
    serialized = "\n".join(str(message.content) for message in messages)

    assert state["phase"] == "researching"
    assert state["current_candidate"] == {
        "strategy_name": "moving_average",
        "parameters": {"short_window": 20, "long_window": 100},
    }
    assert state["tested_candidates"][0]["training_metrics"]["total_return"] == 0.12
    assert "test_metrics" not in json.dumps(state)
    assert "9876.54321" not in serialized
    assert "SECRET_PRICE_COLUMN" not in serialized
    assert "FULL_REPORT_MUST_NOT_APPEAR" not in serialized
    assert state["budget"]["tool_calls"] == {"used": 3, "remaining": 27}


def test_locked_context_pins_only_the_selected_test_metrics() -> None:
    selected = MovingAverageConfig(short_window=20, long_window=100)
    other = RsiConfig(rsi_window=14, entry_threshold=30, exit_threshold=70)
    workspace = Workspace(
        phase="locked",
        current_candidate=selected,
        training_evaluations={
            LOCKED_FINGERPRINT: training_evaluation(
                selected,
                total_return=0.12,
                max_drawdown=-0.10,
            ),
            OTHER_FINGERPRINT: training_evaluation(
                other,
                total_return=0.20,
                max_drawdown=-0.25,
            ),
        },
        final_evaluations={
            LOCKED_FINGERPRINT: final_evaluation(
                selected,
                training_return=0.12,
                test_return=0.06,
            ),
            OTHER_FINGERPRINT: final_evaluation(
                other,
                training_return=0.20,
                test_return=9876.54321,
            ),
        },
        locked_fingerprint=LOCKED_FINGERPRINT,
        final_batch_completed=True,
        events=[event(index) for index in range(1, 7)],
    )

    messages = build_research_messages(workspace)
    state = json_content(messages[-1])["authoritative_current_state"]
    locked = state["locked_candidate"]
    serialized = json.dumps(state)

    assert locked["fingerprint"] == LOCKED_FINGERPRINT
    assert locked["training_metrics"]["total_return"] == 0.12
    assert locked["test_metrics"]["total_return"] == 0.06
    assert "9876.54321" not in serialized
    assert serialized.count('"test_metrics"') == 1
    assert "observation-1" not in "\n".join(str(message.content) for message in messages)


def test_locked_no_selection_exposes_no_candidate_test_metrics() -> None:
    config = MovingAverageConfig(short_window=20, long_window=100)
    workspace = Workspace(
        phase="locked",
        training_evaluations={
            LOCKED_FINGERPRINT: training_evaluation(
                config,
                total_return=0.12,
                max_drawdown=-0.10,
            )
        },
        final_evaluations={
            LOCKED_FINGERPRINT: final_evaluation(
                config,
                training_return=0.12,
                test_return=9876.54321,
            )
        },
        locked_fingerprint=None,
        final_batch_completed=True,
    )

    state = json_content(build_research_messages(workspace)[-1])["authoritative_current_state"]

    assert state["locked_candidate"] == {"fingerprint": None, "selection": "no_strategy"}
    assert "test_metrics" not in json.dumps(state)
    assert "9876.54321" not in json.dumps(state)


def test_tested_candidates_are_capped_at_eight() -> None:
    config = MovingAverageConfig(short_window=20, long_window=100)
    evaluations = {
        f"{index:064x}": training_evaluation(
            config,
            total_return=index / 100,
            max_drawdown=-0.10,
        )
        for index in range(9)
    }
    workspace = Workspace(training_evaluations=evaluations)

    state = json_content(build_research_messages(workspace)[-1])["authoritative_current_state"]

    assert len(state["tested_candidates"]) == 8
