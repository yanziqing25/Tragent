"""Offline tests for one structured goal parse and deterministic defaults."""

from __future__ import annotations

import copy
import json
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    convert_to_openai_messages,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langsmith import tracing_context
from pydantic import PrivateAttr

import trading.agent.finalization as finalization_module
import trading.agent.graph as graph_module
import trading.agent.runtime as runtime_module
from trading.agent.events import AgentEvent
from trading.agent.graph import parse_goal
from trading.agent.models import AgentBudget, GoalDraft
from trading.agent.state import GraphState, validate_graph_state
from trading.quant import DataQualityReport, MarketData
from trading.reporting import load_report_view

type FakeMode = Literal[
    "valid",
    "no_tool",
    "multiple_tools",
    "invalid_tool",
    "wrong_tool",
    "transport_error",
    "bind_error",
]

_INTERVAL_INDEPENDENT_DEFAULTED_FIELDS = [
    "allowed_strategies",
    "position_tranches",
    "transaction_cost_bps",
    "slippage_bps",
]
_ALL_DEFAULTED_FIELDS = [
    *_INTERVAL_INDEPENDENT_DEFAULTED_FIELDS,
    "benchmark_symbol",
]
_GOAL_PARSE_FAILURE = "The research goal could not be parsed safely."
_RESEARCH_TOOL_NAMES = (
    "get_market_data",
    "calculate_asset_statistics",
    "run_backtest",
    "analyze_drawdown",
    "compare_with_benchmark",
    "submit_research_draft",
)
_RESEARCH_REVISION_TOOL_NAMES = (
    "analyze_drawdown",
    "compare_with_benchmark",
    "submit_research_draft",
)
_CRITIC_TOOL_NAMES = (
    "inspect_selected_drawdown",
    "compare_selected_with_benchmark",
    "submit_critic_review",
)


def _nullable_goal_response(**changes: object) -> dict[str, object]:
    response = {field_name: None for field_name in GoalDraft.model_fields}
    response.update(changes)
    return response


class GoalParserFakeChatModel(BaseChatModel):
    """Single-purpose fake that emits native ToolCalls for the bound goal schema."""

    _mode: FakeMode = PrivateAttr()
    _draft: dict[str, object] = PrivateAttr(default_factory=dict)
    _call_id: str = PrivateAttr(default="goal-call-1")
    _bind_count: int = PrivateAttr(default=0)
    _messages_seen: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tool_names: tuple[str, ...] = PrivateAttr(default=())
    _binding_options: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        draft: Mapping[str, object] | None = None,
        *,
        mode: FakeMode = "valid",
        call_id: str = "goal-call-1",
    ) -> None:
        super().__init__()
        self._mode = mode
        self._draft = dict(draft or {})
        self._call_id = call_id

    @property
    def _llm_type(self) -> str:
        return "goal-parser-fake-chat-model"

    @property
    def bind_count(self) -> int:
        return self._bind_count

    @property
    def invoke_count(self) -> int:
        return len(self._messages_seen)

    @property
    def messages_seen(self) -> list[list[BaseMessage]]:
        return self._messages_seen

    @property
    def bound_tool_names(self) -> tuple[str, ...]:
        return self._bound_tool_names

    @property
    def binding_options(self) -> list[dict[str, Any]]:
        return self._binding_options

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> GoalParserFakeChatModel:
        options = dict(kwargs)
        if tool_choice is not None:
            options["tool_choice"] = tool_choice
        self._binding_options.append(options)
        self._bind_count += 1
        if self._mode == "bind_error":
            raise RuntimeError("unsafe-goal-bind-secret")
        self._bound_tool_names = tuple(_bound_tool_name(tool) for tool in tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self._messages_seen.append(list(messages))
        if self._mode == "transport_error":
            raise RuntimeError("unsafe-goal-transport-secret")
        if len(self._bound_tool_names) != 1:
            raise AssertionError("the goal parser must bind exactly one schema")

        tool_name = self._bound_tool_names[0]
        if self._mode == "no_tool":
            message = AIMessage(content="unsafe-free-text-goal-answer")
        elif self._mode == "multiple_tools":
            message = AIMessage(
                content="",
                tool_calls=[
                    _tool_call(tool_name, self._draft, "goal-call-1"),
                    _tool_call(tool_name, self._draft, "goal-call-2"),
                ],
            )
        elif self._mode == "invalid_tool":
            message = AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": tool_name,
                        "args": "{unsafe-invalid-goal-arguments",
                        "id": "goal-invalid-1",
                        "error": "unsafe-goal-parser-detail",
                        "type": "invalid_tool_call",
                    }
                ],
            )
        elif self._mode == "wrong_tool":
            message = AIMessage(
                content="",
                tool_calls=[_tool_call("wrong_goal_tool", self._draft, "goal-call-1")],
            )
        else:
            message = AIMessage(
                content="unsafe-free-text-must-not-be-saved",
                tool_calls=[_tool_call(tool_name, self._draft, self._call_id)],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


class SequencedGoalParserFakeChatModel(GoalParserFakeChatModel):
    """Use one native goal draft per parse call so edit can be tested end to end."""

    _draft_sequence: list[dict[str, object]] = PrivateAttr()

    def __init__(self, drafts: Sequence[Mapping[str, object]]) -> None:
        if not drafts:
            raise ValueError("drafts must not be empty")
        super().__init__(drafts[0])
        self._draft_sequence = [dict(draft) for draft in drafts]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        call_index = self.invoke_count
        if call_index >= len(self._draft_sequence):
            raise AssertionError("the graph parsed the goal more often than expected")
        self._draft = dict(self._draft_sequence[call_index])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


type ResearchDecider = Callable[[list[BaseMessage]], AIMessage]


class DynamicResearchGraphFakeChatModel(GoalParserFakeChatModel):
    """Parse one goal, then choose each research Action from current observations."""

    _research_decider: ResearchDecider = PrivateAttr()
    _critic_decider: ResearchDecider = PrivateAttr()
    _research_messages_seen: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _critic_messages_seen: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _binding_history: list[tuple[str, ...]] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        draft: Mapping[str, object],
        research_decider: ResearchDecider,
        critic_decider: ResearchDecider | None = None,
    ) -> None:
        super().__init__(draft)
        self._research_decider = research_decider
        self._critic_decider = critic_decider or _accept_critic_decider

    @property
    def research_messages_seen(self) -> list[list[BaseMessage]]:
        return self._research_messages_seen

    @property
    def binding_history(self) -> list[tuple[str, ...]]:
        return self._binding_history

    @property
    def critic_messages_seen(self) -> list[list[BaseMessage]]:
        return self._critic_messages_seen

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> DynamicResearchGraphFakeChatModel:
        super().bind_tools(tools, tool_choice=tool_choice, **kwargs)
        self._binding_history.append(self.bound_tool_names)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.bound_tool_names == ("GoalDraft",):
            return super()._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        del stop, run_manager, kwargs
        snapshot = list(messages)
        self._messages_seen.append(snapshot)
        if self.bound_tool_names in {_RESEARCH_TOOL_NAMES, _RESEARCH_REVISION_TOOL_NAMES}:
            self._research_messages_seen.append(snapshot)
            decision = self._research_decider(snapshot)
        elif self.bound_tool_names and set(self.bound_tool_names).issubset(_CRITIC_TOOL_NAMES):
            self._critic_messages_seen.append(snapshot)
            decision = self._critic_decider(snapshot)
        else:
            raise AssertionError("the graph bound an unexpected Agent Tool set")
        return ChatResult(generations=[ChatGeneration(message=decision)])


class NeverMarketDataProvider:
    """Count and reject any market-data access before or inside the HITL boundary."""

    def __init__(self) -> None:
        self.calls = 0

    def get_daily_bars(self, *_args: object, **_kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("goal confirmation must precede every market-data access")


class GraphMarketDataProvider:
    """Return deterministic aligned inputs while recording the confirmed request scope."""

    def __init__(self, asset_symbol: str = "AAPL") -> None:
        self.calls: list[tuple[str, date, date]] = []
        self._outcomes = {
            asset_symbol: _graph_market_data(asset_symbol),
            "SPY": _graph_market_data("SPY", falling=True),
        }

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> MarketData:
        self.calls.append((symbol, start_date, end_date))
        return self._outcomes[symbol]


@pytest.fixture(autouse=True)
def disable_external_tracing():
    with tracing_context(enabled=False):
        yield


def _bound_tool_name(tool: dict[str, Any] | type | Callable[..., Any] | BaseTool) -> str:
    if isinstance(tool, BaseTool):
        return tool.name
    if isinstance(tool, type):
        return tool.__name__
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            return function["name"]
        if isinstance(tool.get("name"), str):
            return tool["name"]
    name = getattr(tool, "__name__", None)
    if isinstance(name, str):
        return name
    raise AssertionError("the goal parser bound an unsupported schema shape")


def _tool_call(
    name: str,
    arguments: Mapping[str, object],
    call_id: str,
) -> dict[str, object]:
    return {
        "name": name,
        "args": dict(arguments),
        "id": call_id,
        "type": "tool_call",
    }


def _graph_market_data(symbol: str, *, falling: bool = False) -> MarketData:
    index = pd.date_range("2024-01-01", periods=30, freq="D")
    closes = [100.0 + offset + (offset % 4) for offset in range(len(index))]
    if falling:
        closes = list(reversed(closes))
    bars = pd.DataFrame(
        {
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": [1_000.0] * len(index),
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


def _react_state(messages: list[BaseMessage]) -> dict[str, Any]:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage) or not isinstance(message.content, str):
            continue
        payload = json.loads(message.content)
        state = payload.get("authoritative_current_state")
        if isinstance(state, dict):
            return state
    raise AssertionError("the Graph did not supply authoritative ReAct state")


def _critic_payload(messages: list[BaseMessage], message_type: str) -> dict[str, Any]:
    for message in messages:
        if not isinstance(message, HumanMessage) or not isinstance(message.content, str):
            continue
        payload = json.loads(message.content)
        section = payload.get(message_type)
        if isinstance(section, dict):
            return section
    raise AssertionError(f"the Graph did not supply {message_type}")


def _react_observations(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    for message_type in ("authoritative_current_state", "critic_current_state"):
        try:
            state = _critic_payload(messages, message_type)
        except AssertionError:
            continue
        observations = list(state.get("completed_context", []))
        observations.extend(action["observation"] for action in state.get("recent_actions", []))
        return observations
    raise AssertionError("the Graph did not supply structured ReAct observations")


def _react_call(
    messages: list[BaseMessage],
    name: str,
    arguments: dict[str, object],
) -> AIMessage:
    step = _react_state(messages)["budget"]["agent_steps"]["used"]
    return AIMessage(
        content="ignored model prose",
        tool_calls=[_tool_call(name, arguments, f"research-call-{step}")],
    )


def _single_market_data_decider(messages: list[BaseMessage]) -> AIMessage:
    state = _react_state(messages)
    assert state["phase"] == "researching"
    assert state["tested_candidates"] == []
    return _react_call(messages, "get_market_data", {})


def _market_data_then_training_decider(messages: list[BaseMessage]) -> AIMessage:
    state = _react_state(messages)
    observations = _react_observations(messages)
    if not any(item["tool_name"] == "get_market_data" for item in observations):
        return _react_call(messages, "get_market_data", {})
    assert state["tested_candidates"] == []
    return _react_call(
        messages,
        "run_backtest",
        {
            "stage": "training",
            "strategy_name": "moving_average",
            "parameters": {"short_window": 2, "long_window": 3},
        },
    )


def _must_not_redecide(messages: list[BaseMessage]) -> AIMessage:
    del messages
    raise AssertionError("a recovered pending Action must execute without model re-decision")


class FailOnceResearchDecision:
    """Raise one transient model error, then decide from the durable current state."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("unsafe-transient-model-detail")
        return _single_market_data_decider(messages)


def _worst_case_budget_decider(messages: list[BaseMessage]) -> AIMessage:
    state = _react_state(messages)
    tool_calls = state["budget"]["tool_calls"]["used"]
    return _react_call(
        messages,
        "analyze_drawdown",
        {
            "fingerprint": f"{tool_calls + 1:064x}",
            "segment": "training",
        },
    )


def _accept_critic_decider(messages: list[BaseMessage]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            _tool_call(
                "submit_critic_review",
                {
                    "verdict": "accept",
                    "summary": "The frozen evidence and conclusion are consistent.",
                    "issues": [],
                    "required_revision": None,
                },
                f"critic-call-{len(messages)}",
            )
        ],
    )


def _evidence_then_accept_critic_decider(messages: list[BaseMessage]) -> AIMessage:
    observations = _react_observations(messages)
    state = _critic_payload(messages, "critic_current_state")
    step = state["budget"]["agent_steps"]["used"]
    if step == 1:
        return AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "inspect_selected_drawdown",
                    {"segment": "out_of_sample"},
                    "critic-call-drawdown",
                ),
            ],
        )
    drawdown = next(
        (
            item
            for item in observations
            if item["tool_name"] == "inspect_selected_drawdown" and item["status"] == "success"
        ),
        None,
    )
    if drawdown is None:
        return AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "inspect_selected_drawdown",
                    {"segment": "out_of_sample"},
                    f"critic-call-{step}",
                )
            ],
        )
    assert drawdown["status"] == "success"
    assert drawdown["identifiers"]["segment"] == "test"
    return AIMessage(
        content="",
        tool_calls=[
            _tool_call(
                "submit_critic_review",
                {
                    "verdict": "accept",
                    "summary": "The inspected drawdown supports the bounded conclusion.",
                    "issues": [],
                    "required_revision": None,
                },
                f"critic-call-{step}",
            )
        ],
    )


def _pause_after_final_decider(messages: list[BaseMessage]) -> AIMessage:
    state = _react_state(messages)
    observations = _react_observations(messages)
    candidates = state["tested_candidates"]
    if not any(item["tool_name"] == "get_market_data" for item in observations) and not candidates:
        return _react_call(messages, "get_market_data", {})
    if not candidates:
        return _react_call(
            messages,
            "run_backtest",
            {
                "stage": "training",
                "strategy_name": "moving_average",
                "parameters": {"short_window": 2, "long_window": 3},
            },
        )
    fingerprint = candidates[0]["fingerprint"]
    if state["phase"] == "researching":
        return _react_call(
            messages,
            "run_backtest",
            {"stage": "final", "selected_fingerprint": fingerprint},
        )
    assert state["phase"] == "locked"
    assert state["locked_candidate"]["fingerprint"] == fingerprint
    assert "test_metrics" in state["locked_candidate"]
    assert all("test_metrics" not in item for item in candidates)
    raise RuntimeError("unsafe-process-boundary-after-final")


def _submit_locked_decider(messages: list[BaseMessage]) -> AIMessage:
    state = _react_state(messages)
    assert state["phase"] == "locked"
    locked = state["locked_candidate"]
    assert isinstance(locked["fingerprint"], str)
    assert "test_metrics" in locked
    assert all("test_metrics" not in item for item in state["tested_candidates"])
    return _react_call(
        messages,
        "submit_research_draft",
        {
            "outcome": "selected",
            "selected_fingerprint": locked["fingerprint"],
            "conclusion": "The restored final lock remains the only selectable candidate.",
            "limitations": ["Daily bars only."],
        },
    )


def _dynamic_selected_decider(messages: list[BaseMessage]) -> AIMessage:
    """Use evidence to change parameters, then strategy, before the final lock."""

    state = _react_state(messages)
    observations = _react_observations(messages)
    candidates = state["tested_candidates"]
    data_ready = any(
        item["tool_name"] == "get_market_data" and item["status"] == "success"
        for item in observations
    )
    if not data_ready and not candidates:
        return _react_call(messages, "get_market_data", {})
    if not candidates:
        return _react_call(
            messages,
            "run_backtest",
            {
                "stage": "training",
                "strategy_name": "moving_average",
                "parameters": {"short_window": 2, "long_window": 3},
            },
        )

    drawdown_observation = next(
        (
            item
            for item in observations
            if item["tool_name"] == "analyze_drawdown" and item["status"] == "success"
        ),
        None,
    )
    if len(candidates) == 1 and drawdown_observation is None:
        return _react_call(
            messages,
            "analyze_drawdown",
            {"fingerprint": candidates[0]["fingerprint"], "segment": "training"},
        )
    if len(candidates) == 1:
        assert drawdown_observation is not None
        assert drawdown_observation["identifiers"]["fingerprint"] == candidates[0]["fingerprint"]
        return _react_call(
            messages,
            "run_backtest",
            {
                "stage": "training",
                "strategy_name": "moving_average",
                "parameters": {"short_window": 2, "long_window": 4},
            },
        )
    if len(candidates) == 2:
        return _react_call(
            messages,
            "run_backtest",
            {
                "stage": "training",
                "strategy_name": "rsi",
                "parameters": {
                    "rsi_window": 2,
                    "entry_threshold": 20.0,
                    "exit_threshold": 80.0,
                },
            },
        )

    # Select the stronger of the two moving-average candidates. The RSI probe is
    # deliberately non-positive, so selecting it would violate the production
    # evidence rule that asks for a second materially different RSI candidate.
    fingerprint = candidates[1]["fingerprint"]
    if state["phase"] == "researching":
        return _react_call(
            messages,
            "run_backtest",
            {"stage": "final", "selected_fingerprint": fingerprint},
        )
    compared = any(
        item["tool_name"] == "compare_with_benchmark"
        and item["status"] == "success"
        and item["identifiers"].get("segment") == "test"
        for item in observations
    )
    if not compared:
        return _react_call(
            messages,
            "compare_with_benchmark",
            {"fingerprint": fingerprint, "segment": "test"},
        )
    return _react_call(
        messages,
        "submit_research_draft",
        {
            "outcome": "selected",
            "selected_fingerprint": fingerprint,
            "conclusion": "The dynamically chosen frozen candidate passed final review.",
            "limitations": ["Daily bars only."],
        },
    )


def _no_strategy_decider(messages: list[BaseMessage]) -> AIMessage:
    state = _react_state(messages)
    observations = _react_observations(messages)
    if not any(item["tool_name"] == "get_market_data" for item in observations):
        return _react_call(messages, "get_market_data", {})
    if not state["tested_candidates"]:
        return _react_call(
            messages,
            "run_backtest",
            {
                "stage": "training",
                "strategy_name": "moving_average",
                "parameters": {"short_window": 2, "long_window": 3},
            },
        )
    if state["phase"] == "researching":
        fingerprint = state["tested_candidates"][0]["fingerprint"]
        return _react_call(
            messages,
            "run_backtest",
            {"stage": "final", "selected_fingerprint": fingerprint},
        )
    fingerprint = state["tested_candidates"][0]["fingerprint"]
    return _react_call(
        messages,
        "submit_research_draft",
        {
            "outcome": "no_strategy",
            "selected_fingerprint": fingerprint,
            "conclusion": "No candidate was selected after the frozen final batch.",
            "limitations": [],
        },
    )


def _revision_research_decider(messages: list[BaseMessage]) -> AIMessage:
    state = _react_state(messages)
    observations = _react_observations(messages)
    candidates = state["tested_candidates"]
    if not candidates and not any(item["tool_name"] == "get_market_data" for item in observations):
        return _react_call(messages, "get_market_data", {})
    if not candidates:
        return _react_call(
            messages,
            "run_backtest",
            {
                "stage": "training",
                "strategy_name": "moving_average",
                "parameters": {"short_window": 2, "long_window": 3},
            },
        )
    fingerprint = candidates[0]["fingerprint"]
    if state["phase"] == "researching":
        return _react_call(
            messages,
            "run_backtest",
            {"stage": "final", "selected_fingerprint": fingerprint},
        )
    revision = state["critic_revision"]
    if revision is not None and not any(
        item["tool_name"] == "compare_with_benchmark"
        and item["identifiers"].get("segment") == "test"
        for item in observations
    ):
        return _react_call(
            messages,
            "compare_with_benchmark",
            {"segment": "out_of_sample"},
        )
    return _react_call(
        messages,
        "submit_research_draft",
        {
            "outcome": "selected",
            "selected_fingerprint": fingerprint,
            "conclusion": (
                "The revised conclusion now states the frozen benchmark evidence."
                if revision is not None
                else "The frozen candidate meets the stated constraint."
            ),
            "limitations": ["Daily bars only."],
        },
    )


def _revise_then_accept_critic_decider(messages: list[BaseMessage]) -> AIMessage:
    packet = _critic_payload(messages, "critic_review_packet")
    state = _critic_payload(messages, "critic_current_state")
    step = state["budget"]["agent_steps"]["used"]
    if packet["review_round"] == 1:
        arguments = {
            "verdict": "revise",
            "summary": "The conclusion needs one frozen benchmark check.",
            "issues": ["The benchmark comparison is not explicit."],
            "required_revision": "Inspect and state the frozen sample-out benchmark comparison.",
        }
    else:
        assert packet["review_round"] == 2
        assert packet["previous_critic_review"]["verdict"] == "revise"
        assert "revised conclusion" in packet["research_draft"]["conclusion"]
        arguments = {
            "verdict": "accept",
            "summary": "The bounded revision now matches the frozen evidence.",
            "issues": [],
            "required_revision": None,
        }
    return AIMessage(
        content="",
        tool_calls=[_tool_call("submit_critic_review", arguments, f"critic-call-{step}")],
    )


def _seed(
    *,
    request: str = "Research AAPL with transparent long-only strategies.",
    reference_date: str = "2026-07-13",
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> dict[str, object]:
    return {
        "thread_id": str(uuid4()),
        "user_id": 1,
        "request": request,
        "reference_date": reference_date,
        "cost_defaults": {
            "transaction_cost_bps": transaction_cost_bps,
            "slippage_bps": slippage_bps,
        },
        "position_tranches_default": 5,
        "budget": AgentBudget().model_dump(mode="json"),
    }


def _explicit_draft() -> dict[str, object]:
    return {
        "symbol": "AAPL",
        "bar_interval": "1d",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "allowed_strategies": ["moving_average"],
        "max_drawdown_limit": 0.15,
        "position_tranches": 4,
        "transaction_cost_bps": 2.0,
        "slippage_bps": 3.0,
        "benchmark_symbol": "SPY",
        "request_issue": None,
    }


def _graph_research_draft(symbol: str = "AAPL") -> dict[str, object]:
    return {
        "symbol": symbol,
        "bar_interval": "1d",
        "start_date": "2024-01-01",
        "end_date": "2024-01-30",
        "allowed_strategies": ["moving_average", "rsi", "macd"],
        "max_drawdown_limit": 0.99,
        "transaction_cost_bps": 5.0,
        "slippage_bps": 5.0,
        "benchmark_symbol": "SPY",
        "request_issue": None,
    }


def _draft_goal(state: GraphState) -> dict[str, object]:
    draft = state["draft_goal"]
    assert isinstance(draft, dict)
    return draft


def _interrupt_goal(output: dict[str, Any]) -> dict[str, object]:
    payload = _interrupt_payload(output)
    draft = payload["draft_goal"]
    assert isinstance(draft, dict)
    return draft


def _assert_goal_parsing_failure(state: GraphState) -> None:
    assert validate_graph_state(state) == state
    assert state["status"] == "failed"
    assert state["failure_stage"] == "goal_parsing"
    assert state["failure_summary"] == _GOAL_PARSE_FAILURE
    assert state["draft_goal"] is None
    assert state["confirmed_goal"] is None
    assert state["defaulted_fields"] == []
    assert state["phase"] is None
    assert state["terminal_result"] is None
    assert state["agent_steps"] == state["tool_calls"] == state["training_backtests"] == 0
    assert state["message_journal"] == []
    assert state["data_snapshot"] is None
    assert state["validated_configs"] == {}
    assert state["training_evaluations"] == []
    assert state["final_evaluations"] == []


def _build_production_graph(*, checkpointer: Any | None = None) -> Any:
    factory = getattr(graph_module, "build_research_graph", None)
    assert callable(factory), "Step 6 requires build_research_graph(*, checkpointer)"
    return factory(
        checkpointer=InMemorySaver() if checkpointer is None else checkpointer,
    )


def _start_to_confirmation(
    tmp_path: Path,
    *,
    model: GoalParserFakeChatModel,
    seed: dict[str, object] | None = None,
) -> tuple[
    Any,
    dict[str, object],
    dict[str, object],
    Path,
    NeverMarketDataProvider,
    dict[str, Any],
]:
    initial_seed = seed or _seed()
    thread_id = initial_seed["thread_id"]
    assert isinstance(thread_id, str)
    report_dir = tmp_path / thread_id
    provider = NeverMarketDataProvider()
    context: dict[str, object] = {
        "model": model,
        "provider": provider,
        "output_dir": report_dir,
    }
    config: dict[str, object] = {"configurable": {"thread_id": thread_id}}
    graph = _build_production_graph()

    output = graph.invoke(
        initial_seed,
        config,
        context=context,
        durability="sync",
    )
    assert isinstance(output, dict)
    return graph, config, context, report_dir, provider, output


def _interrupt_payload(output: Mapping[str, object]) -> dict[str, object]:
    interrupts = output.get("__interrupt__")
    assert isinstance(interrupts, (list, tuple)) and len(interrupts) == 1
    value = getattr(interrupts[0], "value", None)
    assert isinstance(value, Mapping)
    return dict(value)


def _checkpoint_state(graph: Any, config: dict[str, object]) -> GraphState:
    snapshot = graph.get_state(config)
    return validate_graph_state(dict(snapshot.values))


def _checkpoint_error_envelope(
    saver: InMemorySaver,
    graph: Any,
    config: dict[str, object],
) -> str:
    saved = saver.get_tuple(config)
    assert saved is not None
    assert saved.pending_writes is not None
    assert "__error__" in repr(saved.pending_writes)
    return repr(
        {
            "checkpoint": saved.checkpoint,
            "metadata": saved.metadata,
            "pending_writes": saved.pending_writes,
            "tasks": graph.get_state(config).tasks,
        }
    )


def _active_agent_workspace_checkpoint(
    saver: InMemorySaver,
    graph: Any,
    config: dict[str, object],
) -> GraphState:
    snapshot = graph.get_state(config)
    assert len(snapshot.tasks) == 1
    task = snapshot.tasks[0]
    assert task.name in {"research_agent", "critic_agent"}
    child_config = {
        "configurable": {
            "thread_id": config["configurable"]["thread_id"],
            "checkpoint_ns": f"{task.name}:{task.id}",
        }
    }
    saved = saver.get_tuple(child_config)
    assert saved is not None
    workspace_state = saved.checkpoint["channel_values"].get("workspace_state")
    assert isinstance(workspace_state, Mapping)
    return validate_graph_state(workspace_state)


def _assert_research_side_effects_are_zero(state: GraphState) -> None:
    assert state["agent_steps"] == 0
    assert state["tool_calls"] == 0
    assert state["training_backtests"] == 0
    assert state["download_failures"] == 0
    assert state["message_journal"] == []
    assert state["data_snapshot"] is None
    assert state["current_candidate"] is None
    assert state["validated_configs"] == {}
    assert state["training_evaluations"] == []
    assert state["final_evaluations"] == []
    assert state["locked_fingerprint"] is None
    assert state["final_batch_completed"] is False
    assert state["terminal_result"] is None


def _run_approved_graph_to_terminal(
    tmp_path: Path,
    *,
    model: DynamicResearchGraphFakeChatModel,
    provider: GraphMarketDataProvider,
    budget: AgentBudget,
    events: list[AgentEvent] | None = None,
) -> tuple[GraphState, Path]:
    seed = _seed(
        request="Research AAPL dynamically with the supported long-only strategies.",
        reference_date="2024-01-30",
    )
    seed["budget"] = budget.model_dump(mode="json")
    thread_id = seed["thread_id"]
    assert isinstance(thread_id, str)
    report_dir = tmp_path / thread_id
    context: dict[str, object] = {
        "model": model,
        "provider": provider,
        "output_dir": report_dir,
    }
    config: dict[str, object] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 96,
    }
    graph = _build_production_graph()

    def stream(value: object) -> None:
        for part in graph.stream(
            value,
            config,
            context=context,
            durability="sync",
            stream_mode="custom",
            version="v2",
            subgraphs=True,
        ):
            if events is not None:
                events.append(AgentEvent.model_validate(part["data"]))

    stream(seed)
    awaiting_confirmation = _checkpoint_state(graph, config)
    assert awaiting_confirmation["status"] == "awaiting_confirmation"
    assert _draft_goal(awaiting_confirmation)["symbol"] == "AAPL"

    stream(Command(resume={"action": "approve"}))
    assert graph.get_state(config).next == ()
    return _checkpoint_state(graph, config), report_dir


def _start_durable_research_graph(
    tmp_path: Path,
    *,
    model: DynamicResearchGraphFakeChatModel,
    provider: GraphMarketDataProvider,
    budget: AgentBudget,
    checkpointer: InMemorySaver,
    expected_symbol: str = "AAPL",
) -> tuple[Any, dict[str, object], dict[str, object], Path]:
    seed = _seed(
        request=f"Research {expected_symbol} through one durable ReAct Action.",
        reference_date="2024-01-30",
    )
    seed["budget"] = budget.model_dump(mode="json")
    thread_id = seed["thread_id"]
    assert isinstance(thread_id, str)
    report_dir = tmp_path / thread_id
    context: dict[str, object] = {
        "model": model,
        "provider": provider,
        "output_dir": report_dir,
    }
    config: dict[str, object] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 96,
    }
    graph = _build_production_graph(checkpointer=checkpointer)
    output = graph.invoke(
        seed,
        config,
        context=context,
        durability="sync",
    )
    assert _interrupt_goal(output)["symbol"] == expected_symbol
    return graph, config, context, report_dir


def test_parse_goal_uses_one_structured_model_call_and_returns_valid_draft() -> None:
    seed = _seed(request="Research AAPL with a maximum drawdown of 15%.")
    model = GoalParserFakeChatModel(_explicit_draft())

    state = parse_goal(seed, model=model)

    assert validate_graph_state(state) == state
    assert state["status"] == "awaiting_confirmation"
    assert state["failure_stage"] is None
    assert state["failure_summary"] is None
    assert state["confirmed_goal"] is None
    assert state["missing_goal_fields"] == []
    assert state["defaulted_fields"] == []
    goal = _draft_goal(state)
    assert goal == {
        "symbol": "AAPL",
        "bar_interval": "1d",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "training_sessions": None,
        "test_sessions": None,
        "benchmark_symbol": "SPY",
        "allowed_strategies": ["moving_average"],
        "max_drawdown_limit": 0.15,
        "position_tranches": 4,
        "transaction_cost_bps": 2.0,
        "slippage_bps": 3.0,
        "request_issue": None,
    }
    assert model.bind_count == 1
    assert model.invoke_count == 1
    assert len(model.bound_tool_names) == 1
    assert model.bound_tool_names == ("GoalDraft",)
    assert convert_to_openai_messages([model.messages_seen[0][0]])[0]["role"] == "developer"
    encoded_messages = json.dumps(
        [message.content for message in model.messages_seen[0]],
        ensure_ascii=False,
    )
    assert seed["request"] in encoded_messages
    assert seed["reference_date"] in encoded_messages
    assert "1% = 100 bps" in encoded_messages
    assert "基点" in encoded_messages
    assert "moving_average for dual/double moving average or 双均线" in encoded_messages
    strategy_schema = graph_module.GoalDraft.model_json_schema()["$defs"]["StrategyName"]
    assert strategy_schema["enum"] == ["moving_average", "rsi", "macd"]
    assert "unsafe-free-text-must-not-be-saved" not in json.dumps(state)


@pytest.mark.parametrize(
    (
        "draft_changes",
        "expected_interval",
        "expected_start",
        "expected_end",
        "expected_missing",
        "defaulted",
    ),
    [
        (
            {},
            None,
            None,
            None,
            ["bar_interval"],
            _INTERVAL_INDEPENDENT_DEFAULTED_FIELDS,
        ),
        (
            {"bar_interval": "1d"},
            "1d",
            None,
            None,
            ["start_date", "end_date"],
            _ALL_DEFAULTED_FIELDS,
        ),
        (
            {
                "bar_interval": "1d",
                "start_date": "2023-07-13",
                "end_date": "2026-07-13",
                "benchmark_symbol": "QQQ",
            },
            "1d",
            "2023-07-13",
            "2026-07-13",
            [],
            _INTERVAL_INDEPENDENT_DEFAULTED_FIELDS,
        ),
    ],
)
def test_parse_goal_requires_interval_and_daily_period_while_applying_other_defaults(
    draft_changes: dict[str, object],
    expected_interval: str | None,
    expected_start: str | None,
    expected_end: str | None,
    expected_missing: list[str],
    defaulted: list[str],
) -> None:
    draft = {"symbol": "AAPL", **draft_changes}

    state = parse_goal(
        _seed(reference_date="2026-07-13"),
        model=GoalParserFakeChatModel(draft),
    )

    goal = _draft_goal(state)
    assert goal["bar_interval"] == expected_interval
    assert goal["start_date"] == expected_start
    assert goal["end_date"] == expected_end
    assert goal["allowed_strategies"] == ["moving_average", "rsi", "macd"]
    assert goal["max_drawdown_limit"] is None
    expected_benchmark = "QQQ" if draft_changes.get("benchmark_symbol") else "SPY"
    assert goal["benchmark_symbol"] == (expected_benchmark if expected_interval == "1d" else None)
    assert goal["transaction_cost_bps"] == 5.0
    assert goal["slippage_bps"] == 5.0
    assert state["missing_goal_fields"] == expected_missing
    assert state["status"] == (
        "awaiting_clarification" if expected_missing else "awaiting_confirmation"
    )
    assert state["defaulted_fields"] == defaulted
    assert ("benchmark_symbol" in state["defaulted_fields"]) == (
        expected_interval == "1d" and "benchmark_symbol" not in draft_changes
    )


def test_parse_goal_merges_one_explicit_cost_with_thread_defaults() -> None:
    state = parse_goal(
        _seed(
            request="Research AAPL with a 0.075% transaction fee.",
            transaction_cost_bps=3.0,
            slippage_bps=4.0,
        ),
        model=GoalParserFakeChatModel({"symbol": "AAPL", "transaction_cost_bps": 7.5}),
    )

    assert _draft_goal(state)["transaction_cost_bps"] == 7.5
    assert _draft_goal(state)["slippage_bps"] == 4.0
    assert "transaction_cost_bps" not in state["defaulted_fields"]
    assert "slippage_bps" in state["defaulted_fields"]


def test_goal_domain_gaps_clarify_while_schema_failure_stays_safe(tmp_path: Path) -> None:
    domain_gaps = (
        ({}, "symbol"),
        ({"symbol": "AAPL,MSFT"}, "symbol"),
        (
            {
                "symbol": "AAPL",
                "bar_interval": "1d",
                "start_date": "2023-07-13",
                "end_date": "2026-07-13",
                "benchmark_symbol": "Q/Q",
            },
            "benchmark_symbol",
        ),
        (
            {
                "symbol": "AAPL",
                "bar_interval": "1m",
                "training_sessions": 2,
                "test_sessions": 1,
                "benchmark_symbol": "QQQ",
            },
            "benchmark_symbol",
        ),
    )
    for draft, missing in domain_gaps:
        model = GoalParserFakeChatModel(draft)
        state = parse_goal(_seed(), model=model)
        assert validate_graph_state(state) == state
        assert state["status"] == "awaiting_clarification"
        assert missing in state["missing_goal_fields"]
        assert model.bind_count == model.invoke_count == 1

    model = GoalParserFakeChatModel({"symbol": "AAPL", "unexpected": "unsafe-extra-field"})
    state = parse_goal(_seed(), model=model)
    _assert_goal_parsing_failure(state)
    assert "unsafe-extra-field" not in json.dumps(state)

    ambiguous = parse_goal(
        _seed(request="研究闪迪的分线"),
        model=GoalParserFakeChatModel({"symbol": "SNDK", "request_issue": "ambiguous_fields"}),
    )
    assert ambiguous["status"] == "awaiting_clarification"
    assert ambiguous["missing_goal_fields"] == ["bar_interval"]
    assert _draft_goal(ambiguous)["request_issue"] == "ambiguous_fields"

    unrelated = parse_goal(
        _seed(request="今天天气怎么样"),
        model=GoalParserFakeChatModel({"request_issue": "unrelated_request"}),
    )
    assert unrelated["missing_goal_fields"] == ["request_details"]
    assert unrelated["defaulted_fields"] == []
    assert _draft_goal(unrelated)["allowed_strategies"] is None
    assert _draft_goal(unrelated)["transaction_cost_bps"] is None

    unsupported_model = SequencedGoalParserFakeChatModel(
        [
            {
                "symbol": "SNDK",
                "bar_interval": "1d",
                "start_date": "2023-07-13",
                "end_date": "2026-07-13",
                "request_issue": "unsupported_request",
            },
            {"symbol": "MSFT", "request_issue": "unsupported_request"},
            {
                "bar_interval": "1d",
                "start_date": "2023-07-13",
                "end_date": "2026-07-13",
                "request_issue": None,
            },
        ]
    )
    graph, config, context, _, _, _ = _start_to_confirmation(
        tmp_path,
        model=unsupported_model,
        seed=_seed(request="研究闪迪的期权日线，最近三年"),
    )
    blocked = _checkpoint_state(graph, config)
    assert blocked["missing_goal_fields"] == ["request_details"]
    assert blocked["defaulted_fields"] == []
    assert blocked["draft_goal"]["symbol"] == "SNDK"
    assert blocked["draft_goal"]["bar_interval"] == "1d"

    graph.invoke(
        Command(resume={"answer": "把标的改成 MSFT"}),
        config,
        context=context,
        durability="sync",
    )
    still_blocked = _checkpoint_state(graph, config)
    assert still_blocked["draft_goal"]["symbol"] == "MSFT"
    assert still_blocked["draft_goal"]["request_issue"] == "unsupported_request"
    assert still_blocked["missing_goal_fields"] == ["request_details"]

    graph.invoke(
        Command(resume={"answer": "改为研究 MSFT 最近三年的日线"}),
        config,
        context=context,
        durability="sync",
    )
    corrected = _checkpoint_state(graph, config)
    assert corrected["status"] == "awaiting_confirmation"
    assert corrected["draft_goal"]["request_issue"] is None
    assert corrected["missing_goal_fields"] == []


def test_start_interrupts_with_defaulted_goal_before_any_research_side_effect(
    tmp_path: Path,
) -> None:
    seed = _seed()
    model = GoalParserFakeChatModel(
        {
            "symbol": "AAPL",
            "bar_interval": "1d",
            "start_date": "2023-07-13",
            "end_date": "2026-07-13",
        }
    )

    graph, config, _context, report_dir, provider, output = _start_to_confirmation(
        tmp_path,
        model=model,
        seed=seed,
    )

    state = _checkpoint_state(graph, config)
    payload = _interrupt_payload(output)
    assert state["status"] == "awaiting_confirmation"
    assert state["confirmed_goal"] is None
    assert state["confirmation_error"] is None
    assert payload == {
        "thread_id": seed["thread_id"],
        "status": "awaiting_confirmation",
        "draft_goal": _draft_goal(state),
        "missing_fields": [],
        "defaulted_fields": _ALL_DEFAULTED_FIELDS,
        "allowed_actions": ["answer", "approve", "reject"],
    }
    assert graph.get_state(config).next == ("confirm_goal",)
    assert model.bind_count == model.invoke_count == 1
    assert provider.calls == 0
    assert not report_dir.exists()
    _assert_research_side_effects_are_zero(state)


def test_approve_freezes_the_draft_before_research_can_run(tmp_path: Path) -> None:
    model = GoalParserFakeChatModel(_explicit_draft())
    graph, config, context, report_dir, provider, _ = _start_to_confirmation(
        tmp_path,
        model=model,
    )
    draft = copy.deepcopy(_checkpoint_state(graph, config)["draft_goal"])

    graph.invoke(
        Command(resume={"action": "approve"}),
        config,
        context=context,
        durability="sync",
        interrupt_after=["confirm_goal"],
    )

    state = _checkpoint_state(graph, config)
    assert state["status"] == "running"
    assert state["phase"] == "researching"
    confirmed = state["confirmed_goal"]
    assert confirmed is not None
    assert confirmed["symbol"] == draft["symbol"]
    assert confirmed["bar_interval"] == draft["bar_interval"]
    assert confirmed["costs"] == {
        "transaction_cost_bps": draft["transaction_cost_bps"],
        "slippage_bps": draft["slippage_bps"],
    }
    assert state["confirmation_error"] is None
    assert model.bind_count == model.invoke_count == 1
    assert provider.calls == 0
    assert not report_dir.exists()
    _assert_research_side_effects_are_zero(state)


def test_answer_updates_only_explicit_goal_fields_and_reuses_reference_date(
    tmp_path: Path,
) -> None:
    original_seed = _seed(
        request="Research AAPL using the fixed defaults.",
        reference_date="2024-02-29",
    )
    period_answer = "最近三年"
    answer = "把标的改成 MSFT"
    model = SequencedGoalParserFakeChatModel(
        [
            {
                "symbol": "AAPL",
                "bar_interval": "1d",
            },
            {
                "start_date": "2021-02-28",
                "end_date": "2024-02-29",
            },
            {"symbol": "MSFT"},
        ]
    )
    graph, config, context, report_dir, provider, first_output = _start_to_confirmation(
        tmp_path,
        model=model,
        seed=original_seed,
    )
    assert _interrupt_goal(first_output)["symbol"] == "AAPL"
    initial = _checkpoint_state(graph, config)
    assert initial["status"] == "awaiting_clarification"
    assert initial["missing_goal_fields"] == ["start_date", "end_date"]

    period_output = graph.invoke(
        Command(resume={"answer": period_answer}),
        config,
        context=context,
        durability="sync",
    )
    assert isinstance(period_output, dict)
    period_goal = _draft_goal(_checkpoint_state(graph, config))
    assert period_goal["start_date"] == "2021-02-28"
    assert period_goal["end_date"] == "2024-02-29"

    edited_output = graph.invoke(
        Command(resume={"answer": answer}),
        config,
        context=context,
        durability="sync",
    )

    assert isinstance(edited_output, dict)
    edited_payload = _interrupt_payload(edited_output)
    state = _checkpoint_state(graph, config)
    goal = _draft_goal(state)
    assert state["status"] == "awaiting_confirmation"
    assert period_answer in state["request"]
    assert answer in state["request"]
    assert state["reference_date"] == "2024-02-29"
    assert state["confirmed_goal"] is None
    assert state["confirmation_error"] is None
    assert goal["symbol"] == "MSFT"
    assert goal["start_date"] == "2021-02-28"
    assert goal["end_date"] == "2024-02-29"
    assert edited_payload["draft_goal"]["symbol"] == "MSFT"
    assert edited_payload["draft_goal"]["start_date"] == "2021-02-28"
    assert edited_payload["draft_goal"]["end_date"] == "2024-02-29"
    assert graph.get_state(config).next == ("confirm_goal",)
    assert model.bind_count == model.invoke_count == 3
    assert "2024-02-29" in json.dumps(
        [message.content for message in model.messages_seen[1]],
        ensure_ascii=False,
    )
    assert provider.calls == 0
    assert not report_dir.exists()
    _assert_research_side_effects_are_zero(state)

    minute_model = SequencedGoalParserFakeChatModel(
        [
            _nullable_goal_response(
                symbol="NOW",
                bar_interval="1m",
                allowed_strategies=["rsi", "macd"],
            ),
            _nullable_goal_response(request_issue="unrelated_request"),
            _nullable_goal_response(training_sessions=2, test_sessions=1),
            _nullable_goal_response(symbol="AAPL"),
        ]
    )
    minute_graph, minute_config, minute_context, _, _, minute_output = _start_to_confirmation(
        tmp_path / "minute",
        model=minute_model,
        seed=_seed(request="研究 NOW 分钟级别的数据"),
    )
    minute_state = _checkpoint_state(minute_graph, minute_config)
    assert minute_state["status"] == "awaiting_clarification"
    assert minute_state["missing_goal_fields"] == ["training_sessions", "test_sessions"]

    minute_graph.invoke(
        Command(resume={"answer": "今天天气如何？"}),
        minute_config,
        context=minute_context,
        durability="sync",
    )
    unchanged = _checkpoint_state(minute_graph, minute_config)
    assert unchanged["draft_goal"] == minute_state["draft_goal"]
    assert unchanged["request"] == minute_state["request"]

    minute_graph.invoke(
        Command(resume={"answer": "1外，2内"}),
        minute_config,
        context=minute_context,
        durability="sync",
    )
    complete = _checkpoint_state(minute_graph, minute_config)
    assert complete["status"] == "awaiting_confirmation"
    assert complete["draft_goal"]["training_sessions"] == 2
    assert complete["draft_goal"]["test_sessions"] == 1
    assert complete["draft_goal"]["symbol"] == "NOW"
    assert complete["draft_goal"]["bar_interval"] == "1m"
    assert complete["draft_goal"]["allowed_strategies"] == ["rsi", "macd"]
    assert complete["draft_goal"]["benchmark_symbol"] is None

    minute_graph.invoke(
        Command(resume={"answer": "把标的改成 AAPL"}),
        minute_config,
        context=minute_context,
        durability="sync",
    )
    corrected = _checkpoint_state(minute_graph, minute_config)
    assert corrected["draft_goal"]["symbol"] == "AAPL"
    assert minute_model.bind_count == minute_model.invoke_count == 4
    minute_graph.invoke(
        Command(resume={"action": "approve"}),
        minute_config,
        context=minute_context,
        durability="sync",
        interrupt_after=["confirm_goal"],
    )
    confirmed = _checkpoint_state(minute_graph, minute_config)["confirmed_goal"]
    assert confirmed is not None
    assert confirmed["bar_interval"] == "1m"
    assert confirmed["training_sessions"] == 2
    assert confirmed["test_sessions"] == 1


def test_reject_ends_without_confirmed_goal_result_or_report(tmp_path: Path) -> None:
    model = GoalParserFakeChatModel({"symbol": "AAPL"})
    graph, config, context, report_dir, provider, _ = _start_to_confirmation(
        tmp_path,
        model=model,
    )

    graph.invoke(
        Command(resume={"action": "reject"}),
        config,
        context=context,
        durability="sync",
    )

    state = _checkpoint_state(graph, config)
    assert state["status"] == "rejected"
    assert state["confirmed_goal"] is None
    assert state["phase"] is None
    assert state["terminal_result"] is None
    assert state["confirmation_error"] is None
    assert graph.get_state(config).next == ()
    assert model.bind_count == model.invoke_count == 1
    assert provider.calls == 0
    assert not report_dir.exists()
    _assert_research_side_effects_are_zero(state)


def test_research_graph_has_only_four_business_nodes_and_no_fixed_tool_workflow() -> None:
    graph = _build_production_graph().get_graph()
    business_nodes = set(graph.nodes) - {"__start__", "__end__"}

    assert business_nodes == {"parse_goal", "confirm_goal", "research_agent", "critic_agent"}
    assert business_nodes.isdisjoint(_RESEARCH_TOOL_NAMES)


def test_dynamic_react_graph_completes_selected_with_switches_and_final_isolation(
    tmp_path: Path,
) -> None:
    model = DynamicResearchGraphFakeChatModel(
        _graph_research_draft(),
        _dynamic_selected_decider,
        _evidence_then_accept_critic_decider,
    )
    provider = GraphMarketDataProvider()
    events: list[AgentEvent] = []

    state, report_dir = _run_approved_graph_to_terminal(
        tmp_path,
        model=model,
        provider=provider,
        budget=AgentBudget(
            max_agent_steps=11,
            max_tool_calls=11,
            max_training_backtests=3,
        ),
        events=events,
    )

    assert state["status"] == "selected"
    assert state["phase"] == "completed"
    assert state["agent_steps"] == state["tool_calls"] == 8
    assert state["training_backtests"] == 3
    assert state["final_batch_completed"] is True
    assert len(state["training_evaluations"]) == 3
    assert set(state["final_evaluations"]) == set(state["training_evaluations"])
    messages = {event.message for event in events}
    assert "Goal Parser · Parsing the research request..." in messages
    assert "Research Agent · Deciding the next Action..." in messages
    assert "Critic Agent · Reviewing the research draft..." in messages
    assert {event.source for event in events} == {
        "goal_parser",
        "workflow",
        "research_agent",
        "critic_agent",
    }
    assert {event.kind for event in events} == {
        "status",
        "decision",
        "tool",
        "observation",
    }
    assert all(event.tool_name for event in events if event.kind in {"tool", "observation"})

    workflow_events = [event for event in events if event.source in {"goal_parser", "workflow"}]
    research_events = [event for event in events if event.source == "research_agent"]
    critic_events = [event for event in events if event.source == "critic_agent"]
    assert all(event.agent_status is None for event in workflow_events)
    assert all(event.agent_status is not None for event in research_events + critic_events)

    first_research_status = research_events[0].agent_status
    final_research_status = research_events[-1].agent_status
    first_critic_status = critic_events[0].agent_status
    final_critic_status = critic_events[-1].agent_status
    assert first_research_status is not None
    assert final_research_status is not None
    assert first_critic_status is not None
    assert final_critic_status is not None
    assert first_research_status.budget_used.model_dump() == {
        "agent_steps": 1,
        "tool_calls": 0,
        "training_backtests": 0,
    }
    assert final_research_status.budget_used.model_dump() == {
        "agent_steps": 8,
        "tool_calls": 8,
        "training_backtests": 3,
    }
    assert final_research_status.budget_remaining.model_dump() == {
        "agent_steps": 3,
        "tool_calls": 3,
        "training_backtests": 0,
    }
    assert first_critic_status.critic_round == 1
    assert first_critic_status.critic_budget_used.model_dump() == {
        "agent_steps": 1,
        "tool_calls": 0,
    }
    assert final_critic_status.critic_verdict == "accept"
    assert final_critic_status.critic_budget_used.model_dump() == {
        "agent_steps": 2,
        "tool_calls": 2,
    }
    assert final_critic_status.critic_budget_remaining.model_dump() == {
        "agent_steps": 8,
        "tool_calls": 4,
    }

    journal = state["message_journal"]
    assert [entry["tool_name"] for entry in journal] == [
        "get_market_data",
        "run_backtest",
        "analyze_drawdown",
        "run_backtest",
        "run_backtest",
        "run_backtest",
        "compare_with_benchmark",
        "submit_research_draft",
    ]
    training_entries = [
        entry
        for entry in journal
        if entry["tool_name"] == "run_backtest" and entry["arguments"]["stage"] == "training"
    ]
    assert [entry["arguments"]["strategy_name"] for entry in training_entries] == [
        "moving_average",
        "moving_average",
        "rsi",
    ]
    assert training_entries[0]["arguments"]["parameters"] == {
        "short_window": 2,
        "long_window": 3,
    }
    assert training_entries[1]["arguments"]["parameters"] == {
        "short_window": 2,
        "long_window": 4,
    }
    first_fingerprint = training_entries[0]["observation"]["identifiers"]["fingerprint"]
    assert journal[2]["observation"]["identifiers"]["fingerprint"] == first_fingerprint

    locked_fingerprint = state["locked_fingerprint"]
    assert isinstance(locked_fingerprint, str)
    assert training_entries[1]["observation"]["identifiers"]["fingerprint"] == locked_fingerprint
    terminal = state["terminal_result"]
    assert terminal is not None
    assert terminal["status"] == "selected"
    assert terminal["selected_fingerprint"] == locked_fingerprint
    assert terminal["evaluated_fingerprint"] == locked_fingerprint
    assert (
        terminal["test_metrics"]["asset_buy_and_hold_return"]
        != terminal["test_metrics"]["benchmark_return"]
    )

    comparison = journal[-2]["observation"]
    assert comparison["status"] == "success"
    assert set(comparison["metrics"]) == {
        "strategy_total_return",
        "asset_buy_and_hold_return",
        "excess_vs_asset_buy_and_hold",
        "benchmark_return",
        "excess_vs_benchmark",
    }
    assert (
        comparison["metrics"]["asset_buy_and_hold_return"]
        != comparison["metrics"]["benchmark_return"]
    )

    researching_turns = 0
    locked_turns = 0
    for messages in model.research_messages_seen:
        assert len(messages) == 2
        assert convert_to_openai_messages([messages[0]])[0]["role"] == "developer"
        react_state = _react_state(messages)
        assert all("test_metrics" not in item for item in react_state["tested_candidates"])
        encoded = "\n".join(
            message.content for message in messages if isinstance(message.content, str)
        )
        if react_state["phase"] == "researching":
            researching_turns += 1
            assert '"test_metrics"' not in encoded
            assert react_state["locked_candidate"] is None
        else:
            locked_turns += 1
            assert react_state["phase"] == "locked"
            assert react_state["locked_candidate"]["fingerprint"] == locked_fingerprint
            assert "test_metrics" in react_state["locked_candidate"]
    assert researching_turns == 6
    assert locked_turns == 2

    assert _RESEARCH_TOOL_NAMES in model.binding_history
    assert _CRITIC_TOOL_NAMES in model.binding_history
    assert len(model.critic_messages_seen) == 2
    for messages in model.critic_messages_seen:
        assert len(messages) == 2
        assert convert_to_openai_messages([messages[0]])[0]["role"] == "developer"
    assert model.bound_tool_names == _CRITIC_TOOL_NAMES
    assert all(
        options
        == {
            "parallel_tool_calls": False,
            "strict": False,
            "tool_choice": "required",
        }
        for options in model.binding_options[1:]
    )

    critic_round = state["critic_rounds"][0]
    assert critic_round["agent_steps"] == 2
    assert critic_round["tool_calls"] == 2
    assert [entry["tool_name"] for entry in critic_round["journal"]] == [
        "inspect_selected_drawdown",
        "submit_critic_review",
    ]
    assert ("inspect_selected_drawdown",) not in model.binding_history
    assert critic_round["review"]["verdict"] == "accept"
    assert [call[0] for call in provider.calls] == ["AAPL", "SPY"]
    assert {path.name for path in report_dir.iterdir()} == {
        "report.json",
    }
    report = load_report_view(report_dir / "report.json")
    assert report.critic_review_status == "accepted"
    assert report.critic_reviews[-1].verdict == "accept"


def test_dynamic_react_graph_can_finish_with_normal_no_strategy_result(
    tmp_path: Path,
) -> None:
    model = DynamicResearchGraphFakeChatModel(
        _graph_research_draft(),
        _no_strategy_decider,
    )
    provider = GraphMarketDataProvider()

    state, report_dir = _run_approved_graph_to_terminal(
        tmp_path,
        model=model,
        provider=provider,
        budget=AgentBudget(
            max_agent_steps=6,
            max_tool_calls=6,
            max_training_backtests=1,
        ),
    )

    terminal = state["terminal_result"]
    assert terminal is not None
    assert state["status"] == terminal["status"] == "no_strategy"
    assert state["phase"] == "completed"
    assert state["agent_steps"] == state["tool_calls"] == 4
    assert state["training_backtests"] == 1
    fingerprint = next(iter(state["training_evaluations"]))
    assert state["locked_fingerprint"] == fingerprint
    assert state["final_batch_completed"] is True
    assert len(state["training_evaluations"]) == 1
    assert set(state["final_evaluations"]) == set(state["training_evaluations"])
    assert terminal["selected_fingerprint"] is None
    assert terminal["evaluated_fingerprint"] == fingerprint
    assert terminal["training_metrics"]
    assert terminal["test_metrics"]
    assert [entry["tool_name"] for entry in state["message_journal"]] == [
        "get_market_data",
        "run_backtest",
        "run_backtest",
        "submit_research_draft",
    ]
    assert [call[0] for call in provider.calls] == ["AAPL", "SPY"]
    assert {path.name for path in report_dir.iterdir()} == {
        "report.json",
    }


def test_one_critic_revision_reuses_the_frozen_candidate_then_accepts(
    tmp_path: Path,
) -> None:
    model = DynamicResearchGraphFakeChatModel(
        _graph_research_draft(),
        _revision_research_decider,
        _revise_then_accept_critic_decider,
    )
    state, report_dir = _run_approved_graph_to_terminal(
        tmp_path,
        model=model,
        provider=GraphMarketDataProvider(),
        budget=AgentBudget(
            max_agent_steps=10,
            max_tool_calls=10,
            max_training_backtests=1,
        ),
    )

    assert state["status"] == "selected"
    assert len(state["critic_rounds"]) == 2
    assert [review_round["review"]["verdict"] for review_round in state["critic_rounds"]] == [
        "revise",
        "accept",
    ]
    fingerprints = {
        review_round["draft"]["selected_fingerprint"] for review_round in state["critic_rounds"]
    }
    assert fingerprints == {state["locked_fingerprint"]}
    assert [entry["tool_name"] for entry in state["message_journal"]] == [
        "get_market_data",
        "run_backtest",
        "run_backtest",
        "submit_research_draft",
        "compare_with_benchmark",
        "submit_research_draft",
    ]
    assert _RESEARCH_REVISION_TOOL_NAMES in model.binding_history
    report = load_report_view(report_dir / "report.json")
    assert [review.verdict for review in report.critic_reviews] == ["revise", "accept"]


def test_retryable_research_model_error_keeps_checkpointed_step_then_empty_resume(
    tmp_path: Path,
) -> None:
    saver = InMemorySaver()
    decision = FailOnceResearchDecision()
    model = DynamicResearchGraphFakeChatModel(
        _graph_research_draft(),
        decision,
    )
    provider = GraphMarketDataProvider()
    graph, config, context, report_dir = _start_durable_research_graph(
        tmp_path,
        model=model,
        provider=provider,
        budget=AgentBudget(
            max_agent_steps=1,
            max_tool_calls=1,
            max_training_backtests=1,
        ),
        checkpointer=saver,
    )

    with pytest.raises(graph_module.RecoverableResearchError) as caught:
        graph.invoke(
            Command(resume={"action": "approve"}),
            config,
            context=context,
            durability="sync",
        )

    assert caught.value.stage == "model"
    assert caught.value.args == ("The research model call failed safely.",)
    assert str(caught.value) == "The research model call failed safely."
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "unsafe-" not in repr(getattr(caught.value, "__notes__", ()))
    assert "unsafe-transient-model-detail" not in "".join(traceback.format_exception(caught.value))
    assert "unsafe-transient-model-detail" not in _checkpoint_error_envelope(
        saver,
        graph,
        config,
    )
    failed_state = _checkpoint_state(graph, config)
    active_state = _active_agent_workspace_checkpoint(saver, graph, config)
    assert failed_state["status"] == "running"
    assert failed_state["phase"] == "researching"
    assert active_state["status"] == "running"
    assert active_state["agent_steps"] == 1
    assert active_state["tool_calls"] == 0
    assert graph.get_state(config).next == ("research_agent",)
    assert model.invoke_count == 2
    assert len(model.research_messages_seen) == decision.calls == 1
    assert provider.calls == []
    assert not report_dir.exists()
    _assert_research_side_effects_are_zero(failed_state)
    assert "unsafe-transient-model-detail" not in json.dumps(failed_state)

    graph.invoke(
        None,
        config,
        context=context,
        durability="sync",
    )

    resumed_state = _checkpoint_state(graph, config)
    assert resumed_state["status"] == "budget_exhausted"
    assert resumed_state["agent_steps"] == resumed_state["tool_calls"] == 1
    assert [entry["tool_name"] for entry in resumed_state["message_journal"]] == ["get_market_data"]
    assert model.invoke_count == 3
    assert len(model.research_messages_seen) == decision.calls == 2
    assert [call[0] for call in provider.calls] == ["AAPL", "SPY"]


def test_retryable_research_tool_error_resumes_without_redeciding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saver = InMemorySaver()
    model = DynamicResearchGraphFakeChatModel(
        _graph_research_draft(),
        _single_market_data_decider,
    )
    provider = GraphMarketDataProvider()
    graph, config, context, report_dir = _start_durable_research_graph(
        tmp_path,
        model=model,
        provider=provider,
        budget=AgentBudget(
            max_agent_steps=1,
            max_tool_calls=1,
            max_training_backtests=1,
        ),
        checkpointer=saver,
    )
    real_get_market_data = runtime_module.get_market_data
    operation_calls = 0

    def transient_get_market_data(workspace: Any) -> Any:
        nonlocal operation_calls
        operation_calls += 1
        if operation_calls == 1:
            error = graph_module.RecoverableResearchError(stage="tool")
            error.add_note("unsafe-transient-tool-detail")
            raise error
        return real_get_market_data(workspace)

    monkeypatch.setattr(runtime_module, "get_market_data", transient_get_market_data)

    with pytest.raises(graph_module.RecoverableResearchError) as caught:
        graph.invoke(
            Command(resume={"action": "approve"}),
            config,
            context=context,
            durability="sync",
        )

    assert caught.value.stage == "tool"
    assert caught.value.args == ("The research Tool failed temporarily.",)
    assert str(caught.value) == "The research Tool failed temporarily."
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "unsafe-" not in repr(getattr(caught.value, "__notes__", ()))
    assert "unsafe-transient-tool-detail" not in "".join(traceback.format_exception(caught.value))
    assert "unsafe-transient-tool-detail" not in _checkpoint_error_envelope(
        saver,
        graph,
        config,
    )
    failed_state = _checkpoint_state(graph, config)
    active_state = _active_agent_workspace_checkpoint(saver, graph, config)
    assert failed_state["status"] == "running"
    assert active_state["status"] == "running"
    assert active_state["agent_steps"] == 1
    assert active_state["tool_calls"] == 0
    assert active_state["message_journal"] == []
    assert graph.get_state(config).next == ("research_agent",)
    assert failed_state["message_journal"] == []
    assert len(model.research_messages_seen) == 1
    assert operation_calls == 1
    assert provider.calls == []
    assert not report_dir.exists()

    graph.invoke(
        None,
        config,
        context=context,
        durability="sync",
    )

    resumed_state = _checkpoint_state(graph, config)
    assert resumed_state["status"] == "budget_exhausted"
    assert resumed_state["agent_steps"] == resumed_state["tool_calls"] == 1
    assert [entry["tool_name"] for entry in resumed_state["message_journal"]] == ["get_market_data"]
    assert len(model.research_messages_seen) == 1
    assert operation_calls == 2
    assert [call[0] for call in provider.calls] == ["AAPL", "SPY"]


def test_two_threads_keep_state_provider_and_report_directories_isolated(
    tmp_path: Path,
) -> None:
    saver = InMemorySaver()
    first_budget = AgentBudget(
        max_agent_steps=1,
        max_tool_calls=1,
        max_training_backtests=1,
    )
    second_budget = AgentBudget(
        max_agent_steps=2,
        max_tool_calls=2,
        max_training_backtests=1,
    )
    first_model = DynamicResearchGraphFakeChatModel(
        _graph_research_draft("AAPL"),
        _single_market_data_decider,
    )
    second_model = DynamicResearchGraphFakeChatModel(
        _graph_research_draft("MSFT"),
        _single_market_data_decider,
    )
    first_provider = GraphMarketDataProvider("AAPL")
    second_provider = GraphMarketDataProvider("MSFT")
    first_graph, first_config, first_context, first_report_dir = _start_durable_research_graph(
        tmp_path,
        model=first_model,
        provider=first_provider,
        budget=first_budget,
        checkpointer=saver,
        expected_symbol="AAPL",
    )
    second_graph, second_config, second_context, second_report_dir = _start_durable_research_graph(
        tmp_path,
        model=second_model,
        provider=second_provider,
        budget=second_budget,
        checkpointer=saver,
        expected_symbol="MSFT",
    )

    first_graph.invoke(
        Command(resume={"action": "approve"}),
        first_config,
        context=first_context,
        durability="sync",
    )
    second_graph.invoke(
        Command(resume={"action": "approve"}),
        second_config,
        context=second_context,
        durability="sync",
    )

    first_state = _checkpoint_state(first_graph, first_config)
    second_state = _checkpoint_state(second_graph, second_config)
    assert first_state["thread_id"] != second_state["thread_id"]
    assert first_state["confirmed_goal"]["symbol"] == "AAPL"
    assert second_state["confirmed_goal"]["symbol"] == "MSFT"
    assert first_state["data_snapshot"]["asset_symbol"] == "AAPL"
    assert second_state["data_snapshot"]["asset_symbol"] == "MSFT"
    assert first_state["status"] == second_state["status"] == "budget_exhausted"
    assert first_state["agent_steps"] == first_state["tool_calls"] == 1
    assert second_state["agent_steps"] == second_state["tool_calls"] == 2
    assert [call[0] for call in first_provider.calls] == ["AAPL", "SPY"]
    assert [call[0] for call in second_provider.calls] == ["MSFT", "SPY"]
    assert first_report_dir != second_report_dir
    assert first_report_dir.parent == second_report_dir.parent == tmp_path
    assert {path.name for path in first_report_dir.iterdir()} == {
        "report.json",
    }
    assert {path.name for path in second_report_dir.iterdir()} == {
        "report.json",
    }
    assert load_report_view(first_report_dir / "report.json").symbol == "AAPL"
    assert load_report_view(second_report_dir / "report.json").symbol == "MSFT"


def test_terminal_empty_resume_does_not_rerun_model_tool_or_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DynamicResearchGraphFakeChatModel(
        _graph_research_draft(),
        _single_market_data_decider,
    )
    provider = GraphMarketDataProvider()
    graph, config, context, report_dir = _start_durable_research_graph(
        tmp_path,
        model=model,
        provider=provider,
        budget=AgentBudget(
            max_agent_steps=1,
            max_tool_calls=1,
            max_training_backtests=1,
        ),
        checkpointer=InMemorySaver(),
    )
    graph.invoke(
        Command(resume={"action": "approve"}),
        config,
        context=context,
        durability="sync",
    )
    before_snapshot = graph.get_state(config)
    before_state = validate_graph_state(dict(before_snapshot.values))
    before_checkpoint_id = before_snapshot.config["configurable"]["checkpoint_id"]
    model_calls = model.invoke_count
    provider_calls = list(provider.calls)
    report_contents = {path.name: path.read_bytes() for path in sorted(report_dir.iterdir())}
    report_calls = 0

    def must_not_generate_report(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal report_calls
        report_calls += 1
        raise AssertionError("a terminal resume must not regenerate its report")

    monkeypatch.setattr(finalization_module, "generate_report", must_not_generate_report)

    graph.invoke(
        None,
        config,
        context=context,
        durability="sync",
    )

    after_snapshot = graph.get_state(config)
    assert validate_graph_state(dict(after_snapshot.values)) == before_state
    assert after_snapshot.next == before_snapshot.next == ()
    assert after_snapshot.config["configurable"]["checkpoint_id"] == before_checkpoint_id
    assert model.invoke_count == model_calls
    assert provider.calls == provider_calls
    assert report_calls == 0
    assert {
        path.name: path.read_bytes() for path in sorted(report_dir.iterdir())
    } == report_contents


def test_maximum_tool_path_stops_by_budget_within_recursion_limit(
    tmp_path: Path,
) -> None:
    model = DynamicResearchGraphFakeChatModel(
        _graph_research_draft(),
        _worst_case_budget_decider,
    )
    provider = GraphMarketDataProvider()
    graph, config, context, _report_dir = _start_durable_research_graph(
        tmp_path,
        model=model,
        provider=provider,
        budget=AgentBudget(),
        checkpointer=InMemorySaver(),
    )

    graph.invoke(
        Command(resume={"action": "approve"}),
        config,
        context=context,
        durability="sync",
    )

    state = _checkpoint_state(graph, config)
    assert state["status"] == "budget_exhausted"
    assert state["agent_steps"] == state["tool_calls"] == 30
    assert state["training_backtests"] == 0
    assert state["validated_configs"] == {}
    assert state["training_evaluations"] == state["final_evaluations"] == []
    assert len(state["message_journal"]) == 30
    assert [entry["tool_name"] for entry in state["message_journal"]] == ["analyze_drawdown"] * 30
    assert all(entry["observation"]["status"] == "rejected" for entry in state["message_journal"])
    terminal = state["terminal_result"]
    assert terminal is not None
    assert terminal["status"] == "budget_exhausted"
    assert terminal["budget_usage"] == {
        "agent_steps": 30,
        "tool_calls": 30,
        "training_backtests": 0,
    }
    assert len(model.research_messages_seen) == 30
    assert provider.calls == []


def test_final_lock_recompile_preserves_test_visibility_and_submits_without_download(
    tmp_path: Path,
) -> None:
    saver = InMemorySaver()
    first_model = DynamicResearchGraphFakeChatModel(
        _graph_research_draft(),
        _pause_after_final_decider,
    )
    first_provider = GraphMarketDataProvider()
    graph, config, context, report_dir = _start_durable_research_graph(
        tmp_path,
        model=first_model,
        provider=first_provider,
        budget=AgentBudget(
            max_agent_steps=4,
            max_tool_calls=4,
            max_training_backtests=1,
        ),
        checkpointer=saver,
    )

    with pytest.raises(graph_module.RecoverableResearchError) as caught:
        graph.invoke(
            Command(resume={"action": "approve"}),
            config,
            context=context,
            durability="sync",
        )

    assert caught.value.stage == "model"
    assert caught.value.__context__ is None
    locked_state = _checkpoint_state(graph, config)
    assert locked_state["status"] == "running"
    assert locked_state["phase"] == "researching"
    assert graph.get_state(config).next == ("research_agent",)
    assert [call[0] for call in first_provider.calls] == ["AAPL", "SPY"]
    assert not report_dir.exists()
    before_prompt_state = _react_state(first_model.research_messages_seen[-1])
    locked_fingerprint = before_prompt_state["locked_candidate"]["fingerprint"]
    assert isinstance(locked_fingerprint, str)

    resumed_model = DynamicResearchGraphFakeChatModel(
        _graph_research_draft(),
        _submit_locked_decider,
    )
    resumed_provider = GraphMarketDataProvider()
    resumed_context = {
        **context,
        "model": resumed_model,
        "provider": resumed_provider,
    }
    resumed_graph = _build_production_graph(checkpointer=saver)
    resumed_graph.invoke(
        None,
        config,
        context=resumed_context,
        durability="sync",
    )

    state = _checkpoint_state(resumed_graph, config)
    assert state["status"] == "selected"
    assert state["phase"] == "completed"
    assert state["locked_fingerprint"] == locked_fingerprint
    assert state["final_evaluations"] == [locked_fingerprint]
    terminal = state["terminal_result"]
    assert terminal is not None
    assert terminal["selected_fingerprint"] == locked_fingerprint
    assert terminal["evaluated_fingerprint"] == locked_fingerprint
    after_prompt_state = _react_state(resumed_model.research_messages_seen[0])
    assert after_prompt_state["locked_candidate"] == before_prompt_state["locked_candidate"]
    assert after_prompt_state["tested_candidates"] == before_prompt_state["tested_candidates"]
    assert resumed_provider.calls == []
    assert {path.name for path in report_dir.iterdir()} == {
        "report.json",
    }
