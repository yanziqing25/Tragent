"""Tests for the fixed, strict StructuredTool boundary."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import ValidationError

import tragent.research.agent.tools as tools_module
from tragent.research.agent.models import AgentBudget, ResearchGoal, ToolObservation
from tragent.research.agent.tools import (
    RunBacktestArgs,
    SubmitResearchDraftArgs,
    build_research_tools,
)
from tragent.research.agent.workspace import ResearchGraphContext, ResearchWorkspace

FINGERPRINT = "a" * 64
TOOL_NAMES = (
    "get_market_data",
    "calculate_asset_statistics",
    "run_backtest",
    "analyze_drawdown",
    "compare_with_benchmark",
    "search_market_context",
    "submit_research_draft",
)


@pytest.fixture
def runtime_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ResearchWorkspace, list[tuple[str, object, dict]]]:
    workspace = ResearchWorkspace(
        goal=ResearchGoal(
            objective="Exercise the Tool boundary.",
            symbol="AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            allowed_strategies=("moving_average", "rsi", "macd"),
        ),
        budget=AgentBudget(),
    )
    calls: list[tuple[str, object, dict]] = []
    operations: dict[str, Any] = {}

    for tool_name in TOOL_NAMES:

        def operation(
            received_workspace: object,
            *,
            _tool_name: str = tool_name,
            **arguments: object,
        ) -> ToolObservation:
            calls.append((_tool_name, received_workspace, arguments))
            return ToolObservation(
                tool_name=_tool_name,
                status="success",
                summary=f"{_tool_name} completed.",
            )

        operations[tool_name] = operation

    monkeypatch.setattr(tools_module, "runtime", SimpleNamespace(**operations))
    return workspace, calls


def _tools_by_name() -> dict[str, StructuredTool]:
    return {tool.name: tool for tool in build_research_tools()}


def _call(
    tool: StructuredTool,
    workspace: ResearchWorkspace,
    arguments: dict[str, object],
    events: list[object] | None = None,
) -> Any:
    """Invoke a Tool the way ToolNode does, with the injected ToolRuntime."""

    tool_call_id = f"call-{tool.name}"
    runtime = ToolRuntime(
        state={"workspace": workspace.model_dump(mode="json")},
        context=ResearchGraphContext(
            model=MagicMock(), provider=MagicMock(), artifacts="research/t"
        ),
        config={},
        stream_writer=(events if events is not None else []).append,
        tool_call_id=tool_call_id,
        store=None,
    )
    return tool.invoke(
        {
            "name": tool.name,
            "args": {**arguments, "tool_runtime": runtime},
            "id": tool_call_id,
            "type": "tool_call",
        }
    )


def test_builds_the_research_agent_toolset(runtime_spy) -> None:
    tools = build_research_tools()

    assert tuple(tool.name for tool in tools) == TOOL_NAMES
    run_backtest_description = next(
        tool.description for tool in tools if tool.name == "run_backtest"
    )
    for parameter_name in ("short_window", "rsi_window", "fast_period"):
        assert parameter_name in run_backtest_description


@pytest.mark.parametrize("bad_value", [True, float("nan")])
def test_strategy_parameters_are_strict_numbers(bad_value: object) -> None:
    with pytest.raises(ValidationError):
        RunBacktestArgs.model_validate(
            {
                "stage": "training",
                "strategy_name": "moving_average",
                "parameters": {"short_window": bad_value, "long_window": 100},
            }
        )

    parsed = RunBacktestArgs.model_validate(
        {
            "stage": "training",
            "strategy_name": "rsi",
            "parameters": {
                "rsi_window": 14,
                "entry_threshold": 30.0,
                "exit_threshold": 70,
            },
        }
    )
    assert parsed.parameters == {
        "rsi_window": 14,
        "entry_threshold": 30.0,
        "exit_threshold": 70,
    }


def test_strategy_parameter_mapping_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        RunBacktestArgs.model_validate(
            {
                "stage": "training",
                "strategy_name": "moving_average",
                "parameters": {},
            }
        )


@pytest.mark.parametrize("policy", ["intraday_rsi", "rsi_price_exit", "rsi_vwap_exit"])
def test_fixed_policy_tool_accepts_only_empty_parameters(runtime_spy, policy) -> None:
    workspace, calls = runtime_spy
    args = {"stage": "training", "strategy_name": policy, "parameters": {}}
    _call(_tools_by_name()["run_backtest"], workspace, args)
    assert calls[-1][0] == "run_backtest"
    assert calls[-1][2] == {**args, "selected_fingerprint": None}
    with pytest.raises(ValidationError, match="fixed intraday"):
        RunBacktestArgs.model_validate({**args, "parameters": {"rsi_window": 7}})


@pytest.mark.parametrize(
    "payload",
    [
        {"stage": "training"},
        {
            "stage": "final",
            "strategy_name": "moving_average",
            "parameters": {"short_window": 20, "long_window": 100},
        },
    ],
)
def test_run_backtest_schema_rejects_invalid_stage_combinations(payload: dict) -> None:
    with pytest.raises(ValidationError):
        RunBacktestArgs.model_validate(payload)


def test_run_backtest_schema_accepts_training_and_tested_final_candidate() -> None:
    training = RunBacktestArgs.model_validate(
        {
            "stage": "training",
            "strategy_name": "moving_average",
            "parameters": {"short_window": 20, "long_window": 100},
        }
    )
    selected = RunBacktestArgs.model_validate(
        {"stage": "final", "selected_fingerprint": FINGERPRINT}
    )
    assert training.stage == "training"
    assert selected.selected_fingerprint == FINGERPRINT


def test_submit_schema_enforces_terminal_text_and_selected_fingerprint() -> None:
    with pytest.raises(ValidationError, match="selected_fingerprint"):
        SubmitResearchDraftArgs.model_validate(
            {"outcome": "selected", "conclusion": "Choose the candidate."}
        )
    with pytest.raises(ValidationError):
        SubmitResearchDraftArgs.model_validate(
            {
                "outcome": "no_strategy",
                "selected_fingerprint": FINGERPRINT,
                "conclusion": "   ",
            }
        )
    with pytest.raises(ValidationError, match="at least 1 character"):
        SubmitResearchDraftArgs.model_validate(
            {
                "outcome": "no_strategy",
                "conclusion": "No candidate qualified.",
                "limitations": ["   "],
            }
        )

    parsed = SubmitResearchDraftArgs.model_validate(
        {
            "outcome": "selected",
            "selected_fingerprint": FINGERPRINT,
            "conclusion": "  Choose the frozen candidate.  ",
            "limitations": ["Daily bars only."],
        }
    )
    assert parsed.conclusion == "Choose the frozen candidate."
    assert parsed.limitations == ("Daily bars only.",)


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("get_market_data", {}),
        ("calculate_asset_statistics", {}),
        (
            "run_backtest",
            {
                "stage": "training",
                "strategy_name": "moving_average",
                "parameters": {"short_window": 20, "long_window": 100},
            },
        ),
        ("analyze_drawdown", {"fingerprint": FINGERPRINT, "segment": "training"}),
        ("compare_with_benchmark", {"fingerprint": FINGERPRINT, "segment": "test"}),
        (
            "search_market_context",
            {"queries": {"sec": "recent risk disclosures"}},
        ),
        (
            "submit_research_draft",
            {
                "outcome": "no_strategy",
                "selected_fingerprint": FINGERPRINT,
                "conclusion": "No candidate met the confirmed objective.",
                "limitations": ["Daily bars only."],
            },
        ),
    ],
)
def test_full_tool_call_preserves_protocol_and_delegates_once(
    runtime_spy,
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    workspace, calls = runtime_spy
    events: list[object] = []

    result = _call(_tools_by_name()[tool_name], workspace, arguments, events)

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == f"call-{tool_name}"
    observation = ToolObservation.model_validate_json(message.content)
    assert observation.tool_name == tool_name
    assert observation.status == "success"
    assert len(calls) == 1
    assert calls[0][0] == tool_name
    assert isinstance(calls[0][1], ResearchWorkspace)
    expected_runtime_arguments = dict(arguments)
    if tool_name == "run_backtest":
        expected_runtime_arguments["selected_fingerprint"] = None
    elif tool_name == "submit_research_draft":
        expected_runtime_arguments["limitations"] = tuple(arguments["limitations"])
        expected_runtime_arguments["market_context_summary"] = None
    elif tool_name == "search_market_context":
        expected_runtime_arguments["revision_reason"] = None
    assert calls[0][2] == expected_runtime_arguments
    updated = ResearchWorkspace.model_validate(result.update["workspace"])
    assert updated.tool_calls == 1
    assert [(record.tool_name, record.observation) for record in updated.journal] == [
        (tool_name, observation)
    ]
    assert [event["kind"] for event in events] == ["tool", "observation"]
