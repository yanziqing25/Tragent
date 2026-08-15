"""Tests for the fixed, strict StructuredTool boundary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

import trading.agent as agent_package
from trading.agent.models import ToolObservation
from trading.agent.tools import (
    EvaluationArgs,
    NoArgs,
    RunBacktestArgs,
    SubmitResearchDraftArgs,
    build_research_tools,
)

FINGERPRINT = "a" * 64
TOOL_NAMES = (
    "get_market_data",
    "calculate_asset_statistics",
    "run_backtest",
    "analyze_drawdown",
    "compare_with_benchmark",
    "submit_research_draft",
)


@pytest.fixture
def runtime_spy(monkeypatch: pytest.MonkeyPatch) -> tuple[object, list[tuple[str, object, dict]]]:
    workspace = object()
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

    fake_runtime = SimpleNamespace(**operations)
    monkeypatch.setattr(agent_package, "runtime", fake_runtime, raising=False)
    return workspace, calls


def _tools_by_name(workspace: object) -> dict[str, StructuredTool]:
    return {
        tool.name: tool
        for tool in build_research_tools(lambda: workspace)  # type: ignore[return-value]
    }


def test_builds_exactly_six_tools_with_strict_openai_schemas(runtime_spy) -> None:
    workspace, _ = runtime_spy
    tools = build_research_tools(lambda: workspace)  # type: ignore[return-value]

    assert tuple(tool.name for tool in tools) == TOOL_NAMES
    assert all(isinstance(tool, StructuredTool) for tool in tools)
    run_backtest_description = next(
        tool.description for tool in tools if tool.name == "run_backtest"
    )
    for parameter_name in ("short_window", "rsi_window", "fast_period"):
        assert parameter_name in run_backtest_description

    forbidden_model_fields = {
        "symbol",
        "start_date",
        "end_date",
        "benchmark_symbol",
        "output_dir",
        "metrics",
        "python_code",
    }
    for tool in tools:
        assert tool.args_schema is not None
        assert tool.return_direct is False
        assert tool.args_schema.model_json_schema()["additionalProperties"] is False
        converted = convert_to_openai_tool(tool)
        parameters = converted["function"]["parameters"]
        assert parameters["type"] == "object"
        assert forbidden_model_fields.isdisjoint(parameters.get("properties", {}))


@pytest.mark.parametrize(
    "args_schema",
    [NoArgs, RunBacktestArgs, EvaluationArgs, SubmitResearchDraftArgs],
)
def test_all_tool_schemas_forbid_extra_fields(args_schema) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        args_schema.model_validate({"unexpected": "value"})


@pytest.mark.parametrize("bad_value", [True, "20", float("nan"), float("inf")])
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


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"a": 1, "b": 2, "c": 3, "d": 4},
        {"x" * 65: 1},
    ],
)
def test_strategy_parameter_mapping_is_compact(parameters: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        RunBacktestArgs.model_validate(
            {
                "stage": "training",
                "strategy_name": "moving_average",
                "parameters": parameters,
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"stage": "training"},
        {
            "stage": "training",
            "strategy_name": "moving_average",
            "parameters": {"short_window": 20, "long_window": 100},
            "selected_fingerprint": FINGERPRINT,
        },
        {"stage": "final", "strategy_name": "moving_average"},
        {"stage": "final", "parameters": {"short_window": 20, "long_window": 100}},
        {"stage": "final"},
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
    with pytest.raises(ValidationError, match="blank"):
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
    tool = _tools_by_name(workspace)[tool_name]
    tool_call_id = f"call-{tool_name}"

    result = tool.invoke(
        {
            "name": tool_name,
            "args": arguments,
            "id": tool_call_id,
            "type": "tool_call",
        }
    )

    assert isinstance(result, ToolMessage)
    assert result.name == tool_name
    assert result.tool_call_id == tool_call_id
    assert result.status == "success"
    assert isinstance(result.content, str)
    observation = ToolObservation.model_validate_json(result.content)
    assert observation.tool_name == tool_name
    assert observation.status == "success"
    assert len(calls) == 1
    assert calls[0][0] == tool_name
    assert calls[0][1] is workspace
    expected_runtime_arguments = dict(arguments)
    if tool_name == "run_backtest":
        expected_runtime_arguments["selected_fingerprint"] = None
    elif tool_name == "submit_research_draft":
        expected_runtime_arguments["limitations"] = tuple(arguments["limitations"])
    assert calls[0][2] == expected_runtime_arguments
