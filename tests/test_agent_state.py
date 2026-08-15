"""Plain-JSON checkpoint State, journal, and terminal-result adapter tests."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from trading.agent.models import (
    AgentBudget,
    GoalDraft,
    ResearchGoal,
    ToolObservation,
)
from trading.agent.prompt import build_research_messages
from trading.agent.state import (
    GraphState,
    aligned_data_to_snapshot,
    restore_workspace_from_state,
    validate_graph_state,
    workspace_to_state,
)
from trading.agent.workspace import ResearchWorkspace, record_event
from trading.quant import AlignedMarketData, DataQualityReport, MarketData


class NeverProvider:
    def __init__(self) -> None:
        self.calls = 0

    def get_daily_bars(self, *_args: object, **_kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("State adapter tests must not access market data")


def _goal() -> ResearchGoal:
    return ResearchGoal(
        objective="Compare transparent long-only strategies.",
        symbol="AAPL",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 30),
        allowed_strategies=("moving_average", "rsi", "macd"),
        max_drawdown_limit=0.2,
    )


def _draft() -> dict[str, object]:
    goal = _goal()
    return GoalDraft(
        symbol=goal.symbol,
        bar_interval=goal.bar_interval,
        start_date=goal.start_date.isoformat() if goal.start_date else None,
        end_date=goal.end_date.isoformat() if goal.end_date else None,
        allowed_strategies=list(goal.allowed_strategies),
        max_drawdown_limit=goal.max_drawdown_limit,
        position_tranches=goal.position_tranches,
        transaction_cost_bps=goal.costs.transaction_cost_bps,
        slippage_bps=goal.costs.slippage_bps,
        benchmark_symbol=goal.benchmark_symbol,
    ).model_dump(mode="json")


def _aligned_data() -> AlignedMarketData:
    index = pd.date_range("2024-01-01", periods=30, freq="D")
    asset_values = [100.0 + offset for offset in range(len(index))]
    benchmark_values = [200.0 + offset * 0.5 for offset in range(len(index))]

    def build(symbol: str, values: list[float], *, optional_nan: bool) -> MarketData:
        high = [value + 1.0 for value in values]
        volume = [1_000.0] * len(values)
        if optional_nan:
            high[3] = float("nan")
            volume[4] = float("nan")
        bars = pd.DataFrame(
            {
                "open": values,
                "high": high,
                "low": [value - 1.0 for value in values],
                "close": values,
                "volume": volume,
            },
            index=index,
        )
        return MarketData(
            symbol=symbol,
            bars=bars,
            quality=DataQualityReport(
                symbol=symbol,
                valid_rows=len(index),
                start_date=index[0],
                end_date=index[-1],
                duplicate_rows_removed=1 if symbol == "AAPL" else 0,
                missing_value_counts={
                    "open": 0,
                    "high": 1 if optional_nan else 0,
                    "low": 0,
                    "close": 0,
                    "volume": 1 if optional_nan else 0,
                },
                invalid_price_rows_removed=0,
                warnings=(f"{symbol}: preserved test quality metadata.",),
                is_valid=True,
            ),
        )

    return AlignedMarketData(
        asset=build("AAPL", asset_values, optional_nan=True),
        benchmark=build("SPY", benchmark_values, optional_nan=False),
        warnings=("The two test frames were already aligned.",),
    )


def _state(*, status: str = "awaiting_confirmation") -> GraphState:
    goal = _goal().model_dump(mode="json")
    draft = _draft()
    confirmed = status not in {"awaiting_confirmation", "rejected"}
    terminal = None
    phase = "researching" if status == "running" else None
    if status in {"selected", "no_strategy", "failed", "budget_exhausted"}:
        phase = "completed"
        terminal = {
            "status": status,
            "selected_fingerprint": None,
            "evaluated_fingerprint": None,
            "training_metrics": {},
            "test_metrics": {},
            "conclusion": "The research run stopped safely without selecting a strategy.",
            "stop_reason": "The research run stopped safely.",
            "warnings": [],
            "limitations": [],
            "budget_usage": {
                "agent_steps": 0,
                "tool_calls": 0,
                "training_backtests": 0,
            },
        }
    return {
        "thread_id": str(uuid4()),
        "user_id": 1,
        "request": _goal().objective,
        "reference_date": "2026-07-13",
        "status": status,
        "failure_stage": None,
        "confirmation_error": None,
        "failure_summary": None,
        "cost_defaults": {
            "transaction_cost_bps": 5.0,
            "slippage_bps": 5.0,
        },
        "position_tranches_default": 5,
        "draft_goal": draft if status == "awaiting_confirmation" else None,
        "confirmed_goal": goal if confirmed else None,
        "missing_goal_fields": [],
        "defaulted_fields": ["start_date", "end_date"],
        "budget": AgentBudget().model_dump(mode="json"),
        "phase": phase,
        "active_agent": "research" if status == "running" else None,
        "critic_rounds": [],
        "agent_steps": 0,
        "tool_calls": 0,
        "training_backtests": 0,
        "download_failures": 0,
        "message_journal": [],
        "unresolved_issues": [],
        "runtime_warnings": [],
        "fatal_reason": None,
        "data_snapshot": None,
        "current_candidate": None,
        "validated_configs": {},
        "training_evaluations": [],
        "final_evaluations": [],
        "locked_fingerprint": None,
        "final_batch_completed": False,
        "terminal_result": terminal,
    }


def _report_dir(tmp_path: Path) -> Path:
    report_dir = tmp_path / str(uuid4())
    report_dir.mkdir()
    for name in ("report.json",):
        (report_dir / name).write_text("test", encoding="utf-8")
    return report_dir


def test_graph_state_round_trips_with_strict_msgpack_as_plain_json() -> None:
    script = textwrap.dedent(
        """
        from uuid import uuid4

        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph

        from trading.agent.models import AgentBudget, GoalDraft, ResearchGoal
        from trading.agent.state import GraphState, validate_graph_state

        goal = ResearchGoal(
            objective="Compare transparent long-only strategies.",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-30",
            allowed_strategies=("moving_average", "rsi", "macd"),
            max_drawdown_limit=0.2,
        ).model_dump(mode="json")
        draft = GoalDraft(
            symbol=goal["symbol"],
            bar_interval=goal["bar_interval"],
            start_date=goal["start_date"],
            end_date=goal["end_date"],
            allowed_strategies=goal["allowed_strategies"],
            max_drawdown_limit=goal["max_drawdown_limit"],
            position_tranches=goal["position_tranches"],
            transaction_cost_bps=goal["costs"]["transaction_cost_bps"],
            slippage_bps=goal["costs"]["slippage_bps"],
            benchmark_symbol=goal["benchmark_symbol"],
        ).model_dump(mode="json")
        state = {
            "thread_id": str(uuid4()),
            "user_id": 1,
            "request": "Compare transparent long-only strategies.",
            "reference_date": "2026-07-13",
            "status": "awaiting_confirmation",
            "failure_stage": None,
            "confirmation_error": None,
            "failure_summary": None,
            "cost_defaults": {"transaction_cost_bps": 5.0, "slippage_bps": 5.0},
            "position_tranches_default": 5,
            "draft_goal": draft,
            "confirmed_goal": None,
            "missing_goal_fields": [],
            "defaulted_fields": ["start_date", "end_date"],
            "budget": AgentBudget().model_dump(mode="json"),
            "phase": None,
            "active_agent": None,
            "critic_rounds": [],
            "agent_steps": 0,
            "tool_calls": 0,
            "training_backtests": 0,
            "download_failures": 0,
            "message_journal": [],
            "unresolved_issues": [],
            "runtime_warnings": [],
            "fatal_reason": None,
            "data_snapshot": None,
            "current_candidate": None,
            "validated_configs": {},
            "training_evaluations": [],
            "final_evaluations": [],
            "locked_fingerprint": None,
            "final_batch_completed": False,
            "terminal_result": None,
        }
        state = validate_graph_state(state)
        builder = StateGraph(GraphState)
        builder.add_node("validate", validate_graph_state)
        builder.add_edge(START, "validate")
        builder.add_edge("validate", END)
        graph = builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": state["thread_id"]}}
        output = graph.invoke(state, config, durability="sync")
        snapshot = graph.get_state(config)
        assert output == state
        assert snapshot.values == state

        invalid = dict(state)
        invalid["data_snapshot"] = {"huge_integer": 2**64}
        try:
            validate_graph_state(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("strict-msgpack-incompatible integer was accepted")
        """
    )
    environment = dict(os.environ)
    environment["LANGGRAPH_STRICT_MSGPACK"] = "true"
    environment["LANGSMITH_TRACING"] = "false"

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_validate_graph_state_returns_detached_plain_json() -> None:
    source = _state()

    validated = validate_graph_state(source)
    source["defaulted_fields"].append("benchmark_symbol")

    assert validated["defaulted_fields"] == ["start_date", "end_date"]


def test_missing_required_state_field_is_not_silently_defaulted() -> None:
    state = _state()
    state.pop("agent_steps")

    with pytest.raises(ValueError, match="Graph State failed validation"):
        validate_graph_state(state)


def test_journal_round_trip_restores_events_trace_and_prompt(tmp_path: Path) -> None:
    report_dir = _report_dir(tmp_path)
    workspace = ResearchWorkspace(
        goal=_goal(),
        budget=AgentBudget(),
        provider=NeverProvider(),
        output_dir=report_dir,
    )
    workspace.agent_steps = 1
    observation = ToolObservation(
        tool_name="submit_research_draft",
        status="success",
        summary="The frozen draft was submitted for Critic review.",
    )
    record_event(
        workspace,
        tool_name="submit_research_draft",
        normalized_arguments={"outcome": "no_strategy"},
        observation=observation,
        tool_call_id="call-1",
        replayable=True,
    )
    original_messages = build_research_messages(workspace)

    state = _state(status="running")
    state.update(workspace_to_state(workspace))
    _, rebuilt = restore_workspace_from_state(
        validate_graph_state(state),
        provider=NeverProvider(),
        output_dir=report_dir,
    )

    assert rebuilt.events[0].observation == observation
    assert rebuilt.trace == workspace.trace
    assert build_research_messages(rebuilt) == original_messages


def test_snapshot_round_trip_rebuilds_workspace_without_provider_or_counter_side_effects(
    tmp_path: Path,
) -> None:
    state = _state(status="running")
    original = _aligned_data()
    state["data_snapshot"] = aligned_data_to_snapshot(original)
    provider = NeverProvider()
    output_dir = tmp_path / "not-created-by-rehydrate"

    _, rebuilt = restore_workspace_from_state(state, provider=provider, output_dir=output_dir)

    assert rebuilt.aligned_data is not None
    pd.testing.assert_frame_equal(
        rebuilt.aligned_data.asset.bars,
        original.asset.bars,
        check_freq=False,
        check_names=False,
    )
    pd.testing.assert_frame_equal(
        rebuilt.aligned_data.benchmark.bars,
        original.benchmark.bars,
        check_freq=False,
        check_names=False,
    )
    assert rebuilt.aligned_data.asset.quality == original.asset.quality
    assert rebuilt.aligned_data.benchmark.quality == original.benchmark.quality
    assert rebuilt.aligned_data.warnings == original.warnings
    assert provider.calls == 0
    assert rebuilt.agent_steps == rebuilt.tool_calls == rebuilt.training_backtests == 0
    assert rebuilt.events == rebuilt.trace == []
    assert output_dir.exists() is False
