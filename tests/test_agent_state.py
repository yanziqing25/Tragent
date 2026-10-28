"""Plain-JSON checkpoint State, workspace journal, and market-data snapshot tests."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pandas as pd
import pytest

from tragent.quant import AlignedMarketData, DataQualityReport, MarketData
from tragent.research.agent.market_snapshot import load_market_data, save_market_data
from tragent.research.agent.models import (
    AgentBudget,
    GoalDraft,
    ResearchGoal,
    ToolObservation,
)
from tragent.research.agent.prompt import research_system_prompt
from tragent.research.agent.state import validate_graph_state
from tragent.research.agent.workspace import (
    ResearchGraphContext,
    ResearchWorkspace,
    record_tool_call,
)


class NeverProvider:
    def __init__(self) -> None:
        self.calls = 0

    def get_daily_bars(self, *_args: object, **_kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("State tests must not access market data")


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


def _state(*, workspace: ResearchWorkspace | None = None) -> dict[str, object]:
    return {
        "thread_id": str(uuid4()),
        "user_id": 1,
        "request": _goal().objective,
        "reference_date": "2026-07-13",
        "status": "awaiting_confirmation" if workspace is None else "running",
        "cost_defaults": {"transaction_cost_bps": 5.0, "slippage_bps": 5.0},
        "position_tranches_default": 5,
        "budget": AgentBudget().model_dump(mode="json"),
        "draft_goal": _draft() if workspace is None else None,
        "defaulted_fields": ["start_date", "end_date"],
        "workspace": None if workspace is None else workspace.model_dump(mode="json"),
    }


def test_graph_state_round_trips_with_strict_msgpack_as_plain_json() -> None:
    script = textwrap.dedent(
        """
        from uuid import uuid4

        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph

        from tragent.research.agent.models import AgentBudget, ResearchGoal
        from tragent.research.agent.state import GraphState, validate_graph_state
        from tragent.research.agent.workspace import ResearchWorkspace

        goal = ResearchGoal(
            objective="Compare transparent long-only strategies.",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-30",
            allowed_strategies=("moving_average", "rsi", "macd"),
            max_drawdown_limit=0.2,
        )
        state = validate_graph_state(
            {
                "thread_id": str(uuid4()),
                "user_id": 1,
                "request": "Compare transparent long-only strategies.",
                "reference_date": "2026-07-13",
                "status": "running",
                "cost_defaults": {"transaction_cost_bps": 5.0, "slippage_bps": 5.0},
                "position_tranches_default": 5,
                "budget": AgentBudget().model_dump(mode="json"),
                "workspace": ResearchWorkspace(goal=goal, budget=AgentBudget()).model_dump(
                    mode="json"
                ),
            }
        )
        builder = StateGraph(GraphState)
        builder.add_node(
            "validate",
            lambda current: validate_graph_state(current).model_dump(mode="json"),
        )
        builder.add_edge(START, "validate")
        builder.add_edge("validate", END)
        graph = builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": state.thread_id}}
        expected = state.model_dump(mode="json")
        assert graph.invoke(expected, config, durability="sync") == expected
        assert graph.get_state(config).values == expected
        """
    )
    environment = dict(os.environ)
    environment["LANGGRAPH_STRICT_MSGPACK"] = "true"
    environment["LANGSMITH_TRACING"] = "false"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

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

    assert validated.defaulted_fields == ("start_date", "end_date")


def test_missing_required_state_field_is_not_silently_defaulted() -> None:
    state = _state()
    state.pop("budget")

    with pytest.raises(ValueError, match="Graph State failed validation"):
        validate_graph_state(state)


def test_journal_round_trip_rebuilds_the_identical_model_context() -> None:
    context = ResearchGraphContext(
        model=MagicMock(), provider=NeverProvider(), artifacts="research/t"
    )
    workspace = ResearchWorkspace(goal=_goal(), budget=AgentBudget(), agent_steps=1).bind(context)
    observation = ToolObservation(
        tool_name="submit_research_draft",
        status="success",
        summary="The frozen draft was submitted for Critic review.",
    )
    record_tool_call(
        workspace.journal,
        step=1,
        tool_name="submit_research_draft",
        arguments={"outcome": "no_strategy"},
        observation=observation,
    )

    rebuilt = validate_graph_state(_state(workspace=workspace)).workspace

    assert rebuilt is not None
    assert rebuilt.journal == workspace.journal
    assert research_system_prompt(rebuilt) == research_system_prompt(workspace)


def test_market_data_snapshot_round_trips_without_provider_access() -> None:
    original = _aligned_data()

    save_market_data("research/t", original)
    rebuilt = load_market_data("research/t")

    for name in ("asset", "benchmark"):
        pd.testing.assert_frame_equal(
            getattr(rebuilt, name).bars,
            getattr(original, name).bars,
            check_freq=False,
            check_names=False,
        )
        assert getattr(rebuilt, name).quality == getattr(original, name).quality
    assert rebuilt.warnings == original.warnings
