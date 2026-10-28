"""Tests for the compact, JSON-facing Research Agent contracts."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from tragent.quant import BacktestCosts
from tragent.research.agent.models import (
    AgentBudget,
    BudgetUsage,
    ResearchGoal,
    ResearchResult,
    ToolObservation,
)

FINGERPRINT = "a" * 64
OTHER_FINGERPRINT = "b" * 64


def goal(**changes: object) -> ResearchGoal:
    values: dict[str, object] = {
        "objective": "Find a long-only strategy with controlled drawdown.",
        "symbol": " aapl ",
        "start_date": date(2020, 1, 1),
        "end_date": date(2024, 12, 31),
        "allowed_strategies": ("moving_average", "rsi", "macd"),
        "max_drawdown_limit": 0.20,
    }
    values.update(changes)
    return ResearchGoal(**values)


def metrics(**changes: int | float | None) -> dict[str, int | float | None]:
    values: dict[str, int | float | None] = {
        "total_return": 0.12,
        "annualized_return": 0.08,
        "annualized_volatility": 0.15,
        "sharpe_ratio": 0.53,
        "max_drawdown": -0.10,
        "trade_count": 4,
        "turnover": 7.0,
        "asset_buy_and_hold_return": 0.09,
        "benchmark_return": 0.07,
    }
    values.update(changes)
    return values


def observation() -> ToolObservation:
    return ToolObservation(
        tool_name="run_backtest",
        status="success",
        summary="Training evaluation completed.",
        identifiers={"fingerprint": FINGERPRINT, "segment": "training"},
        metrics={"total_return": 0.12, "trade_count": 4},
    )


def selected_result(**changes: object) -> ResearchResult:
    values: dict[str, object] = {
        "status": "selected",
        "selected_fingerprint": FINGERPRINT,
        "evaluated_fingerprint": FINGERPRINT,
        "training_metrics": metrics(),
        "test_metrics": metrics(total_return=0.06),
        "conclusion": "The frozen candidate met the drawdown constraint.",
        "limitations": ("Daily adjusted data only.",),
        "budget_usage": BudgetUsage(agent_steps=5, tool_calls=5, training_backtests=1),
    }
    values.update(changes)
    return ResearchResult(**values)


def test_research_goal_normalizes_and_deduplicates_supported_strategies() -> None:
    research_goal = goal(
        objective="  Compare transparent strategies.  ",
        allowed_strategies=("rsi", "moving_average", "rsi"),
    )

    assert research_goal.objective == "Compare transparent strategies."
    assert research_goal.symbol == "AAPL"
    assert research_goal.benchmark_symbol == "SPY"
    assert goal(benchmark_symbol=" qqq ").benchmark_symbol == "QQQ"
    assert research_goal.allowed_strategies == ("rsi", "moving_average")
    assert research_goal.costs == BacktestCosts(5.0, 5.0)
    assert goal(max_drawdown_limit=None).max_drawdown_limit is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"objective": " "}, "objective"),
        ({"start_date": date(2025, 1, 1)}, "start_date"),
    ],
)
def test_research_goal_rejects_invalid_inputs(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        goal(**changes)


def test_research_goal_accepts_explicit_costs() -> None:
    research_goal = goal(costs=BacktestCosts(2.0, 3.0))

    assert research_goal.costs.total_bps == 5.0


def test_agent_budget_defaults_are_the_hard_caps() -> None:
    budget = AgentBudget()

    assert budget.max_agent_steps == 40
    assert budget.max_tool_calls == 30
    assert budget.max_training_backtests == 11


@pytest.mark.parametrize(
    "values",
    [
        {"max_agent_steps": 41},
        {"max_tool_calls": True},
    ],
)
def test_agent_budget_rejects_invalid_or_above_cap_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AgentBudget(**values)


def test_tool_observation_is_compact_json_and_truncates_optional_text() -> None:
    compact = ToolObservation(
        tool_name="get_market_data",
        status="success",
        summary="s" * 900,
        identifiers={"segment": "training"},
        metrics={"valid_rows": 500},
        warnings=tuple(["w" * 350] + [f"warning-{index}" for index in range(11)]),
    )

    encoded = compact.to_compact_json()
    payload = json.loads(encoded)

    assert payload["tool_name"] == "get_market_data"
    assert len(compact.summary) <= 800
    assert len(compact.warnings) <= 10
    assert all(len(item) <= 300 for item in compact.warnings)
    assert any("truncated" in item.lower() for item in compact.warnings)
    assert len(encoded) <= 6_000


@pytest.mark.parametrize(
    "changes",
    [
        {"identifiers": {str(index): str(index) for index in range(5)}},
        {"metrics": {"frame": pd.DataFrame({"x": [1]})}},
    ],
)
def test_tool_observation_rejects_invalid_mandatory_content(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "tool_name": "run_backtest",
        "status": "success",
        "summary": "Completed.",
    }
    values.update(changes)

    with pytest.raises(ValidationError):
        ToolObservation(**values)


def test_selected_research_result_keeps_trusted_metrics_and_audit_fingerprint() -> None:
    result = selected_result()

    assert result.selected_fingerprint == FINGERPRINT
    assert result.evaluated_fingerprint == FINGERPRINT
    assert result.training_metrics["asset_buy_and_hold_return"] == pytest.approx(0.09)
    assert result.test_metrics["benchmark_return"] == pytest.approx(0.07)


def test_no_strategy_can_retain_the_locked_fingerprint_for_audit() -> None:
    result = ResearchResult(
        status="no_strategy",
        evaluated_fingerprint=FINGERPRINT,
        training_metrics=metrics(),
        test_metrics=metrics(max_drawdown=-0.30),
        conclusion="The frozen candidate failed the test drawdown constraint.",
        stop_reason="No frozen candidate passed the out-of-sample check.",
        budget_usage=BudgetUsage(agent_steps=5, tool_calls=5, training_backtests=1),
    )

    assert result.selected_fingerprint is None
    assert result.evaluated_fingerprint == FINGERPRINT


def test_no_strategy_without_evaluated_candidate_cannot_carry_candidate_metrics() -> None:
    common: dict[str, object] = {
        "status": "no_strategy",
        "conclusion": "No candidate was selected for final evaluation.",
        "stop_reason": "The final batch completed with an explicit no-selection lock.",
    }

    with pytest.raises(ValidationError, match="without an evaluated fingerprint"):
        ResearchResult(training_metrics=metrics(), **common)
    with pytest.raises(ValidationError, match="without an evaluated fingerprint"):
        ResearchResult(test_metrics=metrics(), **common)


@pytest.mark.parametrize(
    "changes",
    [
        {"selected_fingerprint": None},
        {"evaluated_fingerprint": OTHER_FINGERPRINT},
        {"training_metrics": {}},
        {"stop_reason": "Selected results must not have a stop reason."},
    ],
)
def test_selected_research_result_rejects_inconsistent_fields(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        selected_result(**changes)


def test_non_selected_results_require_a_stop_reason_and_never_select() -> None:
    common: dict[str, object] = {
        "conclusion": "The run stopped safely.",
        "budget_usage": BudgetUsage(),
    }

    for status in ("no_strategy", "failed", "budget_exhausted"):
        with pytest.raises(ValidationError):
            ResearchResult(status=status, **common)
        with pytest.raises(ValidationError):
            ResearchResult(
                status=status,
                selected_fingerprint=FINGERPRINT,
                stop_reason="Stopped.",
                **common,
            )
