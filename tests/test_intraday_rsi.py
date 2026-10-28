"""Execution, causality and application contracts for the fixed intraday policies."""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tragent.quant import strategies
from tragent.quant.backtest import run_final_evaluation, run_training_evaluation
from tragent.quant.models import (
    INTRADAY_RSI_STRATEGIES,
    AlignedMarketData,
    BacktestCosts,
    DataQualityReport,
    IntradayRsiConfig,
    MarketData,
    build_strategy_config,
    strategy_name,
    strategy_parameters,
)
from tragent.research.agent.models import AgentBudget, GoalDraft, ResearchGoal
from tragent.research.agent.runtime import get_market_data, run_backtest
from tragent.research.agent.workspace import ResearchWorkspace, record_tool_call
from tragent.research.reporting import ReportInput, generate_report, load_report_view


def session(day="2024-01-02", *, oscillating=False):
    index = pd.date_range(f"{day} 14:30:00+00:00", periods=390, freq="1min")
    close = 100 + 3 * np.sin(np.arange(390) / 7) if oscillating else np.full(390, 100.0)
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1000.0},
        index=index,
    )


def aligned():
    days = ("2024-01-02", "2024-01-03", "2024-01-04")
    bars = pd.concat([session(day, oscillating=True) for day in days])
    labels = tuple(day for day in days for _ in range(390))
    quality = DataQualityReport("AAPL", len(bars), bars.index[0], bars.index[-1])
    asset = MarketData("AAPL", bars, quality, bar_interval="1m", session_labels=labels)
    return AlignedMarketData(
        asset, None, bar_interval="1m", session_labels=labels, training_sessions=2, test_sessions=1
    )


class MinuteProvider:
    def get_minute_bars(self, *_args):
        return aligned().asset


_ARTIFACTS = "research/intraday"


def minute_workspace(goal, artifacts):
    from test_agent_state_rehydrate import bound

    workspace = bound(
        ResearchWorkspace(goal=goal, budget=AgentBudget()), MinuteProvider(), artifacts
    )
    observation = get_market_data(workspace)
    assert observation.status == "success"
    record_tool_call(
        workspace.journal,
        step=1,
        tool_name="get_market_data",
        arguments={},
        observation=observation,
    )
    return workspace


def replayed_finals(workspace):
    return {
        candidate.fingerprint: run_final_evaluation(
            workspace.aligned_data, candidate.config, costs=workspace.goal.costs
        )
        for candidate in workspace.candidates
    }


def mock_rsi(monkeypatch, values):
    def calculate(close, _config):
        result = pd.Series(50.0, index=close.index)
        for index, value in values.items():
            result.iloc[index] = value
        return result

    monkeypatch.setattr(strategies, "calculate_rsi", calculate)


def session_targets(bars, config):
    return strategies.intraday_rsi_targets(bars, ("session",) * len(bars), config)


@pytest.mark.parametrize("policy", INTRADAY_RSI_STRATEGIES)
def test_configs_are_fixed_and_round_trip(policy):
    config = build_strategy_config(policy, {})
    assert config == IntradayRsiConfig(policy)
    assert strategy_name(config) == policy
    assert strategy_parameters(config) == {}
    with pytest.raises(ValueError, match="no parameters"):
        build_strategy_config(policy, {"rsi_window": 7})


def test_exit_type_is_locked_at_entry_and_below_filter_skips_above(monkeypatch):
    mock_rsi(monkeypatch, {59: 20, 75: 80})
    below = session()
    below.iloc[59, below.columns.get_loc("close")] = 99
    for policy in ("rsi_price_exit", "rsi_vwap_exit"):
        targets = session_targets(below, IntradayRsiConfig(policy))
        assert targets.iloc[59] == 1
        assert targets.iloc[60] == 0  # recovers to VWAP, next-open sell
    above = session()
    above.iloc[59:, above.columns.get_loc("close")] = 101
    price = session_targets(above, IntradayRsiConfig("rsi_price_exit"))
    baseline = session_targets(above, IntradayRsiConfig("intraday_rsi"))
    pd.testing.assert_series_equal(price, baseline)
    assert price.iloc[59:75].eq(1).all()  # being above VWAP cannot override RSI exit
    assert price.iloc[75] == 0
    assert not session_targets(above, IntradayRsiConfig("rsi_vwap_exit")).any()


def test_entry_clock_cooldown_and_final_minute_liquidation(monkeypatch):
    mock_rsi(monkeypatch, {58: 20, 60: 20, 61: 80, 63: 20, 66: 20})
    targets = session_targets(session(), IntradayRsiConfig("intraday_rsi"))
    assert not targets.iloc[:60].any()  # early crossing does not create a delayed entry
    assert targets.iloc[60] == 1  # buy on minute 62 open
    assert targets.iloc[61:66].eq(0).all()  # sell at 62, cooldown rejects signal 63
    assert targets.iloc[66:389].eq(1).all()  # buy at 67; the engine flattens the session
    mock_rsi(monkeypatch, {359: 20})
    assert not session_targets(session(), IntradayRsiConfig("intraday_rsi")).any()


def test_undefined_vwap_is_not_below_vwap(monkeypatch):
    mock_rsi(monkeypatch, {59: 20, 75: 80})
    bars = session().assign(volume=0.0)
    assert not session_targets(bars, IntradayRsiConfig("rsi_vwap_exit")).any()
    pd.testing.assert_series_equal(
        session_targets(bars, IntradayRsiConfig("rsi_price_exit")),
        session_targets(bars, IntradayRsiConfig("intraday_rsi")),
    )


@pytest.mark.parametrize("policy", INTRADAY_RSI_STRATEGIES)
def test_future_prices_and_volumes_do_not_change_past_targets(policy):
    bars = session(oscillating=True)
    changed = bars.copy()
    changed.iloc[200:, :4] *= 1.8
    changed.iloc[200:, 4] *= 20
    config = IntradayRsiConfig(policy)
    pd.testing.assert_series_equal(
        session_targets(bars, config).iloc[:200],
        session_targets(changed, config).iloc[:200],
    )


@pytest.mark.parametrize("policy", INTRADAY_RSI_STRATEGIES)
def test_daily_reset_next_open_ledger_exposure_and_costs(policy):
    data, config = aligned(), IntradayRsiConfig(policy)
    result = run_final_evaluation(data, config, costs=BacktestCosts(0, 1), position_tranches=5)
    assert result.training.metrics.trade_count > 0
    for segment in (result.training, result.test):
        assert segment.metrics.annualized_return is None
        assert segment.positions.max() == pytest.approx(0.2)
        bars = data.asset.bars.loc[segment.equity_curve.index]
        cash, shares, exposure = 10000.0, 0.0, []
        trades = {trade.date: trade for trade in segment.trades}
        for timestamp, row in bars.iterrows():
            if timestamp in trades:
                trade = trades[timestamp]
                assert trade.price == row.open
                if trade.action == "buy":
                    assert shares == 0  # no pyramiding
                    assert trade.net_cash == -2000
                    assert trade.trading_cost == pytest.approx(0.2)
                    shares = 1999.8 / row.open
                else:
                    assert trade.quantity == pytest.approx(shares)
                    assert trade.trading_cost == pytest.approx(shares * row.open * 0.0001)
                    shares = 0
                cash += trade.net_cash
            equity = cash + shares * row.close
            assert segment.equity_curve.loc[timestamp] == pytest.approx(equity / 10000)
            exposure.append(shares * row.close / equity)
        assert segment.mean_exposure == pytest.approx(np.mean(exposure))
        for _, positions in segment.positions.groupby(segment.positions.index.date):
            assert positions.iloc[:60].eq(0).all()
            assert positions.iloc[-1] == 0
        assert shares == 0
    # Identical days start fresh despite earlier bars and a train/test boundary.
    train = result.training.positions
    np.testing.assert_array_equal(train.iloc[:390], train.iloc[390:])
    np.testing.assert_array_equal(train.iloc[:390], result.test.positions)
    higher_cost = run_final_evaluation(data, config, costs=BacktestCosts(0, 2))
    assert higher_cost.test.metrics.total_return < result.test.metrics.total_return
    changed = data.asset.bars.copy()
    changed.iloc[780:, :4] *= 2
    modified = replace(data, asset=replace(data.asset, bars=changed))
    pd.testing.assert_series_equal(
        result.training.equity_curve,
        run_training_evaluation(modified, config, costs=BacktestCosts(0, 1)).training.equity_curve,
    )


def test_session_end_sell_uses_open_even_when_close_changes(monkeypatch):
    mock_rsi(monkeypatch, {59: 20})
    data = aligned()
    data.asset.bars.iloc[389::390, data.asset.bars.columns.get_loc("close")] = 70
    result = run_final_evaluation(data, IntradayRsiConfig("intraday_rsi"))
    buy, sell = result.test.trades
    assert buy.date == data.asset.bars.index[840]
    assert sell.date == data.asset.bars.index[-1]
    assert sell.price == data.asset.bars.open.iloc[-1]
    assert sell.price != 70


def test_minute_goals_add_matched_baseline_and_daily_goals_reject():
    values = dict(
        objective="Compare RSI exits.",
        symbol="AAPL",
        allowed_strategies=("rsi_price_exit", "rsi_vwap_exit"),
    )
    goal = ResearchGoal(
        **values,
        bar_interval="1m",
        training_sessions=2,
        test_sessions=1,
        end_date=date(2024, 1, 4),
        benchmark_symbol=None,
    )
    assert goal.allowed_strategies == ("rsi_price_exit", "rsi_vwap_exit", "intraday_rsi")
    with pytest.raises(ValueError, match="at most 20 sessions"):
        goal.model_validate({**goal.model_dump(), "training_sessions": 15, "test_sessions": 6})
    assert (
        len(
            GoalDraft(
                allowed_strategies=["moving_average", "rsi", "macd", *INTRADAY_RSI_STRATEGIES]
            ).allowed_strategies
        )
        == 6
    )
    with pytest.raises(ValueError, match="require 1m"):
        ResearchGoal(**values, start_date=date(2024, 1, 1), end_date=date(2024, 2, 1))


def test_agent_baseline_freeze_checkpoint_and_report():
    from test_agent_state_rehydrate import NeverProvider, _graph_state, _restored

    goal = ResearchGoal(
        objective="Compare RSI exits.",
        symbol="AAPL",
        bar_interval="1m",
        training_sessions=2,
        test_sessions=1,
        end_date=date(2024, 1, 4),
        benchmark_symbol=None,
        allowed_strategies=("rsi_price_exit", "rsi_vwap_exit"),
        costs=BacktestCosts(0, 1),
    )
    workspace = minute_workspace(goal, _ARTIFACTS)
    rejected = run_backtest(
        workspace, stage="training", strategy_name="rsi_price_exit", parameters={}
    )
    assert rejected.status == "rejected"
    assert "intraday_rsi" in rejected.summary
    for policy in INTRADAY_RSI_STRATEGIES:
        observation = run_backtest(workspace, stage="training", strategy_name=policy, parameters={})
        assert observation.status == "success"
    selected = observation.identifiers["fingerprint"]
    final = run_backtest(workspace, stage="final", selected_fingerprint=selected)
    assert final.status == "success"
    assert sum(candidate.test is not None for candidate in workspace.candidates) == 3
    assert (
        run_backtest(
            workspace, stage="training", strategy_name="intraday_rsi", parameters={}
        ).status
        == "rejected"
    )
    provider = NeverProvider()
    restored = _restored(_graph_state(workspace), provider, _ARTIFACTS)
    assert provider.calls == 0
    assert restored.locked_fingerprint == selected
    finals, original = replayed_finals(restored), replayed_finals(workspace)
    for fingerprint, evaluation in original.items():
        assert finals[fingerprint].test.trades == evaluation.test.trades
        pd.testing.assert_series_equal(
            finals[fingerprint].test.equity_curve, evaluation.test.equity_curve
        )
    path = generate_report(
        ReportInput(
            goal.objective,
            "AAPL",
            None,
            finals,
            _ARTIFACTS,
            bar_interval="1m",
            selected_fingerprint=selected,
        )
    )
    report = load_report_view(path)
    assert {candidate.strategy_name for candidate in report.candidates} == set(
        INTRADAY_RSI_STRATEGIES
    )
    for candidate in report.candidates:
        assert candidate.parameters == {}
        assert candidate.slippage_bps == 1
        for segment in candidate.segments:
            assert 0 < segment.mean_exposure < 0.25


@pytest.mark.parametrize("interval", ["1m", "1d"])
def test_goal_parser_preserves_fixed_rules_and_clarifies_daily_conflict(interval):
    from test_agent_graph import GoalParserFakeChatModel, _nullable_goal_response, _task

    from tragent.research.agent.graph import parse_goal

    fields = (
        {"training_sessions": 2, "test_sessions": 1}
        if interval == "1m"
        else {"start_date": "2024-01-01"}
    )
    model = GoalParserFakeChatModel(
        _nullable_goal_response(
            symbol="AAPL",
            bar_interval=interval,
            end_date="2024-01-04",
            allowed_strategies=["rsi_price_exit", "rsi_vwap_exit"],
            **fields,
        )
    )
    state = parse_goal(_task(request="Compare the two fixed RSI exit policies."), model=model)
    assert state.draft_goal.allowed_strategies == [
        "rsi_price_exit",
        "rsi_vwap_exit",
        "intraday_rsi",
    ]
    if interval == "1m":
        assert state.status == "awaiting_confirmation"
        assert state.missing_goal_fields == ()
    else:
        assert state.status == "awaiting_clarification"
        assert "bar_interval" in state.missing_goal_fields


@pytest.mark.parametrize("initial_interval", [None, "1d", "1m"])
@pytest.mark.parametrize("explicit", [False, True])
def test_strategy_defaults_follow_clarified_interval_and_preserve_explicit_scope(
    initial_interval, explicit
):
    from test_agent_graph import GoalParserFakeChatModel, _nullable_goal_response, _task

    from tragent.research.agent.goal import apply_goal_answer, parse_goal

    original = ["rsi", "macd"] if explicit else None
    state = parse_goal(
        _task(),
        model=GoalParserFakeChatModel(
            _nullable_goal_response(
                symbol="AAPL", bar_interval=initial_interval, allowed_strategies=original
            )
        ),
    )
    minute = apply_goal_answer(
        state,
        "Use minute bars, training 2 sessions and testing 1 session.",
        model=GoalParserFakeChatModel(
            {"bar_interval": "1m", "training_sessions": 2, "test_sessions": 1}
        ),
    )
    assert minute.status == "awaiting_confirmation"
    assert minute.draft_goal.allowed_strategies == (
        original or ["moving_average", "rsi", "macd", *INTRADAY_RSI_STRATEGIES]
    )
    daily = apply_goal_answer(
        minute,
        "Use daily bars from 2024-01-01 through 2024-01-04.",
        model=GoalParserFakeChatModel(
            {"bar_interval": "1d", "start_date": "2024-01-01", "end_date": "2024-01-04"}
        ),
    )
    assert daily.status == "awaiting_confirmation"
    assert daily.draft_goal.allowed_strategies == (original or ["moving_average", "rsi", "macd"])
    assert ("allowed_strategies" in daily.defaulted_fields) is not explicit


def test_default_minute_goal_trains_and_reports_all_six_within_existing_budget():
    from test_agent_graph import GoalParserFakeChatModel, _nullable_goal_response, _task

    from tragent.research.agent.goal import materialize_goal, parse_goal

    task = _task(slippage_bps=1.0, transaction_cost_bps=0.0)
    state = parse_goal(
        task,
        model=GoalParserFakeChatModel(
            _nullable_goal_response(
                symbol="AAPL",
                bar_interval="1m",
                training_sessions=2,
                test_sessions=1,
                end_date="2024-01-04",
            )
        ),
    )
    goal = materialize_goal(task, state.draft_goal)
    assert goal.allowed_strategies == ("moving_average", "rsi", "macd", *INTRADAY_RSI_STRATEGIES)
    assert "allowed_strategies" in state.defaulted_fields
    workspace = minute_workspace(goal, _ARTIFACTS)
    params = {
        "moving_average": {"short_window": 10, "long_window": 30},
        "rsi": {"rsi_window": 14, "entry_threshold": 30, "exit_threshold": 70},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    }
    for name in goal.allowed_strategies:
        observation = run_backtest(
            workspace, stage="training", strategy_name=name, parameters=params.get(name, {})
        )
        assert observation.status == "success", observation.summary
    assert workspace.training_backtests == 6
    assert workspace.training_backtests < workspace.budget.max_training_backtests
    selected = observation.identifiers["fingerprint"]
    assert run_backtest(workspace, stage="final", selected_fingerprint=selected).status == "success"
    path = generate_report(
        ReportInput(
            goal.objective,
            "AAPL",
            None,
            replayed_finals(workspace),
            _ARTIFACTS,
            bar_interval="1m",
            selected_fingerprint=selected,
        )
    )
    report = load_report_view(path)
    assert {candidate.strategy_name for candidate in report.candidates} == set(
        goal.allowed_strategies
    )
