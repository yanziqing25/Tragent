"""Small immutable data contracts shared by the quantitative modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isclose, isfinite
from numbers import Integral, Real
from typing import Literal

from pandas import DataFrame, Series, Timestamp, isna

TRADE_DISPLAY_INITIAL_CAPITAL = 10_000.0


def _integer_parameter(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


@dataclass(frozen=True, slots=True)
class MovingAverageConfig:
    """Parameters for a long-only moving-average crossover strategy."""

    short_window: int
    long_window: int

    def __post_init__(self) -> None:
        short_window = _integer_parameter("short_window", self.short_window, minimum=2)
        long_window = _integer_parameter("long_window", self.long_window, minimum=2)
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        object.__setattr__(self, "short_window", short_window)
        object.__setattr__(self, "long_window", long_window)


@dataclass(frozen=True, slots=True)
class RsiConfig:
    """Parameters for the long-only Wilder RSI state strategy."""

    rsi_window: int
    entry_threshold: float
    exit_threshold: float

    def __post_init__(self) -> None:
        rsi_window = _integer_parameter("rsi_window", self.rsi_window, minimum=2)
        entry_threshold = _finite_number("entry_threshold", self.entry_threshold)
        exit_threshold = _finite_number("exit_threshold", self.exit_threshold)
        if not 0 < entry_threshold < exit_threshold < 100:
            raise ValueError(
                "RSI thresholds must satisfy 0 < entry_threshold < exit_threshold < 100"
            )
        object.__setattr__(self, "rsi_window", rsi_window)
        object.__setattr__(self, "entry_threshold", entry_threshold)
        object.__setattr__(self, "exit_threshold", exit_threshold)


@dataclass(frozen=True, slots=True)
class MacdConfig:
    """Parameters for a long-only MACD crossover strategy."""

    fast_period: int
    slow_period: int
    signal_period: int

    def __post_init__(self) -> None:
        fast_period = _integer_parameter("fast_period", self.fast_period, minimum=2)
        slow_period = _integer_parameter("slow_period", self.slow_period, minimum=2)
        signal_period = _integer_parameter("signal_period", self.signal_period, minimum=2)
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        object.__setattr__(self, "fast_period", fast_period)
        object.__setattr__(self, "slow_period", slow_period)
        object.__setattr__(self, "signal_period", signal_period)


type StrategyConfig = MovingAverageConfig | RsiConfig | MacdConfig
type StrategyName = Literal["moving_average", "rsi", "macd"]


def strategy_name(config: StrategyConfig) -> StrategyName:
    """Return the stable external name for a supported strategy configuration."""

    if isinstance(config, MovingAverageConfig):
        return "moving_average"
    if isinstance(config, RsiConfig):
        return "rsi"
    if isinstance(config, MacdConfig):
        return "macd"
    raise TypeError(f"unsupported strategy config: {type(config).__name__}")


def strategy_parameters(config: StrategyConfig) -> dict[str, int | float]:
    """Return primitive parameters suitable for reports and Agent observations."""

    if isinstance(config, MovingAverageConfig):
        return {"short_window": config.short_window, "long_window": config.long_window}
    if isinstance(config, RsiConfig):
        return {
            "rsi_window": config.rsi_window,
            "entry_threshold": config.entry_threshold,
            "exit_threshold": config.exit_threshold,
        }
    if isinstance(config, MacdConfig):
        return {
            "fast_period": config.fast_period,
            "slow_period": config.slow_period,
            "signal_period": config.signal_period,
        }
    raise TypeError(f"unsupported strategy config: {type(config).__name__}")


def build_strategy_config(
    requested_strategy: str,
    parameters: Mapping[str, object],
) -> StrategyConfig:
    """Construct one supported strategy config from its external name and parameters."""

    expected_parameters = {
        "moving_average": frozenset({"short_window", "long_window"}),
        "rsi": frozenset({"rsi_window", "entry_threshold", "exit_threshold"}),
        "macd": frozenset({"fast_period", "slow_period", "signal_period"}),
    }
    expected = expected_parameters.get(requested_strategy)
    if expected is None:
        raise ValueError("unsupported strategy")
    if not isinstance(parameters, Mapping) or frozenset(parameters) != expected:
        raise ValueError("strategy parameters must exactly match the supported parameter names")
    values = dict(parameters)
    if requested_strategy == "moving_average":
        return MovingAverageConfig(
            short_window=values["short_window"],  # type: ignore[arg-type]
            long_window=values["long_window"],  # type: ignore[arg-type]
        )
    if requested_strategy == "rsi":
        return RsiConfig(
            rsi_window=values["rsi_window"],  # type: ignore[arg-type]
            entry_threshold=values["entry_threshold"],  # type: ignore[arg-type]
            exit_threshold=values["exit_threshold"],  # type: ignore[arg-type]
        )
    return MacdConfig(
        fast_period=values["fast_period"],  # type: ignore[arg-type]
        slow_period=values["slow_period"],  # type: ignore[arg-type]
        signal_period=values["signal_period"],  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class BacktestCosts:
    """Per-turnover transaction cost and slippage expressed in basis points."""

    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        transaction_cost_bps = _finite_number("transaction_cost_bps", self.transaction_cost_bps)
        slippage_bps = _finite_number("slippage_bps", self.slippage_bps)
        if transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if transaction_cost_bps + slippage_bps >= 10_000:
            raise ValueError("transaction_cost_bps and slippage_bps must total less than 10000")
        object.__setattr__(self, "transaction_cost_bps", transaction_cost_bps)
        object.__setattr__(self, "slippage_bps", slippage_bps)

    @property
    def total_bps(self) -> float:
        return self.transaction_cost_bps + self.slippage_bps

    @property
    def cost_rate(self) -> float:
        return self.total_bps / 10_000


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Structured summary of market-data normalization."""

    symbol: str
    valid_rows: int
    start_date: Timestamp | None
    end_date: Timestamp | None
    duplicate_rows_removed: int = 0
    missing_value_counts: Mapping[str, int] = field(default_factory=dict)
    invalid_price_rows_removed: int = 0
    warnings: tuple[str, ...] = ()
    is_valid: bool = True


@dataclass(frozen=True, slots=True)
class MarketData:
    """Normalized daily or minute bars and their quality summary."""

    symbol: str
    bars: DataFrame
    quality: DataQualityReport
    bar_interval: Literal["1d", "1m"] = "1d"
    session_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AlignedMarketData:
    """One research-ready asset series with an optional aligned benchmark."""

    asset: MarketData
    benchmark: MarketData | None
    warnings: tuple[str, ...] = ()
    bar_interval: Literal["1d", "1m"] = "1d"
    session_labels: tuple[str, ...] = ()
    training_sessions: int | None = None
    test_sessions: int | None = None


@dataclass(frozen=True, slots=True)
class AssetStatistics:
    """Deterministic descriptive statistics for one normalized asset."""

    symbol: str
    observations: int
    start_date: Timestamp
    end_date: Timestamp
    total_return: float
    annualized_return: float | None
    annualized_volatility: float | None
    max_drawdown: float


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Metrics for one independently funded backtest segment."""

    total_return: float
    annualized_return: float | None
    annualized_volatility: float | None
    sharpe_ratio: float | None
    max_drawdown: float
    trade_count: int
    turnover: float
    asset_buy_and_hold_return: float
    benchmark_return: float | None


@dataclass(frozen=True, slots=True)
class DrawdownAnalysis:
    """Dates and duration of the maximum drawdown episode."""

    max_drawdown: float
    peak_date: Timestamp
    trough_date: Timestamp
    recovery_date: Timestamp | None
    underwater_bars: int


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """One deterministic Buy or Sell transaction and its post-trade balances."""

    date: Timestamp
    action: Literal["buy", "sell"]
    price: float
    quantity: float
    gross_value: float
    trading_cost: float
    net_cash: float
    realized_profit: float | None
    average_cost_after: float | None
    position_quantity_after: float
    cash_balance_after: float

    def __post_init__(self) -> None:
        date = Timestamp(self.date)
        if isna(date):
            raise ValueError("date must be a valid timestamp")
        if self.action not in {"buy", "sell"}:
            raise ValueError("action must be buy or sell")
        price = _finite_number("price", self.price)
        quantity = _finite_number("quantity", self.quantity)
        gross_value = _finite_number("gross_value", self.gross_value)
        trading_cost = _finite_number("trading_cost", self.trading_cost)
        net_cash = _finite_number("net_cash", self.net_cash)
        position_quantity_after = _finite_number(
            "position_quantity_after", self.position_quantity_after
        )
        cash_balance_after = _finite_number("cash_balance_after", self.cash_balance_after)
        if price <= 0 or quantity <= 0 or gross_value <= 0 or trading_cost < 0:
            raise ValueError("transaction values must be positive with non-negative cost")
        if position_quantity_after < 0 or cash_balance_after < 0:
            raise ValueError("post-trade balances must be non-negative")
        if not isclose(gross_value, price * quantity, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("gross_value must equal price times quantity")

        expected_net_cash = (
            -(gross_value + trading_cost) if self.action == "buy" else gross_value - trading_cost
        )
        if not isclose(net_cash, expected_net_cash, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("net_cash is inconsistent with transaction values")
        if self.action == "buy":
            if self.realized_profit is not None:
                raise ValueError("Buy events must not contain realized profit")
        elif self.realized_profit is None:
            raise ValueError("Sell events require realized profit")
        else:
            object.__setattr__(
                self,
                "realized_profit",
                _finite_number("realized_profit", self.realized_profit),
            )

        average_cost_after = self.average_cost_after
        if position_quantity_after == 0:
            if average_cost_after is not None:
                raise ValueError("flat positions must not retain an average cost")
        elif average_cost_after is None:
            raise ValueError("open positions require an average cost")
        else:
            average_cost_after = _finite_number("average_cost_after", average_cost_after)
            if average_cost_after <= 0:
                raise ValueError("average_cost_after must be positive")

        object.__setattr__(self, "date", date)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "gross_value", gross_value)
        object.__setattr__(self, "trading_cost", trading_cost)
        object.__setattr__(self, "net_cash", net_cash)
        object.__setattr__(self, "average_cost_after", average_cost_after)
        object.__setattr__(self, "position_quantity_after", position_quantity_after)
        object.__setattr__(self, "cash_balance_after", cash_balance_after)


type SegmentName = Literal["training", "test"]


@dataclass(frozen=True, slots=True)
class SegmentBacktestResult:
    """Series and metrics produced for one train or test segment."""

    segment: SegmentName
    bar_returns: Series
    equity_curve: Series
    drawdown_curve: Series
    positions: Series
    asset_buy_and_hold_bar_returns: Series
    asset_buy_and_hold_equity_curve: Series
    asset_buy_and_hold_drawdown_curve: Series
    benchmark_bar_returns: Series | None
    benchmark_equity_curve: Series | None
    benchmark_drawdown_curve: Series | None
    metrics: PerformanceMetrics
    drawdown: DrawdownAnalysis
    trades: tuple[TradeEvent, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainingEvaluation:
    """Training-only result; intentionally has no test-result field."""

    symbol: str
    benchmark_symbol: str | None
    strategy_name: StrategyName
    config: StrategyConfig
    costs: BacktestCosts
    data_start: Timestamp
    data_end: Timestamp
    split_index: int
    split_date: Timestamp
    training: SegmentBacktestResult
    asset_quality: DataQualityReport
    benchmark_quality: DataQualityReport | None
    bar_interval: Literal["1d", "1m"] = "1d"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FinalEvaluation:
    """Frozen-parameter result containing both training and test evaluation."""

    symbol: str
    benchmark_symbol: str | None
    strategy_name: StrategyName
    config: StrategyConfig
    costs: BacktestCosts
    data_start: Timestamp
    data_end: Timestamp
    split_index: int
    split_date: Timestamp
    training: SegmentBacktestResult
    test: SegmentBacktestResult
    asset_quality: DataQualityReport
    benchmark_quality: DataQualityReport | None
    bar_interval: Literal["1d", "1m"] = "1d"
    warnings: tuple[str, ...] = ()
