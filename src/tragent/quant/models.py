"""Small immutable data contracts shared by the quantitative modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Annotated, Literal, Self

from pandas import DataFrame, Series, Timestamp
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass as validated_dataclass

TRADE_DISPLAY_INITIAL_CAPITAL = 10_000.0

_Config = ConfigDict(extra="forbid")
_Window = Annotated[int, Field(strict=True, ge=2)]
_RsiThreshold = Annotated[float, Field(gt=0, lt=100, allow_inf_nan=False)]
_Bps = Annotated[float, Field(ge=0, allow_inf_nan=False)]


@validated_dataclass(frozen=True, slots=True, config=_Config)
class MovingAverageConfig:
    """Parameters for a long-only moving-average crossover strategy."""

    short_window: _Window
    long_window: _Window

    @model_validator(mode="after")
    def windows_are_ordered(self) -> Self:
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        return self


@validated_dataclass(frozen=True, slots=True, config=_Config)
class RsiConfig:
    """Parameters for the long-only Wilder RSI state strategy."""

    rsi_window: _Window
    entry_threshold: _RsiThreshold
    exit_threshold: _RsiThreshold

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> Self:
        if self.entry_threshold >= self.exit_threshold:
            raise ValueError("entry_threshold must be less than exit_threshold")
        return self


@validated_dataclass(frozen=True, slots=True, config=_Config)
class MacdConfig:
    """Parameters for a long-only MACD crossover strategy."""

    fast_period: _Window
    slow_period: _Window
    signal_period: _Window

    @model_validator(mode="after")
    def periods_are_ordered(self) -> Self:
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        return self


type IntradayRsiName = Literal["intraday_rsi", "rsi_price_exit", "rsi_vwap_exit"]
INTRADAY_RSI_BASELINE = "intraday_rsi"
INTRADAY_RSI_VARIANTS = ("rsi_price_exit", "rsi_vwap_exit")
INTRADAY_RSI_STRATEGIES = (INTRADAY_RSI_BASELINE, *INTRADAY_RSI_VARIANTS)


@validated_dataclass(frozen=True, slots=True, config=_Config)
class IntradayRsiConfig:
    """Fixed RSI(14,30,70) daily-reset policies, sharing one-tranche execution."""

    policy: IntradayRsiName


type StrategyConfig = MovingAverageConfig | RsiConfig | MacdConfig | IntradayRsiConfig
type StrategyName = Literal[
    "moving_average", "rsi", "macd", "intraday_rsi", "rsi_price_exit", "rsi_vwap_exit"
]
_TUNABLE_CONFIGS: dict[str, type[MovingAverageConfig | RsiConfig | MacdConfig]] = {
    "moving_average": MovingAverageConfig,
    "rsi": RsiConfig,
    "macd": MacdConfig,
}


def strategy_name(config: StrategyConfig) -> StrategyName:
    """Return the stable external name for a supported strategy configuration."""

    if isinstance(config, IntradayRsiConfig):
        return config.policy
    return next(name for name, kind in _TUNABLE_CONFIGS.items() if isinstance(config, kind))


def strategy_parameters(config: StrategyConfig) -> dict[str, int | float]:
    """Return primitive parameters suitable for reports and Agent observations."""

    return {} if isinstance(config, IntradayRsiConfig) else asdict(config)


def build_strategy_config(
    requested_strategy: str,
    parameters: Mapping[str, object],
) -> StrategyConfig:
    """Construct one supported strategy config from its external name and parameters."""

    if requested_strategy in INTRADAY_RSI_STRATEGIES:
        if parameters:
            raise ValueError("fixed intraday RSI policies take no parameters")
        return IntradayRsiConfig(policy=requested_strategy)  # type: ignore[arg-type]
    config_type = _TUNABLE_CONFIGS.get(requested_strategy)
    if config_type is None:
        raise ValueError("unsupported strategy")
    return config_type(**parameters)  # type: ignore[arg-type]


@validated_dataclass(frozen=True, slots=True, config=_Config)
class BacktestCosts:
    """Per-turnover transaction cost and slippage expressed in basis points."""

    transaction_cost_bps: _Bps = 5.0
    slippage_bps: _Bps = 5.0

    @model_validator(mode="after")
    def total_is_below_one_hundred_percent(self) -> Self:
        if self.total_bps >= 10_000:
            raise ValueError("transaction_cost_bps and slippage_bps must total less than 10000")
        return self

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
    mean_exposure: float
    trades: tuple[TradeEvent, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class _EvaluationBase:
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


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingEvaluation(_EvaluationBase):
    """Training-only result; intentionally has no test-result field."""


@dataclass(frozen=True, slots=True, kw_only=True)
class FinalEvaluation(_EvaluationBase):
    """Frozen-parameter result containing both training and test evaluation."""

    test: SegmentBacktestResult

    def training_evaluation(self) -> TrainingEvaluation:
        """Return the embedded training run; it equals a separate training evaluation."""
        return TrainingEvaluation(
            **{item.name: getattr(self, item.name) for item in fields(TrainingEvaluation)}
        )
