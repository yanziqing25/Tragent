"""Daily market-data loading, normalization, alignment, and basic statistics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Protocol

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf

from trading.quant.models import (
    AlignedMarketData,
    AssetStatistics,
    DataQualityReport,
    MarketData,
)

STANDARD_COLUMNS = ("open", "high", "low", "close", "volume")
REQUIRED_COLUMNS = frozenset({"open", "close"})

DateInput = date | str
Downloader = Callable[..., pd.DataFrame]


class MarketDataError(ValueError):
    """Market data could not be safely prepared for deterministic analysis."""


class MarketDataDownloadError(MarketDataError):
    """A transient provider download failed before data normalization."""


class MarketDataProvider(Protocol):
    """Minimal boundary used by later tools instead of importing yfinance."""

    def get_daily_bars(
        self,
        symbol: str,
        start_date: DateInput,
        end_date: DateInput,
    ) -> MarketData:
        """Return normalized daily bars for an inclusive date range."""

    def get_minute_bars(
        self,
        symbol: str,
        cutoff_date: DateInput,
        session_count: int,
    ) -> MarketData:
        """Return completed XNYS regular-session minute bars before a cutoff date."""


class YFinanceProvider:
    """Load adjusted daily or minute bars from yfinance."""

    def __init__(self, downloader: Downloader | None = None) -> None:
        self._downloader = downloader

    def get_daily_bars(
        self,
        symbol: str,
        start_date: DateInput,
        end_date: DateInput,
    ) -> MarketData:
        """Download and normalize daily bars, treating both dates as inclusive."""

        normalized_symbol = _normalize_symbol(symbol)
        start = _parse_date(start_date, "start_date")
        end = _parse_date(end_date, "end_date")
        if start > end:
            raise MarketDataError("start_date must be on or before end_date")

        try:
            exclusive_end = end + timedelta(days=1)
        except OverflowError:
            raise MarketDataError("end_date is outside the supported date range") from None

        download = self._downloader or yf.download
        try:
            raw = download(
                tickers=normalized_symbol,
                start=start.isoformat(),
                end=exclusive_end.isoformat(),
                interval="1d",
                auto_adjust=True,
                actions=False,
                threads=False,
                progress=False,
                multi_level_index=False,
            )
        except Exception:
            raise MarketDataDownloadError(
                f"Unable to download daily market data for {normalized_symbol}"
            ) from None

        normalized = normalize_market_data(raw, normalized_symbol)
        return _restrict_to_requested_dates(normalized, start, end)

    def get_minute_bars(
        self,
        symbol: str,
        cutoff_date: DateInput,
        session_count: int,
    ) -> MarketData:
        """Download completed XNYS regular sessions through ``cutoff_date``."""

        normalized_symbol = _normalize_symbol(symbol)
        cutoff = _parse_date(cutoff_date, "cutoff_date")
        if isinstance(session_count, bool) or not isinstance(session_count, int):
            raise MarketDataError("session_count must be an integer")
        if not 2 <= session_count <= 120:
            raise MarketDataError("session_count must be between 2 and 120")

        calendar = xcals.get_calendar("XNYS")
        available = calendar.sessions_in_range(cutoff - timedelta(days=370), cutoff)
        now_utc = _current_utc_time()
        completed = pd.DatetimeIndex(
            [
                session
                for session in available
                if calendar.session_close(session).tz_convert("UTC") <= now_utc
            ]
        )
        if len(completed) < session_count:
            raise MarketDataError("The requested completed XNYS sessions are not available")
        sessions = completed[-session_count:]
        first_session = pd.Timestamp(sessions[0]).date()
        last_session = pd.Timestamp(sessions[-1]).date()

        download = self._downloader or yf.download
        try:
            raw = download(
                tickers=normalized_symbol,
                start=first_session.isoformat(),
                end=(last_session + timedelta(days=1)).isoformat(),
                interval="1m",
                auto_adjust=True,
                actions=False,
                threads=False,
                progress=False,
                prepost=False,
                multi_level_index=False,
            )
        except Exception:
            raise MarketDataDownloadError(
                f"Unable to download minute market data for {normalized_symbol}"
            ) from None

        normalized = normalize_market_data(raw, normalized_symbol, bar_interval="1m")
        pieces: list[pd.DataFrame] = []
        labels: list[str] = []
        warnings = list(normalized.quality.warnings)
        for session in sessions:
            session_label = pd.Timestamp(session).date().isoformat()
            session_open = calendar.session_open(session).tz_convert("UTC")
            session_close = calendar.session_close(session).tz_convert("UTC")
            frame = normalized.bars.loc[
                (normalized.bars.index >= session_open) & (normalized.bars.index < session_close)
            ].copy()
            expected_rows = int((session_close - session_open) / pd.Timedelta(minutes=1))
            expected_last = session_close - pd.Timedelta(minutes=1)
            if frame.empty or frame.index[0] != session_open or frame.index[-1] < expected_last:
                raise MarketDataError(
                    f"{normalized_symbol} has an incomplete regular session on {session_label}"
                )
            if len(frame) < expected_rows:
                warnings.append(
                    f"{normalized_symbol}: {session_label} is missing "
                    f"{expected_rows - len(frame)} regular-session minute bar(s)"
                )
            pieces.append(frame)
            labels.extend([session_label] * len(frame))

        bars = pd.concat(pieces).sort_index(kind="stable")
        quality = replace(
            normalized.quality,
            valid_rows=len(bars),
            start_date=pd.Timestamp(bars.index[0]),
            end_date=pd.Timestamp(bars.index[-1]),
            warnings=tuple(warnings[:20]),
        )
        return MarketData(
            symbol=normalized_symbol,
            bars=bars,
            quality=quality,
            bar_interval="1m",
            session_labels=tuple(labels),
        )


def _current_utc_time() -> pd.Timestamp:
    """Return the current instant used to exclude an open XNYS session."""

    return pd.Timestamp.now(tz="UTC")


def normalize_market_data(
    raw: pd.DataFrame,
    symbol: str,
    *,
    bar_interval: str = "1d",
) -> MarketData:
    """Convert one-symbol yfinance output into the canonical bar contract."""

    normalized_symbol = _normalize_symbol(symbol)
    if bar_interval not in {"1d", "1m"}:
        raise MarketDataError("bar_interval must be 1d or 1m")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise MarketDataError(f"No market data was returned for {normalized_symbol}")

    bars, missing_optional = _select_price_columns(raw, normalized_symbol)
    bars.index = _normalize_date_index(
        bars.index,
        normalized_symbol,
        intraday=bar_interval == "1m",
    )

    duplicate_rows = int(bars.index.duplicated(keep="last").sum())
    if duplicate_rows:
        bars = bars.loc[~bars.index.duplicated(keep="last")]
    bars = bars.sort_index(kind="stable")

    for column in STANDARD_COLUMNS:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")

    missing_value_counts = {column: int(bars[column].isna().sum()) for column in STANDARD_COLUMNS}
    required_values = bars[["open", "close"]].to_numpy(dtype=float, na_value=np.nan)
    invalid_prices = (~np.isfinite(required_values)).any(axis=1) | (required_values <= 0).any(
        axis=1
    )
    invalid_price_rows = int(invalid_prices.sum())
    if invalid_price_rows:
        bars = bars.loc[~invalid_prices]

    warnings: list[str] = []
    if missing_optional:
        names = ", ".join(missing_optional)
        warnings.append(f"{normalized_symbol}: optional columns missing: {names}")
    if duplicate_rows:
        warnings.append(
            f"{normalized_symbol}: removed {duplicate_rows} duplicate date row(s), keeping last"
        )
    nonzero_missing = {column: count for column, count in missing_value_counts.items() if count > 0}
    if nonzero_missing:
        detail = ", ".join(f"{column}={count}" for column, count in nonzero_missing.items())
        warnings.append(f"{normalized_symbol}: missing values detected ({detail})")
    if invalid_price_rows:
        warnings.append(
            f"{normalized_symbol}: removed {invalid_price_rows} row(s) with invalid open/close"
        )

    if len(bars) < 2:
        unit = "daily bars" if bar_interval == "1d" else "minute bars"
        raise MarketDataError(
            f"{normalized_symbol} has fewer than two valid {unit} after normalization"
        )

    bars = bars.loc[:, STANDARD_COLUMNS].copy()
    quality = DataQualityReport(
        symbol=normalized_symbol,
        valid_rows=len(bars),
        start_date=pd.Timestamp(bars.index[0]),
        end_date=pd.Timestamp(bars.index[-1]),
        duplicate_rows_removed=duplicate_rows,
        missing_value_counts=missing_value_counts,
        invalid_price_rows_removed=invalid_price_rows,
        warnings=tuple(warnings),
        is_valid=True,
    )
    return MarketData(
        symbol=normalized_symbol,
        bars=bars,
        quality=quality,
        bar_interval=bar_interval,  # type: ignore[arg-type]
    )


def align_market_data(asset: MarketData, benchmark: MarketData) -> AlignedMarketData:
    """Align an asset and benchmark to exactly the same common trading dates."""

    _validate_normalized_bars(asset)
    _validate_normalized_bars(benchmark)

    common_dates = asset.bars.index.intersection(benchmark.bars.index).sort_values()
    if len(common_dates) < 2:
        raise MarketDataError(
            f"{asset.symbol} and {benchmark.symbol} have fewer than two common trading days"
        )

    aligned_asset_bars = asset.bars.loc[common_dates].copy()
    aligned_benchmark_bars = benchmark.bars.loc[common_dates].copy()
    start_date = pd.Timestamp(common_dates[0])
    end_date = pd.Timestamp(common_dates[-1])

    asset_removed = len(asset.bars) - len(common_dates)
    benchmark_removed = len(benchmark.bars) - len(common_dates)

    asset_warnings = list(asset.quality.warnings)
    benchmark_warnings = list(benchmark.quality.warnings)
    alignment_warnings: list[str] = []
    if asset_removed:
        warning = f"{asset.symbol}: alignment removed {asset_removed} non-common trading day(s)"
        asset_warnings.append(warning)
        alignment_warnings.append(warning)
    if benchmark_removed:
        warning = (
            f"{benchmark.symbol}: alignment removed {benchmark_removed} non-common trading day(s)"
        )
        benchmark_warnings.append(warning)
        alignment_warnings.append(warning)

    aligned_asset = MarketData(
        symbol=asset.symbol,
        bars=aligned_asset_bars,
        quality=replace(
            asset.quality,
            valid_rows=len(common_dates),
            start_date=start_date,
            end_date=end_date,
            warnings=tuple(asset_warnings),
        ),
    )
    aligned_benchmark = MarketData(
        symbol=benchmark.symbol,
        bars=aligned_benchmark_bars,
        quality=replace(
            benchmark.quality,
            valid_rows=len(common_dates),
            start_date=start_date,
            end_date=end_date,
            warnings=tuple(benchmark_warnings),
        ),
    )
    return AlignedMarketData(
        asset=aligned_asset,
        benchmark=aligned_benchmark,
        warnings=tuple(alignment_warnings),
    )


def calculate_asset_statistics(data: MarketData) -> AssetStatistics:
    """Calculate deterministic close-to-close statistics without downloading data."""

    _validate_normalized_bars(data)
    close = data.bars["close"].astype(float)
    if len(close) < 2:
        raise MarketDataError(f"{data.symbol} needs at least two closes for asset statistics")
    if (~np.isfinite(close.to_numpy())).any() or (close <= 0).any():
        raise MarketDataError(f"{data.symbol} close prices must be finite and positive")

    growth = float(close.iloc[-1] / close.iloc[0])
    total_return = growth - 1.0
    bar_returns = close.pct_change(fill_method=None).iloc[1:]
    if data.bar_interval == "1d":
        periods = len(close) - 1
        annualized_return = float(growth ** (252.0 / periods) - 1.0)
        annualized_volatility = (
            None if len(bar_returns) < 2 else float(bar_returns.std(ddof=1) * np.sqrt(252.0))
        )
    else:
        annualized_return = None
        annualized_volatility = None

    normalized_equity = close / close.iloc[0]
    drawdown = normalized_equity / normalized_equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    numeric_results = (total_return, max_drawdown)
    if (
        not all(np.isfinite(value) for value in numeric_results)
        or (annualized_return is not None and not np.isfinite(annualized_return))
        or (annualized_volatility is not None and not np.isfinite(annualized_volatility))
    ):
        raise MarketDataError(f"{data.symbol} statistics produced a non-finite result")

    return AssetStatistics(
        symbol=data.symbol,
        observations=len(close),
        start_date=pd.Timestamp(close.index[0]),
        end_date=pd.Timestamp(close.index[-1]),
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        max_drawdown=max_drawdown,
    )


def _select_price_columns(raw: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    positions: dict[str, int] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        flat_columns = list(raw.columns.to_flat_index())
        price_levels: list[tuple[int, int]] = []
        for level in range(raw.columns.nlevels):
            names = {_normalize_column_name(column[level]) for column in flat_columns}
            if REQUIRED_COLUMNS.issubset(names):
                price_levels.append((len(names.intersection(STANDARD_COLUMNS)), level))
        if not price_levels:
            raise MarketDataError(f"{symbol} data is missing required open/close columns")

        _, price_level = max(price_levels)
        symbol_key = symbol.casefold()
        symbol_is_present = any(
            any(
                level != price_level and str(column[level]).strip().casefold() == symbol_key
                for level in range(len(column))
            )
            for column in flat_columns
        )

        for position, column in enumerate(flat_columns):
            if symbol_is_present and not any(
                level != price_level and str(column[level]).strip().casefold() == symbol_key
                for level in range(len(column))
            ):
                continue
            normalized_name = _normalize_column_name(column[price_level])
            if normalized_name in STANDARD_COLUMNS:
                if normalized_name in positions:
                    raise MarketDataError(
                        f"{symbol} data contains ambiguous {normalized_name} columns"
                    )
                positions[normalized_name] = position
    else:
        for position, column in enumerate(raw.columns):
            normalized_name = _normalize_column_name(column)
            if normalized_name in STANDARD_COLUMNS:
                if normalized_name in positions:
                    raise MarketDataError(
                        f"{symbol} data contains ambiguous {normalized_name} columns"
                    )
                positions[normalized_name] = position

    missing_required = REQUIRED_COLUMNS.difference(positions)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise MarketDataError(f"{symbol} data is missing required column(s): {missing}")

    result = pd.DataFrame(index=raw.index)
    missing_optional: list[str] = []
    for column in STANDARD_COLUMNS:
        if column in positions:
            result[column] = raw.iloc[:, positions[column]]
        else:
            result[column] = np.nan
            missing_optional.append(column)
    return result, tuple(missing_optional)


def _normalize_date_index(
    index: pd.Index,
    symbol: str,
    *,
    intraday: bool = False,
) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        raise MarketDataError(f"{symbol} data must use a one-dimensional date index")
    if not isinstance(index, pd.DatetimeIndex) and pd.api.types.is_numeric_dtype(index.dtype):
        raise MarketDataError(f"{symbol} data index must contain dates")

    try:
        normalized = pd.DatetimeIndex(pd.to_datetime(index, errors="raise"))
    except (TypeError, ValueError, OverflowError):
        raise MarketDataError(f"{symbol} data index contains an invalid date") from None
    if normalized.hasnans:
        raise MarketDataError(f"{symbol} data index contains an invalid date")
    if intraday:
        if normalized.tz is None:
            normalized = normalized.tz_localize("America/New_York")
        return normalized.tz_convert("UTC")
    if normalized.tz is not None:
        normalized = normalized.tz_localize(None)
    return normalized.normalize()


def _restrict_to_requested_dates(data: MarketData, start: date, end: date) -> MarketData:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    within_range = (data.bars.index >= start_timestamp) & (data.bars.index <= end_timestamp)
    bars = data.bars.loc[within_range].copy()
    if len(bars) < 2:
        raise MarketDataError(
            f"{data.symbol} has fewer than two valid daily bars in the requested date range"
        )

    removed = len(data.bars) - len(bars)
    warnings = list(data.quality.warnings)
    if removed:
        warnings.append(f"{data.symbol}: ignored {removed} row(s) outside the requested range")
    quality = replace(
        data.quality,
        valid_rows=len(bars),
        start_date=pd.Timestamp(bars.index[0]),
        end_date=pd.Timestamp(bars.index[-1]),
        warnings=tuple(warnings),
    )
    return MarketData(symbol=data.symbol, bars=bars, quality=quality)


def _validate_normalized_bars(data: MarketData) -> None:
    bars = data.bars
    if not isinstance(bars, pd.DataFrame) or not isinstance(bars.index, pd.DatetimeIndex):
        raise MarketDataError(f"{data.symbol} bars must use a DatetimeIndex")
    invalid_timezone = (
        bars.index.tz is not None if data.bar_interval == "1d" else bars.index.tz is None
    )
    if invalid_timezone or not bars.index.is_monotonic_increasing:
        raise MarketDataError(f"{data.symbol} bar timestamps have an invalid timezone or order")
    if not bars.index.is_unique:
        raise MarketDataError(f"{data.symbol} bar dates must be unique")
    missing = REQUIRED_COLUMNS.difference(bars.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise MarketDataError(f"{data.symbol} bars are missing required column(s): {names}")


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not normalized:
        raise MarketDataError("symbol must not be empty")
    return normalized


def _parse_date(value: DateInput, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise MarketDataError(f"{field_name} must be an ISO date in YYYY-MM-DD format")


def _normalize_column_name(value: object) -> str:
    return str(value).strip().casefold()
