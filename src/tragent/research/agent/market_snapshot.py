"""The immutable market-data snapshot saved once among each task's artifacts."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from tragent import storage
from tragent.quant import AlignedMarketData, DataQualityReport, MarketData

_BAR_COLUMNS = ("open", "high", "low", "close", "volume")
MARKET_DATA_FILENAME = "market_data.json"


class _SnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _BarSnapshot(_SnapshotModel):
    open: list[float]
    high: list[float | None]
    low: list[float | None]
    close: list[float]
    volume: list[float | None]


class _QualitySnapshot(_SnapshotModel):
    symbol: str
    valid_rows: int
    start_date: str | None
    end_date: str | None
    duplicate_rows_removed: int
    missing_value_counts: dict[str, int]
    invalid_price_rows_removed: int
    warnings: tuple[str, ...]
    is_valid: bool


class DataSnapshotDTO(_SnapshotModel):
    """Normalized OHLCV bars and quality reports for one confirmed study."""

    bar_interval: Literal["1d", "1m"]
    asset_symbol: str
    benchmark_symbol: str | None
    dates: list[str]
    session_labels: tuple[str, ...]
    training_sessions: int | None
    test_sessions: int | None
    asset: _BarSnapshot
    benchmark: _BarSnapshot | None
    asset_quality: _QualitySnapshot
    benchmark_quality: _QualitySnapshot | None
    warnings: tuple[str, ...]


def save_market_data(artifacts: str, data: AlignedMarketData) -> None:
    """Freeze the downloaded bars so every later Agent turn reads identical data."""

    storage.put(
        f"{artifacts}/{MARKET_DATA_FILENAME}",
        _snapshot(data).model_dump_json().encode(),
        "application/json",
    )


def load_market_data(artifacts: str) -> AlignedMarketData:
    """Restore the frozen bars without another provider download."""

    raw = storage.get(f"{artifacts}/{MARKET_DATA_FILENAME}")
    return _aligned_data(DataSnapshotDTO.model_validate_json(raw))


def _snapshot(data: AlignedMarketData) -> DataSnapshotDTO:
    dates, asset = _bars_to_snapshot(data.asset)
    benchmark = None
    benchmark_quality = None
    if data.benchmark is not None:
        _benchmark_dates, benchmark = _bars_to_snapshot(data.benchmark)
        benchmark_quality = _quality_to_snapshot(data.benchmark.quality)

    return DataSnapshotDTO(
        bar_interval=data.bar_interval,
        asset_symbol=data.asset.symbol,
        benchmark_symbol=data.benchmark.symbol if data.benchmark else None,
        dates=dates,
        session_labels=data.session_labels,
        training_sessions=data.training_sessions,
        test_sessions=data.test_sessions,
        asset=asset,
        benchmark=benchmark,
        asset_quality=_quality_to_snapshot(data.asset.quality),
        benchmark_quality=benchmark_quality,
        warnings=data.warnings,
    )


def _aligned_data(snapshot: DataSnapshotDTO) -> AlignedMarketData:
    index = pd.DatetimeIndex(snapshot.dates)

    def market_data(
        symbol: str,
        bars: _BarSnapshot,
        quality: _QualitySnapshot,
        *,
        session_labels: tuple[str, ...] = (),
    ) -> MarketData:
        return MarketData(
            symbol=symbol,
            bars=pd.DataFrame(
                {column: getattr(bars, column) for column in _BAR_COLUMNS},
                index=index.copy(),
                dtype="float64",
            ),
            quality=_quality_from_snapshot(quality),
            bar_interval=snapshot.bar_interval,
            session_labels=session_labels,
        )

    benchmark = (
        None
        if snapshot.benchmark is None
        or snapshot.benchmark_symbol is None
        or snapshot.benchmark_quality is None
        else market_data(
            snapshot.benchmark_symbol,
            snapshot.benchmark,
            snapshot.benchmark_quality,
        )
    )
    return AlignedMarketData(
        asset=market_data(
            snapshot.asset_symbol,
            snapshot.asset,
            snapshot.asset_quality,
            session_labels=snapshot.session_labels,
        ),
        benchmark=benchmark,
        warnings=snapshot.warnings,
        bar_interval=snapshot.bar_interval,
        session_labels=snapshot.session_labels,
        training_sessions=snapshot.training_sessions,
        test_sessions=snapshot.test_sessions,
    )


def _bars_to_snapshot(data: MarketData) -> tuple[list[str], dict[str, object]]:
    index = data.bars.index
    dates = [
        pd.Timestamp(value).date().isoformat()
        if data.bar_interval == "1d"
        else pd.Timestamp(value).isoformat()
        for value in index
    ]
    columns = {
        column: [None if pd.isna(value) else float(value) for value in data.bars[column]]
        for column in _BAR_COLUMNS
    }
    return dates, columns


def _quality_to_snapshot(quality: DataQualityReport) -> _QualitySnapshot:
    return _QualitySnapshot(
        symbol=quality.symbol,
        valid_rows=quality.valid_rows,
        start_date=quality.start_date.isoformat() if quality.start_date else None,
        end_date=quality.end_date.isoformat() if quality.end_date else None,
        duplicate_rows_removed=quality.duplicate_rows_removed,
        missing_value_counts=dict(quality.missing_value_counts),
        invalid_price_rows_removed=quality.invalid_price_rows_removed,
        warnings=quality.warnings,
        is_valid=quality.is_valid,
    )


def _quality_from_snapshot(quality: _QualitySnapshot) -> DataQualityReport:
    return DataQualityReport(
        symbol=quality.symbol,
        valid_rows=quality.valid_rows,
        start_date=pd.Timestamp(quality.start_date) if quality.start_date else None,
        end_date=pd.Timestamp(quality.end_date) if quality.end_date else None,
        duplicate_rows_removed=quality.duplicate_rows_removed,
        missing_value_counts=quality.missing_value_counts,
        invalid_price_rows_removed=quality.invalid_price_rows_removed,
        warnings=quality.warnings,
        is_valid=quality.is_valid,
    )


__all__ = ["MARKET_DATA_FILENAME", "load_market_data", "save_market_data"]
