from __future__ import annotations

from datetime import date
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

import tragent.quant.market_data as market_data_module
from tragent.quant.market_data import (
    MarketDataDownloadError,
    MarketDataError,
    YFinanceProvider,
    align_market_data,
    calculate_asset_statistics,
    latest_completed_us_market_session,
    normalize_market_data,
)


def make_raw_bars(
    dates: list[str] | pd.DatetimeIndex,
    *,
    opens: list[object] | None = None,
    closes: list[object] | None = None,
) -> pd.DataFrame:
    rows = len(dates)
    open_values = opens if opens is not None else [100.0 + index for index in range(rows)]
    close_values = closes if closes is not None else [101.0 + index for index in range(rows)]
    return pd.DataFrame(
        {
            "Open": open_values,
            "High": [105.0 + index for index in range(rows)],
            "Low": [95.0 + index for index in range(rows)],
            "Close": close_values,
            "Volume": [1_000 + index for index in range(rows)],
        },
        index=pd.DatetimeIndex(dates),
    )


def test_provider_uses_inclusive_end_and_normalizes_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    raw = make_raw_bars(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])

    def fake_download(**kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs)
        return raw

    result = YFinanceProvider(fake_download).get_daily_bars(
        " aapl ", date(2024, 1, 1), "2024-01-03"
    )

    assert calls == [
        {
            "tickers": "AAPL",
            "start": "2024-01-01",
            "end": "2024-01-04",
            "interval": "1d",
            "auto_adjust": True,
            "actions": False,
            "threads": False,
            "progress": False,
            "multi_level_index": False,
        }
    ]
    assert result.symbol == "AAPL"
    assert result.bars.index.equals(pd.date_range("2024-01-01", "2024-01-03"))
    assert result.quality.valid_rows == 3
    assert any("outside the requested range" in warning for warning in result.quality.warnings)

    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2024-07-02", "2024-07-03")
    minute_index = pd.DatetimeIndex(
        timestamp
        for session in sessions
        for timestamp in pd.date_range(
            calendar.session_open(session),
            calendar.session_close(session) - pd.Timedelta(minutes=1),
            freq="1min",
        )
    )
    minute_calls: list[dict[str, Any]] = []

    def minute_download(**kwargs: Any) -> pd.DataFrame:
        minute_calls.append(kwargs)
        return make_raw_bars(minute_index)

    monkeypatch.setattr(
        market_data_module,
        "_current_utc_time",
        lambda: pd.Timestamp("2024-07-04 00:00:00+00:00"),
    )
    minute = YFinanceProvider(minute_download).get_minute_bars("aapl", "2024-07-03", 2)

    assert minute.bar_interval == "1m"
    assert tuple(dict.fromkeys(minute.session_labels)) == ("2024-07-02", "2024-07-03")
    assert minute.bars.index.tz is not None
    assert minute_calls[0]["interval"] == "1m"
    assert minute_calls[0]["prepost"] is False
    assert len(minute.bars) == 390 + 210

    completed_sessions = calendar.sessions_in_range("2024-07-01", "2024-07-02")
    completed_index = pd.DatetimeIndex(
        timestamp
        for session in completed_sessions
        for timestamp in pd.date_range(
            calendar.session_open(session),
            calendar.session_close(session) - pd.Timedelta(minutes=1),
            freq="1min",
        )
    )
    monkeypatch.setattr(
        market_data_module,
        "_current_utc_time",
        lambda: pd.Timestamp("2024-07-03 16:00:00+00:00"),
    )
    while_market_is_open = YFinanceProvider(
        lambda **_kwargs: make_raw_bars(completed_index)
    ).get_minute_bars("aapl", "2024-07-03", 2)

    assert tuple(dict.fromkeys(while_market_is_open.session_labels)) == (
        "2024-07-01",
        "2024-07-02",
    )


def regular_session_minutes(start: str, exclusive_end: str) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(start, pd.Timestamp(exclusive_end) - pd.Timedelta(days=1))
    return pd.DatetimeIndex(
        timestamp
        for session in sessions
        for timestamp in pd.date_range(
            calendar.session_open(session),
            calendar.session_close(session) - pd.Timedelta(minutes=1),
            freq="1min",
        )
    )


def test_provider_downloads_minute_sessions_within_yahoo_request_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def minute_download(**kwargs: Any) -> pd.DataFrame:
        calls.append((kwargs["start"], kwargs["end"]))
        return make_raw_bars(regular_session_minutes(kwargs["start"], kwargs["end"]))

    monkeypatch.setattr(
        market_data_module,
        "_current_utc_time",
        lambda: pd.Timestamp("2024-07-27 00:00:00+00:00"),
    )
    minute = YFinanceProvider(minute_download).get_minute_bars("AAPL", "2024-07-26", 15)

    assert calls == [
        ("2024-07-08", "2024-07-16"),
        ("2024-07-16", "2024-07-24"),
        ("2024-07-24", "2024-07-27"),
    ]
    labels = tuple(dict.fromkeys(minute.session_labels))
    assert (len(labels), labels[0], labels[-1]) == (15, "2024-07-08", "2024-07-26")
    assert len(minute.bars) == 15 * 390
    assert minute.bars.index.is_monotonic_increasing


def test_provider_limits_minute_sessions_to_yahoo_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_download(**_kwargs: Any) -> pd.DataFrame:
        raise AssertionError("no download is expected")

    monkeypatch.setattr(
        market_data_module,
        "_current_utc_time",
        lambda: pd.Timestamp("2024-07-27 00:00:00+00:00"),
    )
    provider = YFinanceProvider(unexpected_download)

    # 2024-06-28 through 2024-07-26 hold the 20 sessions inside the last 30 days.
    with pytest.raises(MarketDataError, match="last 30 days of minute data"):
        provider.get_minute_bars("AAPL", "2024-07-26", 21)
    with pytest.raises(MarketDataError, match="last 30 days of minute data"):
        provider.get_minute_bars("AAPL", "2024-06-27", 1)


def test_provider_rejects_invalid_dates_before_downloading() -> None:
    called = False

    def fake_download(**_kwargs: Any) -> pd.DataFrame:
        nonlocal called
        called = True
        return make_raw_bars(["2024-01-01", "2024-01-02"])

    provider = YFinanceProvider(fake_download)

    with pytest.raises(MarketDataError, match="start_date must be on or before end_date"):
        provider.get_daily_bars("AAPL", "2024-02-01", "2024-01-01")
    with pytest.raises(MarketDataError, match="start_date must be an ISO date"):
        provider.get_daily_bars("AAPL", "01/01/2024", "2024-02-01")

    assert called is False


@pytest.mark.parametrize(
    ("now_utc", "expected"),
    [
        ("2026-09-17 16:06:51+00:00", date(2026, 9, 16)),
        ("2026-09-17 20:00:00+00:00", date(2026, 9, 17)),
        ("2026-09-19 12:00:00+00:00", date(2026, 9, 18)),
    ],
)
def test_latest_completed_us_market_session(
    monkeypatch: pytest.MonkeyPatch,
    now_utc: str,
    expected: date,
) -> None:
    monkeypatch.setattr(
        market_data_module,
        "_current_utc_time",
        lambda: pd.Timestamp(now_utc),
    )

    assert latest_completed_us_market_session() == expected


def test_provider_hides_downloader_exception_details() -> None:
    def failing_download(**_kwargs: Any) -> pd.DataFrame:
        raise RuntimeError("request failed with secret-token")

    with pytest.raises(MarketDataDownloadError) as error:
        YFinanceProvider(failing_download).get_daily_bars("AAPL", "2024-01-01", "2024-01-31")

    assert "Unable to download daily market data for AAPL" in str(error.value)
    assert "secret-token" not in str(error.value)


def test_provider_treats_empty_download_as_retryable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        market_data_module,
        "_current_utc_time",
        lambda: pd.Timestamp("2024-07-04 00:00:00+00:00"),
    )
    provider = YFinanceProvider(lambda **_kwargs: pd.DataFrame())

    with pytest.raises(MarketDataDownloadError, match="daily market data for AAPL"):
        provider.get_daily_bars("AAPL", "2024-01-01", "2024-01-31")
    with pytest.raises(MarketDataDownloadError, match="minute market data for AAPL"):
        provider.get_minute_bars("AAPL", "2024-07-03", 2)


def test_normalization_sorts_dates_keeps_last_duplicate_and_removes_bad_prices() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-01-03 16:00:00+00:00",
            "2024-01-02 09:00:00+00:00",
            "2024-01-02 16:00:00+00:00",
            "2024-01-04 16:00:00+00:00",
            "2024-01-05 16:00:00+00:00",
        ]
    )
    raw = make_raw_bars(
        index,
        opens=[30.0, 10.0, 20.0, 0.0, 50.0],
        closes=[31.0, 11.0, 21.0, 41.0, np.inf],
    )
    raw.loc[index[0], "Volume"] = np.nan

    result = normalize_market_data(raw, "AAPL")

    assert result.bars.index.equals(pd.DatetimeIndex(["2024-01-02", "2024-01-03"]))
    assert result.bars.index.tz is None
    assert list(result.bars.columns) == ["open", "high", "low", "close", "volume"]
    assert result.bars.loc[pd.Timestamp("2024-01-02"), "close"] == 21.0
    assert result.quality.duplicate_rows_removed == 1
    assert result.quality.invalid_price_rows_removed == 2
    assert result.quality.missing_value_counts["volume"] == 1
    assert result.quality.valid_rows == 2
    assert result.quality.is_valid is True
    assert len(result.quality.warnings) >= 3


@pytest.mark.parametrize(
    "raw, message",
    [
        (
            make_raw_bars(
                ["2024-01-01", "2024-01-02"],
                opens=[0.0, 2.0],
                closes=[1.0, 2.0],
            ),
            "fewer than two valid daily bars",
        ),
    ],
)
def test_normalization_rejects_fatal_data(raw: pd.DataFrame, message: str) -> None:
    with pytest.raises(MarketDataError, match=message) as error:
        normalize_market_data(raw, "AAPL")

    assert not isinstance(error.value, MarketDataDownloadError)


def test_alignment_uses_only_common_dates_and_updates_quality() -> None:
    asset = normalize_market_data(make_raw_bars(["2024-01-01", "2024-01-02", "2024-01-03"]), "AAPL")
    benchmark = normalize_market_data(
        make_raw_bars(["2024-01-02", "2024-01-03", "2024-01-04"]), "SPY"
    )

    aligned = align_market_data(asset, benchmark)

    expected_index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    assert aligned.asset.bars.index.equals(expected_index)
    assert aligned.benchmark.bars.index.equals(expected_index)
    assert aligned.asset.quality.valid_rows == 2
    assert aligned.asset.quality.start_date == pd.Timestamp("2024-01-02")
    assert aligned.asset.quality.end_date == pd.Timestamp("2024-01-03")
    assert len(aligned.warnings) == 2
    assert len(asset.bars) == 3
    assert len(benchmark.bars) == 3


def test_alignment_rejects_an_insufficient_intersection() -> None:
    asset = normalize_market_data(make_raw_bars(["2024-01-01", "2024-01-02"]), "AAPL")
    benchmark = normalize_market_data(make_raw_bars(["2024-01-02", "2024-01-03"]), "SPY")

    with pytest.raises(MarketDataError, match="fewer than two common trading days"):
        align_market_data(asset, benchmark)


def test_asset_statistics_use_n_minus_one_periods_and_sample_volatility() -> None:
    data = normalize_market_data(
        make_raw_bars(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            opens=[100.0, 110.0, 99.0],
            closes=[100.0, 110.0, 99.0],
        ),
        "AAPL",
    )

    statistics = calculate_asset_statistics(data)

    assert statistics.observations == 3
    assert statistics.start_date == pd.Timestamp("2024-01-01")
    assert statistics.end_date == pd.Timestamp("2024-01-03")
    assert statistics.total_return == pytest.approx(-0.01)
    assert statistics.annualized_return == pytest.approx(0.99**126 - 1.0)
    assert statistics.annualized_volatility == pytest.approx(
        pd.Series([0.1, -0.1]).std(ddof=1) * np.sqrt(252.0)
    )
    assert statistics.max_drawdown == pytest.approx(-0.1)


def test_asset_statistics_return_none_when_sample_volatility_is_undefined() -> None:
    data = normalize_market_data(
        make_raw_bars(
            ["2024-01-01", "2024-01-02"],
            opens=[100.0, 101.0],
            closes=[100.0, 101.0],
        ),
        "AAPL",
    )

    assert calculate_asset_statistics(data).annualized_volatility is None
