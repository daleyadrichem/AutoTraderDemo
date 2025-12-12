from __future__ import annotations

from typing import Optional, List, Union
from pathlib import Path
import hashlib
import json

import pandas as pd
import yfinance as yf

PathLike = Union[str, Path]


# ------------------------------------------------------------------------------
# Internal helper utilities
# ------------------------------------------------------------------------------

def _ensure_cache_dir(cache_dir: PathLike) -> Path:
    """
    Ensure the cache directory exists.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _cache_key_from_params(params: dict) -> str:
    """
    Convert any dict of parameters to a stable hash for use as cache filename.
    """
    params_json = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(params_json.encode("utf-8")).hexdigest()


def _read_pickle_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_pickle(path)
        except Exception:
            return None
    return None


# ------------------------------------------------------------------------------
# Period-based mode
# ------------------------------------------------------------------------------

def download_price_data_period(
    ticker: str,
    interval: str = "1d",
    period: str = "2y",
    auto_adjust: bool = True,
    cache: bool = True,
    cache_dir: PathLike = ".data_cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV price data for a ticker using period-based mode, with pickle
    caching.

    Uses:
        yf.Ticker(ticker).history(period=..., interval=...)

    Parameters
    ----------
    ticker : str
        The ticker symbol.
    interval : str
        Bar size, e.g. "1d", "1h", "30m".
    period : str
        Lookback period, e.g. "1y", "5y", "max".
    auto_adjust : bool
        Adjust OHLC for splits and dividends.
    cache : bool
        Enable on-disk caching.
    cache_dir : str or Path
        Directory used for cache files.
    force_refresh : bool
        Ignore existing cache and re-download.

    Returns
    -------
    pandas.DataFrame
    """
    cache_path = None
    if cache:
        cache_dir_path = _ensure_cache_dir(cache_dir)
        params = {
            "mode": "period",
            "ticker": ticker,
            "interval": interval,
            "period": period,
            "auto_adjust": auto_adjust,
        }
        key = _cache_key_from_params(params)
        cache_path = cache_dir_path / f"{key}.pkl"

        if not force_refresh:
            cached = _read_pickle_if_exists(cache_path)
            if cached is not None:
                return cached

    # Download with Ticker().history()
    ticker_data = yf.Ticker(ticker=ticker)
    data = ticker_data.history(
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
    ).dropna()

    if data.empty:
        raise RuntimeError(
            f"No data returned for ticker={ticker!r} with "
            f"period={period!r} and interval={interval!r}."
        )

    data = data.sort_index()

    if cache and cache_path is not None:
        data.to_pickle(cache_path)

    return data


# ------------------------------------------------------------------------------
# Date-range mode with chunking
# ------------------------------------------------------------------------------

def download_price_data_daterange(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str = "1d",
    max_chunk_days: int = 60,
    auto_adjust: bool = True,
    cache: bool = True,
    cache_dir: PathLike = ".data_cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV data based on explicit start/end date, chunked into smaller
    windows for intraday interval safety, using Ticker().history().
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    if start_ts >= end_ts:
        raise ValueError(f"`start` must be earlier than `end` (got {start_ts} >= {end_ts}).")

    cache_path = None
    if cache:
        cache_dir_path = _ensure_cache_dir(cache_dir)
        params = {
            "mode": "daterange",
            "ticker": ticker,
            "interval": interval,
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "auto_adjust": auto_adjust,
            "max_chunk_days": max_chunk_days,
        }
        key = _cache_key_from_params(params)
        cache_path = cache_dir_path / f"{key}.pkl"

        if not force_refresh:
            cached = _read_pickle_if_exists(cache_path)
            if cached is not None:
                return cached

    # Download in chunks
    all_chunks: List[pd.DataFrame] = []
    current_start = start_ts

    ticker_data = yf.Ticker(ticker=ticker)

    while current_start < end_ts:
        current_end = min(current_start + pd.Timedelta(days=max_chunk_days), end_ts)

        chunk = ticker_data.history(
            start=current_start,
            end=current_end,
            interval=interval,
            auto_adjust=auto_adjust,
        ).dropna()

        if not chunk.empty:
            all_chunks.append(chunk)

        current_start = current_end

    if not all_chunks:
        raise RuntimeError(
            f"No data returned for {ticker!r} between {start_ts.date()} "
            f"and {end_ts.date()} using interval={interval!r}."
        )

    data = pd.concat(all_chunks)
    data = data[~data.index.duplicated(keep="last")]  # drop duplicates at boundaries
    data = data.sort_index()

    if cache and cache_path is not None:
        data.to_pickle(cache_path)

    return data


# ------------------------------------------------------------------------------
# Wrapper
# ------------------------------------------------------------------------------

def download_price_data(
    ticker: str,
    interval: str = "1d",
    period: Optional[str] = None,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
    max_chunk_days: int = 60,
    auto_adjust: bool = True,
    cache: bool = True,
    cache_dir: PathLike = ".data_cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Wrapper: dispatch to either period-based or date-range mode.
    """
    if period is not None and start is None and end is None:
        return download_price_data_period(
            ticker=ticker,
            interval=interval,
            period=period,
            auto_adjust=auto_adjust,
            cache=cache,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )

    if start is not None and end is not None:
        return download_price_data_daterange(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
            max_chunk_days=max_chunk_days,
            auto_adjust=auto_adjust,
            cache=cache,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )

    raise ValueError(
        "Invalid combination. Use either:\n"
        "- period='5y' (no start/end)\n"
        "- start='YYYY-MM-DD', end='YYYY-MM-DD' (no period)\n"
    )
