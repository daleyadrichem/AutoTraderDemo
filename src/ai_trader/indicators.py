from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd
import numpy as np


class IndicatorEngineer:
    """
    Helper class for computing and managing technical indicators on price data.

    This class is initialized with a pandas DataFrame containing at least a
    closing price column (default: "Close"). Methods can then be used to add
    various indicators as new columns to the internal DataFrame.

    Examples
    --------
    >>> df = download_price_data("SPY", period="5y", interval="1d")
    >>> ie = IndicatorEngineer(df, price_col="Close")
    >>> ie.add_simple_return()
    >>> ie.add_rolling_return_mean(window=5)
    >>> ie.add_moving_average(window=20)
    >>> ie.add_moving_average(window=50)
    >>> ie.add_distance_from_ma(ma_col="MA20")
    >>> ie.add_distance_from_ma(ma_col="MA50")
    >>> ie.add_volatility(window=10)
    >>> ie.add_rsi(period=14)
    >>> ie.add_macd()
    >>> features = ie.get_indicators(["Return_1d", "MA20", "RSI", "MACD"])
    """

    def __init__(self, data: pd.DataFrame, price_col: str = "Close") -> None:
        """
        Parameters
        ----------
        data : pandas.DataFrame
            Input price data. Must contain at least a closing price column.
        price_col : str, optional
            Name of the closing price column in `data`.
        """
        if price_col not in data.columns:
            raise ValueError(
                f"price_col={price_col!r} not found in DataFrame columns: {list(data.columns)}"
            )

        self._df = data.copy()
        self.price_col = price_col

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns
        -------
        pandas.DataFrame
            The full DataFrame, including any indicators that have been added.
        """
        return self._df

    # -------------------------------------------------------------------------
    # Basic returns & moving averages
    # -------------------------------------------------------------------------
    def add_simple_return(
        self,
        periods: int = 1,
        col_name: Optional[str] = None,
    ) -> IndicatorEngineer:
        """
        Add simple percentage return over a given number of periods.

        Parameters
        ----------
        periods : int, optional
            Number of periods over which to compute the percent change.
        col_name : str or None, optional
            Name of the new column. If None, uses "Return_{periods}d".

        Returns
        -------
        IndicatorEngineer
            The instance (for method chaining).
        """
        if col_name is None:
            col_name = f"Return_{periods}d"

        self._df[col_name] = self._df[self.price_col].pct_change(periods=periods)
        return self

    def add_rolling_return_mean(
        self,
        window: int,
        source_col: str = "Return_1d",
        col_name: Optional[str] = None,
    ) -> IndicatorEngineer:
        """
        Add a rolling mean of returns.

        Parameters
        ----------
        window : int
            Rolling window size (in rows).
        source_col : str, optional
            Column name from which to compute the rolling mean (typically a
            return column created by `add_simple_return`).
        col_name : str or None, optional
            Name of the new column. If None, uses "Return_{window}d_mean".

        Returns
        -------
        IndicatorEngineer
            The instance (for method chaining).
        """
        if source_col not in self._df.columns:
            raise ValueError(f"source_col={source_col!r} not found in DataFrame.")

        if col_name is None:
            col_name = f"Return_{window}d_mean"

        self._df[col_name] = self._df[source_col].rolling(window=window).mean()
        return self

    def add_moving_average(
        self,
        window: int,
        col_name: Optional[str] = None,
    ) -> IndicatorEngineer:
        """
        Add a simple moving average (SMA) of the closing price.

        Parameters
        ----------
        window : int
            Rolling window size (in rows).
        col_name : str or None, optional
            Name of the new column. If None, uses "MA{window}".

        Returns
        -------
        IndicatorEngineer
            The instance (for method chaining).
        """
        if col_name is None:
            col_name = f"MA{window}"

        self._df[col_name] = self._df[self.price_col].rolling(window=window).mean()
        return self

    def add_distance_from_ma(
        self,
        ma_col: str,
        col_name: Optional[str] = None,
    ) -> IndicatorEngineer:
        """
        Add the distance from a moving average, as a percentage of that MA.

        The distance is defined as:
            (price - MA) / MA

        Parameters
        ----------
        ma_col : str
            Name of the moving average column (e.g. "MA20", "MA50").
        col_name : str or None, optional
            Name of the new column. If None, uses "Dist_{ma_col}".

        Returns
        -------
        IndicatorEngineer
            The instance (for method chaining).
        """
        if ma_col not in self._df.columns:
            raise ValueError(f"ma_col={ma_col!r} not found in DataFrame.")

        if col_name is None:
            col_name = f"Dist_{ma_col}"

        ma_values = self._df[ma_col]
        self._df[col_name] = (self._df[self.price_col] - ma_values) / ma_values
        return self

    # -------------------------------------------------------------------------
    # Volatility
    # -------------------------------------------------------------------------
    def add_volatility(
        self,
        window: int = 10,
        return_col: Optional[str] = "Return_1d",
        col_name: Optional[str] = None,
    ) -> IndicatorEngineer:
        """
        Add a rolling volatility estimate based on returns.

        By default, uses the 1-period return column ("Return_1d"). If this
        column does not exist yet, it will be created automatically.

        Parameters
        ----------
        window : int, optional
            Rolling window size for the standard deviation.
        return_col : str or None, optional
            Name of the return column to use. If None, uses the percentage
            change of `price_col` on the fly.
        col_name : str or None, optional
            Name of the new column. If None, uses "Volatility_{window}d".

        Returns
        -------
        IndicatorEngineer
            The instance (for method chaining).
        """
        if col_name is None:
            col_name = f"Volatility_{window}d"

        if return_col is not None:
            if return_col not in self._df.columns:
                # auto-create simple 1-period return if requested column is "Return_1d"
                if return_col == "Return_1d":
                    self.add_simple_return(periods=1, col_name="Return_1d")
                else:
                    raise ValueError(f"return_col={return_col!r} not found in DataFrame.")
            returns = self._df[return_col]
        else:
            returns = self._df[self.price_col].pct_change()

        self._df[col_name] = returns.rolling(window=window).std()
        return self

    # -------------------------------------------------------------------------
    # RSI
    # -------------------------------------------------------------------------
    def add_rsi(
        self,
        period: int = 14,
        col_name: str = "RSI",
    ) -> IndicatorEngineer:
        """
        Add the Relative Strength Index (RSI) indicator.

        RSI is computed using the classic Wilder's smoothing method approximation
        via simple rolling averages of gains and losses.

        Parameters
        ----------
        period : int, optional
            Lookback period for RSI.
        col_name : str, optional
            Name of the new column.

        Returns
        -------
        IndicatorEngineer
            The instance (for method chaining).
        """
        close = self._df[self.price_col]
        delta = close.diff()

        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        self._df[col_name] = rsi
        return self

    # -------------------------------------------------------------------------
    # MACD
    # -------------------------------------------------------------------------
    def add_macd(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        macd_col: str = "MACD",
        signal_col: str = "MACD_Signal",
        hist_col: str = "MACD_Hist",
    ) -> IndicatorEngineer:
        """
        Add MACD (Moving Average Convergence Divergence) indicator.

        MACD is defined as the difference between a "fast" EMA and a "slow" EMA:
            MACD = EMA_fast(price) - EMA_slow(price)
        A "signal" line is then computed as an EMA of the MACD itself.

        The histogram is:
            MACD_Hist = MACD - MACD_Signal

        Parameters
        ----------
        fast_period : int, optional
            Span for the fast EMA.
        slow_period : int, optional
            Span for the slow EMA.
        signal_period : int, optional
            Span for the signal line EMA.
        macd_col : str, optional
            Column name for the MACD line.
        signal_col : str, optional
            Column name for the signal line.
        hist_col : str, optional
            Column name for the MACD histogram.

        Returns
        -------
        IndicatorEngineer
            The instance (for method chaining).
        """
        close = self._df[self.price_col]

        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal

        self._df[macd_col] = macd
        self._df[signal_col] = signal
        self._df[hist_col] = hist

        return self

    # -------------------------------------------------------------------------
    # Convenience: add the "basic set" used in your demo
    # -------------------------------------------------------------------------
    def add_basic_demo_indicators(self) -> IndicatorEngineer:
        """
        Add a basic set of indicators commonly used in the trading demo:

        - 1-day return
        - 5-day and 20-day rolling mean of returns
        - 20-day and 50-day moving averages
        - Distance from MA20 and MA50
        - 10-day volatility (std of returns)
        - 14-period RSI
        - MACD (12, 26, 9)

        Returns
        -------
        IndicatorEngineer
            The instance (for method chaining).
        """
        # Returns and rolling means
        self.add_simple_return(periods=1, col_name="Return_1d")
        self.add_rolling_return_mean(window=5, source_col="Return_1d", col_name="Return_5d_mean")
        self.add_rolling_return_mean(window=20, source_col="Return_1d", col_name="Return_20d_mean")

        # Moving averages
        self.add_moving_average(window=20, col_name="MA20")
        self.add_moving_average(window=50, col_name="MA50")

        # Distance from MAs
        self.add_distance_from_ma(ma_col="MA20", col_name="Dist_MA20")
        self.add_distance_from_ma(ma_col="MA50", col_name="Dist_MA50")

        # Volatility, RSI, MACD
        self.add_volatility(window=10, return_col="Return_1d", col_name="Volatility_10d")
        self.add_rsi(period=14, col_name="RSI")
        self.add_macd(fast_period=12, slow_period=26, signal_period=9)

        return self

    # -------------------------------------------------------------------------
    # Getter for a subset of indicators
    # -------------------------------------------------------------------------
    def get_indicators(
        self,
        indicators: Sequence[str],
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        Return a DataFrame containing only the selected indicator columns.

        Parameters
        ----------
        indicators : sequence of str
            List of column names (indicators) to extract.
        dropna : bool, optional
            If True, drop any rows that contain NaN in the selected columns.
            If False, return the raw subset with NaNs preserved.

        Returns
        -------
        pandas.DataFrame
            DataFrame with the requested indicator columns.

        Raises
        ------
        KeyError
            If any requested indicator column does not exist.
        """
        missing = [col for col in indicators if col not in self._df.columns]
        if missing:
            raise KeyError(f"Requested indicator columns not found: {missing}")

        df_subset = self._df.loc[:, indicators].copy()

        if dropna:
            df_subset = df_subset.dropna()

        return df_subset
