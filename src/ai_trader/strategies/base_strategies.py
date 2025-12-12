from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, TypeVar, Generic, Optional
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report


EstimatorT = TypeVar("EstimatorT", bound=ClassifierMixin)


@dataclass
class StrategyResult(Generic[EstimatorT]):
    """
    Container for results of an ML-based trading strategy.
    """
    model: EstimatorT
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    metrics: Dict[str, Any]
    equity_curve_strategy: pd.Series
    equity_curve_buy_hold: pd.Series
    returns_strategy: pd.Series
    returns_buy_hold: pd.Series


class BaseMLTradingStrategy(Generic[EstimatorT]):
    """
    Base class for simple ML-based trading strategies.

    Workflow
    --------
    1. Initialize with:
       - DataFrame containing price + indicators
       - Feature column names
       - Price column name
    2. Call `prepare_data()`:
       - Builds a binary target:
         * 1 → tomorrow's close > today's close
         * 0 → otherwise
       - Creates time-based train/test split.
    3. Call `fit_model()`:
       - Instantiates and fits the underlying sklearn model.
    4. Call `run_backtest()`:
       - Computes strategy returns:
         * If model predicts 1 → take next-day return (long)
         * Else → 0 return (flat)
       - Computes buy-and-hold returns and equity curves.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: Sequence[str],
        price_col: str = "Close",
        train_ratio: float = 0.7,
        target_col: str = "Target_Up_1d",
    ) -> None:
        """
        Parameters
        ----------
        data : pandas.DataFrame
            Input data with price and indicators. Index should be chronological
            (datetime index recommended).
        feature_cols : sequence of str
            Columns to use as model features.
        price_col : str, optional
            Column with closing price.
        train_ratio : float, optional
            Fraction of data used for training (rest is test).
        target_col : str, optional
            Name of the target column to be created.
        """
        if price_col not in data.columns:
            raise ValueError(
                f"price_col={price_col!r} not found in DataFrame columns: {list(data.columns)}"
            )

        missing_features = [c for c in feature_cols if c not in data.columns]
        if missing_features:
            raise ValueError(
                f"The following feature columns are missing from data: {missing_features}"
            )

        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1.")

        self._df = data.copy()
        self.feature_cols = list(feature_cols)
        self.price_col = price_col
        self.train_ratio = train_ratio
        self.target_col = target_col

        # To be filled during prepare/fit
        self._X_train: Optional[pd.DataFrame] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series] = None
        self._y_test: Optional[pd.Series] = None
        self._model: Optional[EstimatorT] = None

    # ------------------------------------------------------------------
    # Methods to be overridden by subclasses
    # ------------------------------------------------------------------
    def _build_model(self) -> EstimatorT:
        """
        Construct and return the underlying sklearn classifier.

        Subclasses must override this to provide a concrete estimator.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public pipeline methods
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        """
        Build the target, compute returns, drop NaNs, and create train/test split.
        """
        df = self._df

        # Target: is tomorrow's close higher than today's?
        df["Tomorrow_Close"] = df[self.price_col].shift(-1)
        df[self.target_col] = (df["Tomorrow_Close"] > df[self.price_col]).astype(int)

        # Returns: 1-day percentage change (for backtest)
        df["Return_1d"] = df[self.price_col].pct_change()

        # Drop NaNs created by shift / pct_change / indicators
        df_ml = df.dropna().copy()
        self._df = df_ml

        X = df_ml[self.feature_cols]
        y = df_ml[self.target_col]

        split_idx = int(len(df_ml) * self.train_ratio)
        self._X_train = X.iloc[:split_idx]
        self._X_test = X.iloc[split_idx:]
        self._y_train = y.iloc[:split_idx]
        self._y_test = y.iloc[split_idx:]

    def fit_model(self) -> None:
        """
        Fit the underlying ML model on the training data.
        """
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Call `prepare_data()` before `fit_model()`.")

        model = self._build_model()
        model.fit(self._X_train, self._y_train)
        self._model = model

    def run_backtest(self) -> StrategyResult[EstimatorT]:
        """
        Run a simple long-or-flat backtest on the test set.

        Strategy:
        ---------
        - If model predicts 1 (up):
            take the next day's return (long)
        - If model predicts 0:
            stay flat (return = 0)

        Buy-and-hold:
        -------------
        - Always take the asset's next-day return.

        Returns
        -------
        StrategyResult
        """
        if any(
            v is None
            for v in (self._X_train, self._X_test, self._y_train, self._y_test, self._model)
        ):
            raise RuntimeError("Call `prepare_data()` and `fit_model()` before `run_backtest()`.")

        X_train = self._X_train  # type: ignore[assignment]
        X_test = self._X_test    # type: ignore[assignment]
        y_train = self._y_train  # type: ignore[assignment]
        y_test = self._y_test    # type: ignore[assignment]
        model = self._model      # type: ignore[assignment]

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics: Dict[str, Any] = {
            "accuracy": acc,
            "classification_report": report,
        }

        # Backtest
        df = self._df
        test_index = X_test.index

        returns = df.loc[test_index, "Return_1d"]

        # Strategy: exposure = prediction (0 or 1)
        signals = pd.Series(y_pred, index=test_index)
        returns_strategy = returns * signals

        # Buy & hold: always exposed
        returns_buy_hold = returns.copy()

        equity_curve_strategy = (1.0 + returns_strategy).cumprod()
        equity_curve_buy_hold = (1.0 + returns_buy_hold).cumprod()

        return StrategyResult(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            metrics=metrics,
            equity_curve_strategy=equity_curve_strategy,
            equity_curve_buy_hold=equity_curve_buy_hold,
            returns_strategy=returns_strategy,
            returns_buy_hold=returns_buy_hold,
        )