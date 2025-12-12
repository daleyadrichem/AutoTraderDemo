from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


@dataclass
class SimpleStrategyResult:
    """
    Container for results of the simple ML-based trading strategy.
    """
    model: LogisticRegression
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


class SimpleMLStrategy:
    """
    Simple machine-learning-based trading strategy using next-day direction
    as the target.

    This class assumes the input DataFrame contains:

    - A price column (default: "Close")
    - A set of feature columns (e.g. indicators added by `IndicatorEngineer`)

    The strategy:

    1. Builds a binary target:
       - 1 if tomorrow's close > today's close
       - 0 otherwise
    2. Splits the data into a time-based train and test set.
    3. Trains a Logistic Regression classifier.
    4. On the test set, goes long on days where the model predicts "up"
       and stays in cash otherwise.
    5. Computes cumulative returns for:
       - The strategy
       - A buy-and-hold benchmark

    Notes
    -----
    This is intentionally simple and for educational/demo purposes only.
    It is not a production trading strategy or financial advice.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: Sequence[str],
        price_col: str = "Close",
    ) -> None:
        """
        Parameters
        ----------
        data : pandas.DataFrame
            Input data containing price and indicator columns. Index is assumed
            to be chronological (datetime index is ideal).
        feature_cols : sequence of str
            Column names to use as features in the model.
        price_col : str, optional
            Name of the close price column used to compute returns/targets.
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

        self._df = data.copy()
        self.feature_cols = list(feature_cols)
        self.price_col = price_col

        # Will be set later
        self._model: LogisticRegression | None = None
        self._X_train: pd.DataFrame | None = None
        self._X_test: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None
        self._y_test: pd.Series | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        train_ratio: float = 0.7,
        target_col: str = "Target_Up_1d",
    ) -> None:
        """
        Build the target column and create time-based train/test splits.

        Parameters
        ----------
        train_ratio : float, optional
            Fraction of data to use for training (between 0 and 1).
        target_col : str, optional
            Name of the target column to create (1 = up, 0 = not up).
        """
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1.")

        df = self._df

        # 1-day ahead close
        df["Tomorrow_Close"] = df[self.price_col].shift(-1)
        df[target_col] = (df["Tomorrow_Close"] > df[self.price_col]).astype(int)

        # Compute simple 1-day returns for later backtest
        df["Return_1d"] = df[self.price_col].pct_change()

        # Drop rows with NaN (from shift / pct_change / indicators)
        df_ml = df.dropna().copy()

        # Save back
        self._df = df_ml

        # Features and target
        X = df_ml[self.feature_cols]
        y = df_ml[target_col]

        # Time-based split (no shuffling)
        split_idx = int(len(df_ml) * train_ratio)
        self._X_train = X.iloc[:split_idx]
        self._X_test = X.iloc[split_idx:]
        self._y_train = y.iloc[:split_idx]
        self._y_test = y.iloc[split_idx:]

    def fit_model(self) -> None:
        """
        Fit a simple Logistic Regression classifier on the training data.

        Raises
        ------
        RuntimeError
            If `prepare_data` has not been called first.
        """
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Call `prepare_data` before `fit_model`.")

        model = LogisticRegression(max_iter=1000)
        model.fit(self._X_train, self._y_train)

        self._model = model

    def run_backtest(self) -> SimpleStrategyResult:
        """
        Run a simple backtest on the test set.

        The strategy:

        - If model predicts "up" (1), take the next day's return.
        - If model predicts "not up" (0), stay in cash (return = 0).
        - Buy-and-hold benchmark: always take the next day's return.

        Returns
        -------
        SimpleStrategyResult
            Object containing predictions, metrics, and equity curves.

        Raises
        ------
        RuntimeError
            If `prepare_data` or `fit_model` have not been called.
        """
        if any(v is None for v in (self._X_train, self._X_test, self._y_train, self._y_test)):
            raise RuntimeError("Call `prepare_data` before `run_backtest`.")
        if self._model is None:
            raise RuntimeError("Call `fit_model` before `run_backtest`.")

        X_train = self._X_train  # type: ignore[assignment]
        X_test = self._X_test    # type: ignore[assignment]
        y_train = self._y_train  # type: ignore[assignment]
        y_test = self._y_test    # type: ignore[assignment]
        model = self._model

        # Predictions on test set
        y_pred = model.predict(X_test)

        # Classification metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        metrics: Dict[str, Any] = {
            "accuracy": acc,
            "classification_report": report,
        }

        # Align returns with X_test index
        df = self._df
        test_index = X_test.index

        # 1-day returns over the whole df (already computed in prepare_data)
        returns = df.loc[test_index, "Return_1d"]

        # Strategy: only take return where prediction == 1 (else 0)
        pred_series = pd.Series(y_pred, index=test_index)
        returns_strategy = returns * pred_series

        # Buy-and-hold: always take return
        returns_buy_hold = returns.copy()

        # Equity curves (starting at 1.0)
        equity_curve_strategy = (1.0 + returns_strategy).cumprod()
        equity_curve_buy_hold = (1.0 + returns_buy_hold).cumprod()

        return SimpleStrategyResult(
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
