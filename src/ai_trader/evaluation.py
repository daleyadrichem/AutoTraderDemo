from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


@dataclass
class ModelPerformance:
    """
    Container for performance metrics and curves of a single model.
    """
    name: str
    y_true: pd.Series
    y_pred: pd.Series
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    returns_strategy: pd.Series
    returns_buy_hold: pd.Series
    equity_strategy: pd.Series
    equity_buy_hold: pd.Series


class ModelPerformanceAnalyzer:
    """
    General analyzer for comparing multiple classification-based trading models.

    Assumptions
    -----------
    - You have a time index (typically datetime) and a series of realized returns
      for the underlying asset (e.g. 1-day returns).
    - Each model produces a series of *signals* aligned with that index:
        - 1  → long / buy
        - 0  → flat (no position)
        - -1 → short / sell (optional; if present, interpreted as short exposure)
    - Strategy PnL is computed as:
        returns_strategy = returns * signals

      where `signals` is either in {0, 1} (long or flat) or {-1, 0, 1} (short/flat/long).

    This class:
    - Stores multiple models by name.
    - Computes classification metrics and equity curves per model.
    - Provides a plotting helper to visualize:
        * price + buy/sell markers
        * strategy equity vs buy-and-hold
    """

    def __init__(
        self,
        returns: pd.Series,
        price: pd.Series,
        y_true: Optional[pd.Series] = None,
    ) -> None:
        """
        Parameters
        ----------
        returns : pandas.Series
            Realized returns of the underlying asset (e.g. 1-day returns),
            indexed by time.
        price : pandas.Series
            Price series (e.g. closing price), indexed by time.
        y_true : pandas.Series or None, optional
            True labels (e.g. 0/1 direction) aligned with `returns`. If None,
            you can still use the analyzer for PnL / equity comparisons but
            classification metrics will be unavailable unless you supply y_true
            per model.
        """
        if not returns.index.equals(price.index):
            raise ValueError("`returns` and `price` must have the same index.")

        self.returns = returns
        self.price = price
        self.global_y_true = y_true
        self._models: Dict[str, ModelPerformance] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_model(
        self,
        name: str,
        signals: pd.Series,
        y_true: Optional[pd.Series] = None,
        positive_label: int = 1,
    ) -> None:
        """
        Register a model's signals and compute its performance.

        Parameters
        ----------
        name : str
            Name of the model (used as key and for plots).
        signals : pandas.Series
            Trading signals aligned with `returns.index`. Values typically in
            {0, 1} (flat vs long) or {-1, 0, 1} (short/flat/long).
        y_true : pandas.Series or None, optional
            True labels for classification metrics. If None, uses `self.global_y_true`.
            If both are None, classification metrics will not be computed
            (they will be set to NaN).
        positive_label : int, optional
            Which label to treat as the "positive" class when computing
            precision/recall/F1 (default is 1).
        """
        # Align signals to returns index
        signals = signals.reindex(self.returns.index).fillna(0)

        # Strategy returns: multiply by signal (assumes signal is exposure)
        returns_strategy = self.returns * signals

        # Buy & hold returns: always take the underlying return
        returns_buy_hold = self.returns.copy()

        equity_strategy = (1.0 + returns_strategy).cumprod()
        equity_buy_hold = (1.0 + returns_buy_hold).cumprod()

        # Classification metrics
        if y_true is None:
            y_true = self.global_y_true

        if y_true is not None:
            y_true = y_true.reindex(self.returns.index)
            # Convert signals to discrete labels in {0,1} if needed
            y_pred_labels = (signals > 0).astype(int)
            acc = accuracy_score(y_true, y_pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred_labels,
                average="binary",
                pos_label=positive_label,
                zero_division=0,
            )
            confusion = confusion_matrix(y_true, y_pred_labels, labels=[0, positive_label])
        else:
            # No true labels provided
            y_true = pd.Series(index=self.returns.index, dtype="float64")
            y_pred_labels = (signals > 0).astype(int)
            acc = np.nan
            precision = np.nan
            recall = np.nan
            f1 = np.nan
            confusion = np.full((2, 2), np.nan)

        mp = ModelPerformance(
            name=name,
            y_true=y_true,
            y_pred=y_pred_labels,
            accuracy=acc,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion=confusion,
            returns_strategy=returns_strategy,
            returns_buy_hold=returns_buy_hold,
            equity_strategy=equity_strategy,
            equity_buy_hold=equity_buy_hold,
        )
        self._models[name] = mp

    def get_model_performance(self, name: str) -> ModelPerformance:
        """
        Retrieve the performance object for a given model.

        Parameters
        ----------
        name : str
            Model name.

        Returns
        -------
        ModelPerformance
        """
        if name not in self._models:
            raise KeyError(f"Model {name!r} has not been added.")
        return self._models[name]

    def summary_table(self) -> pd.DataFrame:
        """
        Return a summary table with classification metrics and final equity
        for each registered model.

        Returns
        -------
        pandas.DataFrame
            One row per model, columns include:
            ["accuracy", "precision", "recall", "f1",
             "final_equity_strategy", "final_equity_buy_hold"]
        """
        rows = []
        for name, mp in self._models.items():
            rows.append(
                {
                    "model": name,
                    "accuracy": mp.accuracy,
                    "precision": mp.precision,
                    "recall": mp.recall,
                    "f1": mp.f1,
                    "final_equity_strategy": mp.equity_strategy.iloc[-1],
                    "final_equity_buy_hold": mp.equity_buy_hold.iloc[-1],
                }
            )
        return pd.DataFrame(rows).set_index("model")

    def equity_curves(self) -> pd.DataFrame:
        """
        Return a DataFrame of equity curves for all models plus buy-and-hold.

        Returns
        -------
        pandas.DataFrame
            Columns: one per model (strategy equity) + "BuyHold".
        """
        eq = pd.DataFrame(index=self.returns.index)
        # All models share the same buy-and-hold curve, so pick one
        buy_hold = None
        for name, mp in self._models.items():
            eq[name] = mp.equity_strategy
            if buy_hold is None:
                buy_hold = mp.equity_buy_hold
        if buy_hold is not None:
            eq["BuyHold"] = buy_hold
        return eq

    # ------------------------------------------------------------------
    # Plotting helper: signals + profit
    # ------------------------------------------------------------------
    def plot_signals_and_equity(
        self,
        model_name: str,
        figsize: tuple[float, float] = (12.0, 8.0),
        show_buy_sell_markers: bool = True,
        title: Optional[str] = None,
    ) -> Figure:
        """
        Plot price with buy/sell markers and equity curve for a given model.

        Top panel:
        ----------
        - Price (line)
        - Buy markers where signal goes from <= 0 to > 0
        - Sell markers where signal goes from > 0 to <= 0

        Bottom panel:
        -------------
        - Strategy equity curve
        - Buy-and-hold equity curve

        Parameters
        ----------
        model_name : str
            Name of the model to plot (must have been added via `add_model`).
        figsize : (float, float), optional
            Figure size.
        show_buy_sell_markers : bool, optional
            If True, draw markers at buy & sell transitions.
        title : str or None, optional
            Figure title. If None, a default will be used.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if model_name not in self._models:
            raise KeyError(f"Model {model_name!r} has not been added.")

        mp = self._models[model_name]

        fig, (ax_price, ax_equity) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, constrained_layout=True
        )

        # ---- Top panel: price + buy/sell markers ----
        ax_price.plot(self.price.index, self.price, label="Price")

        if show_buy_sell_markers:
            # Reconstruct signals from y_pred (1 = long, 0 = flat)
            signals = mp.y_pred.reindex(self.price.index).fillna(0)
            signals_shift = signals.shift(1).fillna(0)

            # Buy where we go from <=0 to >0
            buy_points = (signals > 0) & (signals_shift <= 0)
            # Sell where we go from >0 to <=0
            sell_points = (signals <= 0) & (signals_shift > 0)

            ax_price.scatter(
                self.price.index[buy_points],
                self.price[buy_points],
                marker="^",
                color="green",
                label="Buy",
                zorder=5,
            )
            ax_price.scatter(
                self.price.index[sell_points],
                self.price[sell_points],
                marker="v",
                color="red",
                label="Sell",
                zorder=5,
            )

        ax_price.set_ylabel("Price")
        ax_price.legend(loc="best")
        ax_price.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # ---- Bottom panel: equity curves ----
        ax_equity.plot(mp.equity_strategy.index, mp.equity_strategy, label=f"{model_name} strategy")
        ax_equity.plot(mp.equity_buy_hold.index, mp.equity_buy_hold, label="Buy & Hold", linestyle="--")

        ax_equity.set_ylabel("Equity")
        ax_equity.set_xlabel("Time")
        ax_equity.legend(loc="best")
        ax_equity.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        if title is None:
            title = f"Model: {model_name} – Signals & Strategy Performance"
        fig.suptitle(title)

        # Rotate x labels for readability
        for label in ax_equity.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

        return fig
