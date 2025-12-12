from __future__ import annotations

from typing import Sequence
import pandas as pd
from ai_trader.strategies.base_strategies import BaseMLTradingStrategy
from sklearn.linear_model import LogisticRegression

class LogisticRegressionStrategy(BaseMLTradingStrategy[LogisticRegression]):
    """
    ML trading strategy using LogisticRegression as the classifier.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: Sequence[str],
        price_col: str = "Close",
        train_ratio: float = 0.7,
        target_col: str = "Target_Up_1d",
        max_iter: int = 1000,
        C: float = 1.0,
    ) -> None:
        super().__init__(data, feature_cols, price_col, train_ratio, target_col)
        self.max_iter = max_iter
        self.C = C

    def _build_model(self) -> LogisticRegression:
        return LogisticRegression(max_iter=self.max_iter, C=self.C)

