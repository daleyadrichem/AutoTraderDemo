from __future__ import annotations

from typing import Sequence, Optional
import pandas as pd
from ai_trader.strategies.base_strategies import BaseMLTradingStrategy
from sklearn.ensemble import RandomForestClassifier


class RandomForestStrategy(BaseMLTradingStrategy[RandomForestClassifier]):
    """
    ML trading strategy using RandomForestClassifier as the classifier.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: Sequence[str],
        price_col: str = "Close",
        train_ratio: float = 0.7,
        target_col: str = "Target_Up_1d",
        n_estimators: int = 200,
        max_depth: Optional[int] = 6,
        random_state: int = 42,
    ) -> None:
        super().__init__(data, feature_cols, price_col, train_ratio, target_col)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _build_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )

