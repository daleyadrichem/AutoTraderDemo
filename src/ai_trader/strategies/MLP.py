from __future__ import annotations

from typing import Sequence
import pandas as pd
from ai_trader.strategies.base_strategies import BaseMLTradingStrategy
from sklearn.neural_network import MLPClassifier


class MLPStrategy(BaseMLTradingStrategy[MLPClassifier]):
    """
    ML trading strategy using a simple feedforward neural network (MLPClassifier).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: Sequence[str],
        price_col: str = "Close",
        train_ratio: float = 0.7,
        target_col: str = "Target_Up_1d",
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        activation: str = "relu",
        max_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        super().__init__(data, feature_cols, price_col, train_ratio, target_col)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state

    def _build_model(self) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )