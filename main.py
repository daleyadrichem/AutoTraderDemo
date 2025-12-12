from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ai_trader.data_grabber import download_price_data
from ai_trader.indicators import IndicatorEngineer
from ai_trader.plotting import IndicatorPlotter
from ai_trader.evaluation import ModelPerformanceAnalyzer
from ai_trader.strategies.logic_regression import LogisticRegressionStrategy
from ai_trader.strategies.random_forest import RandomForestStrategy
from ai_trader.strategies.MLP import MLPStrategy

def run_pipeline(
    ticker: str = "SPY",
    period: str = "5y",
    interval: str = "1d",
    train_ratio: float = 0.7,
) -> None:
    """
    End-to-end pipeline:
    - download data
    - build indicators
    - plot indicators
    - train 3 ML trading strategies
    - compare models
    - plot signals + equity curves
    """
    print(f"Downloading data for {ticker} ({period}, {interval})...")
    price_df = download_price_data(
        ticker=ticker,
        interval=interval,
        period=period,
        cache=True,
        force_refresh=False,
    )

    # ---------------------------------------------------------------------
    # 1. Build indicators
    # ---------------------------------------------------------------------
    print("Building indicators...")
    ie = IndicatorEngineer(price_df, price_col="Close")
    ie.add_basic_demo_indicators()
    data_with_indicators = ie.data

    feature_cols = [
        "Return_1d",
        "Return_5d_mean",
        "Return_20d_mean",
        "MA20",
        "MA50",
        "Dist_MA20",
        "Dist_MA50",
        "Volatility_10d",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
    ]

    # ---------------------------------------------------------------------
    # 2. Plot indicators (price + MACD + RSI etc.)
    # ---------------------------------------------------------------------
    print("Plotting indicators...")
    plotter = IndicatorPlotter(data_with_indicators, price_col="Close")
    fig_indicators = plotter.plot_panels(title=f"{ticker} – Price & Indicators")
    plt.show(block=False)

    # ---------------------------------------------------------------------
    # 3. Train strategies
    # ---------------------------------------------------------------------
    print("Training Logistic Regression strategy...")
    logreg_strategy = LogisticRegressionStrategy(
        data_with_indicators,
        feature_cols=feature_cols,
        price_col="Close",
        train_ratio=train_ratio,
    )
    logreg_strategy.prepare_data()
    logreg_strategy.fit_model()
    logreg_result = logreg_strategy.run_backtest()

    print("Training Random Forest strategy...")
    rf_strategy = RandomForestStrategy(
        data_with_indicators,
        feature_cols=feature_cols,
        price_col="Close",
        train_ratio=train_ratio,
    )
    rf_strategy.prepare_data()
    rf_strategy.fit_model()
    rf_result = rf_strategy.run_backtest()

    print("Training MLP strategy...")
    mlp_strategy = MLPStrategy(
        data_with_indicators,
        feature_cols=feature_cols,
        price_col="Close",
        train_ratio=train_ratio,
    )
    mlp_strategy.prepare_data()
    mlp_strategy.fit_model()
    mlp_result = mlp_strategy.run_backtest()

    # All strategies share the same cleaned df after prepare_data()
    df_ml = logreg_strategy._df  # if you like, expose via a property later

    # ---------------------------------------------------------------------
    # 4. Build analyzer on TEST set only
    # ---------------------------------------------------------------------
    print("Building performance analyzer...")

    test_index = logreg_result.X_test.index
    price_test = df_ml.loc[test_index, "Close"]
    returns_test = df_ml.loc[test_index, "Return_1d"]
    y_true_test = logreg_result.y_test  # same target for all strategies

    analyzer = ModelPerformanceAnalyzer(
        returns=returns_test,
        price=price_test,
        y_true=y_true_test,
    )

    # Convert predictions to signals (1 = long, 0 = flat)
    signals_logreg = pd.Series(logreg_result.y_pred, index=test_index)
    signals_rf = pd.Series(rf_result.y_pred, index=test_index)
    signals_mlp = pd.Series(mlp_result.y_pred, index=test_index)

    analyzer.add_model("LogReg", signals=signals_logreg)
    analyzer.add_model("RandomForest", signals=signals_rf)
    analyzer.add_model("MLP", signals=signals_mlp)

    # ---------------------------------------------------------------------
    # 5. Print comparison table
    # ---------------------------------------------------------------------
    print("\nModel comparison:")
    print(analyzer.summary_table())

    # ---------------------------------------------------------------------
    # 6. Plot signals + equity per model
    # ---------------------------------------------------------------------
    print("Plotting model signals and equity curves...")

    fig_logreg = analyzer.plot_signals_and_equity(
        "LogReg",
        title=f"{ticker} – Logistic Regression Signals & Performance",
    )
    plt.show(block=False)

    fig_rf = analyzer.plot_signals_and_equity(
        "RandomForest",
        title=f"{ticker} – Random Forest Signals & Performance",
    )
    plt.show(block=False)

    fig_mlp = analyzer.plot_signals_and_equity(
        "MLP",
        title=f"{ticker} – MLP Signals & Performance",
    )
    plt.show(block=False)

    # ---------------------------------------------------------------------
    # 7. Plot all equity curves together
    # ---------------------------------------------------------------------
    print("Plotting combined equity curves...")
    eq_all = analyzer.equity_curves()

    plt.figure(figsize=(10, 6))
    for col in eq_all.columns:
        plt.plot(eq_all.index, eq_all[col], label=col)
    plt.title(f"{ticker} – Equity Curves (All Models vs Buy & Hold)")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    print("Done.")


def main() -> None:
    # You can later hook argparse here; for now hard-coded defaults:
    run_pipeline(
        ticker="SPY",
        period="5y",
        interval="1d",
        train_ratio=0.7,
    )


if __name__ == "__main__":
    main()