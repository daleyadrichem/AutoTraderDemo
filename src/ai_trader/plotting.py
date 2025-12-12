from __future__ import annotations

from typing import Sequence, List, Optional, Iterable

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class IndicatorPlotter:
    """
    Helper class for plotting price data and technical indicators.

    This class is initialized with a pandas DataFrame (typically coming from
    `IndicatorEngineer.data`) and provides a flexible method to plot multiple
    indicators, arranged in configurable panels (subplots) within a figure.

    Panels are defined as a sequence of lists of column names. Each inner list
    represents one subplot and contains the names of the indicators/columns to
    plot together on that axis.

    Examples
    --------
    Basic usage with default layout (if columns exist):

    >>> plotter = IndicatorPlotter(df, price_col="Close")
    >>> fig = plotter.plot_panels()

    Custom layout: price + MAs on top, MACD in the middle, RSI at the bottom:

    >>> panels = [
    ...     ["Close", "MA20", "MA50"],
    ...     ["MACD", "MACD_Signal", "MACD_Hist"],
    ...     ["RSI"],
    ... ]
    >>> fig = plotter.plot_panels(panels=panels, title="SPY with indicators")

    Only plot MACD-related indicators in a single panel:

    >>> fig = plotter.plot_panels(panels=[["MACD", "MACD_Signal", "MACD_Hist"]])
    """

    def __init__(self, data: pd.DataFrame, price_col: str = "Close") -> None:
        """
        Parameters
        ----------
        data : pandas.DataFrame
            Input data containing price and indicator columns. Index is expected
            to be a datetime index for best results.
        price_col : str, optional
            Name of the main price column (used in default panel configuration).
        """
        if price_col not in data.columns:
            raise ValueError(
                f"price_col={price_col!r} not found in DataFrame columns: {list(data.columns)}"
            )

        self._df = data
        self.price_col = price_col

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns
        -------
        pandas.DataFrame
            The underlying data used for plotting.
        """
        return self._df

    # ---------------------------------------------------------------------
    # Main plotting API
    # ---------------------------------------------------------------------
    def plot_panels(
        self,
        panels: Optional[Sequence[Sequence[str]]] = None,
        figsize: tuple[float, float] = (12.0, 8.0),
        sharex: bool = True,
        title: Optional[str] = None,
        legend: bool = True,
        grid: bool = True,
        macd_hist_alpha: float = 0.4,
    ) -> Figure:
        """
        Plot indicators organized into vertically stacked panels (subplots).

        Parameters
        ----------
        panels : sequence of sequences of str, optional
            Panel configuration. Each inner sequence is a list of column names
            to plot on the same axis. If None, a default layout is used:
            - Panel 1: price_col, MA20, MA50 (if present)
            - Panel 2: MACD, MACD_Signal, MACD_Hist (if present)
            - Panel 3: RSI (if present)
        figsize : (float, float), optional
            Figure size passed to matplotlib.
        sharex : bool, optional
            If True, all subplots share the same x-axis.
        title : str or None, optional
            Global figure title. If None, no title is set.
        legend : bool, optional
            If True, show a legend on each axis.
        grid : bool, optional
            If True, enable grid on each axis.
        macd_hist_alpha : float, optional
            Transparency level for MACD histogram bars.

        Returns
        -------
        matplotlib.figure.Figure
            The created matplotlib Figure object.

        Raises
        ------
        KeyError
            If any of the requested columns are not found in the DataFrame.
        """
        df = self._df

        if panels is None:
            panels = self._default_panels()

        # Filter out panels that end up empty (e.g. indicators not present)
        filtered_panels: List[List[str]] = []
        for panel in panels:
            existing_cols = [c for c in panel if c in df.columns]
            if existing_cols:
                filtered_panels.append(existing_cols)

        if not filtered_panels:
            raise KeyError(
                "No valid indicators found for plotting. "
                "Check your panel configuration and DataFrame columns."
            )

        n_panels = len(filtered_panels)
        fig, axes = plt.subplots(
            n_panels, 1, sharex=sharex, figsize=figsize, constrained_layout=True
        )

        # axes can be a single Axes if n_panels == 1
        if n_panels == 1:
            axes_list = [axes]  # type: ignore[list-item]
        else:
            axes_list = list(axes)

        for ax, panel_cols in zip(axes_list, filtered_panels):
            self._plot_panel(ax, df, panel_cols, grid=grid, legend=legend, macd_hist_alpha=macd_hist_alpha)

        if title is not None:
            fig.suptitle(title)

        # Rotate x-axis labels for readability if index is datetime-like
        if sharex and hasattr(df.index, "to_pydatetime"):
            axes_list[-1].tick_params(axis="x", rotation=30)

        return fig

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _default_panels(self) -> List[List[str]]:
        """
        Construct a default panel configuration based on commonly used columns.

        Returns
        -------
        list of list of str
            Default panel configuration.
        """
        df_cols = set(self._df.columns)
        panels: List[List[str]] = []

        # Panel 1: price and moving averages if they exist
        price_panel = [self.price_col]
        if "MA20" in df_cols:
            price_panel.append("MA20")
        if "MA50" in df_cols:
            price_panel.append("MA50")
        if price_panel:
            panels.append(price_panel)

        # Panel 2: MACD family
        macd_panel: List[str] = []
        for col in ["MACD", "MACD_Signal", "MACD_Hist"]:
            if col in df_cols:
                macd_panel.append(col)
        if macd_panel:
            panels.append(macd_panel)

        # Panel 3: RSI
        if "RSI" in df_cols:
            panels.append(["RSI"])

        return panels

    def _plot_panel(
        self,
        ax: Axes,
        df: pd.DataFrame,
        columns: Sequence[str],
        grid: bool,
        legend: bool,
        macd_hist_alpha: float,
    ) -> None:
        """
        Plot a set of columns on a single Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis on which to plot.
        df : pandas.DataFrame
            Source data.
        columns : sequence of str
            Column names to plot on this axis.
        grid : bool
            Whether to show a grid.
        legend : bool
            Whether to show a legend.
        macd_hist_alpha : float
            Transparency for MACD histogram bars.
        """
        index = df.index

        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column {col!r} not found in DataFrame.")

            # Special handling for some typical indicators
            if col.lower().startswith("macd_hist") or col == "MACD_Hist":
                # MACD histogram as bar plot
                ax.bar(index, df[col], alpha=macd_hist_alpha, label=col)
            elif col.lower().startswith("volume"):
                # Volume as bar plot
                ax.bar(index, df[col], alpha=0.5, label=col)
            else:
                # Default to line plot
                ax.plot(index, df[col], label=col)

        if grid:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        if legend:
            ax.legend(loc="best")

        # Auto-set y-label as a comma-separated list of columns
        ax.set_ylabel(", ".join(columns))
