"""Independent summary plot for categorical data distribution."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors


# ── Shared colour utility ───────────────────────────────────


def is_light_color(color: str) -> bool:
    """Return *True* when overlaid text should be black (light background)."""
    r, g, b = mcolors.to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.7


# ── Result dataclass ────────────────────────────────────────


@dataclass
class SummaryPlotResult:
    """Return value of :meth:`SummaryPlot.plot`."""

    ax: plt.Axes
    category_percentages: pd.Series


# ── Main class ──────────────────────────────────────────────


class SummaryPlot:
    """
    Stacked bar chart for categorical data distribution.

    Supports both horizontal (default) and vertical orientation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        target_column: str | None = None,
        bins: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
        orientation: str | None = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        self._data = data
        self._target_column = target_column
        self._bins = list(bins) if bins is not None else None
        self._labels = list(labels) if labels is not None else None
        self._colors = list(colors) if colors is not None else None
        self._orientation = orientation
        self._last_result: SummaryPlotResult | None = None

    # ── Data management ─────────────────────────────────────

    def update(self, data: pd.DataFrame) -> SummaryPlot:
        """Replace the source data.  Returns *self* for chaining."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        self._data = data
        self._last_result = None
        return self

    @property
    def last_result(self) -> SummaryPlotResult | None:
        """The result from the most recent ``plot()`` call, or *None*."""
        return self._last_result

    # ── Public API ──────────────────────────────────────────

    def plot(
        self,
        target_column: str | None = None,
        bins: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
        ax: plt.Axes | None = None,
        title: str | None = "Category Distribution",
        show_percentages: bool = True,
        orientation: str | None = None,
    ) -> SummaryPlotResult:
        """
        Render a stacked bar chart.

        Parameters
        ----------
        orientation : "horizontal" or "vertical" or None
            None falls back to the value stored in ``__init__``,
            then to ``"horizontal"``.
        """
        # Resolve: param → stored → default
        tc = target_column or self._target_column
        b = list(bins) if bins is not None else self._bins
        lb = list(labels) if labels is not None else self._labels
        cl = list(colors) if colors is not None else self._colors
        ori = orientation or self._orientation or "horizontal"

        if tc is None:
            raise ValueError(
                "target_column must be provided in __init__() or plot()."
            )
        if b is None or lb is None or cl is None:
            raise ValueError(
                "bins, labels, and colors must all be provided "
                "in __init__() or plot()."
            )
        if tc not in self._data.columns:
            raise ValueError(f"Column '{tc}' not found in dataframe.")
        if ori not in ("horizontal", "vertical"):
            raise ValueError(
                f"orientation must be 'horizontal' or 'vertical', got '{ori}'."
            )

        if ax is None:
            figsize = (8, 2) if ori == "horizontal" else (2.5, 6)
            _, ax = plt.subplots(figsize=figsize)

        # Categorize — local Series, self._data is never mutated
        categories = pd.cut(
            self._data[tc], bins=b, labels=lb, right=False,
        )
        pct = (
            categories.value_counts(normalize=True)
            .reindex(lb)
            .fillna(0.0)
            * 100
        )

        if ori == "horizontal":
            self._setup_axes_h(ax, title)
            self._render_bar_h(ax, pct, lb, cl, show_percentages)
        else:
            self._setup_axes_v(ax, title)
            self._render_bar_v(ax, pct, lb, cl, show_percentages)

        result = SummaryPlotResult(ax=ax, category_percentages=pct)
        self._last_result = result
        return result

    # ── Horizontal rendering ────────────────────────────────

    @staticmethod
    def _setup_axes_h(ax: plt.Axes, title: str | None) -> None:
        if title:
            ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    @staticmethod
    def _render_bar_h(
        ax: plt.Axes,
        pct: pd.Series,
        labels: Sequence[str],
        colors: Sequence[str],
        show_percentages: bool,
    ) -> None:
        left = 0.0
        bar_y, bar_h = 0.0, 0.4

        for label, color in zip(labels, colors):
            value = float(pct[label])
            ax.barh(
                y=bar_y, width=value, left=left, height=bar_h,
                color=color, edgecolor="white", linewidth=1.0,
            )
            if value > 0:
                if show_percentages:
                    tc = "black" if is_light_color(color) else "white"
                    ax.text(
                        left + value / 2, bar_y, f"{value:.1f}%",
                        ha="center", va="center",
                        fontsize=12, fontweight="bold", color=tc,
                    )
                lc = "dimgray" if is_light_color(color) else color
                ax.text(
                    left + value / 2, bar_y - 0.28, label,
                    ha="center", va="top", fontsize=11, color=lc,
                )
            left += value

    # ── Vertical rendering ──────────────────────────────────

    @staticmethod
    def _setup_axes_v(ax: plt.Axes, title: str | None) -> None:
        if title:
            ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlim(-0.5, 0.9)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    @staticmethod
    def _render_bar_v(
        ax: plt.Axes,
        pct: pd.Series,
        labels: Sequence[str],
        colors: Sequence[str],
        show_percentages: bool,
    ) -> None:
        bottom = 0.0
        bar_x, bar_w = 0.0, 0.4

        for label, color in zip(labels, colors):
            value = float(pct[label])
            ax.bar(
                x=bar_x, height=value, bottom=bottom, width=bar_w,
                color=color, edgecolor="white", linewidth=1.0,
            )
            if value > 0:
                if show_percentages:
                    tc = "black" if is_light_color(color) else "white"
                    ax.text(
                        bar_x, bottom + value / 2, f"{value:.1f}%",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color=tc,
                        rotation=90,
                    )
                lc = "dimgray" if is_light_color(color) else color
                ax.text(
                    bar_x + 0.30, bottom + value / 2, label,
                    ha="left", va="center", fontsize=9, color=lc,
                )
            bottom += value
