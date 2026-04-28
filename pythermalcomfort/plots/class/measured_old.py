"""Measured-data helpers that compose with existing RangeScene plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import cycle
from typing import Any

import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from pythermalcomfort.models import pmv_ppd_iso


@dataclass
class MeasuredSummaryResult:
    fig: Figure
    ax_left: Axes
    ax_right: Axes
    plot: Any
    data: pd.DataFrame
    category_percentages: pd.Series
    scatter_artists: list[PathCollection]


def _validate_measured_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = ["tdb", "rh", "tr", "vr", "met", "clo", "wme"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"df is missing required column(s): {missing_str}")


def _validate_plot_result(plot: Any) -> Axes:
    ax = getattr(plot, "ax", None)
    if not isinstance(ax, Axes):
        raise TypeError("plot must provide a matplotlib Axes via plot.ax.")
    return ax


def _validate_category_inputs(
    category_bins: Sequence[float],
    category_labels: Sequence[str],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    bins = tuple(category_bins)
    labels = tuple(category_labels)

    if len(bins) < 2:
        raise ValueError("category_bins must contain at least two bin edges.")
    if len(labels) != len(bins) - 1:
        raise ValueError(
            "category_labels must contain exactly len(category_bins) - 1 labels."
        )

    return bins, labels


def _compute_measured_pmv(df_copy: pd.DataFrame) -> pd.DataFrame:
    df_copy["pmv"] = df_copy.apply(
        lambda row: pmv_ppd_iso(
            tdb=row["tdb"],
            tr=row["tr"],
            vr=row["vr"],
            rh=row["rh"],
            met=row["met"],
            clo=row["clo"],
            wme=row["wme"],
        ).pmv,
        axis=1,
    )
    return df_copy


def _categorize_measured_pmv(
    df_copy: pd.DataFrame,
    category_bins: Sequence[float],
    category_labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    df_copy["category"] = pd.cut(
        df_copy["pmv"],
        bins=category_bins,
        labels=category_labels,
        right=False,
    )

    category_percentages = (
        df_copy["category"]
        .value_counts(normalize=True)
        .reindex(category_labels, fill_value=0.0)
        .mul(100)
        .round(1)
    )

    return df_copy, category_percentages


def _resolve_category_colors(
    user_colors: Mapping[str, str] | None,
    category_labels: Sequence[str],
    default_palette: Sequence[str],
) -> dict[str, str]:
    resolved = {
        label: color
        for label, color in zip(category_labels, cycle(default_palette), strict=False)
    }
    if user_colors is not None:
        resolved.update(dict(user_colors))
    return resolved


def _is_light_color(color: str) -> bool:
    red, green, blue = mcolors.to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return luminance > 0.7


def _clear_previous_measured_overlay(ax_left: Axes) -> None:
    previous = getattr(ax_left, "_ptc_measured_scatter_artists", [])
    for artist in previous:
        try:
            artist.remove()
        except ValueError:
            continue
    ax_left._ptc_measured_scatter_artists = []


def _prepare_summary_layout(fig: Figure, ax_left: Axes) -> Axes:
    for axis in list(fig.axes):
        if getattr(axis, "_ptc_measured_summary_axis", False):
            axis.remove()

    original_bounds = getattr(ax_left, "_ptc_measured_original_bounds", None)
    if original_bounds is None:
        original_bounds = tuple(ax_left.get_position().bounds)
        ax_left._ptc_measured_original_bounds = original_bounds

    x0, y0, width, height = original_bounds
    gap = min(0.04, width * 0.08)
    left_width = width * 0.62
    right_width = max(width - left_width - gap, width * 0.22)
    max_right_width = max(width - left_width - gap, 0.0)
    right_width = min(right_width, max_right_width)

    ax_left.set_position([x0, y0, left_width, height])

    right_x0 = x0 + left_width + gap
    right_y0 = y0 + height * 0.28
    right_height = height * 0.36
    ax_right = fig.add_axes([right_x0, right_y0, right_width, right_height])
    ax_right._ptc_measured_summary_axis = True
    return ax_right


def _plot_measured_overlay(
    ax_left: Axes,
    df_copy: pd.DataFrame,
    overlay_colors: Mapping[str, str],
    category_labels: Sequence[str],
    scatter_kws: Mapping[str, Any] | None,
) -> list[PathCollection]:
    scatter_defaults: dict[str, Any] = {"edgecolor": "black", "s": 70}
    if scatter_kws is not None:
        scatter_defaults.update(dict(scatter_kws))

    artists: list[PathCollection] = []
    for category in category_labels:
        subset = df_copy[df_copy["category"] == category]
        if subset.empty:
            continue

        artist = ax_left.scatter(
            subset["tdb"],
            subset["rh"],
            **{
                **scatter_defaults,
                "color": overlay_colors[category],
            },
        )
        artist._ptc_measured_overlay = True
        artists.append(artist)

    ax_left._ptc_measured_scatter_artists = artists
    return artists


def _plot_summary_panel(
    ax_right: Axes,
    category_percentages: pd.Series,
    category_labels: Sequence[str],
    summary_colors: Mapping[str, str],
    summary_title: str,
) -> None:
    ax_right.set_title(summary_title, fontsize=13, pad=10)
    ax_right.set_xlim(0, 100)
    ax_right.set_ylim(-0.6, 0.6)
    ax_right.set_xticks([])
    ax_right.set_yticks([])

    for spine in ax_right.spines.values():
        spine.set_visible(False)

    left = 0.0
    bar_y = 0.0
    bar_height = 0.36

    for category in category_labels:
        value = float(category_percentages[category])
        color = summary_colors[category]

        ax_right.barh(
            y=bar_y,
            width=value,
            left=left,
            height=bar_height,
            color=color,
            edgecolor="white",
            linewidth=1.0,
        )

        if value > 0:
            text_color = "black" if _is_light_color(color) else "white"
            ax_right.text(
                left + value / 2,
                bar_y,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )

        left += value

    label_left = 0.0
    for category in category_labels:
        value = float(category_percentages[category])
        if value > 0:
            label_color = (
                "dimgray"
                if _is_light_color(summary_colors[category])
                else summary_colors[category]
            )
            ax_right.text(
                label_left + value / 2,
                0.34,
                category,
                ha="center",
                va="bottom",
                fontsize=11,
                color=label_color,
            )
        label_left += value


def _display_figure_if_notebook(fig: Figure) -> None:
    canvas = getattr(fig, "canvas", None)
    if canvas is not None:
        try:
            canvas.draw_idle()
        except Exception:
            pass

    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return

    shell = get_ipython()
    if shell is None or getattr(shell, "kernel", None) is None:
        return

    display(fig)


def summary(
    df: pd.DataFrame,
    plot: Any,
    *,
    category_bins: Sequence[float] = (-float("inf"), -0.5, 0.5, float("inf")),
    category_labels: Sequence[str] = ("Cold", "Neutral", "Hot"),
    overlay_colors: Mapping[str, str] | None = None,
    summary_colors: Mapping[str, str] | None = None,
    scatter_kws: Mapping[str, Any] | None = None,
    summary_title: str = "Measured PMV Summary",
) -> MeasuredSummaryResult:
    """Overlay measured PMV points and add a compact summary panel to an existing plot."""

    _validate_measured_dataframe(df)
    ax_left = _validate_plot_result(plot)
    bins, labels = _validate_category_inputs(category_bins, category_labels)

    df_copy = df.copy()
    df_copy = _compute_measured_pmv(df_copy)
    df_copy, category_percentages = _categorize_measured_pmv(df_copy, bins, labels)

    fig = ax_left.figure
    ax_right = _prepare_summary_layout(fig, ax_left)
    _clear_previous_measured_overlay(ax_left)

    resolved_overlay_colors = _resolve_category_colors(
        user_colors=overlay_colors,
        category_labels=labels,
        default_palette=("#4c78a8", "#54a24b", "#e15759"),
    )
    resolved_summary_colors = _resolve_category_colors(
        user_colors=summary_colors,
        category_labels=labels,
        default_palette=("#4c78a8", "#d9d9d9", "#c9415f"),
    )

    scatter_artists = _plot_measured_overlay(
        ax_left=ax_left,
        df_copy=df_copy,
        overlay_colors=resolved_overlay_colors,
        category_labels=labels,
        scatter_kws=scatter_kws,
    )
    _plot_summary_panel(
        ax_right=ax_right,
        category_percentages=category_percentages,
        category_labels=labels,
        summary_colors=resolved_summary_colors,
        summary_title=summary_title,
    )
    _display_figure_if_notebook(fig)

    return MeasuredSummaryResult(
        fig=fig,
        ax_left=ax_left,
        ax_right=ax_right,
        plot=plot,
        data=df_copy,
        category_percentages=category_percentages,
        scatter_artists=scatter_artists,
    )
