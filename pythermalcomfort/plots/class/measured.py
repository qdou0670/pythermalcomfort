"""Measured-data helpers that compose with existing RangeScene plots."""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from pythermalcomfort.models import pmv_ppd_iso

from summary_plot import SummaryPlot


# ═══════════════════════════════════════════════════════════
#  Protocol
# ═══════════════════════════════════════════════════════════


@runtime_checkable
class HasAxes(Protocol):
    """Any object that exposes a matplotlib ``Axes`` via ``.ax``."""

    @property
    def ax(self) -> Axes: ...


# ── Constants ───────────────────────────────────────────────

REQUIRED_COLUMNS: tuple[str, ...] = (
    "tdb", "rh", "tr", "vr", "met", "clo", "wme",
)

_DEFAULT_BINS: tuple[float, ...] = (float("-inf"), -0.5, 0.5, float("inf"))
_DEFAULT_LABELS: tuple[str, ...] = ("Cold", "Neutral", "Hot")

_DEFAULT_OVERLAY_PALETTE: tuple[str, ...] = ("#4c78a8", "#54a24b", "#e15759")
_DEFAULT_SUMMARY_PALETTE: tuple[str, ...] = ("#4c78a8", "#d9d9d9", "#c9415f")


# ═══════════════════════════════════════════════════════════
#  Layout configuration
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LayoutConfig:
    """
    Controls how :func:`summary` splits the figure into a main axes
    and a right-hand summary panel.

    All ratios are relative to the *original* axes bounds.
    """

    left_ratio: float = 0.62
    """Fraction of original width retained by *ax_left*."""

    gap_ratio: float = 0.08
    """Gap between *ax_left* and *ax_right*, as fraction of original width."""

    max_gap: float = 0.04
    """Absolute maximum gap in figure coordinates."""

    panel_y_offset: float = 0.28
    """Vertical offset of the panel, as fraction of original height."""

    panel_height_ratio: float = 0.36
    """Height of the panel, as fraction of original height."""


_DEFAULT_LAYOUT = LayoutConfig()

_DEFAULT_VERTICAL_LAYOUT = LayoutConfig(
    left_ratio=0.85,
    gap_ratio=0.03,
    max_gap=0.02,
    panel_y_offset=0.05,
    panel_height_ratio=0.90,
)


# ═══════════════════════════════════════════════════════════
#  Result dataclass
# ═══════════════════════════════════════════════════════════


@dataclass
class MeasuredSummaryResult:
    """
    Return value of :func:`summary`.

    Holds every mutable reference created during the overlay so that
    the caller can cleanly undo / replace it.
    """

    fig: Figure
    ax_left: Axes
    ax_right: Axes
    plot: HasAxes
    data: pd.DataFrame
    category_percentages: pd.Series
    scatter_artists: list[PathCollection]
    original_bounds: tuple[float, float, float, float]
    _layout_managed: bool = field(default=True, repr=False)

    # ── Lifecycle helpers ───────────────────────────────────

    def clear_overlay(self) -> None:
        """Remove scatter artists from *ax_left*."""
        for artist in self.scatter_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.scatter_artists.clear()

    def clear_summary(self) -> None:
        """
        Undo the summary panel.

        - **Auto-layout** (``_layout_managed=True``): remove *ax_right*
          from the figure and restore *ax_left* to its original size.
        - **User-managed** (``_layout_managed=False``): clear the
          content of *ax_right* but leave the axes in the figure.
        """
        if self._layout_managed:
            if self.ax_right in self.fig.axes:
                self.ax_right.remove()
            self.ax_left.set_position(list(self.original_bounds))
        else:
            self.ax_right.cla()

    def clear(self) -> None:
        """Remove all measured overlays and restore layout."""
        self.clear_overlay()
        self.clear_summary()


# ═══════════════════════════════════════════════════════════
#  Composable public functions
# ═══════════════════════════════════════════════════════════


def compute_pmv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ISO-7730 PMV for every row.

    Returns a **new** DataFrame with an added ``pmv`` column.
    Tries vectorised evaluation first; falls back to row-wise ``apply``.
    """
    _validate_columns(df)
    out = df.copy()

    try:
        result = pmv_ppd_iso(
            tdb=out["tdb"].values,
            tr=out["tr"].values,
            vr=out["vr"].values,
            rh=out["rh"].values,
            met=out["met"].values,
            clo=out["clo"].values,
            wme=out["wme"].values,
        )
        raw = result.pmv if hasattr(result, "pmv") else result
        out["pmv"] = np.asarray(raw, dtype=float)

    except (TypeError, ValueError, AttributeError):
        warnings.warn(
            "Vectorised pmv_ppd_iso call failed — falling back to "
            "row-wise evaluation.  This is slower but produces "
            "identical results.",
            stacklevel=2,
        )
        out["pmv"] = out.apply(
            lambda r: pmv_ppd_iso(
                tdb=r["tdb"], tr=r["tr"], vr=r["vr"],
                rh=r["rh"], met=r["met"], clo=r["clo"], wme=r["wme"],
            ).pmv,
            axis=1,
        )

    return out


def categorize_pmv(
    df: pd.DataFrame,
    bins: Sequence[float] = _DEFAULT_BINS,
    labels: Sequence[str] = _DEFAULT_LABELS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Bin the ``pmv`` column into categories.

    Returns ``(df_out, category_percentages)``.
    """
    if "pmv" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'pmv' column.  "
            "Call compute_pmv() first."
        )
    bins_list = list(bins)
    labels_list = list(labels)
    if len(bins_list) < 2:
        raise ValueError("bins must contain at least two edges.")
    if len(labels_list) != len(bins_list) - 1:
        raise ValueError("labels must have exactly len(bins) - 1 elements.")

    out = df.copy()
    out["category"] = pd.cut(
        out["pmv"], bins=bins_list, labels=labels_list, right=False,
    )
    pct = (
        out["category"]
        .value_counts(normalize=True)
        .reindex(labels_list, fill_value=0.0)
        .mul(100)
        .round(1)
    )
    return out, pct


def overlay_scatter(
    ax: Axes,
    df: pd.DataFrame,
    category_labels: Sequence[str],
    colors: Mapping[str, str] | None = None,
    default_palette: Sequence[str] = _DEFAULT_OVERLAY_PALETTE,
    scatter_kws: Mapping[str, Any] | None = None,
) -> list[PathCollection]:
    """Draw categorised scatter points on *ax*."""
    for col in ("tdb", "rh", "category"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column.")

    resolved = resolve_category_colors(colors, category_labels, default_palette)

    defaults: dict[str, Any] = {"edgecolor": "black", "s": 70}
    if scatter_kws:
        defaults.update(dict(scatter_kws))

    artists: list[PathCollection] = []
    for cat in category_labels:
        subset = df[df["category"] == cat]
        if subset.empty:
            continue
        artist = ax.scatter(
            subset["tdb"], subset["rh"],
            **{**defaults, "color": resolved[cat]},
        )
        artists.append(artist)

    return artists


def clear_artists(artists: list[PathCollection]) -> None:
    """Remove *artists* from their axes and empty the list (in-place)."""
    for artist in artists:
        try:
            artist.remove()
        except ValueError:
            pass
    artists.clear()


def resolve_category_colors(
    user_colors: Mapping[str, str] | None,
    category_labels: Sequence[str],
    default_palette: Sequence[str],
) -> dict[str, str]:
    """Merge user overrides onto a cycling default palette."""
    resolved = {
        label: color
        for label, color in zip(
            category_labels, cycle(default_palette), strict=False,
        )
    }
    if user_colors is not None:
        resolved.update(dict(user_colors))
    return resolved


# ═══════════════════════════════════════════════════════════
#  Convenience wrapper
# ═══════════════════════════════════════════════════════════


def summary(
    df: pd.DataFrame,
    plot: HasAxes,
    *,
    category_bins: Sequence[float] = _DEFAULT_BINS,
    category_labels: Sequence[str] = _DEFAULT_LABELS,
    overlay_colors: Mapping[str, str] | None = None,
    summary_colors: Mapping[str, str] | None = None,
    scatter_kws: Mapping[str, Any] | None = None,
    summary_title: str = "Measured PMV Summary",
    previous: MeasuredSummaryResult | None = None,
    auto_display: bool = False,
    ax_right: Axes | None = None,
    layout: LayoutConfig | None = None,
    orientation: str = "horizontal",
) -> MeasuredSummaryResult:
    """
    Overlay measured PMV scatter and a summary panel onto an existing plot.

    Parameters
    ----------
    ax_right : Axes or None
        User-managed axes for the summary panel.  When provided,
        *ax_left* is **not** resized and no new axes are created.
        Recommended for multi-plot figures.
    orientation : "horizontal" or "vertical"
        Orientation of the summary bar chart.  When using auto-layout
        (``ax_right=None``), the panel shape adapts automatically.
    layout : LayoutConfig or None
        Controls auto-layout proportions.  If *None* (default),
        a sensible default is chosen based on *orientation*.
        Ignored when *ax_right* is provided.
    previous : MeasuredSummaryResult or None
        If given, ``previous.clear()`` is called first.
    auto_display : bool
        If *True*, call ``display(fig)`` in Jupyter.  Default *False*.
    """
    # ── Validate ────────────────────────────────────────────
    ax_left = _validate_plot_result(plot)
    fig = ax_left.figure

    # ── Clean up previous ───────────────────────────────────
    if previous is not None:
        previous.clear()
        prev_bounds = previous.original_bounds
    else:
        prev_bounds = None

    # ── Step 1 + 2: compute & categorize ────────────────────
    df_out = compute_pmv(df)
    df_out, pct = categorize_pmv(df_out, category_bins, category_labels)

    # ── Step 3: scatter overlay ─────────────────────────────
    scatter_artists = overlay_scatter(
        ax_left, df_out, category_labels,
        colors=overlay_colors, scatter_kws=scatter_kws,
    )

    # ── Step 4: summary panel ───────────────────────────────
    if ax_right is not None:
        # User-managed: no layout changes
        original_bounds = tuple(ax_left.get_position().bounds)
        layout_managed = False
    else:
        '''# Auto-layout
        ax_right, original_bounds = _prepare_summary_layout(
            fig, ax_left, prev_bounds, layout,
        )
        layout_managed = True'''

        if layout is None:
            layout = (
                _DEFAULT_VERTICAL_LAYOUT
                if orientation == "vertical"
                else _DEFAULT_LAYOUT
            )
        ax_right, original_bounds = _prepare_summary_layout(
            fig, ax_left, prev_bounds, layout,
        )
        layout_managed = True

    resolved_summary = resolve_category_colors(
        summary_colors, category_labels, _DEFAULT_SUMMARY_PALETTE,
    )
    colors_list = [resolved_summary[lbl] for lbl in category_labels]

    SummaryPlot(df_out).plot(
        target_column="pmv",
        bins=list(category_bins),
        labels=list(category_labels),
        colors=colors_list,
        ax=ax_right,
        title=summary_title,
        orientation=orientation,
    )

    # ── Display ─────────────────────────────────────────────
    if auto_display:
        _display_figure_if_notebook(fig)

    return MeasuredSummaryResult(
        fig=fig,
        ax_left=ax_left,
        ax_right=ax_right,
        plot=plot,
        data=df_out,
        category_percentages=pct,
        scatter_artists=scatter_artists,
        original_bounds=original_bounds,
        _layout_managed=layout_managed,
    )


# ═══════════════════════════════════════════════════════════
#  Private helpers
# ═══════════════════════════════════════════════════════════


def _validate_columns(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"df is missing required column(s): {', '.join(missing)}"
        )


def _validate_plot_result(plot: HasAxes) -> Axes:
    ax = getattr(plot, "ax", None)
    if not isinstance(ax, Axes):
        raise TypeError("plot must expose a matplotlib Axes via .ax")
    return ax


def _prepare_summary_layout(
    fig: Figure,
    ax_left: Axes,
    original_bounds: tuple[float, float, float, float] | None = None,
    layout: LayoutConfig = _DEFAULT_LAYOUT,
) -> tuple[Axes, tuple[float, float, float, float]]:
    """Shrink *ax_left* and create a right-hand axes for the summary bar."""
    if original_bounds is None:
        original_bounds = tuple(ax_left.get_position().bounds)

    x0, y0, w, h = original_bounds
    gap = min(layout.max_gap, w * layout.gap_ratio)
    lw = w * layout.left_ratio
    rw = min(max(w - lw - gap, w * 0.22), max(w - lw - gap, 0.0))

    ax_left.set_position([x0, y0, lw, h])
    ax_right = fig.add_axes([
        x0 + lw + gap,
        y0 + h * layout.panel_y_offset,
        rw,
        h * layout.panel_height_ratio,
    ])

    return ax_right, original_bounds


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
