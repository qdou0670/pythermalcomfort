"""Tests for measured.py"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

from matplotlib.collections import PathCollection

from measured import (
    HasAxes,
    LayoutConfig,
    MeasuredSummaryResult,
    REQUIRED_COLUMNS,
    _DEFAULT_BINS,
    _DEFAULT_LABELS,
    _DEFAULT_LAYOUT,
    categorize_pmv,
    clear_artists,
    compute_pmv,
    overlay_scatter,
    resolve_category_colors,
    summary,
)


# ── Mock models ─────────────────────────────────────────────


@dataclass
class _MockResult:
    pmv: float | np.ndarray


def _mock_vectorized(tdb, tr, vr, rh, met, clo, wme):
    return _MockResult(pmv=(np.asarray(tdb, dtype=float) - 24.0) * 0.5)


def _mock_scalar_only(tdb, tr, vr, rh, met, clo, wme):
    if isinstance(tdb, np.ndarray):
        raise TypeError("Scalar only")
    return _MockResult(pmv=(float(tdb) - 24.0) * 0.5)


# ── Auto-cleanup ────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ── Shared fixtures ─────────────────────────────────────────


@pytest.fixture
def sample_df():
    n = 10
    return pd.DataFrame({
        "tdb": np.linspace(20, 30, n),
        "rh":  np.linspace(30, 70, n),
        "tr":  np.linspace(20, 30, n),
        "vr":  np.full(n, 0.1),
        "met": np.full(n, 1.2),
        "clo": np.full(n, 0.5),
        "wme": np.full(n, 0.0),
    })


@pytest.fixture
def pmv_df(sample_df):
    with patch("measured.pmv_ppd_iso", side_effect=_mock_vectorized):
        return compute_pmv(sample_df)


@pytest.fixture
def categorized_df(pmv_df):
    df, pct = categorize_pmv(pmv_df)
    return df, pct


@pytest.fixture
def mock_plot():
    fig, ax = plt.subplots(figsize=(9, 6))
    obj = MagicMock()
    obj.ax = ax
    return obj


@pytest.fixture
def make_result(sample_df, mock_plot):
    def _make(**overrides):
        with patch("measured.pmv_ppd_iso", side_effect=_mock_vectorized):
            return summary(sample_df, mock_plot, **overrides)
    return _make


# ═══════════════════════════════════════════════════════════
#  HasAxes Protocol
# ═══════════════════════════════════════════════════════════


class TestHasAxesProtocol:
    def test_object_with_ax_satisfies(self):
        obj = MagicMock()
        obj.ax = plt.subplots()[1]
        assert isinstance(obj, HasAxes)

    def test_object_without_ax_fails(self):
        obj = MagicMock(spec=[])
        assert not isinstance(obj, HasAxes)

    def test_string_is_not_has_axes(self):
        assert not isinstance("not_a_plot", HasAxes)


# ═══════════════════════════════════════════════════════════
#  compute_pmv
# ═══════════════════════════════════════════════════════════


class TestComputePmv:
    def test_vectorized_path_single_call(self, sample_df):
        with patch("measured.pmv_ppd_iso", side_effect=_mock_vectorized) as m:
            result = compute_pmv(sample_df)
        m.assert_called_once()
        assert "pmv" in result.columns
        expected = (sample_df["tdb"].values - 24.0) * 0.5
        np.testing.assert_allclose(result["pmv"].values, expected)

    def test_fallback_path_warns(self, sample_df):
        with patch("measured.pmv_ppd_iso", side_effect=_mock_scalar_only):
            with pytest.warns(UserWarning, match="Vectorised.*failed"):
                result = compute_pmv(sample_df)
        expected = (sample_df["tdb"].values - 24.0) * 0.5
        np.testing.assert_allclose(result["pmv"].values, expected)

    def test_does_not_mutate_input(self, sample_df):
        cols_before = list(sample_df.columns)
        with patch("measured.pmv_ppd_iso", side_effect=_mock_vectorized):
            compute_pmv(sample_df)
        assert list(sample_df.columns) == cols_before

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing required column"):
            compute_pmv(pd.DataFrame({"tdb": [22], "rh": [50]}))

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            compute_pmv([1, 2, 3])


# ═══════════════════════════════════════════════════════════
#  categorize_pmv
# ═══════════════════════════════════════════════════════════


class TestCategorizePmv:
    def test_default_bins(self, pmv_df):
        df_out, pct = categorize_pmv(pmv_df)
        assert "category" in df_out.columns
        assert set(pct.index) == set(_DEFAULT_LABELS)
        assert abs(pct.sum() - 100.0) < 0.2

    def test_custom_bins_and_labels(self, pmv_df):
        df_out, pct = categorize_pmv(
            pmv_df,
            bins=[float("-inf"), 0.0, float("inf")],
            labels=["Neg", "Pos"],
        )
        assert set(pct.index) == {"Neg", "Pos"}

    def test_does_not_mutate_input(self, pmv_df):
        cols_before = list(pmv_df.columns)
        categorize_pmv(pmv_df)
        assert list(pmv_df.columns) == cols_before
        assert "category" not in pmv_df.columns

    def test_no_pmv_column_raises(self):
        with pytest.raises(ValueError, match="pmv"):
            categorize_pmv(pd.DataFrame({"x": [1]}))

    def test_bad_bins_raises(self, pmv_df):
        with pytest.raises(ValueError, match="at least two"):
            categorize_pmv(pmv_df, bins=[0.0], labels=[])

    def test_mismatched_labels_raises(self, pmv_df):
        with pytest.raises(ValueError, match="len\\(bins\\)"):
            categorize_pmv(
                pmv_df,
                bins=[float("-inf"), 0.0, float("inf")],
                labels=["A", "B", "C"],
            )


# ═══════════════════════════════════════════════════════════
#  overlay_scatter
# ═══════════════════════════════════════════════════════════


class TestOverlayScatter:
    def test_creates_artists(self, categorized_df):
        df, _ = categorized_df
        _, ax = plt.subplots()
        artists = overlay_scatter(ax, df, _DEFAULT_LABELS)
        assert len(artists) > 0
        assert all(isinstance(a, PathCollection) for a in artists)

    def test_no_monkey_patch_on_axes(self, categorized_df):
        df, _ = categorized_df
        _, ax = plt.subplots()
        overlay_scatter(ax, df, _DEFAULT_LABELS)
        ptc = [a for a in dir(ax) if a.startswith("_ptc_")]
        assert ptc == []

    def test_custom_colors(self, categorized_df):
        df, _ = categorized_df
        _, ax = plt.subplots()
        artists = overlay_scatter(
            ax, df, _DEFAULT_LABELS,
            colors={"Cold": "cyan", "Neutral": "lime", "Hot": "magenta"},
        )
        assert len(artists) > 0

    def test_missing_category_raises(self, sample_df):
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="category"):
            overlay_scatter(ax, sample_df, _DEFAULT_LABELS)

    def test_empty_categories_ok(self):
        df = pd.DataFrame({
            "tdb": [22, 23],
            "rh":  [50, 55],
            "category": pd.Categorical(
                ["Neutral", "Neutral"], categories=list(_DEFAULT_LABELS),
            ),
        })
        _, ax = plt.subplots()
        artists = overlay_scatter(ax, df, _DEFAULT_LABELS)
        assert len(artists) == 1


# ═══════════════════════════════════════════════════════════
#  clear_artists
# ═══════════════════════════════════════════════════════════


class TestClearArtists:
    def test_removes_from_axes(self, categorized_df):
        df, _ = categorized_df
        _, ax = plt.subplots()
        artists = overlay_scatter(ax, df, _DEFAULT_LABELS)
        assert len(ax.collections) > 0
        clear_artists(artists)
        assert len(ax.collections) == 0
        assert artists == []

    def test_idempotent(self, categorized_df):
        df, _ = categorized_df
        _, ax = plt.subplots()
        artists = overlay_scatter(ax, df, _DEFAULT_LABELS)
        clear_artists(artists)
        clear_artists(artists)

    def test_empty_list_noop(self):
        clear_artists([])


# ═══════════════════════════════════════════════════════════
#  resolve_category_colors
# ═══════════════════════════════════════════════════════════


class TestResolveCategoryColors:
    def test_defaults_only(self):
        pal = ("#aaa", "#bbb", "#ccc")
        assert resolve_category_colors(None, ["A", "B", "C"], pal) == {
            "A": "#aaa", "B": "#bbb", "C": "#ccc",
        }

    def test_user_override(self):
        pal = ("#aaa", "#bbb", "#ccc")
        result = resolve_category_colors({"B": "#fff"}, ["A", "B", "C"], pal)
        assert result["B"] == "#fff"
        assert result["A"] == "#aaa"

    def test_cycling(self):
        result = resolve_category_colors(None, ["A", "B", "C"], ("#000",))
        assert all(v == "#000" for v in result.values())


# ═══════════════════════════════════════════════════════════
#  LayoutConfig
# ═══════════════════════════════════════════════════════════


class TestLayoutConfig:
    def test_default_values(self):
        lc = LayoutConfig()
        assert lc.left_ratio == 0.62
        assert lc.gap_ratio == 0.08
        assert lc.max_gap == 0.04
        assert lc.panel_y_offset == 0.28
        assert lc.panel_height_ratio == 0.36

    def test_frozen(self):
        lc = LayoutConfig()



# ═══════════════════════════════════════════════════════════
#  vertical
# ═══════════════════════════════════════════════════════════


class TestSummaryVerticalOrientation:

    def test_horizontal_auto_layout(self, sample_df, mock_plot):
        with patch("measured.pmv_ppd_iso", side_effect=_mock_vectorized):
            result = summary(sample_df, mock_plot, orientation="horizontal")
        # Horizontal bar: x-axis is the percentage axis (0–100)
        assert result.ax_right.get_xlim() == (0.0, 100.0)

    def test_vertical_auto_layout(self, sample_df, mock_plot):
        with patch("measured.pmv_ppd_iso", side_effect=_mock_vectorized):
            result = summary(sample_df, mock_plot, orientation="vertical")
        # Vertical bar: y-axis is the percentage axis (0–100)
        assert result.ax_right.get_ylim() == (0.0, 100.0)

    def test_vertical_with_user_ax(self, sample_df):
        fig, (ax_main, ax_panel) = plt.subplots(1, 2, figsize=(10, 6))
        plot_obj = MagicMock()
        plot_obj.ax = ax_main
        with patch("measured.pmv_ppd_iso", side_effect=_mock_vectorized):
            result = summary(
                sample_df, plot_obj,
                ax_right=ax_panel, orientation="vertical",
            )
        assert result.ax_right is ax_panel
        assert len(ax_panel.patches) > 0

    def test_vertical_leaves_main_wider(self, sample_df, mock_plot):
        """Vertical panel is narrower → main plot retains more width."""
        with patch("measured.pmv_ppd_iso", side_effect=_mock_vectorized):
            r_h = summary(sample_df, mock_plot, orientation="horizontal")

        fig2, ax2 = plt.subplots(figsize=(9, 6))
        plot2 = MagicMock()
        plot2.ax = ax2
        with patch("measured.pmv_ppd_iso", side_effect=_mock_vectorized):
            r_v = summary(sample_df, plot2, orientation="vertical")

        w_h = r_h.ax_left.get_position().bounds[2]
        w_v = r_v.ax_left.get_position().bounds[2]
        assert w_v > w_h  # vertical panel leaves more room for main plot
