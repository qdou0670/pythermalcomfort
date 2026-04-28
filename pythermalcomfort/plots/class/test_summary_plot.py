"""Tests for summary_plot.py"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from summary_plot import SummaryPlot, SummaryPlotResult, is_light_color


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {"pmv": [-1.0] * 4 + [0.0] * 4 + [1.0] * 2}
    )


@pytest.fixture
def bins():
    return [float("-inf"), -0.5, 0.5, float("inf")]


@pytest.fixture
def labels():
    return ["Cold", "Neutral", "Hot"]


@pytest.fixture
def colors():
    return ["#4c78a8", "#d9d9d9", "#e15759"]


# ═══════════════════════════════════════════════════════════
#  is_light_color
# ═══════════════════════════════════════════════════════════


class TestIsLightColor:
    def test_white_is_light(self):
        assert is_light_color("white") is True

    def test_black_is_dark(self):
        assert is_light_color("black") is False

    def test_light_gray(self):
        assert is_light_color("#d9d9d9") is True

    def test_dark_blue(self):
        assert is_light_color("#4c78a8") is False


# ═══════════════════════════════════════════════════════════
#  __init__
# ═══════════════════════════════════════════════════════════


class TestSummaryPlotInit:
    def test_accepts_dataframe(self, sample_df):
        sp = SummaryPlot(sample_df)
        assert sp._data is sample_df

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            SummaryPlot([1, 2, 3])

    def test_stores_config(self, sample_df, bins, labels, colors):
        sp = SummaryPlot(
            sample_df,
            target_column="pmv",
            bins=bins,
            labels=labels,
            colors=colors,
        )
        assert sp._target_column == "pmv"
        assert sp._bins == bins
        assert sp._labels == labels
        assert sp._colors == colors

    def test_defaults_none(self, sample_df):
        sp = SummaryPlot(sample_df)
        assert sp._target_column is None
        assert sp._bins is None
        assert sp._labels is None
        assert sp._colors is None


# ═══════════════════════════════════════════════════════════
#  plot — basic
# ═══════════════════════════════════════════════════════════


class TestSummaryPlotPlot:
    def test_returns_result_type(self, sample_df, bins, labels, colors):
        result = SummaryPlot(sample_df).plot("pmv", bins, labels, colors)
        assert isinstance(result, SummaryPlotResult)
        assert isinstance(result.ax, plt.Axes)
        assert isinstance(result.category_percentages, pd.Series)

    def test_percentages_sum_to_100(self, sample_df, bins, labels, colors):
        pct = SummaryPlot(sample_df).plot("pmv", bins, labels, colors).category_percentages
        assert abs(pct.sum() - 100.0) < 0.1

    def test_correct_percentages(self, sample_df, bins, labels, colors):
        pct = SummaryPlot(sample_df).plot("pmv", bins, labels, colors).category_percentages
        assert abs(pct["Cold"] - 40.0) < 0.1
        assert abs(pct["Neutral"] - 40.0) < 0.1
        assert abs(pct["Hot"] - 20.0) < 0.1

    def test_no_data_mutation(self, sample_df, bins, labels, colors):
        original_cols = list(sample_df.columns)
        SummaryPlot(sample_df).plot("pmv", bins, labels, colors)
        assert list(sample_df.columns) == original_cols
        assert "category" not in sample_df.columns

    def test_uses_provided_ax(self, sample_df, bins, labels, colors):
        _, ax = plt.subplots()
        result = SummaryPlot(sample_df).plot("pmv", bins, labels, colors, ax=ax)
        assert result.ax is ax

    def test_creates_ax_when_none(self, sample_df, bins, labels, colors):
        result = SummaryPlot(sample_df).plot("pmv", bins, labels, colors, ax=None)
        assert result.ax is not None

    def test_missing_column_raises(self, bins, labels, colors):
        df = pd.DataFrame({"temperature": [22, 24]})
        with pytest.raises(ValueError, match="not found"):
            SummaryPlot(df).plot("pmv", bins, labels, colors)

    def test_title_none_hides_title(self, sample_df, bins, labels, colors):
        result = SummaryPlot(sample_df).plot(
            "pmv", bins, labels, colors, title=None,
        )
        assert result.ax.get_title() == ""

    def test_all_one_category(self, bins, labels, colors):
        df = pd.DataFrame({"pmv": [0.0, 0.1, -0.2, 0.3]})
        pct = SummaryPlot(df).plot("pmv", bins, labels, colors).category_percentages
        assert abs(pct["Neutral"] - 100.0) < 0.1

    def test_repeated_calls_independent(self, sample_df, bins, labels, colors):
        sp = SummaryPlot(sample_df)
        r1 = sp.plot("pmv", bins, labels, colors)
        wide_bins = [float("-inf"), -5.0, 5.0, float("inf")]
        r2 = sp.plot("pmv", wide_bins, labels, colors)
        assert abs(r1.category_percentages["Cold"] - 40.0) < 0.1
        assert abs(r2.category_percentages["Neutral"] - 100.0) < 0.1


# ═══════════════════════════════════════════════════════════
#  Stored config
# ═══════════════════════════════════════════════════════════


class TestSummaryPlotStoredConfig:
    def test_plot_uses_stored_config(self, sample_df, bins, labels, colors):
        sp = SummaryPlot(
            sample_df,
            target_column="pmv",
            bins=bins,
            labels=labels,
            colors=colors,
        )
        result = sp.plot()  # no args — all from stored config
        assert abs(result.category_percentages["Cold"] - 40.0) < 0.1

    def test_plot_param_overrides_stored(self, sample_df, bins, labels, colors):
        sp = SummaryPlot(
            sample_df,
            target_column="pmv",
            bins=bins,
            labels=labels,
            colors=colors,
        )
        wide_bins = [float("-inf"), -5.0, 5.0, float("inf")]
        result = sp.plot(bins=wide_bins)  # override bins only
        assert abs(result.category_percentages["Neutral"] - 100.0) < 0.1

    def test_partial_stored_rest_from_plot(self, sample_df, bins, labels, colors):
        sp = SummaryPlot(sample_df, target_column="pmv")
        result = sp.plot(bins=bins, labels=labels, colors=colors)
        assert abs(result.category_percentages.sum() - 100.0) < 0.1

    def test_no_config_anywhere_raises(self, sample_df):
        sp = SummaryPlot(sample_df)
        with pytest.raises(ValueError, match="target_column"):
            sp.plot()

    def test_missing_bins_labels_colors_raises(self, sample_df):
        sp = SummaryPlot(sample_df, target_column="pmv")
        with pytest.raises(ValueError, match="bins.*labels.*colors"):
            sp.plot()


# ═══════════════════════════════════════════════════════════
#  update()
# ═══════════════════════════════════════════════════════════


class TestSummaryPlotUpdate:
    def test_update_replaces_data(self, sample_df, bins, labels, colors):
        sp = SummaryPlot(
            sample_df,
            target_column="pmv",
            bins=bins,
            labels=labels,
            colors=colors,
        )
        sp.plot()  # first render
        new_df = pd.DataFrame({"pmv": [0.0] * 10})  # all Neutral
        sp.update(new_df)
        result = sp.plot()
        assert abs(result.category_percentages["Neutral"] - 100.0) < 0.1

    def test_update_clears_last_result(self, sample_df, bins, labels, colors):
        sp = SummaryPlot(
            sample_df,
            target_column="pmv",
            bins=bins,
            labels=labels,
            colors=colors,
        )
        sp.plot()
        assert sp.last_result is not None
        sp.update(pd.DataFrame({"pmv": [0.0]}))
        assert sp.last_result is None

    def test_update_returns_self(self, sample_df):
        sp = SummaryPlot(sample_df)
        assert sp.update(sample_df) is sp

    def test_update_non_dataframe_raises(self, sample_df):
        sp = SummaryPlot(sample_df)
        with pytest.raises(TypeError, match="pandas DataFrame"):
            sp.update("not a dataframe")

    def test_chaining(self, bins, labels, colors):
        df1 = pd.DataFrame({"pmv": [-1.0] * 10})
        df2 = pd.DataFrame({"pmv": [1.0] * 10})
        result = (
            SummaryPlot(df1, target_column="pmv", bins=bins, labels=labels, colors=colors)
            .update(df2)
            .plot()
        )
        assert abs(result.category_percentages["Hot"] - 100.0) < 0.1


# ═══════════════════════════════════════════════════════════
#  last_result
# ═══════════════════════════════════════════════════════════


class TestSummaryPlotLastResult:
    def test_none_before_plot(self, sample_df):
        assert SummaryPlot(sample_df).last_result is None

    def test_set_after_plot(self, sample_df, bins, labels, colors):
        sp = SummaryPlot(sample_df)
        result = sp.plot("pmv", bins, labels, colors)
        assert sp.last_result is result

    def test_updated_after_second_plot(self, sample_df, bins, labels, colors):
        sp = SummaryPlot(sample_df)
        r1 = sp.plot("pmv", bins, labels, colors)
        r2 = sp.plot("pmv", bins, labels, colors)
        assert sp.last_result is r2
        assert sp.last_result is not r1


# ═══════════════════════════════════════════════════════════
#  vertical
# ═══════════════════════════════════════════════════════════


class TestSummaryPlotVertical:

    def test_vertical_returns_result(self, sample_df, bins, labels, colors):
        result = SummaryPlot(sample_df).plot(
            "pmv", bins, labels, colors, orientation="vertical",
        )
        assert isinstance(result, SummaryPlotResult)
        assert abs(result.category_percentages.sum() - 100.0) < 0.1

    def test_vertical_correct_axes_limits(self, sample_df, bins, labels, colors):
        result = SummaryPlot(sample_df).plot(
            "pmv", bins, labels, colors, orientation="vertical",
        )
        assert result.ax.get_ylim() == (0.0, 100.0)

    def test_horizontal_correct_axes_limits(self, sample_df, bins, labels, colors):
        result = SummaryPlot(sample_df).plot(
            "pmv", bins, labels, colors, orientation="horizontal",
        )
        assert result.ax.get_xlim() == (0.0, 100.0)

    def test_vertical_uses_bar_not_barh(self, sample_df, bins, labels, colors):
        result = SummaryPlot(sample_df).plot(
            "pmv", bins, labels, colors, orientation="vertical",
        )
        # bar() creates Rectangle patches; barh() also does, but
        # we verify by checking the orientation of the axes limits
        assert result.ax.get_ylim()[1] == 100.0  # y is the stacking axis
        assert result.ax.get_xlim()[1] < 10      # x is narrow

    def test_stored_orientation(self, sample_df, bins, labels, colors):
        sp = SummaryPlot(
            sample_df,
            target_column="pmv",
            bins=bins, labels=labels, colors=colors,
            orientation="vertical",
        )
        result = sp.plot()
        assert result.ax.get_ylim() == (0.0, 100.0)

    def test_plot_overrides_stored_orientation(self, sample_df, bins, labels, colors):
        sp = SummaryPlot(
            sample_df,
            target_column="pmv",
            bins=bins, labels=labels, colors=colors,
            orientation="vertical",
        )
        result = sp.plot(orientation="horizontal")
        assert result.ax.get_xlim() == (0.0, 100.0)

    def test_invalid_orientation_raises(self, sample_df, bins, labels, colors):
        with pytest.raises(ValueError, match="orientation"):
            SummaryPlot(sample_df).plot(
                "pmv", bins, labels, colors, orientation="diagonal",
            )

    def test_no_data_mutation_vertical(self, sample_df, bins, labels, colors):
        original_cols = list(sample_df.columns)
        SummaryPlot(sample_df).plot(
            "pmv", bins, labels, colors, orientation="vertical",
        )
        assert list(sample_df.columns) == original_cols

    def test_standalone_figsize_vertical(self, sample_df, bins, labels, colors):
        result = SummaryPlot(sample_df).plot(
            "pmv", bins, labels, colors, orientation="vertical", ax=None,
        )
        fig = result.ax.figure
        w, h = fig.get_size_inches()
        assert h > w  # tall and narrow

    def test_standalone_figsize_horizontal(self, sample_df, bins, labels, colors):
        result = SummaryPlot(sample_df).plot(
            "pmv", bins, labels, colors, orientation="horizontal", ax=None,
        )
        fig = result.ax.figure
        w, h = fig.get_size_inches()
        assert w > h  # wide and short
