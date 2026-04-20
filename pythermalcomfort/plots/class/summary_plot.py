"""Independent summary plot for categorical data distribution."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors


class SummaryPlot:
    """
    An independent class to generate a horizontal stacked summary bar chart
    for thermal comfort categories. This decouples the summary visualization
    from the RangeScene base map, offering maximum flexibility to the user.
    """

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")
        self.data = data.copy()

    def _is_light_color(self, color: str) -> bool:
        """
        Calculate luminance to determine if the overlaid text should be
        black or white to maintain optimal contrast.
        """
        red, green, blue = mcolors.to_rgb(color)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return luminance > 0.7

    def plot(
        self,
        target_column: str,
        bins: Sequence[float],
        labels: Sequence[str],
        colors: Sequence[str],
        ax: plt.Axes | None = None,
        title: str | None = "Category Distribution",
        show_percentages: bool = True,
    ) -> plt.Axes:
        """
        Generates the single stacked horizontal bar chart.

        Parameters
        ----------
        target_column : str
            The column in the dataframe to be categorized (e.g., 'pmv').
        bins : Sequence[float]
            The thresholds for categorization.
        labels : Sequence[str]
            The category names corresponding to the bins.
        colors : Sequence[str]
            The colors assigned to each category.
        ax : plt.Axes | None, optional
            A matplotlib axes object to plot on. If None, a new one is created.
        title : str | None, optional
            The title of the plot. Set to None to hide the title.
        show_percentages : bool, optional
            Whether to display the percentage text inside the bars.
        """

        if target_column not in self.data.columns:
            raise ValueError(f"Column '{target_column}' not found in dataframe.")

        if ax is None:
            # Use a wide but flat figure size specifically for a single horizontal bar
            _, ax = plt.subplots(figsize=(8, 2))

        # 1. Categorize the data based on the provided bins
        self.data["category"] = pd.cut(
            self.data[target_column],
            bins=bins,
            labels=labels,
            right=False,
        )

        # 2. Calculate category percentages
        pct = (
            self.data["category"]
            .value_counts(normalize=True)
            .reindex(labels)
            .fillna(0.0)
            * 100
        )

        # 3. Canvas cleanup (Minimalist aesthetic without spines and ticks)
        if title:
            ax.set_title(title, fontsize=13, pad=10)

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        # 4. Render the horizontal stacked bar
        left_pos = 0.0
        bar_y = 0.0
        bar_height = 0.4

        for i, category in enumerate(labels):
            value = float(pct[category])
            color = colors[i]

            # Render the colored segment for the current category
            ax.barh(
                y=bar_y,
                width=value,
                left=left_pos,
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=1.0,
            )

            # Add annotations only if the category has data points
            if value > 0:
                # 5. Render percentage text (if requested)
                if show_percentages:
                    text_color = "black" if self._is_light_color(color) else "white"
                    ax.text(
                        left_pos + value / 2,
                        bar_y,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color=text_color,
                    )

                # 6. Render category label beneath the segment
                label_color = "dimgray" if self._is_light_color(color) else color
                ax.text(
                    left_pos + value / 2,
                    bar_y - 0.28,
                    category,
                    ha="center",
                    va="top",
                    fontsize=11,
                    color=label_color,
                )

            # Shift the starting position for the next segment
            left_pos += value

        return ax
