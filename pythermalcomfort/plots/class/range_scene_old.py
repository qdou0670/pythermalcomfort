"""Class-based threshold-region scene plotting."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Number
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.legend import Legend


@dataclass
class AxisConfig:
    """Configuration for plot axes (min and max boundaries)."""

    name: str
    min_val: float
    max_val: float


@dataclass
class PlotResult:
    """
    Standardized return object for plot rendering.
    Contains editable Matplotlib artists (lines, fills) for post-rendering customization.
    """

    ax: plt.Axes
    lines: list[Any]
    fills: list[PolyCollection]
    legend: Legend | None


def _parse_axis_range(param_name: str, value: Any) -> tuple[float, float]:
    """Validates and parses the axis range input into (min, max) floats."""
    if not isinstance(value, tuple | list) or len(value) != 2:
        msg = f"Axis '{param_name}' must be a tuple/list of length 2: (min, max)."
        raise ValueError(msg)
    try:
        min_val = float(value[0])
        max_val = float(value[1])
    except (TypeError, ValueError) as exc:
        msg = f"Axis '{param_name}' range values must be numeric."
        raise ValueError(msg) from exc
    if min_val >= max_val:
        msg = f"Axis '{param_name}' requires min < max (got {min_val} >= {max_val})."
        raise ValueError(msg)
    return min_val, max_val


def _extract_output_value(result: Any, output: str) -> Any:
    """
    Extracts the targeted output value from the model result.
    Modified to support both scalar extraction (for the iterative solver)
    and array/list extraction (for the vectorized contour approach).
    """
    output_name = output.strip()
    if not output_name:
        raise ValueError("output must be a non-empty string.")

    candidates = [output_name, output_name.lower()]

    for candidate in candidates:
        if hasattr(result, candidate):
            return getattr(result, candidate)

    if isinstance(result, Mapping):
        for candidate in candidates:
            if candidate in result:
                return result[candidate]

    # Supports numpy arrays and lists returned by vectorized model calls
    if isinstance(result, (Number, list, tuple, np.ndarray)) and not isinstance(
        result, bool
    ):
        return result

    raise ValueError(f"Could not extract output '{output_name}' from model result.")


def _default_region_colors(n_regions: int) -> list[str]:
    """Generates default region colors based on the number of specified levels."""
    if n_regions < 1:
        raise ValueError("n_regions must be at least 1.")
    if n_regions == 1:
        return ["#f2f2f2"]
    if n_regions == 2:
        return ["#4c78a8", "#e15759"]
    if n_regions == 3:
        return ["#4c78a8", "#f2f2f2", "#e15759"]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "range_scene_blue_neutral_red",
        ["#4c78a8", "#f2f2f2", "#e15759"],
    )
    positions = np.linspace(0.0, 1.0, n_regions)
    return [mcolors.to_hex(cmap(v)) for v in positions]


class RangeScene:
    """
    Handles the generation of threshold-region base maps for thermal comfort models.
    Supports both an iterative solver approach and a vectorized contour approach.
    """

    def __init__(self, model_func: Any) -> None:
        self.model_func = model_func
        self.x_axis: AxisConfig | None = None
        self.y_axis: AxisConfig | None = None
        self._default_links: dict[str, str] = {"tr": "tdb"}
        self.fixed_values: dict[str, Any] = {}
        self._signature: inspect.Signature | None = None
        self._allowed_args: set[str] = set()
        self._required_args: set[str] = set()
        self._accepts_var_kwargs: bool = False

    def allowed_parameters(self) -> list[str]:
        self._read_model_signature()
        return sorted(self._allowed_args)

    def required_parameters(self) -> list[str]:
        self._read_model_signature()
        return sorted(self._required_args)

    def _extract_axis_spec(
        self,
        axis_label: str,
        axis_range: dict[str, Any],
        other_axis: AxisConfig | None = None,
    ) -> AxisConfig:
        self._read_model_signature()
        if len(axis_range) != 1:
            raise ValueError(f"{axis_label}() requires exactly one keyword range.")
        ((axis_name, value),) = axis_range.items()
        if other_axis is not None and axis_name == other_axis.name:
            raise ValueError("x and y axis parameters must be different.")
        if axis_name not in self._allowed_args:
            raise ValueError(f"{axis_label}() received invalid parameter: {axis_name}.")
        min_val, max_val = _parse_axis_range(axis_name, value)
        return AxisConfig(name=axis_name, min_val=min_val, max_val=max_val)

    def x(self, **axis_range: Any) -> RangeScene:
        self.x_axis = self._extract_axis_spec("x", axis_range, self.y_axis)
        return self

    def y(self, **axis_range: Any) -> RangeScene:
        self.y_axis = self._extract_axis_spec("y", axis_range, self.x_axis)
        return self

    def parameters(self, **kwargs: Any) -> RangeScene:
        self._read_model_signature()
        x_name = self.x_axis.name if self.x_axis else None
        y_name = self.y_axis.name if self.y_axis else None
        for key, value in dict(kwargs).items():
            if key in (x_name, y_name):
                raise ValueError(f"parameters() cannot set axis parameter '{key}'.")
            self.fixed_values[key] = value
        return self

    def _read_model_signature(self) -> None:
        if self._signature is not None:
            return
        self._signature = inspect.signature(self.model_func)
        self._allowed_args = set(self._signature.parameters.keys())
        self._accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in self._signature.parameters.values()
        )
        self._required_args = {
            name
            for name, p in self._signature.parameters.items()
            if p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and p.default is inspect.Signature.empty
        }

    def _validate_call_kwargs(self, kwargs: dict[str, Any]) -> None:
        self._read_model_signature()
        missing = sorted(k for k in self._required_args if k not in kwargs)
        if missing:
            raise ValueError(f"Missing required parameter(s): {', '.join(missing)}")

    def _build_call_kwargs(self, x_value: Any, y_value: Any) -> dict[str, Any]:
        """
        Builds the kwargs dictionary to pass into the model.
        Supports both scalar values (for the iterative solver) and arrays (for vectorization).
        """
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes not set.")
        self._read_model_signature()
        call_kwargs: dict[str, Any] = dict(self.fixed_values)
        call_kwargs[self.x_axis.name] = x_value
        call_kwargs[self.y_axis.name] = y_value

        if "tr" in self._allowed_args and "tdb" in self._allowed_args:
            if "tr" not in call_kwargs and "tdb" in call_kwargs:
                call_kwargs["tr"] = call_kwargs["tdb"]
            elif "tdb" not in call_kwargs and "tr" in call_kwargs:
                call_kwargs["tdb"] = call_kwargs["tr"]
        else:
            for target, source in self._default_links.items():
                if (
                    target in self._allowed_args
                    and source in self._allowed_args
                    and target not in call_kwargs
                    and source in call_kwargs
                ):
                    call_kwargs[target] = call_kwargs[source]
        self._validate_call_kwargs(call_kwargs)
        return call_kwargs

    # =========================================================================
    # APPROACH 1: ORIGINAL ITERATIVE SOLVER
    # =========================================================================
    def _compute_threshold_curves(
        self, *, output: str, thresholds: list[float], x_step: float, y_step: float
    ):
        """Original grid-scanning solver to find region boundaries iteratively."""
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes are not set.")
        self._build_call_kwargs(float(self.x_axis.min_val), float(self.y_axis.min_val))
        x_values = np.arange(self.x_axis.min_val, self.x_axis.max_val + 1e-12, x_step)
        y_values = np.arange(self.y_axis.min_val, self.y_axis.max_val + 1e-12, y_step)
        fill_curves, line_curves = [], []
        x_min, x_max = float(self.x_axis.min_val), float(self.x_axis.max_val)

        for _ in thresholds:
            fill_curves.append(np.full(len(y_values), np.nan, dtype=float))
            line_curves.append(np.full(len(y_values), np.nan, dtype=float))

        for i, y in enumerate(y_values):
            z = np.full(len(x_values), np.nan, dtype=float)
            for j, x in enumerate(x_values):
                kwargs = self._build_call_kwargs(float(x), float(y))
                try:
                    result = self.model_func(**kwargs)
                except Exception:
                    continue
                z[j] = float(_extract_output_value(result, output))

            if not np.isfinite(z).any():
                continue

            for k, threshold in enumerate(thresholds):
                r = z - float(threshold)
                crossing_x = np.nan
                found_crossing = False
                for j in range(len(x_values) - 1):
                    x0, x1, r0, r1 = x_values[j], x_values[j + 1], r[j], r[j + 1]
                    if not np.isfinite(r0) or not np.isfinite(r1):
                        continue
                    if r0 == 0.0:
                        crossing_x, found_crossing = float(x0), True
                        break
                    if r0 * r1 < 0.0:
                        crossing_x, found_crossing = (
                            float(x0 + (-r0) * (x1 - x0) / (r1 - r0)),
                            True,
                        )
                        break
                    if r1 == 0.0:
                        crossing_x, found_crossing = float(x1), True
                        break
                if found_crossing:
                    fill_curves[k][i] = line_curves[k][i] = crossing_x
                    continue
                finite = np.isfinite(r)
                if not finite.any():
                    continue
                rf = r[finite]
                if np.all(rf < 0.0):
                    fill_curves[k][i] = x_max
                elif np.all(rf > 0.0):
                    fill_curves[k][i] = x_min

        return y_values, fill_curves, line_curves

    def plot(
        self,
        *,
        output: str,
        levels: list[float],
        colors: list[str] | None = None,
        x_step: float,
        y_step: float,
        ax: plt.Axes | None = None,
        legend: bool = True,
        show_lines: bool = True,
        line_kws: dict[str, Any] | None = None,
        fill_kws: dict[str, Any] | None = None,
    ) -> PlotResult:
        """Renders the scene using the original iterative solver."""
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes not set.")
        sorted_levels = sorted(float(v) for v in levels)
        region_colors = (
            colors if colors else _default_region_colors(len(sorted_levels) + 1)
        )
        line_opts, fill_opts = dict(line_kws or {}), dict(fill_kws or {})
        fill_opts.setdefault("alpha", 0.65)
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))

        y_values, fill_curves, line_curves = self._compute_threshold_curves(
            output=output, thresholds=sorted_levels, x_step=x_step, y_step=y_step
        )
        x_lo, x_hi = float(self.x_axis.min_val), float(self.x_axis.max_val)
        left_const, right_const = (
            np.full_like(y_values, x_lo, dtype=float),
            np.full_like(y_values, x_hi, dtype=float),
        )

        fills, lines = [], []
        region_pairs = [
            (left_const, fill_curves[0]),
            *[
                (fill_curves[i], fill_curves[i + 1])
                for i in range(len(fill_curves) - 1)
            ],
            (fill_curves[-1], right_const),
        ]

        for i, (x_left, x_right) in enumerate(region_pairs):
            valid = np.isfinite(x_left) & np.isfinite(x_right)
            if valid.any():
                opts = dict(fill_opts)
                opts["color"] = region_colors[i]
                fills.append(
                    ax.fill_betweenx(
                        y_values[valid], x_left[valid], x_right[valid], **opts
                    )
                )

        if show_lines:
            for curve in line_curves:
                valid = np.isfinite(curve)
                if valid.any():
                    (line,) = ax.plot(curve[valid], y_values[valid], **line_opts)
                    lines.append(line)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(float(self.y_axis.min_val), float(self.y_axis.max_val))
        return PlotResult(ax=ax, lines=lines, fills=fills, legend=None)

    # =========================================================================
    # APPROACH 2: VECTORIZED CONTOUR METHOD
    # =========================================================================
    def plot_contour(
        self,
        *,
        output: str,
        levels: list[float],
        colors: list[str] | None = None,
        grid_resolution: int = 150,
        ax: plt.Axes | None = None,
        legend: bool = True,
        show_lines: bool = True,
        line_kws: dict[str, Any] | None = None,
        fill_kws: dict[str, Any] | None = None,
    ) -> PlotResult:
        """
        Renders the scene using a fully vectorized numpy meshgrid and matplotlib contours.
        This approach offers O(1) performance improvements compared to the iterative solver,
        and inherently handles applicability limits by leaving NaN regions blank.
        """
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes not set.")
        sorted_levels = sorted(float(v) for v in levels)
        region_colors = (
            colors if colors else _default_region_colors(len(sorted_levels) + 1)
        )

        line_opts, fill_opts = dict(line_kws or {}), dict(fill_kws or {})
        line_opts.setdefault("color", "black")
        line_opts.setdefault("linewidth", 1.0)
        fill_opts.setdefault("alpha", 0.65)

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))

        # 1. Generate Vectorized Grid
        x_vals = np.linspace(self.x_axis.min_val, self.x_axis.max_val, grid_resolution)
        y_vals = np.linspace(self.y_axis.min_val, self.y_axis.max_val, grid_resolution)
        X, Y = np.meshgrid(x_vals, y_vals)

        # 2. Compute Model Values Vectorized
        kwargs = self._build_call_kwargs(X.ravel().tolist(), Y.ravel().tolist())
        z_flat = _extract_output_value(self.model_func(**kwargs), output)
        Z = np.array(z_flat, dtype=float).reshape(X.shape)

        # 3. Render Filled Contours
        extended_levels = [-np.inf] + sorted_levels + [np.inf]
        cf = ax.contourf(
            X,
            Y,
            Z,
            levels=extended_levels,
            colors=region_colors,
            alpha=fill_opts["alpha"],
            extend="neither",
        )

        # Ensures backward compatibility with Matplotlib updates (where .collections was removed)
        fills = cf.collections if hasattr(cf, "collections") else [cf]

        # 4. Render Boundary Lines
        lines = []
        if show_lines:
            # Utilize the contour engine solely for fast coordinate calculation
            cs = ax.contour(X, Y, Z, levels=sorted_levels)

            # Extract paths from the contour to convert them into standard Line2D objects.
            # This is crucial for returning editable objects (e.g., allowing users to call
            # result.lines[0].set_linewidth() without QuadContourSet errors).
            for path in cs.get_paths():
                segments = path.to_polygons(closed_only=False)
                x_comb, y_comb = [], []
                for seg in segments:
                    x_comb.extend(seg[:, 0].tolist() + [np.nan])
                    y_comb.extend(seg[:, 1].tolist() + [np.nan])

                if x_comb:
                    # Draw as a standard Line2D for PlotResult compatibility
                    (line,) = ax.plot(x_comb, y_comb, **line_opts)
                    lines.append(line)

            # Remove the original contour artist to avoid visual duplication
            cs.remove()

        ax.set_xlim(float(self.x_axis.min_val), float(self.x_axis.max_val))
        ax.set_ylim(float(self.y_axis.min_val), float(self.y_axis.max_val))
        return PlotResult(ax=ax, lines=lines, fills=fills, legend=None)
