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
from matplotlib.lines import Line2D


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
    Contains editable Matplotlib artists and an optional reference
    to the underlying computation grid for programmatic access.
    """

    ax: plt.Axes
    lines: list[Any]
    fills: list[PolyCollection]
    legend: Legend | None
    grid: ComputedGrid | None = None  # ← NEW: backward-compatible default


@dataclass
class ComputedGrid:
    """
    Reusable model computation result, fully decoupled from rendering.

    Holds the raw Z-grid computed by evaluating the model function over
    a 2D parameter grid. This object can be passed to render() or
    render_contour() repeatedly with different visual parameters
    (colors, levels, line styles) without triggering recomputation.

    Attributes
    ----------
    x_values : np.ndarray
        1D array of x-axis sample coordinates, shape (nx,).
    y_values : np.ndarray
        1D array of y-axis sample coordinates, shape (ny,).
    z_grid : np.ndarray
        2D array of model output values, shape (ny, nx).
        NaN entries indicate points where the model raised an exception
        or returned a non-finite value.
    output : str
        Name of the model output that was extracted (e.g. "pmv").
    """

    x_values: np.ndarray
    y_values: np.ndarray
    z_grid: np.ndarray
    output: str

    @property
    def finite_ratio(self) -> float:
        """Fraction of grid points with finite (non-NaN) values."""
        return float(np.isfinite(self.z_grid).mean())

    @property
    def shape(self) -> tuple[int, int]:
        """Grid dimensions as (n_rows, n_cols)."""
        return self.z_grid.shape


@dataclass(frozen=True)
class _ComputeFingerprint:
    """
    Hashable, immutable snapshot of all parameters that affect Z-grid values.

    Two fingerprints are equal iff the corresponding compute() calls
    would produce identical ComputedGrid results (assuming a deterministic model).
    """

    x_name: str
    x_min: float
    x_max: float
    y_name: str
    y_min: float
    y_max: float
    fixed_values_key: str  # sorted repr of fixed_values dict
    output: str
    grid_spec: str  # method-specific: encodes step sizes or resolution


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

        # ── Cache state ──────────────────────────────────────────
        self._cached_grid: ComputedGrid | None = None
        self._cached_fingerprint: _ComputeFingerprint | None = None

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
        self._invalidate_cache()
        return self

    def y(self, **axis_range: Any) -> RangeScene:
        self.y_axis = self._extract_axis_spec("y", axis_range, self.x_axis)
        self._invalidate_cache()
        return self

    def parameters(self, **kwargs: Any) -> RangeScene:
        self._read_model_signature()
        x_name = self.x_axis.name if self.x_axis else None
        y_name = self.y_axis.name if self.y_axis else None
        for key, value in dict(kwargs).items():
            if key in (x_name, y_name):
                raise ValueError(f"parameters() cannot set axis parameter '{key}'.")
            # self.fixed_values[key] = value

        self.fixed_values.update(kwargs)
        self._invalidate_cache()

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

    def _make_fingerprint(self, output: str, **grid_params: Any) -> _ComputeFingerprint:
        """
        Create a hashable fingerprint from the current scene configuration
        and the given compute parameters.

        The fingerprint captures everything that affects the Z-grid values:
        axis definitions, fixed model parameters, output name, and grid geometry.
        Rendering-only parameters (colors, levels, line styles) are deliberately
        excluded — they don't affect the cached computation.
        """
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Both axes must be set before computing.")

        # Stable string representation of fixed_values
        # Uses repr() which is reliable for common types (int, float, str, list, tuple)
        fixed_key = str(sorted((k, repr(v)) for k, v in self.fixed_values.items()))
        grid_spec = str(sorted(grid_params.items()))

        return _ComputeFingerprint(
            x_name=self.x_axis.name,
            x_min=self.x_axis.min_val,
            x_max=self.x_axis.max_val,
            y_name=self.y_axis.name,
            y_min=self.y_axis.min_val,
            y_max=self.y_axis.max_val,
            fixed_values_key=fixed_key,
            output=output,
            grid_spec=grid_spec,
        )

    def _invalidate_cache(self) -> None:
        """Clear cached computation when configuration changes."""
        self._cached_grid = None
        self._cached_fingerprint = None

    def clear_cache(self) -> None:
        """
        Explicitly clear the computation cache.

        Useful when the underlying model function has been modified
        externally (e.g., a lookup table was updated) without changing
        the RangeScene configuration.
        """
        self._invalidate_cache()

    def compute(
        self,
        *,
        output: str,
        x_step: float,
        y_step: float,
    ) -> ComputedGrid:
        """
        Compute the Z-grid using scalar model evaluation at each grid point.

        This is the expensive step (O(nx * ny) model calls). The result is
        cached internally — repeated calls with identical configuration
        return the cached grid instantly.

        Parameters
        ----------
        output : str
            Name of the model output to extract (e.g. "pmv", "ppd").
        x_step : float
            Spacing between x-axis sample points.
        y_step : float
            Spacing between y-axis sample points.

        Returns
        -------
        ComputedGrid
            The computed grid, which can be passed to render() one or more times.
        """
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Both axes must be set before computing.")

        # ── Cache check ──────────────────────────────────────────
        fp = self._make_fingerprint(output, x_step=x_step, y_step=y_step)
        if self._cached_fingerprint == fp and self._cached_grid is not None:
            return self._cached_grid

        # ── Build grid coordinates ───────────────────────────────
        x_values = np.arange(
            self.x_axis.min_val,
            self.x_axis.max_val + 1e-12,
            x_step,
        )
        y_values = np.arange(
            self.y_axis.min_val,
            self.y_axis.max_val + 1e-12,
            y_step,
        )

        # ── Evaluate model at every grid point ───────────────────
        z_grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)

        for i, y in enumerate(y_values):
            for j, x in enumerate(x_values):
                kwargs = self._build_call_kwargs(float(x), float(y))
                try:
                    result = self.model_func(**kwargs)
                except Exception:
                    continue
                z_grid[i, j] = float(_extract_output_value(result, output))

        # ── Cache and return ─────────────────────────────────────
        grid = ComputedGrid(
            x_values=x_values,
            y_values=y_values,
            z_grid=z_grid,
            output=output,
        )
        self._cached_grid = grid
        self._cached_fingerprint = fp
        return grid

    def render(
        self,
        grid: ComputedGrid,
        *,
        levels: list[float],
        colors: list[str] | None = None,
        ax: plt.Axes | None = None,
        legend: bool = True,
        show_lines: bool = True,
        smooth: bool = True,  # ← NEW
        smooth_kws: dict[str, Any] | None = None,  # ← NEW
        line_kws: dict[str, Any] | None = None,
        fill_kws: dict[str, Any] | None = None,
    ) -> PlotResult:
        """
        Render a precomputed grid using the iterative region-fill approach.

        Parameters
        ----------
        smooth : bool
            If True (default), threshold boundary lines are smoothed via
            parametric B-spline fitting.
        smooth_kws : dict | None
            Keyword arguments forwarded to _smooth_curve_points().
        """
        sorted_levels = sorted(float(v) for v in levels)
        n_regions = len(sorted_levels) + 1
        region_colors = colors if colors else _default_region_colors(n_regions)

        line_opts = dict(line_kws or {})
        fill_opts = dict(fill_kws or {})
        alpha = fill_opts.pop("alpha", 0.65)
        fill_opts.pop("color", None)

        resolved_smooth_kws = dict(smooth_kws or {})  # ← NEW

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))

        x_values = grid.x_values
        y_values = grid.y_values
        z_grid = grid.z_grid
        x_min, x_max = float(x_values[0]), float(x_values[-1])
        y_min, y_max = float(y_values[0]), float(y_values[-1])
        y_step = float(y_values[1] - y_values[0]) if len(y_values) > 1 else 1.0

        # ── Region intervals (unchanged) ─────────────────────────
        all_intervals = [
            self._compute_row_region_intervals(
                x_values,
                z_grid[i],
                sorted_levels,
                x_min,
                x_max,
            )
            for i in range(len(y_values))
        ]

        # ── Fill polygons (unchanged) ────────────────────────────
        fills: list[PolyCollection] = []
        region_polys: dict[int, list] = {r: [] for r in range(n_regions)}

        for i in range(len(y_values)):
            y_lo = float(y_values[i])
            y_hi = float(y_values[i + 1]) if i + 1 < len(y_values) else y_lo + y_step
            for x_start, x_end, region in all_intervals[i]:
                region_polys[region].append(
                    [
                        (x_start, y_lo),
                        (x_end, y_lo),
                        (x_end, y_hi),
                        (x_start, y_hi),
                    ]
                )

        for r in range(n_regions):
            if region_polys[r]:
                pc = PolyCollection(
                    region_polys[r],
                    facecolor=region_colors[r],
                    edgecolor="none",
                    alpha=alpha,
                    **fill_opts,
                )
                ax.add_collection(pc)
                fills.append(pc)

        # ── Threshold lines (with smoothing) ─────────────────────
        lines: list[Line2D] = []
        if show_lines:
            for t in sorted_levels:
                crossings_per_y = [
                    self._find_all_crossings_in_row(x_values, z_grid[i], t)
                    for i in range(len(y_values))
                ]
                curves = self._connect_crossings_to_curves(crossings_per_y, y_values)

                for curve in curves:
                    if len(curve) < 2:
                        continue
                    xs = np.array([p[0] for p in curve])
                    ys = np.array([p[1] for p in curve])

                    if smooth:  # ← NEW
                        xs, ys = self._smooth_curve_points(  # ← NEW
                            xs,
                            ys,
                            **resolved_smooth_kws,  # ← NEW
                        )  # ← NEW

                    (line,) = ax.plot(xs, ys, **line_opts)
                    lines.append(line)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        return PlotResult(ax=ax, lines=lines, fills=fills, legend=None, grid=grid)

    @staticmethod
    def _classify_z(z_val: float, sorted_thresholds: list[float]) -> int:
        """
        Classify a z-value into a region index based on sorted thresholds.

        Region layout (N thresholds → N+1 regions):
            Region 0:  z < thresholds[0]
            Region k:  thresholds[k-1] <= z < thresholds[k]   (1 <= k < N)
            Region N:  z >= thresholds[N-1]

        Returns -1 for non-finite (NaN / Inf) values.
        """
        if not np.isfinite(z_val):
            return -1
        for k, t in enumerate(sorted_thresholds):
            if z_val < t:
                return k
        return len(sorted_thresholds)

    @staticmethod
    def _find_all_crossings_in_row(
        x_values: np.ndarray,
        z_row: np.ndarray,
        threshold: float,
    ) -> list[float]:
        """
        Find ALL x-coordinates where z_row crosses the given threshold.

        Scans every consecutive pair of grid points:
          - Exact-on-threshold left endpoints are recorded once.
          - Sign changes trigger linear interpolation.
          - The last grid point is checked separately to avoid omission.

        Returns a sorted list of crossing x-coordinates (deduplicated).
        """
        crossings: list[float] = []
        t = float(threshold)
        n = len(x_values)

        for j in range(n - 1):
            z0, z1 = z_row[j], z_row[j + 1]
            if not np.isfinite(z0) or not np.isfinite(z1):
                continue

            r0, r1 = z0 - t, z1 - t

            if r0 == 0.0:
                # Exact crossing at left endpoint of this segment
                cx = float(x_values[j])
                if not crossings or abs(crossings[-1] - cx) > 1e-12:
                    crossings.append(cx)
            elif r0 * r1 < 0.0:
                # Sign change → interpolate crossing position
                x0, x1 = float(x_values[j]), float(x_values[j + 1])
                cx = x0 + (-r0) * (x1 - x0) / (r1 - r0)
                crossings.append(cx)
            # r1 == 0.0 is naturally handled as r0 in the next segment

        # Handle the very last grid point (not covered as a left endpoint)
        if n > 0 and np.isfinite(z_row[-1]) and z_row[-1] - t == 0.0:
            cx = float(x_values[-1])
            if not crossings or abs(crossings[-1] - cx) > 1e-12:
                crossings.append(cx)

        return crossings

    def _compute_row_region_intervals(
        self,
        x_values: np.ndarray,
        z_row: np.ndarray,
        sorted_thresholds: list[float],
        x_min: float,
        x_max: float,
    ) -> list[tuple[float, float, int]]:
        """
        Compute all region intervals for a single y-row.

        Algorithm:
          1. Collect ALL crossings from ALL thresholds into a single sorted set.
          2. These crossings, together with x_min and x_max, partition the row
             into sub-intervals.
          3. For each sub-interval, interpolate z at its midpoint and classify
             into a region.

        Returns:
            Sorted list of (x_start, x_end, region_index) tuples.
            Empty list if the row has no finite z-values.
        """
        finite_mask = np.isfinite(z_row)
        if not finite_mask.any():
            return []

        # 1. Collect crossings from every threshold
        all_crossings: set[float] = set()
        for t in sorted_thresholds:
            for cx in self._find_all_crossings_in_row(x_values, z_row, t):
                all_crossings.add(cx)

        # 2. Build sorted boundary sequence
        boundaries = sorted({x_min, x_max} | all_crossings)

        # 3. Classify each sub-interval
        x_fin = x_values[finite_mask].astype(float)
        z_fin = z_row[finite_mask].astype(float)

        intervals: list[tuple[float, float, int]] = []
        for b in range(len(boundaries) - 1):
            x_start, x_end = boundaries[b], boundaries[b + 1]
            mid_x = (x_start + x_end) / 2.0
            # Clamp to finite data range for safe np.interp
            mid_x = float(np.clip(mid_x, x_fin[0], x_fin[-1]))
            z_mid = float(np.interp(mid_x, x_fin, z_fin))

            region = self._classify_z(z_mid, sorted_thresholds)
            if region >= 0:
                intervals.append((x_start, x_end, region))

        return intervals

    @staticmethod
    def _connect_crossings_to_curves(
        crossings_per_y: list[list[float]],
        y_values: np.ndarray,
    ) -> list[list[tuple[float, float]]]:
        """
        Connect per-y-row crossing points into continuous line curves.

        Uses greedy nearest-neighbor matching between consecutive y-rows:
          - Each active curve's last x is matched to the closest unmatched
            crossing in the next row.
          - Unmatched active curves are finalized (the curve ends).
          - Unmatched new crossings start fresh curves (a new branch appears).

        This correctly handles:
          - Splits:  1 curve at y[i] → 2 curves at y[i+1]
          - Merges:  2 curves at y[i] → 1 curve at y[i+1]
          - Births:  a new crossing appears at y[i] with no predecessor
          - Deaths:  an active curve has no matching crossing at y[i]

        Returns:
            List of curves; each curve is a list of (x, y) coordinate pairs.
        """
        all_curves: list[list[tuple[float, float]]] = []
        active: list[list[tuple[float, float]]] = []

        for i, y in enumerate(y_values):
            xs = crossings_per_y[i]
            y_f = float(y)

            if not xs and not active:
                continue

            if not xs:
                # No crossings at this row → finalize all active curves
                all_curves.extend(active)
                active = []
                continue

            if not active:
                # No active curves → each crossing starts a new curve
                active = [[(x, y_f)] for x in xs]
                continue

            # ---- Greedy nearest-neighbor matching ----
            last_xs = [c[-1][0] for c in active]
            pairs = sorted(
                (abs(lx - x), ci, xi)
                for ci, lx in enumerate(last_xs)
                for xi, x in enumerate(xs)
            )

            used_curves: set[int] = set()
            used_xs: set[int] = set()
            for _, ci, xi in pairs:
                if ci in used_curves or xi in used_xs:
                    continue
                active[ci].append((xs[xi], y_f))
                used_curves.add(ci)
                used_xs.add(xi)

            # Finalize unmatched active curves (deaths)
            unmatched_active = [
                active[ci] for ci in range(len(active)) if ci not in used_curves
            ]
            all_curves.extend(unmatched_active)

            # Retain matched curves + start new curves for unmatched crossings (births)
            active = [active[ci] for ci in range(len(active)) if ci in used_curves]
            for xi in range(len(xs)):
                if xi not in used_xs:
                    active.append([(xs[xi], y_f)])

        # Finalize remaining active curves
        all_curves.extend(active)
        return all_curves

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
        smooth: bool = True,  # ← NEW
        smooth_kws: dict[str, Any] | None = None,  # ← NEW
        line_kws: dict[str, Any] | None = None,
        fill_kws: dict[str, Any] | None = None,
    ) -> PlotResult:
        grid = self.compute(output=output, x_step=x_step, y_step=y_step)
        return self.render(
            grid,
            levels=levels,
            colors=colors,
            ax=ax,
            legend=legend,
            show_lines=show_lines,
            smooth=smooth,  # ← NEW
            smooth_kws=smooth_kws,  # ← NEW
            line_kws=line_kws,
            fill_kws=fill_kws,
        )

    def compute_contour(
        self,
        *,
        output: str,
        grid_resolution: int = 150,
    ) -> ComputedGrid:
        """
        Compute the Z-grid using vectorized model evaluation.

        Passes numpy arrays directly to the model function. If the model
        does not support array inputs (raises TypeError or ValueError),
        automatically falls back to scalar-per-point evaluation without
        the .tolist() memory penalty.

        Parameters
        ----------
        output : str
            Name of the model output to extract (e.g. "pmv").
        grid_resolution : int
            Number of sample points along each axis.
            - Hard limit: 10,000 (would require ~2.4 GB for the Z-grid alone).
            - Soft limit: 2,000 (a warning is issued above this).
            - For visual smoothness, prefer the `smooth` parameter in render
              methods over brute-force resolution increases.

        Returns
        -------
        ComputedGrid
        """
        import warnings

        # Module-level constants (top of range_scene.py, after imports)

        _MAX_RECOMMENDED_RESOLUTION: int = 2000
        """Grid resolutions above this trigger a performance warning."""

        _ABSOLUTE_MAX_RESOLUTION: int = 10000
        """Grid resolutions above this are rejected outright."""

        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Both axes must be set before computing.")

        # ── Resolution guardrails ────────────────────────────────
        if grid_resolution > _ABSOLUTE_MAX_RESOLUTION:
            total_points = grid_resolution**2
            mem_gb = total_points * 8 * 3 / 1e9  # 3 arrays: X, Y, Z
            raise ValueError(
                f"grid_resolution={grid_resolution:,} exceeds the absolute maximum "
                f"of {_ABSOLUTE_MAX_RESOLUTION:,}. This would create a grid of "
                f"{total_points:,} points requiring ~{mem_gb:.1f} GB of memory. "
                f"Consider using smooth=True in render methods for visual quality."
            )

        if grid_resolution > _MAX_RECOMMENDED_RESOLUTION:
            total_points = grid_resolution**2
            mem_gb = total_points * 8 * 3 / 1e9
            warnings.warn(
                f"grid_resolution={grid_resolution:,} creates {total_points:,} grid "
                f"points (~{mem_gb:.1f} GB). Resolutions above "
                f"{_MAX_RECOMMENDED_RESOLUTION:,} rarely improve visual quality. "
                f"Consider using smooth=True with a moderate resolution instead.",
                stacklevel=2,
            )

        # ── Cache check ──────────────────────────────────────────
        fp = self._make_fingerprint(output, grid_resolution=grid_resolution)
        if self._cached_fingerprint == fp and self._cached_grid is not None:
            return self._cached_grid

        # ── Build meshgrid ───────────────────────────────────────
        x_vals = np.linspace(self.x_axis.min_val, self.x_axis.max_val, grid_resolution)
        y_vals = np.linspace(self.y_axis.min_val, self.y_axis.max_val, grid_resolution)
        X, Y = np.meshgrid(x_vals, y_vals)

        # ── Evaluate model ───────────────────────────────────────
        try:
            # Primary path: pass numpy arrays directly (no .tolist())
            kwargs = self._build_call_kwargs(X.ravel(), Y.ravel())
            z_flat = _extract_output_value(self.model_func(**kwargs), output)
            Z = np.array(z_flat, dtype=float).reshape(X.shape)
        except (TypeError, ValueError, AttributeError) as exc:
            # Fallback: model doesn't support array inputs
            warnings.warn(
                f"Vectorized model call failed ({type(exc).__name__}: {exc}). "
                f"Falling back to scalar evaluation. This is slower but "
                f"produces identical results. To suppress this warning, "
                f"ensure your model function accepts numpy array inputs.",
                stacklevel=2,
            )
            Z = self._fallback_scalar_grid(X, Y, output)

        # ── Cache and return ─────────────────────────────────────
        grid = ComputedGrid(
            x_values=x_vals,
            y_values=y_vals,
            z_grid=Z,
            output=output,
        )
        self._cached_grid = grid
        self._cached_fingerprint = fp
        return grid

    def render_contour(
        self,
        grid: ComputedGrid,
        *,
        levels: list[float],
        colors: list[str] | None = None,
        ax: plt.Axes | None = None,
        legend: bool = True,
        show_lines: bool = True,
        smooth: bool = True,  # ← NEW
        smooth_kws: dict[str, Any] | None = None,  # ← NEW
        line_kws: dict[str, Any] | None = None,
        fill_kws: dict[str, Any] | None = None,
    ) -> PlotResult:
        """
        Render a precomputed grid using matplotlib's contour engine.

        Parameters
        ----------
        smooth : bool
            If True (default), boundary lines are post-processed with
            parametric B-spline fitting, converting the piecewise-linear
            contour segments into mathematically smooth curves.
        smooth_kws : dict | None
            Keyword arguments forwarded to _smooth_curve_points().
            Supported keys: 's' (smoothing factor), 'num_points'
            (output density), 'k' (spline degree).
        """
        sorted_levels = sorted(float(v) for v in levels)
        region_colors = (
            colors if colors else _default_region_colors(len(sorted_levels) + 1)
        )

        line_opts = dict(line_kws or {})
        line_opts.setdefault("color", "black")
        line_opts.setdefault("linewidth", 1.0)
        fill_opts = dict(fill_kws or {})
        fill_opts.setdefault("alpha", 0.65)

        resolved_smooth_kws = dict(smooth_kws or {})  # ← NEW

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))

        X, Y = np.meshgrid(grid.x_values, grid.y_values)
        Z = grid.z_grid

        # ── Filled contours (unchanged) ──────────────────────────
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
        fills = cf.collections if hasattr(cf, "collections") else [cf]

        # ── Boundary lines (with smoothing) ──────────────────────
        lines: list[Line2D] = []
        if show_lines:
            cs = ax.contour(X, Y, Z, levels=sorted_levels)

            for path in cs.get_paths():
                segments = path.to_polygons(closed_only=False)
                x_comb: list[float] = []
                y_comb: list[float] = []

                for seg in segments:
                    seg_x = seg[:, 0]
                    seg_y = seg[:, 1]

                    if smooth:  # ← NEW
                        seg_x, seg_y = self._smooth_curve_points(  # ← NEW
                            seg_x,
                            seg_y,
                            **resolved_smooth_kws,  # ← NEW
                        )  # ← NEW

                    x_comb.extend(seg_x.tolist() + [np.nan])
                    y_comb.extend(seg_y.tolist() + [np.nan])

                if x_comb:
                    (line,) = ax.plot(x_comb, y_comb, **line_opts)
                    lines.append(line)

            cs.remove()

        x_min, x_max = float(grid.x_values[0]), float(grid.x_values[-1])
        y_min, y_max = float(grid.y_values[0]), float(grid.y_values[-1])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        return PlotResult(ax=ax, lines=lines, fills=fills, legend=None, grid=grid)

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
        smooth: bool = True,  # ← NEW
        smooth_kws: dict[str, Any] | None = None,  # ← NEW
        line_kws: dict[str, Any] | None = None,
        fill_kws: dict[str, Any] | None = None,
    ) -> PlotResult:
        grid = self.compute_contour(output=output, grid_resolution=grid_resolution)
        return self.render_contour(
            grid,
            levels=levels,
            colors=colors,
            ax=ax,
            legend=legend,
            show_lines=show_lines,
            smooth=smooth,  # ← NEW
            smooth_kws=smooth_kws,  # ← NEW
            line_kws=line_kws,
            fill_kws=fill_kws,
        )

    @staticmethod
    def _smooth_curve_points(
        x: np.ndarray,
        y: np.ndarray,
        *,
        s: float | None = None,
        num_points: int | None = None,
        k: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Smooth a 2D curve using parametric B-spline fitting.

        Parameters
        ----------
        s : float | None
            Smoothing factor for scipy.interpolate.splprep.
            - None (default): auto-computed from data characteristics.
              The spline is allowed to deviate from each input point by
              approximately 10% of the average inter-point spacing,
              which removes grid-induced positional noise while preserving
              the true boundary shape.
            - 0.0: interpolating spline — passes exactly through every
              input point. Rarely useful for contour smoothing since the
              input points themselves carry grid-quantization noise.
            - >0.0: explicit smoothing budget (sum of squared residuals).
        """
        try:
            from scipy.interpolate import splev, splprep
        except ImportError:
            return x, y

        # ── 1. Clean input ───────────────────────────────────────
        points = np.column_stack([x, y])
        finite_mask = np.all(np.isfinite(points), axis=1)
        points = points[finite_mask]

        if len(points) < k + 1:
            return x, y

        # ── 2. Remove consecutive duplicates ─────────────────────
        diffs = np.diff(points, axis=0)
        segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        keep = np.concatenate([[True], segment_lengths > 1e-12])
        points = points[keep]
        segment_lengths = segment_lengths[segment_lengths > 1e-12]

        if len(points) < k + 1:
            return x, y

        # ── 3. Detect closed curves ──────────────────────────────
        closure_dist = np.hypot(
            points[-1, 0] - points[0, 0],
            points[-1, 1] - points[0, 1],
        )
        typical_spacing = float(np.median(segment_lengths))
        is_closed = bool(closure_dist < typical_spacing * 1.5)

        # ── 4. Auto-compute smoothing factor ─────────────────────
        if s is None:
            n = len(points)
            avg_step = float(np.sum(segment_lengths)) / max(n - 1, 1)

            # Budget: each point may deviate by ~10% of average spacing.
            # Total squared-residual budget = n * (0.1 * avg_step)²
            # This is scale-invariant: adapts to both coordinate range
            # and point density automatically.
            s = n * (0.1 * avg_step) ** 2

        # ── 5. Auto-scale output density ─────────────────────────
        if num_points is None:
            num_points = max(len(points) * 5, 200)

        # ── 6. Fit parametric B-spline ───────────────────────────
        try:
            tck, u = splprep(
                [points[:, 0], points[:, 1]],
                s=s,
                k=k,
                per=is_closed,
            )
            u_fine = np.linspace(0.0, 1.0, num_points)
            x_smooth, y_smooth = splev(u_fine, tck)
            return np.asarray(x_smooth), np.asarray(y_smooth)
        except Exception:
            return x, y

    def _fallback_scalar_grid(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        output: str,
    ) -> np.ndarray:
        """
        Evaluate the model at every grid point using scalar calls.

        Fallback for models that do not accept numpy array inputs.
        Reuses _build_call_kwargs for correctness (tr/tdb linkage, validation),
        but pre-validates once to avoid redundant checks per point.
        """
        # Pre-validate with dummy scalars — fail fast before the slow loop
        self._build_call_kwargs(float(self.x_axis.min_val), float(self.y_axis.min_val))

        x_name = self.x_axis.name
        y_name = self.y_axis.name
        base_kwargs = dict(self.fixed_values)

        # Pre-resolve tr/tdb linkage into base_kwargs
        # so the inner loop only needs to overwrite x/y values
        dummy_kwargs = self._build_call_kwargs(0.0, 0.0)
        for key, val in dummy_kwargs.items():
            if key not in (x_name, y_name):
                base_kwargs[key] = val

        def _eval_single(x_val: float, y_val: float) -> float:
            kwargs = dict(base_kwargs)
            kwargs[x_name] = float(x_val)
            kwargs[y_name] = float(y_val)
            # Maintain tr/tdb link when one of them is an axis
            if x_name in ("tdb", "tr") and y_name not in ("tdb", "tr"):
                linked = "tr" if x_name == "tdb" else "tdb"
                if linked in self._allowed_args and linked not in self.fixed_values:
                    kwargs[linked] = float(x_val)
            if y_name in ("tdb", "tr") and x_name not in ("tdb", "tr"):
                linked = "tr" if y_name == "tdb" else "tdb"
                if linked in self._allowed_args and linked not in self.fixed_values:
                    kwargs[linked] = float(y_val)
            try:
                result = self.model_func(**kwargs)
                return float(_extract_output_value(result, output))
            except Exception:
                return np.nan

        evaluator = np.vectorize(_eval_single, otypes=[float])
        return evaluator(X, Y)
