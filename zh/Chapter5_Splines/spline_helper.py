"""Chapter-local helper for the piecewise-polynomial illustration.

Derived from ``notebooks/scripts/splines.py`` in the English source tree
(SHA-256: 128eb46b5c3e7ae8ccd94d9ae99b4b05c5878ed51622443569d6fc2b6f60f502).
The public ``splines`` behavior is preserved; the linear solve now uses
``numpy.linalg.lstsq`` so rank-deficient exercise variants remain supported, and
``x_max`` is an explicit argument for Exercise 5M9.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray


_FLOAT_ARRAY = NDArray[np.float64]


def basis(x: ArrayLike, degree: int, knots: Sequence[float]) -> _FLOAT_ARRAY:
    """Return a truncated-power basis for a piecewise polynomial."""
    x_array = np.asarray(x, dtype=float)
    columns = [x_array**power for power in range(degree + 1)]
    columns.extend(
        np.where(x_array < knot, 0.0, (x_array - knot) ** degree)
        for knot in knots
    )
    return np.column_stack(columns)


def ols(design: ArrayLike, y: ArrayLike) -> _FLOAT_ARRAY:
    """Return fitted values from a least-squares solution."""
    design_array = np.asarray(design, dtype=float)
    y_array = np.asarray(y, dtype=float)
    coefficients, *_ = np.linalg.lstsq(design_array, y_array, rcond=None)
    return design_array @ coefficients


def splines(
    knots: Sequence[float],
    x_true: ArrayLike | None = None,
    y_true: ArrayLike | None = None,
    *,
    x_max: float = 6.0,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot degree 0--3 piecewise-polynomial approximations.

    ``x_max`` controls the default grid when ``x_true`` is omitted. Returns the
    figure and axes so callers can save or further annotate the plot.
    """
    if x_true is None:
        if not np.isfinite(x_max) or x_max <= 0:
            raise ValueError("x_max must be a positive finite value")
        x_array = np.linspace(0.0, float(x_max), 200)
    else:
        x_array = np.asarray(x_true, dtype=float)

    if y_true is None:
        y_array = np.sin(x_array)
    else:
        y_array = np.asarray(y_true, dtype=float)

    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.shape != y_array.shape:
        raise ValueError("x_true and y_true must be one-dimensional arrays of equal length")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(9, 6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes_array = np.asarray(axes, dtype=object).ravel()

    for axis in axes_array:
        axis.vlines(knots, -1, 1, color="0.55", linestyle="--", linewidth=1.2)
        axis.plot(x_array, y_array, color="#2a78d6", linewidth=2, alpha=0.9)
        axis.set_xticks([])
        axis.set_yticks([])

    labels = ("分段常数", "分段线性", "分段二次", "分段三次")
    for degree, axis, label in zip(range(4), axes_array, labels, strict=True):
        design = basis(x_array, degree, knots)
        axis.plot(x_array, ols(design, y_array), color="black", linewidth=2)
        axis.set_title(label)

    return figure, axes_array


__all__ = ["basis", "ols", "splines"]
