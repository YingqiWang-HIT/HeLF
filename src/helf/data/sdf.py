"""Signed-distance utilities for two-dimensional closed polygons."""

from __future__ import annotations

import numpy as np


def polygon_signed_distance(
    polygon: np.ndarray,
    height: int,
    width: int,
    xlim: tuple[float, float] = (-0.5, 1.5),
    ylim: tuple[float, float] = (-0.75, 0.75),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a signed distance field on a Cartesian grid.

    Negative values are inside the polygon. The implementation is vectorized
    over pixels and loops only over boundary segments.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError("polygon must have shape [N, 2] with N >= 3")
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.concatenate([polygon, polygon[:1]], axis=0)

    xs = np.linspace(*xlim, width, dtype=np.float64)
    ys = np.linspace(*ylim, height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    min_dist2 = np.full(points.shape[0], np.inf, dtype=np.float64)

    inside = np.zeros(points.shape[0], dtype=bool)
    px = points[:, 0]
    py = points[:, 1]

    for p0, p1 in zip(polygon[:-1], polygon[1:]):
        segment = p1 - p0
        denom = float(np.dot(segment, segment)) + 1e-15
        t = np.clip(((points - p0) @ segment) / denom, 0.0, 1.0)
        closest = p0 + t[:, None] * segment
        dist2 = np.sum((points - closest) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)

        y_cross = (p0[1] > py) != (p1[1] > py)
        x_intersection = (p1[0] - p0[0]) * (py - p0[1]) / (p1[1] - p0[1] + 1e-15) + p0[0]
        inside ^= y_cross & (px < x_intersection)

    distance = np.sqrt(min_dist2)
    distance[inside] *= -1.0
    return distance.reshape(height, width).astype(np.float32), grid_x.astype(np.float32), grid_y.astype(np.float32)


def wall_mask_from_sdf(sdf: np.ndarray, band: float | None = None) -> np.ndarray:
    if band is None:
        scale = max(float(np.ptp(sdf)), 1e-6)
        band = 0.005 * scale
    return (np.abs(sdf) <= band).astype(np.float32)


def farfield_mask(height: int, width: int, thickness: int = 2) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:thickness] = 1
    mask[-thickness:] = 1
    mask[:, :thickness] = 1
    mask[:, -thickness:] = 1
    return mask
