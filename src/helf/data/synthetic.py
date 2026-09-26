"""Synthetic airfoil-like fields for functional tests only."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .sdf import farfield_mask, polygon_signed_distance, wall_mask_from_sdf


def naca_like_polygon(
    thickness: float = 0.12,
    camber: float = 0.02,
    camber_position: float = 0.4,
    points: int = 200,
) -> np.ndarray:
    beta = np.linspace(0.0, math.pi, points // 2)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5 * thickness * (
        0.2969 * np.sqrt(np.clip(x, 1e-8, None))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    m = camber
    p = np.clip(camber_position, 0.1, 0.9)
    yc = np.where(
        x < p,
        m / p**2 * (2 * p * x - x**2),
        m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2),
    )
    dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    polygon = np.stack(
        [np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])], axis=1
    )
    return polygon.astype(np.float32)


def pseudo_flow(
    sdf: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mach: float,
    alpha_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.deg2rad(alpha_deg)
    outside = (sdf >= 0).astype(np.float32)
    distance_decay = np.exp(-np.maximum(sdf, 0.0) / 0.12)
    shock_x = 0.62 - 0.22 * (mach - 0.8) - 0.008 * alpha_deg
    shock_width = 0.012 + 0.02 * max(1.0 - mach, 0.0)
    shock_line = shock_x + 0.08 * y
    jump = 0.5 * (1.0 + np.tanh((x - shock_line) / shock_width))
    shock_strength = np.clip(0.10 + 0.24 * (mach - 0.8) + 0.004 * alpha_deg, 0.05, 0.30)

    circulation = 0.08 * alpha_deg / 15.0 * np.exp(-((x - 0.25) ** 2 + y**2) / 0.25)
    rho = 1.0 + 0.08 * distance_decay + shock_strength * 0.45 * jump
    u_inf = mach * np.cos(alpha)
    v_inf = mach * np.sin(alpha)
    u = u_inf * (1.0 - 0.10 * distance_decay) - shock_strength * 0.35 * jump
    v = v_inf + circulation * np.sign(y + 1e-6) - 0.02 * jump
    p = 1.0 + 0.12 * distance_decay + shock_strength * jump

    # Inside-body values are kept finite to simplify tensor learning; masks identify the body.
    rho = rho * outside + 1.0 * (1.0 - outside)
    u = u * outside
    v = v * outside
    p = p * outside + 1.0 * (1.0 - outside)
    flow = np.stack([rho, u, v, p], axis=0).astype(np.float32)

    grad_y, grad_x = np.gradient(p)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    threshold = np.quantile(grad[outside > 0], 0.96)
    shock_mask = ((grad >= threshold) & (outside > 0)).astype(np.float32)
    return flow, shock_mask


def generate_dataset(
    output: str | Path,
    counts: dict[str, int],
    height: int,
    width: int,
    seed: int = 0,
) -> None:
    output = Path(output)
    rng = np.random.default_rng(seed)
    mach_values = np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float32)
    alpha_values = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float32)

    for split, count in counts.items():
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            thickness = float(rng.uniform(0.08, 0.18))
            camber = float(rng.uniform(-0.02, 0.05))
            camber_pos = float(rng.uniform(0.3, 0.6))
            polygon = naca_like_polygon(thickness, camber, camber_pos)
            sdf, x, y = polygon_signed_distance(polygon, height, width)
            mach = float(rng.choice(mach_values))
            alpha = float(rng.choice(alpha_values))
            flow, shock_mask = pseudo_flow(sdf, x, y, mach, alpha)
            wall = wall_mask_from_sdf(sdf, band=2.0 / max(height, width))
            far = farfield_mask(height, width, thickness=max(1, min(height, width) // 32))
            spacing = np.array(
                [(1.5) / max(height - 1, 1), (2.0) / max(width - 1, 1)], dtype=np.float32
            )
            np.savez_compressed(
                split_dir / f"sample_{index:05d}.npz",
                sdf=sdf,
                flow=flow,
                condition=np.array([mach, alpha], dtype=np.float32),
                wall_mask=wall,
                farfield_mask=far,
                shock_mask=shock_mask,
                spacing=spacing,
            )
