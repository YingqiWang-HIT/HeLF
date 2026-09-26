# Data format

## Directory layout

```text
data_root/
├── train/
│   ├── sample_00000.npz
│   └── ...
├── val/
└── test/
```

Each `.npz` file is an independent geometry-condition flow sample.

## Required arrays

| Key | Shape | Description |
|---|---:|---|
| `sdf` | `[H, W]` | Signed distance to the airfoil boundary. Negative inside. |
| `flow` | `[4, H, W]` | Primitive fields ordered as density, x velocity, y velocity, pressure. |
| `condition` | `[2]` | Freestream Mach number and angle of attack in degrees. |

## Optional arrays

| Key | Shape | Description |
|---|---:|---|
| `wall_mask` | `[H, W]` | Airfoil-wall band used for no-penetration diagnostics. |
| `farfield_mask` | `[H, W]` | Outer boundary used for freestream matching. |
| `shock_mask` | `[H, W]` | Reference shock region. When present, evaluation uses it for shock-dependent residuals; otherwise the region is derived from the reference field. |
| `spacing` | `[2]` | Grid spacing `(dy, dx)` in physical coordinates. |
| `geometry_id` | scalar/string | Geometry identifier. |
| `group` | scalar/string | Dataset group such as `NACA0012-CST`, `RAE2822-CST`, or `UIUC`. |

If masks are absent, the loader derives a narrow wall mask from the SDF and an
outer-edge far-field mask. The model predicts saliency and does not require a
reference shock mask during inference.

## Splitting rule

All operating conditions of one airfoil geometry must remain in the same split.
Splitting individual fields at random would leak geometry information and inflate
generalization performance.

## Normalization

The loader supports fixed per-variable mean and standard deviation values from
the YAML configuration. For official experiments, compute them on the training
split only and apply the same values to validation and test fields.

The public review configuration contains neutral defaults. Exact official
statistics will be released after peer review.

## Physical coordinates

The public code assumes a rectangular tensor grid. When physical residuals are
reported, `spacing` should match the physical grid. If omitted, unit spacing is
used, which is suitable only for code testing and relative diagnostics.

## Synthetic data

`scripts/make_synthetic_data.py` creates analytic, airfoil-like fields with a
moving shock transition. These samples validate the software pipeline but are
not substitutes for CFD/LES data and must not be used to claim paper results.
