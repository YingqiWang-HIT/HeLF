# HeLF

**Heterogeneous latent fusion of physical mechanisms and data features for transonic airfoil flow reconstruction across multiple conditions**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-under%20review-orange)](docs/RELEASE_STATUS.md)

> [!NOTE]
> HeLF was previously released under the names **SaFiD** and **APM-Diff**. The computations are unchanged; the package, classes, and documentation were renamed to match the revised manuscript. See [CHANGELOG.md](CHANGELOG.md) for the name mapping.

> [!IMPORTANT]
> **Review-stage release.** The associated manuscript is currently under peer review. This repository provides the complete public software structure, model interfaces, three-stage training pipeline, inference pipeline, latent fusion implementation, synthetic smoke-test data generator, tests, and documentation. A limited set of critical experimental details has intentionally been replaced by clearly marked public defaults. The exact values, dataset split identifiers, preprocessing statistics, selected fusion timesteps, full CFD/LES generation settings, official checkpoints, and complete benchmark metadata will be disclosed after peer review.
>
> Consequently, the present release is suitable for code inspection, extension, and end-to-end functional testing, but it is **not claimed to reproduce the manuscript tables exactly** until the archival release is published.

## Overview

HeLF reconstructs a steady transonic airfoil flow field

\[
\widehat{\mathbf q}=\Phi_\Theta(S_{\mathcal G},\mathbf c),\qquad
\mathbf q=[\rho,u,v,p]^\top,\quad \mathbf c=[Ma_\infty,\alpha]^\top,
\]

from an airfoil signed-distance field and an operating condition. It fuses three kinds of heterogeneous features in the latent space of a diffusion model:

- **Data features**: latent flow features learned from high-fidelity simulation fields by the variational autoencoder and the denoiser.
- **Geometry and condition features**: a spatial signed distance field and two scalar operating parameters, fused by layerwise modulation.
- **Physical mechanism features**: the gradient of a physical energy built from the differential conservation law, the wall and far-field boundary conditions, and the Rankine-Hugoniot jump relations, mapped through the decoder into the latent space.

These features differ in representation, spatial distribution, and origin. Each of the three components below handles one of these differences:

1. **Multiscale geometry encoder with perturbation conditioning** (`MultiscaleGeometryEncoder`): representation
   - multiscale SDF patch embeddings
   - condition and geometry perturbation embedding
   - layerwise scale-and-shift modulation of geometry tokens instead of direct concatenation of a field with scalars

2. **Saliency-gated latent diffusion denoiser** (`SaliencyGatedDenoiser`): spatial distribution
   - variational latent representation (the data features)
   - adaptive group normalization using diffusion time and operating condition
   - soft saliency gate: saliency-conditioned deformable local aggregation for sparse shock features
   - hard saliency gate: top-k sparse global attention linking shocks with smooth flow features

3. **Latent fusion sampler** (`LatentFusionSampler`, `PhysicalMechanismEnergy`): origin, scale, and reliability
   - saliency gate that assigns the shock region `Omega_s` and the smooth region `Omega_c`
   - differential conservation residual in `Omega_c`
   - approximate Rankine-Hugoniot mismatch in `Omega_s`
   - wall and far-field boundary penalties
   - latent physical score `s_p`, the energy gradient mapped through the decoder into the latent space
   - magnitude alignment of `s_p` to the diffusion score
   - reliability gate `omega_t = omega_max * r_t * n_t` over the fusion steps `T_p`

Training is separated into three stages:

- **Stage A:** variational autoencoder training with KL warm-up
- **Stage B:** geometry encoder, denoiser, saliency, and condition-continuity training
- **Stage C:** joint refinement with reconstruction and physical energy terms

## Repository structure

```text
HeLF/
├── baselines/
│   ├── README.md                  # baseline list and comparison protocol
│   └── manifest.yaml              # per-baseline provenance record (to be completed)
├── configs/
│   ├── paper_public.yaml          # manuscript-aligned public configuration
│   ├── smoke.yaml                 # lightweight CPU/GPU functional test
│   ├── ablations/                 # overrides for the partition ablation
│   └── private_override.example.yaml
├── data/                          # ignored placeholders for local datasets
├── docs/
│   ├── DATA_FORMAT.md
│   ├── MANUSCRIPT_CODE_AVAILABILITY.md
│   ├── MODEL_CARD.md
│   ├── RELEASE_STATUS.md
│   └── REPRODUCIBILITY.md
├── scripts/
│   ├── make_synthetic_data.py
│   ├── train.py
│   ├── train_stage_a.py
│   ├── train_stage_b.py
│   ├── train_stage_c.py
│   ├── infer.py
│   ├── evaluate.py
│   ├── evaluate_predictions.py    # scores baseline predictions with the same metrics
│   ├── benchmark.py
│   ├── inspect_checkpoint.py
│   └── validate_dataset.py
├── src/helf/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── config.py
│   ├── evaluation.py
│   ├── metrics.py
│   └── utils.py
├── tests/
├── .github/workflows/ci.yml
├── CITATION.cff
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Installation

The manuscript experiments used Python 3.9, PyTorch 2.0, and CUDA 11.8. Newer compatible PyTorch versions are also supported.

```bash
git clone https://github.com/YingqiWang-HIT/HeLF.git
cd HeLF
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

For development and testing:

```bash
pip install -e ".[dev]"
```

## Functional smoke test

Generate a small synthetic dataset with airfoil-like SDFs and shock-like fields:

```bash
python scripts/make_synthetic_data.py \
  --output data/synthetic \
  --train 32 --val 8 --test 8 \
  --height 64 --width 64
```

Run the three training stages:

```bash
python scripts/train.py --config configs/smoke.yaml --stage A
python scripts/train.py --config configs/smoke.yaml --stage B \
  --resume checkpoints/smoke/stage_a_best.pt
python scripts/train.py --config configs/smoke.yaml --stage C \
  --resume checkpoints/smoke/stage_b_best.pt
```

Run inference:

```bash
python scripts/infer.py \
  --config configs/smoke.yaml \
  --checkpoint checkpoints/smoke/stage_c_best.pt \
  --input data/synthetic/test/sample_00000.npz \
  --output outputs/prediction.npz
```

The output file contains the reconstructed field, the saliency map, and `fusion_diagnostics_json`, which records the energy terms, reliability `r_t`, correction need `n_t`, fusion weight `omega_t`, and calibration temperature `tau_t` at every fusion step.

Evaluate a split and export the predictions:

```bash
python scripts/evaluate.py \
  --config configs/smoke.yaml \
  --checkpoint checkpoints/smoke/stage_c_best.pt \
  --split test \
  --save-predictions outputs/helf_test.npz
```

Run tests:

```bash
pytest -q
```

## Comparing with baselines

All methods should be scored with the same metric code, normalization, and shock region. Export each baseline's test predictions in physical units to an `.npz` file with a `prediction` array of shape `[N, 4, H, W]` (order `rho, u, v, p`) and, preferably, a `paths` array naming the sample files, then run:

```bash
python scripts/evaluate_predictions.py \
  --config configs/paper_public.yaml \
  --predictions outputs/grformer_test.npz \
  --method GrFormer \
  --output outputs/grformer_eval.json
```

By default, shock-dependent residuals are evaluated on a reference shock region taken from the reference field (`--shock-region reference`), so that every method is scored on the same region. `--shock-region prediction` reproduces the review-stage behavior, in which each method is scored on the region it selects itself. The baseline list and the protocol are described in [baselines/README.md](baselines/README.md).

## Partition ablation

The manuscript's partition ablation keeps the trained networks fixed and changes only how physical mechanisms are assigned to shock and smooth regions during sampling:

```bash
# (a) Global conservation: conservation residual on the whole domain, no Rankine-Hugoniot term
python scripts/evaluate.py --config configs/paper_public.yaml \
  --override configs/ablations/partition_global_conservation.yaml \
  --checkpoint <stage_c_best.pt> --split test

# (b) Gradient partition: shock region from the gradient indicator of the decoded field
python scripts/evaluate.py --config configs/paper_public.yaml \
  --override configs/ablations/partition_gradient.yaml \
  --checkpoint <stage_c_best.pt> --split test
```

The default configuration (`shock_partition: saliency`) is the full HeLF model.

## Manuscript-aligned configuration

`configs/paper_public.yaml` records the architecture and training settings stated in the manuscript, including:

- flow channels: `rho, u, v, p`
- latent tensor: 4 channels at approximately 1/8 spatial resolution
- geometry patch scales: 4, 8, and 16
- geometry hidden dimension: 256
- condition representation dimension: 512
- geometry layers/heads: 6/8
- denoiser channels: 128, 256, 384, and 512
- salient fraction: 5%
- diffusion training steps: 1000
- reverse sampling steps: 50
- fusion steps `|T_p|`: 8 (`correction_count`)
- maximum fusion weight `omega_max`: 0.10
- Stage A/B/C epochs: 200/400/300
- batch sizes: 16/8/8

The saliency gate selects the top 5% of the saliency map as the shock region (`shock_quantile: 0.95`) without dilation (`shock_dilation: 0`), which is the behavior of the review-stage release. A dilation radius can be set if the final experiments use one.

Configuration keys from the review-stage release are kept for compatibility with existing private overrides: `correction_count` and `correction_steps` hold the number and set of fusion steps.

Values not explicitly disclosed in the manuscript are marked with `REVIEW_DEFAULT` comments. They are runnable engineering defaults and should not be interpreted as the final experimental values.

A private local override can be created without committing confidential settings:

```bash
cp configs/private_override.example.yaml configs/private_override.yaml
```

Then merge it at runtime:

```bash
python scripts/train.py \
  --config configs/paper_public.yaml \
  --override configs/private_override.yaml \
  --stage A
```

`configs/private_override.yaml` is excluded by `.gitignore`.

## Dataset

The paper evaluates three geometry groups: NACA0012-CST, RAE2822-CST, and UIUC-derived airfoils under multiple Mach numbers and angles of attack. The original CFD/LES fields are not included in this review-stage repository.

Expected sample format:

```python
np.savez_compressed(
    "sample.npz",
    sdf=sdf.astype("float32"),                 # [H, W]
    flow=flow.astype("float32"),               # [4, H, W], rho/u/v/p
    condition=np.array([mach, alpha_deg]),      # [2]
    wall_mask=wall_mask.astype("float32"),     # optional [H, W]
    farfield_mask=farfield_mask.astype("float32"), # optional [H, W]
    shock_mask=shock_mask.astype("float32"),   # optional [H, W], reference shock region
)
```

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the directory layout, masks, normalization, and metadata rules.

## Public/restricted boundary during peer review

The following components are included now:

- executable model definitions
- three-stage optimization code
- public architecture configuration
- DDPM/DDIM utilities
- physical energy, saliency gate, and reliability-gated latent fusion
- synthetic dataset generation
- evaluation metrics, baseline evaluation, and benchmark timing
- unit and smoke tests

The following items are scheduled for the post-review archival release:

- exact per-variable normalization statistics
- exact Stage A/B/C loss coefficients
- precise fusion timestep set `T_p`
- exact reference energy `E_ref`
- official train/validation/test airfoil identifiers
- complete CFD/LES solver, mesh, convergence, and post-processing settings
- official checkpoints and complete result logs
- scripts that reproduce every paper table and figure from the official data

The current public defaults are isolated in YAML rather than hard-coded, so the archival settings can be released without changing the software interface.

## Reproducibility notes

- Use the same seed list across methods for paired statistical comparisons.
- Store each airfoil geometry in only one split to avoid geometry leakage.
- Fit normalization statistics on the training split only.
- Score every method with `scripts/evaluate_predictions.py` and the same shock region.
- Report inference time after warm-up and synchronized device execution.
- Physical residuals are sensitive to grid spacing and field normalization. Use physical units or correctly denormalized fields.
- The included physical operator is an inviscid compressible approximation consistent with the current manuscript scope. It does not represent a full viscous/turbulence closure.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for a detailed checklist.

## Citation

The manuscript is under review. Until a final bibliographic record is available, cite the repository using `CITATION.cff` and identify it as a review-stage software release:

```bibtex
@software{helf_2026,
  author  = {Wang, Yingqi and Liu, Datong and Liu, Lei and Zhang, Yusu and Jia, Shiyan and Xiao, Qianxi and Song, Yuchen},
  title   = {{HeLF}: Heterogeneous latent fusion of physical mechanisms and data features for transonic airfoil flow reconstruction across multiple conditions},
  year    = {2026},
  version = {0.3.0-review},
  url     = {https://github.com/YingqiWang-HIT/HeLF}
}
```

## License

The software is released under the [MIT License](LICENSE). Dataset licenses and third-party baseline licenses remain separate.

## Contact

For issues concerning the public implementation, open a GitHub issue. Questions requiring undisclosed review-stage settings will be answered after the peer-review process is complete.
