# Changelog

## 0.3.0-review - 2026-09-26

Renamed the method from SaFiD to HeLF (heterogeneous latent fusion of physical
mechanisms and data features) to match the revised manuscript. The model,
training, and sampling computations are unchanged: with the same weights and
random state, sampling and all training losses are bit-identical to
0.2.0-review, and existing checkpoints load without changes.

Renamed:

| 0.2.0-review | 0.3.0-review |
|---|---|
| package `safid` | package `helf` |
| `SaFiD` | `HeLF` |
| `SaliencyGatedFusionSampler` | `LatentFusionSampler` |
| `FirstPrinciplesEnergy` | `PhysicalMechanismEnergy` |
| `model.first_principles_energy` | `model.physical_energy` |
| repository `YingqiWang-HIT/SaFiD` | `YingqiWang-HIT/HeLF` |

Unchanged names: `SaliencyGatedDenoiser`, `MultiscaleGeometryEncoder`, all
configuration keys, the `fusion_diagnostics` output, and the command-line
interfaces of every script.

Documentation now describes the three components as the latent fusion of three
kinds of heterogeneous features (data features, geometry and condition
features, and physical mechanism features), following the revised manuscript.
The sampler docstring states how the physical energy gradient is mapped through
the decoder into the latent space.

## 0.2.0-review - 2026-09-24

Renamed the method from APM-Diff to SaFiD to match the revised manuscript. The
model, training, and sampling computations are unchanged: with the same weights
and random state, sampling and all training losses are bit-identical to
0.1.0-review, and existing checkpoints load without changes.

Renamed:

| 0.1.0-review | 0.2.0-review |
|---|---|
| package `apmdiff` | package `safid` |
| `APMDiff` | `SaFiD` |
| `SaliencyGuidedDenoiser` | `SaliencyGatedDenoiser` |
| `AdaptivePhysicalSampler` | `SaliencyGatedFusionSampler` |
| `PhysicalEnergy` | `FirstPrinciplesEnergy` |
| `model.physical_energy` | `model.first_principles_energy` |
| sampler output `physics_diagnostics` | `fusion_diagnostics` |
| `infer.py` output `physics_diagnostics_json` | `fusion_diagnostics_json` |

Added:

- `shock_partition` (`saliency`, `gradient`, `none`) and `shock_dilation` options
  for the saliency gate, with override files in `configs/ablations/` for the
  partition ablation. The defaults reproduce the previous behavior.
- Reliability `r_t`, correction need `n_t`, calibration temperature `tau_t`, and
  shock fraction in the per-step fusion diagnostics.
- `safid.evaluation` and `scripts/evaluate_predictions.py`, which score SaFiD and
  baseline predictions with the same metrics and the same reference shock region.
- `scripts/evaluate.py --save-predictions` and `--shock-region`.
- Baseline list, protocol, and provenance manifest in `baselines/`. TransCFD is
  replaced by GrFormer and FMIGNN by PIDANO.
- `.gitignore` and a CI workflow, both referenced by the documentation but
  missing from 0.1.0-review.

Changed:

- `scripts/evaluate.py` now computes shock-dependent residuals on a reference
  shock region by default. Use `--shock-region prediction` for the previous
  behavior, in which SaFiD was scored on the region selected by its own saliency.

## 0.1.0-review - 2026-07-30

- Added the complete public software structure (released as APM-Diff).
- Added multiscale perturbation-conditioned geometry encoding.
- Added saliency-guided deformable aggregation and sparse attention.
- Added latent diffusion training and DDIM sampling.
- Added adaptive physical-score fusion and physical residual diagnostics.
- Added three-stage training scripts, synthetic data generation, tests, and CI.
- Marked confidential review-stage experimental values as documented public defaults.
