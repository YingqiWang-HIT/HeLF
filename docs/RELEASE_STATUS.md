# Release status

## Current status: `0.3.0-review`

The associated HeLF manuscript is under peer review. HeLF was released
as APM-Diff in `0.1.0-review` and as SaFiD in `0.2.0-review`. This repository is a
review-stage reproducibility preview rather than the archival paper release.

## Available in the review-stage release

- full package and script layout
- executable PyTorch model components
- multiscale SDF geometry encoder with perturbation conditioning
- saliency-gated denoiser with deformable aggregation and sparse attention
- latent diffusion utilities and DDIM inference
- physical energy with conservation, boundary, and shock mismatch terms
- saliency gate for the shock and smooth regions, with partition-ablation options
- latent physical score, magnitude alignment, and reliability-gated latent fusion of physical mechanism and data features
- Stage A/B/C training implementation
- synthetic data generator and public data interface
- PSNR, SSIM, physical metrics, shared baseline evaluation, timing utilities, tests, and CI
- manuscript-reported public architecture and training-budget values

## Intentionally withheld until peer review is complete

The following items are not contained in the public repository at this stage:

1. Exact CFD/LES mesh generation, solver controls, convergence filters, and
   field post-processing parameters.
2. Official geometry identifiers and complete train/validation/test manifests.
3. Exact variable-wise normalization statistics computed from the training set.
4. Exact Stage A/B/C loss weights and all internal calibration constants.
5. Exact fusion timesteps T_p and reference energy E_ref.
6. Official random seed list, checkpoints, training histories, and result files.
7. Table- and figure-specific reproduction scripts tied to the restricted data.

Public YAML values marked `REVIEW_DEFAULT` keep the code executable but are not
asserted to be the values used for the manuscript tables.

## Planned archival release

After the peer-review process is complete, the repository is intended to add:

- `configs/paper_final.yaml`
- official split manifests and normalization statistics
- complete data-preparation documentation
- official checkpoints where redistribution is permitted
- exact evaluation and paper-figure scripts
- final citation metadata and publication link

No release date is promised because it depends on the peer-review and data
clearance processes.
