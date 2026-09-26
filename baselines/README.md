# Baseline integration

The manuscript compares HeLF with ten methods in four categories. Third-party
baseline source code is not copied into this repository because each project has
its own license and versioning. `manifest.yaml` records the provenance and the
adaptation of every baseline and must be completed before results are reported.

## Compared methods

| Category | Method | Reference in the manuscript |
|---|---|---|
| Deterministic model | ARASF | Liu et al., Nat. Mach. Intell. (2025) |
| Deterministic model | GrFormer | Kang et al., Inf. Fusion 125 (2026) 103402 |
| Deterministic model | Flow3DNet | Zuo et al., Aerosp. Sci. Technol. (2025) |
| Graph model | Hines-GNN | Hines and Bekemeyer, Aerosp. Sci. Technol. 137 (2023) 108268 |
| Graph and physics-informed model | PIDANO | Inf. Fusion 136 (2026) 104553 |
| Diffusion and Gaussian model | FlowViT-Diff | Lou et al., Chin. J. Aeronaut. 38 (2025) 103624 |
| Diffusion and Gaussian model | FluidGS | Xie et al., ACM MM (2025) |
| Fusion method | Diff-IF | Yi et al., Inf. Fusion 110 (2024) 102450 |
| Fusion method | LFDT-Fusion | Yang et al., Inf. Fusion 113 (2025) 102639 |
| Fusion method | AtNet | Tang et al., Inf. Fusion 120 (2025) 103045 |

PIDANO replaces FMIGNN, which the manuscript listed as a graph model. PIDANO is a
physics-informed neural operator rather than a graph network, so the manuscript
category should be renamed accordingly (for example, "graph and physics-informed
models") or PIDANO moved to another category.

### Changes from the review-stage baseline list

| Removed | Replaced by | Reason |
|---|---|---|
| TransCFD (Eng. Appl. Artif. Intell., 2023) | GrFormer (Inf. Fusion, 2026) | same class: single-pass deterministic Transformer regression |
| FMIGNN (Phys. Fluids, 2024) | PIDANO (Inf. Fusion, 2026) | same class: physics-constrained deterministic surrogate |

Results obtained for TransCFD and FMIGNN must not be reused for GrFormer or
PIDANO. Both replacements have to be trained and evaluated under the protocol
below, and their parameter count, NFE, and timing measured on the same hardware.

## Protocol

1. Install each baseline from its official source and record the exact commit.
2. Adapt its input and output to this task: input SDF and operating condition,
   output `[4, H, W]` in the order `rho, u, v, p`. Record the adaptation in
   `manifest.yaml`.
3. Use the same geometry-disjoint split and training-only normalization.
4. Export test predictions in physical units to an `.npz` file with
   `prediction` `[N, 4, H, W]` and `paths` (the sample files).
5. Score every method, including HeLF, with the same configuration:

   ```bash
   python scripts/evaluate_predictions.py --config configs/paper_public.yaml \
     --predictions outputs/<method>_test.npz --method <method>
   ```

   HeLF predictions are exported with
   `scripts/evaluate.py --save-predictions outputs/helf_test.npz`.
6. Use the default `--shock-region reference` so that shock-dependent residuals
   are computed on the same region for all methods.
7. Record the parameter count, NFE, and timing protocol (warm-up runs, number of
   timed runs, synchronization, hardware).

## Adapting the two new baselines

These notes describe a reasonable adaptation. Record the one actually used in
`manifest.yaml` and in the manuscript's implementation details.

**GrFormer** (official code: <https://github.com/Shaoyun2023/GrFormer>). It is a
two-input infrared-visible image fusion network. A direct adaptation feeds the
SDF and the operating condition broadcast to constant channels, replaces the
single-channel fused output with a four-channel head, and replaces the
unsupervised fusion losses with the same supervised reconstruction loss on
normalized fields that the other deterministic baselines use.

**PIDANO**. Official code was not located when this note was written; if none is
available, implement the model from the paper and state this in the manuscript.
PIDANO was proposed for physics-informed operator learning trained with physics
losses only. For a fair comparison on labelled flow fields, train it with the
supervised reconstruction loss plus its physics loss, conditioning the operator
on the SDF and the operating condition.
