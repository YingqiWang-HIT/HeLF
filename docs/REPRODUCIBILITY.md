# Reproducibility checklist

## Environment

- Record Python, PyTorch, CUDA, cuDNN, GPU, and CPU versions.
- Use deterministic seeds where supported.
- Log whether deformable convolution used the torchvision operator or fallback.

## Data

- Keep geometry identities disjoint across splits.
- Record all operating conditions represented in every split.
- Fit normalization on training data only.
- Preserve field ordering: density, x velocity, y velocity, pressure.
- Save physical grid spacing and coordinate conventions.

## Training

- Run Stage A before Stage B, and Stage B before Stage C.
- Restore model and optimizer state when continuing an interrupted stage.
- Record configuration files inside every checkpoint.
- Maintain an exponential moving average for trainable parameters.
- Record the complete seed list for repeated trials.

## Evaluation

- Use the same test samples and normalization across all methods.
- Score every method with `scripts/evaluate_predictions.py` and the default
  reference shock region, so shock-dependent residuals use the same region.
- Report PSNR and SSIM per sample before averaging.
- Denormalize fields before evaluating physical residuals.
- Synchronize CUDA before and after timing.
- Perform warm-up runs before latency measurement.
- For paired tests, use matched seeds and matched test samples.

## Review-stage caveat

The current repository can verify tensor shapes, optimization flow, sampling,
latent physical scores, and data interfaces. Exact numerical reproduction of the
manuscript is deferred until official data, final parameters, and checkpoints
are released.
