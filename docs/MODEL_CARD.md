# Model card: HeLF

## Intended use

HeLF is research software for reconstructing steady two-dimensional
transonic airfoil flow fields from an airfoil SDF and freestream operating
condition. It is intended for scientific study, surrogate-model development,
and controlled aerodynamic design experiments.

## Inputs and outputs

- Input geometry: one-channel signed-distance field.
- Input condition: Mach number and angle of attack.
- Output: four primitive fields, density, x velocity, y velocity, and pressure.

## Architecture

The model contains a convolutional variational autoencoder, a multiscale
perturbation-conditioned geometry encoder, a saliency-gated U-Net denoiser,
and a DDIM-like sampler that fuses physical mechanisms into the latent data
features. A learned shock saliency map assigns the conservation law to smooth
regions and the Rankine-Hugoniot relations to shock regions, and a reliability
gate sets the fusion weight at each fusion step.

## Limitations

- The physical mechanisms are inviscid compressible conservation and approximate
  shock and boundary constraints.
- Viscosity, turbulence closure, transition, and three-dimensional effects are
  not modeled explicitly by the public physical operator.
- The manuscript focuses on steady two-dimensional airfoil flow.
- Reliability outside the geometry and operating-condition range represented by
  the training data has not been established.
- The review-stage repository does not contain the official data or checkpoints.

## Safety and engineering use

This software is not a certified CFD solver and must not be used as the sole
basis for flight control, structural certification, or safety-critical design
decisions. Predictions should be checked against trusted numerical or
experimental references.

## Review-stage disclosure

Selected settings required for exact paper reproduction remain withheld during
peer review. See `RELEASE_STATUS.md`.
