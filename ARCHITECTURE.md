# GravLensAI Architecture

This document summarizes the current implementation of GravLensAI in a concise, repository-friendly format.

## Overview

GravLensAI combines simulated lens generation, a Vision Transformer classifier, a ResNet-based regressor, and optional domain adaptation to detect strong gravitational lenses and estimate lens parameters from 64×64 grayscale images.

## Core Components

### Simulation

- `gravlensai/simulate/lens_generator.py` builds synthetic lens and non-lens images.
- `lenstronomy` provides ray-tracing and lensing physics.
- `GalSim` supports realistic source galaxies, PSF effects, and noise.

### Data

- `gravlensai/data/dataset.py` implements simulated and HST datasets.
- `gravlensai/data/normalisation.py` applies arcsinh-based normalisation.
- `gravlensai/data/augmentation.py` provides geometric augmentation.

### Models

- `gravlensai/models/classifier.py` uses a ViT-Tiny backbone for binary lens classification.
- `gravlensai/models/regressor.py` uses a ResNet-18 backbone for 6-parameter regression.
- `gravlensai/models/domain_adapt.py` contains MMD and DANN utilities.

### Evaluation

- `gravlensai/evaluate/metrics.py` computes classification and regression metrics.
- `gravlensai/evaluate/grad_cam.py` supports interpretability.
- `gravlensai/evaluate/lenstool_compare.py` compares against classical modelling.

## Model Outputs

- Classifier output: lens probability.
- Regressor outputs: Einstein radius, ellipticity components, shear components, and subhalo mass.

## Performance Notes

- Simulated benchmark metrics are reported in `README.md`.
- Real-data validation is explicitly separated from simulated results.

## Repository Hygiene

- Keep generated artifacts under `results/`.
- Keep documentation that explains implementation decisions in this file or the README.