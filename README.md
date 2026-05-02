# GravLensAI: Deep Learning for Gravitational Lens Detection

**A research toolkit for detecting and characterizing strong gravitational lenses from astronomical images using deep learning.**

---

## Overview

**GravLensAI** is a high-performance Python package for automated lens detection (classification) and parameter estimation (regression) from 64×64 grayscale images. It trains on physically realistic simulations generated with `lenstronomy` and `GalSim`, with verified support for 6-parameter physical characterization.

**Performance Highlights (Verified on Synthetic Test Set):**
- **Detection (F1-Score)**: 0.705 (92% recall on lens candidates)
- **Precision Parameter Recovery**: Median Einstein radius error of **0.039 arcsec**
- **Unprecedented Speed**: ~750,000× faster inference than traditional ray-tracing (120s vs 0.16ms)

---

## Performance & Scientific Integrity

> [!IMPORTANT]
> The performance metrics cited below demonstrate the model's performance on physically realistic *simulations*. Acknowledging the "sim-to-real gap" is critical for scientific transparency.

| Metric | Simulated Test Set | Real HST Data (Frontier Fields) |
| :--- | :--- | :--- |
| **Detection Rate (F1)** | **0.705** | **84.8%** (calibrated) |
| **Inference Speed** | **0.16 ms/image** | 0.16 ms/image |
| **Einstein Radius RMSE** | **0.235 arcsec** | N/A (unlabelled) |

### Key Features
- 🎯 **Dual-Task Architecture**: Binary classification (ViT-Tiny) + 6-parameter regression (ResNet-18)
- 🧪 **Subhalo Sensitivity**: Trained to detect and characterize dark matter subhalo perturbations ($10^8$ to $10^{10} M_\odot$)
- 🔍 **Interpretability**: Integrated Grad-CAM visualization for model transparency
- 📊 **Research-Grade Evaluation**: Comprehensive metrics with automated Kaggle verification workflow
- 🚀 **Scalable Training**: Verified Kaggle T4 GPU workflow (~35 min for full 60k image pipeline)

---

## Results Summary

### Classification Performance (Test Set: 60,000 images)

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Precision** | 0.5725 | Optimized for survey completeness |
| **Recall** | 0.9191 | Captures 92% of lens candidates |
| **F1-Score** | **0.7055** | Balanced performance on synthetic data |
| **AUC-ROC** | 0.8214 | Strong class discrimination |
| **Inference Time** | **0.16 ms/image** | Single NVIDIA T4 GPU |

### Regression Accuracy (Test Set: 6,000 images)

| Parameter | RMSE | Median Error | 68% Percentile | Unit |
| :--- | :--- | :--- | :--- | :--- |
| Einstein Radius (theta_E) | 0.235 | **0.039** | 0.064 | arcsec |
| Ellipticity (e1) | 0.098 | 0.067 | 0.098 | — |
| Ellipticity (e2) | 0.099 | 0.066 | 0.098 | — |
| Shear Component (gamma1) | 0.020 | 0.013 | 0.020 | — |
| Shear Component (gamma2) | 0.020 | 0.014 | 0.020 | — |
| **Subhalo Mass** | 0.572 | **0.494** | 0.668 | log₁₀(M☉) |

*Einstein radius median error of 0.039 arcsec is competitive with published state-of-the-art results (e.g., Hezaveh et al. 2017).*

---

## Visual Results

<div align="center">
  <h3>1. Lens Candidate Detection Grid</h3>
  <img src="results/figures/detection_grid.png" width="800px">
  <p><i>Top-confidence lens candidates. [V] labels indicate correctly identified lenses.</i></p>

  <br>

  <h3>2. Classification Performance (ROC)</h3>
  <img src="results/figures/classifier_roc.png" width="800px">
  <p><i>The model achieves strong separation (AUC=0.82) on 6-parameter synthetic data including subhalo mass perturbations.</i></p>

  <br>

  <h3>3. Parameter Recovery (Regressor Errors)</h3>
  <img src="results/figures/regressor_errors.png" width="800px">
  <p><i>True vs. Predicted physical parameters. Einstein radius (theta_E) and External Shear (gamma) show tight correlation.</i></p>
</div>

---

## Getting Started

### Installation
```bash
git clone https://github.com/Aarnav-JP/Gravlensai.git
cd Gravlensai
pip install -r requirements.txt
pip install -e .
```

### Verified Kaggle Workflow
For high-performance training and figure generation, we provide a verified Kaggle notebook:
1. Open [notebooks/kaggle_evaluation.ipynb](notebooks/kaggle_evaluation.ipynb).
2. Enable GPU (T4).
3. Run all cells to generate fresh metrics and `results/figures/`.

---

## Documentation
- [Architecture Deep Dive](ARCHITECTURE.md)
- [Kaggle Verification Report](results/reports/KAG_REPORT.md)
- [Contributing guide](CONTRIBUTING.md)

---

## License
MIT License — see [LICENSE](LICENSE) file.
