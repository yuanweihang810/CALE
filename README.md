# CALE: Condition-Aware Label Expansion for Multi-Label Facial Expression Recognition

## Description
This repository contains the official code, data split indices, and documentation for the paper:

> **Aggregation-induced calibration failure in sum-based label expansion for multi-label facial expression recognition: a diagnostic study**

The paper presents a diagnostic empirical study of Condition-Aware Label Expansion (CALE), a label-space organization strategy that expands seven basic emotions into a 28-dimensional joint output (7 emotions × 4 image-quality conditions). The study reports predominantly negative results and identifies an aggregation-induced calibration failure that is general to sum-based label expansion.

## Dataset Information
The dataset consists of:
- **Training set**: 30,240 images (7 emotions × 4 quality conditions, with 20 variants per source image)
- **Test set**: 3,460 independently collected multi-label images

The dataset is permanently archived on Zenodo:

- **DOI**: [https://doi.org/10.5281/zenodo.23252243](https://doi.org/10.5281/zenodo.23252243)

The dataset repository contains the training set and the independent multi-label test set. Model weights are available separately (see below).

## Model Weights
Trained model weights for the paper are archived on Figshare. The archive contains the weights for ResNet-18, ResNet-34, ResNet-50, MobileNetV2, and EfficientNet-B0 backbones under different data regimes.

- **Figshare DOI**: [https://doi.org/10.6084/m9.figshare.34284447](https://doi.org/10.6084/m9.figshare.34284447)

## Code Information
The repository contains the following scripts (all comments and outputs are in English):

- `run_baseline_10pct_1fold.py` – Baseline (7-dim) with ResNet-18 on 10% data
- `run_baseline_10pct_1fold_resnet34.py` – Baseline with ResNet-34 on 10% data
- `run_baseline_ml_10pct_1fold.py` – 7-dim sigmoid multi-label baseline on 10% data
- `run_cale_v2_10pct_1fold.py` – CALE V2 on 10% data
- `run_cale_v2_10pct_1fold_resnet34.py` – CALE V2 with ResNet-34
- `run_cale_v2_10pct_1fold_resnet50.py` – CALE V2 with ResNet-50
- `run_cale_v2_10pct_1fold_mobilenetv2.py` – CALE V2 with MobileNetV2
- `run_cale_v2_10pct_1fold_efficientnetb0.py` – CALE V2 with EfficientNet-B0
- `run_cale_v2_10pct_1fold_softlabel0.py` – CALE V2 with SOFT_LABEL=0.0 (diagnostic)
- `run_baseline_singlelabel.py` – Baseline V0/V1/V3/V4 5-fold single-label training
- `run_cale_singlelabel.py` – CALE V2/V5/V6/V7 5-fold single-label training

Data-split indices (grouped by source image to prevent leakage) are included in the `splits/` folder.

## Usage Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yuanweihang810/CALE.git
   cd CALE
