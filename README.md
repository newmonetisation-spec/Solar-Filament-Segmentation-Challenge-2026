# Solar Filament Segmentation Challenge 2026

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official source code and solution overview for the **Solar Filament Segmentation Challenge 2026** hosted on Kaggle. This repository provides an end-to-end computer vision pipeline for automated detection and instance segmentation of solar filaments from Global Oscillation Network Group (GONG) $H\alpha$ observations.

---

## 📌 Overview & Background
Accurate monitoring of solar filaments is essential for space weather forecasting and solar physics research. GONG $H\alpha$ imagery presents significant segmentation challenges due to low contrast, complex background solar noise, and the delicate, elongated geometry of filaments. 

This solution implements a robust deep learning pipeline combining advanced data preprocessing, a high-capacity encoder-decoder segmentation architecture, and morphological post-processing to maximize Panoptic Quality.


## 🚀 Key Features & Methodology

### 1. Preprocessing & Data Augmentation
* **Contrast Enhancement & Normalization:** Custom histogram scaling optimized for solar disc features.
* **Limb Masking:** Application of circular boundary constraints to eliminate false-positive detections outside the active solar radius.
* **Robust Augmentations:** Implemented using [Albumentations](https://albumentations.ai/), incorporating random spatial rotations, flips, and intensity variations to ensure model invariance.

### 2. Model Architecture
* **Encoder-Backbone:** High-resolution feature extraction using residual blocks to preserve fine structural boundaries.
* **Loss Configuration:** Hybrid loss function combining Binary Cross-Entropy (BCE) and Soft-Dice Loss to effectively handle extreme class imbalance between sparse filament pixels and the background.

### 3. Post-Processing & Refinement
* **Dynamic Probability Thresholding:** Optimized via local validation splits to balance precision and recall.
* **Morphological Filtering:** Use of mathematical morphology (opening and closing kernels) to remove isolated pixel noise and bridge fragmented filament segments.

---

