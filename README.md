# EEG Motor Imagery BCI

A complete end-to-end pipeline for EEG-based motor imagery classification using the [PhysioNet EEGBCI dataset](https://physionet.org/content/eegmmidb/1.0.0/).

## Overview

This project implements a Brain-Computer Interface (BCI) pipeline that classifies left vs. right hand motor imagery from EEG signals. It covers the full workflow from raw data loading to model evaluation.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01-loading-ipynb.ipynb` | Data loading from PhysioNet, electrode montage, event extraction |
| `02-preprocessing-ipynb.ipynb` | Bandpass filtering, notch filter, ICA artifact removal, epoching |
| `03-visualization-ipynb.ipynb` | PSD, topographic maps, ERD/ERS time-frequency analysis |
| `04-augmentation-ipynb.ipynb` | Gaussian noise, amplitude scaling, and mixup augmentation |
| `05-classification.ipynb` | CSP + LDA, SVM-RBF, cross-validation, confusion matrix, ROC curve |

## Dataset

- **Source:** PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI)
- **Subjects:** 3 subjects (expandable)
- **Runs:** 6 (real movement), 10 (imagery), 14 (imagery + rest)
- **Channels:** 64 EEG electrodes (10-20 montage, 160 Hz)
- **Classes:** Left hand (T1) vs. Right hand (T2)

## Methods

- **Preprocessing:** 1–40 Hz bandpass, 50 Hz notch, average reference, ICA
- **Feature extraction:** Common Spatial Patterns (CSP, 6 components) + band power (delta/theta/alpha/beta/gamma) on C3, C4, Cz
- **Classifiers:** CSP + LDA (standard BCI pipeline), SVM with RBF kernel
- **Evaluation:** 5-fold stratified cross-validation, GroupKFold (anti-leakage), accuracy / F1 / AUC-ROC

## Setup

```bash
pip install -r requirements.txt
```

Then open the notebooks in order (01 → 05). The first run will automatically download the EEGBCI dataset via MNE.


## Requirements

- Python 3.10+
- MNE-Python 1.5+
- scikit-learn 1.3+
- See `requirements.txt` for full list
