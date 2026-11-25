# EEG Motor Imagery Classification

An end-to-end research project for decoding imagined left- versus right-hand
movements from EEG using PhysioNet's EEG Motor Movement/Imagery Dataset
(EEGBCI), MNE-Python, scikit-learn, and PyTorch.

The project is organized as reusable Python code, runnable notebooks, tests,
and an optional Streamlit demo. It compares classical EEG/ML baselines with a
compact EEGNet-style CNN and reports both subject-dependent and
subject-independent results.

## Dataset and task

The default experiment is **binary left- versus right-hand motor imagery**.
It uses EEGBCI runs **4, 8, and 12**. Do not mix these with runs 6, 10, and 14:
those runs represent imagined hands-versus-feet tasks, and the meanings of the
T1/T2 event markers depend on the run paradigm.

The dataset contains recordings from 109 participants, 64 EEG channels, and a
160 Hz sampling rate. MNE downloads files on first use. The project keeps the
dataset out of source control and does not bundle participant recordings.

## Repository layout

```text
.
├── app.py
├── docs/
│   └── methodology.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eeg_preprocessing.ipynb
│   ├── 03_signal_analysis.ipynb
│   ├── 04_feature_extraction.ipynb
│   ├── 05_classical_ml.ipynb
│   └── 06_deep_learning.ipynb
├── results/
│   └── figures/
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── models.py
│   ├── evaluation.py
│   └── train.py
├── tests/
└── requirements.txt
```

Previous tutorial notebooks are kept in `notebooks/archive/`; the new
notebooks use the reusable modules above.

## Setup

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the experiment

Download and evaluate participants 1–5 (subject-independent GroupKFold plus
within-subject stratified CV):

```bash
python -m src.train --subjects 1 2 3 4 5
```

Results are written to `results/`. Add `--skip-eegnet` for a faster run that
omits deep learning. The default EEGNet training runs on CPU; its epoch count
can be adjusted with `--eegnet-epochs`.

To run the Streamlit demo:

```bash
streamlit run app.py
```

Select at least two participants for subject-independent evaluation. The
application downloads data through MNE on first use; the first run may take a
while.

Run focused tests with:

```bash
python -m pytest
```

## Evaluation and interpretation

- **Subject-dependent:** each participant's trials are split into stratified
  folds. This measures performance when some labeled data from that person is
  available during training.
- **Subject-independent:** GroupKFold keeps every trial from a participant
  together, so test participants are unseen during training.
- **Models:** CSP+LDA on raw epochs, plus Logistic Regression, RBF-SVM, and
  Random Forest on log band-power features. EEGNet is an optional raw-epoch
  deep-learning baseline.
- **Metrics:** accuracy, balanced accuracy, macro F1, and ROC AUC.

Subject-independent performance is the more demanding deployment estimate.
Scores depend on selected participants, preprocessing, and the small number of
participants used. A research result should report participant-wise scores and
uncertainty, not only one aggregate accuracy. This project is for research and
education; it is not a medical device or a validated clinical system.

See [docs/methodology.md](docs/methodology.md) for protocol choices, signal
processing, validation design, and limitations.

## Citation

Please cite the EEG Motor Movement/Imagery Dataset and its original BCI2000
publication when using these data. See the PhysioNet dataset page:
https://physionet.org/content/eegmmidb/1.0.0/
