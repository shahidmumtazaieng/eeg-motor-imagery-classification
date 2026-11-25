# Methodology

## Research question

Can EEG epochs from imagined left- and right-hand movement be distinguished,
and how does performance differ when evaluating trials from a known participant
versus a participant absent from training?

## Dataset protocol

The project uses the PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI)
through MNE. The default run set is 4, 8, and 12, the dataset's repeated
left-versus-right hand motor-imagery task. In this paradigm, T1 indicates the
left hand and T2 indicates the right hand. Run sets 6, 10, and 14 represent
imagined hands versus feet; their event markers must not be interpreted as
left/right labels. This implementation deliberately restricts the run argument
to 4, 8, and 12 to prevent accidental label mixing.

EEGBCI contains 64 EEG channels sampled at 160 Hz. MNE downloads recordings
into its configured data directory. Dataset files are not included or
committed.

## Preprocessing

1. Standardize EEGBCI channel names and attach the standard 10-20 montage.
2. Apply a zero-phase 1–45 Hz FIR bandpass and a 50 Hz notch filter to EEG
   channels.
3. Apply an average EEG reference.
4. Extract T1/T2 event epochs from 0 to 4 seconds after the event. Bad
   concatenation-boundary annotations are rejected by MNE.

The pipeline intentionally does not auto-remove ICA components. Automatic
component selection without review can remove neural activity as well as
artifacts. A future study should document artifact handling and rejection
thresholds explicitly and consistently.

## Features and models

- **CSP + LDA:** learns spatial filters from the raw multichannel epochs.
  CSP is part of the estimator pipeline and is fitted separately inside each
  training fold.
- **Logistic Regression, RBF-SVM, Random Forest:** use log mean PSD within
  delta, theta, alpha, beta, and gamma bands from C3, C4, and Cz when available.
  Feature scaling is fitted on training folds only.
- **EEGNet-style CNN:** operates on raw epochs and standardizes each channel
  using training-fold statistics only. It is an optional baseline; the fixed
  epoch count is not tuned on the test participant.

There is no synthetic augmentation in evaluation. If augmentation is added,
each synthetic example must remain grouped with its source trial and be
generated from training-fold data only.

## Validation and metrics

### Subject-dependent

Each participant is evaluated independently with shuffled, stratified
five-fold CV (or fewer folds when a participant has fewer trials in one class).
This estimates performance when training data from that person is available.

### Subject-independent

GroupKFold assigns complete participants to held-out test folds. No epochs
from a test participant enter that fold's training set. This estimates
generalization to unseen participants and is the more demanding of the two
settings. It is not a substitute for an external-site or prospective study.

For both designs the code reports accuracy, balanced accuracy, macro F1, and
ROC AUC at fold level. Fold means and standard deviations are descriptive:
folds are not necessarily independent, especially in participant-wise
evaluation.

## Limitations

- EEGBCI is a controlled public dataset; real-world BCI performance may differ.
- This implementation handles two imagery classes only.
- Participants, recording sessions, and trial counts are heterogeneous.
- Standard filtering and average reference do not remove every artifact.
- The default EEGNet hyperparameters are a baseline, not an optimized model.
- Results should include participant-level variation, class counts, and
  uncertainty; the highest accuracy alone is not a meaningful conclusion.
- This project is not a medical device and must not be used for diagnosis or
  treatment.

## Data citation

Schalk, G. (2009). *EEG Motor Movement/Imagery Dataset* (version 1.0.0).
PhysioNet. https://doi.org/10.13026/C28G6P

Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R.
(2004). BCI2000: A general-purpose brain-computer interface (BCI) system.
*IEEE Transactions on Biomedical Engineering, 51*(6), 1034–1043.
https://doi.org/10.1109/TBME.2004.827072
