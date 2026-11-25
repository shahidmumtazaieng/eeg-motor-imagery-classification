"""Leakage-aware subject-dependent and subject-independent evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .feature_extraction import extract_band_power, motor_channel_indices
from .models import EEGNet, make_classical_models, make_csp_lda


def _validate_dataset(
    data: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(data)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if data.ndim != 3 or labels.ndim != 1 or groups.ndim != 1:
        raise ValueError("Expected data (epochs, channels, time), labels, and groups.")
    if len(data) == 0 or len(data) != len(labels) or len(labels) != len(groups):
        raise ValueError("Data, labels, and groups must have the same nonzero length.")
    if not np.isfinite(data).all():
        raise ValueError("EEG data contains non-finite values.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Labels must contain both binary classes encoded as 0 and 1.")
    return data, labels, groups


def _record_metrics(
    model_name: str,
    evaluation: str,
    held_out: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_score: np.ndarray,
) -> dict:
    return {
        "model": model_name,
        "evaluation": evaluation,
        "held_out": held_out,
        "n_test": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, positive_score),
    }


def evaluate_subject_independent(
    data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    channel_names: list[str],
    sampling_frequency: float,
    include_eegnet: bool = True,
    eegnet_epochs: int = 30,
) -> pd.DataFrame:
    """Hold out complete participants in each fold; no participant overlap."""
    data, labels, groups = _validate_dataset(data, labels, groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Subject-independent evaluation requires at least two subjects.")
    features = extract_band_power(
        data,
        sampling_frequency,
        motor_channel_indices(channel_names),
    )
    cv = GroupKFold(n_splits=len(unique_groups))
    records: list[dict] = []
    models = make_classical_models()

    for fold, (train, test) in enumerate(cv.split(data, labels, groups), start=1):
        if set(np.unique(labels[train])) != {0, 1} or set(np.unique(labels[test])) != {0, 1}:
            raise ValueError(
                "Each subject-independent fold must contain both classes in train and test."
            )
        held_out = ",".join(map(str, np.unique(groups[test]).tolist()))
        estimators = [("csp_lda", make_csp_lda(), data)]
        estimators.extend((name, model, features) for name, model in models.items())
        for name, estimator, inputs in estimators:
            estimator.fit(inputs[train], labels[train])
            predictions = estimator.predict(inputs[test])
            scores = estimator.predict_proba(inputs[test])[:, 1]
            records.append(
                _record_metrics(
                    name,
                    "subject_independent",
                    held_out,
                    labels[test],
                    predictions,
                    scores,
                )
            )

        if include_eegnet:
            predictions, scores = EEGNet.fit_predict(
                data[train],
                labels[train],
                data[test],
                epochs=eegnet_epochs,
                seed=42 + fold,
            )
            records.append(
                _record_metrics(
                    "eegnet",
                    "subject_independent",
                    held_out,
                    labels[test],
                    predictions,
                    scores,
                )
            )
    return pd.DataFrame.from_records(records)


def subject_independent_predictions(
    data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    channel_names: list[str],
    sampling_frequency: float,
    model_name: str = "csp_lda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return out-of-participant predictions for confusion and ROC plots."""
    data, labels, groups = _validate_dataset(data, labels, groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Subject-independent evaluation requires at least two subjects.")
    if model_name == "csp_lda":
        model = make_csp_lda()
        inputs = data
    elif model_name in make_classical_models():
        model = make_classical_models()[model_name]
        inputs = extract_band_power(
            data,
            sampling_frequency,
            motor_channel_indices(channel_names),
        )
    else:
        raise ValueError(f"Unknown classical model: {model_name}")

    truth, predictions, scores = [], [], []
    cv = GroupKFold(n_splits=len(unique_groups))
    for train, test in cv.split(data, labels, groups):
        if set(np.unique(labels[train])) != {0, 1} or set(np.unique(labels[test])) != {0, 1}:
            raise ValueError(
                "Each subject-independent fold must contain both classes in train and test."
            )
        model.fit(inputs[train], labels[train])
        truth.append(labels[test])
        predictions.append(model.predict(inputs[test]))
        scores.append(model.predict_proba(inputs[test])[:, 1])
    return np.concatenate(truth), np.concatenate(predictions), np.concatenate(scores)


def evaluate_subject_dependent(
    data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    channel_names: list[str],
    sampling_frequency: float,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Stratified CV within each participant, using only original trials."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two.")
    data, labels, groups = _validate_dataset(data, labels, groups)
    features = extract_band_power(
        data,
        sampling_frequency,
        motor_channel_indices(channel_names),
    )
    records: list[dict] = []
    models = make_classical_models()

    for subject in np.unique(groups):
        subject_indices = np.flatnonzero(groups == subject)
        subject_labels = labels[subject_indices]
        counts = np.bincount(subject_labels, minlength=2)
        folds = min(n_splits, int(counts.min()))
        if folds < 2:
            raise ValueError(
                f"Subject {subject} has too few trials per class for stratified CV."
            )
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        subject_data = data[subject_indices]
        subject_features = features[subject_indices]

        for fold, (train, test) in enumerate(cv.split(subject_data, subject_labels), start=1):
            estimators = [("csp_lda", make_csp_lda(), subject_data)]
            estimators.extend(
                (name, model, subject_features) for name, model in models.items()
            )
            for name, estimator, inputs in estimators:
                estimator.fit(inputs[train], subject_labels[train])
                predictions = estimator.predict(inputs[test])
                scores = estimator.predict_proba(inputs[test])[:, 1]
                records.append(
                    _record_metrics(
                        name,
                        "subject_dependent",
                        f"{subject}:fold{fold}",
                        subject_labels[test],
                        predictions,
                        scores,
                    )
                )
    return pd.DataFrame.from_records(records)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level metrics while retaining between-fold variation."""
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]
    return (
        results.groupby(["evaluation", "model"])[metrics]
        .agg(["mean", "std"])
        .round(4)
    )
