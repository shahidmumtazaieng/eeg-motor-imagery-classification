"""EEGBCI acquisition and subject-wise dataset assembly."""

from pathlib import Path
from typing import Iterable

import numpy as np

from .preprocessing import load_subject_epochs


def load_dataset(
    subjects: Iterable[int],
    data_dir: str | Path | None = None,
    runs: tuple[int, ...] = (4, 8, 12),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """Load binary left/right imagery epochs and participant group labels."""
    subject_ids = list(dict.fromkeys(int(subject) for subject in subjects))
    if not subject_ids:
        raise ValueError("Provide at least one subject ID.")
    if any(subject < 1 or subject > 109 for subject in subject_ids):
        raise ValueError("EEGBCI subject IDs must be between 1 and 109.")

    all_data: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    channel_names: list[str] | None = None
    sampling_frequency: float | None = None

    for subject in subject_ids:
        epochs = load_subject_epochs(subject, data_dir=data_dir, runs=runs)
        if channel_names is None:
            channel_names = list(epochs.ch_names)
            sampling_frequency = float(epochs.info["sfreq"])
        elif channel_names != list(epochs.ch_names):
            raise ValueError(f"Channel layout differs for subject {subject}.")

        labels = np.where(epochs.events[:, -1] == epochs.event_id["left_hand"], 0, 1)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(
                f"Subject {subject} does not contain both hand-imagery classes."
            )
        all_data.append(epochs.get_data())
        all_labels.append(labels.astype(np.int64))
        all_groups.append(np.full(len(labels), subject, dtype=np.int64))

    if channel_names is None or sampling_frequency is None:
        raise RuntimeError("No EEG epochs were loaded.")
    return (
        np.concatenate(all_data),
        np.concatenate(all_labels),
        np.concatenate(all_groups),
        channel_names,
        sampling_frequency,
    )
