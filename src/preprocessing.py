"""Preprocessing for the PhysioNet EEGBCI binary hand-imagery task."""

from pathlib import Path


def load_subject_epochs(
    subject: int,
    data_dir: str | Path | None = None,
    runs: tuple[int, ...] = (4, 8, 12),
    tmin: float = 0.0,
    tmax: float = 4.0,
):
    """Download, preprocess, and epoch one participant's left/right imagery."""
    if subject < 1 or subject > 109:
        raise ValueError("EEGBCI subject IDs must be between 1 and 109.")
    if tuple(runs) != (4, 8, 12):
        raise ValueError(
            "This binary hand-imagery pipeline supports EEGBCI runs 4, 8, and 12."
        )
    if tmin < 0 or tmax <= tmin:
        raise ValueError("Epoch bounds must satisfy 0 <= tmin < tmax.")

    import mne
    from mne.datasets import eegbci

    data_path = None if data_dir is None else str(Path(data_dir).expanduser())
    paths = eegbci.load_data(
        subject,
        list(runs),
        path=data_path,
        update_path=False,
        verbose="ERROR",
    )
    recordings = [
        mne.io.read_raw_edf(path, preload=True, verbose="ERROR") for path in paths
    ]
    raw = mne.concatenate_raws(recordings, verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage(
        mne.channels.make_standard_montage("colin27_1020"),
        on_missing="ignore",
        verbose="ERROR",
    )

    raw.filter(
        l_freq=1.0,
        h_freq=45.0,
        picks="eeg",
        method="fir",
        phase="zero",
        verbose="ERROR",
    )
    raw.notch_filter(
        freqs=[50.0],
        picks="eeg",
        method="fir",
        verbose="ERROR",
    )
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    missing = {"T1", "T2"} - event_id.keys()
    if missing:
        raise ValueError(
            f"Subject {subject} is missing expected hand labels: {sorted(missing)}."
        )

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"left_hand": event_id["T1"], "right_hand": event_id["T2"]},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise ValueError(f"No valid hand-imagery epochs found for subject {subject}.")
    return epochs
