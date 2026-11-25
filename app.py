"""Streamlit interface for subject-wise EEGBCI experiments."""

import streamlit as st

from src.data import load_dataset
from src.evaluation import evaluate_subject_dependent, evaluate_subject_independent


@st.cache_data(show_spinner=False)
def cached_dataset(subject_ids: tuple[int, ...], data_directory: str | None):
    return load_dataset(subject_ids, data_dir=data_directory)


st.set_page_config(page_title="EEG Motor Imagery BCI", layout="wide")
st.title("EEG Motor Imagery Classification")
st.caption(
    "PhysioNet EEGBCI | imagined left vs. right hand | research and education only"
)

with st.sidebar:
    st.header("Experiment")
    subjects = st.multiselect(
        "Participant IDs",
        options=list(range(1, 110)),
        default=[1, 2, 3],
        help="MNE downloads selected EEGBCI runs on first use.",
    )
    evaluation = st.selectbox(
        "Validation",
        ["Subject-independent", "Subject-dependent"],
        help=(
            "Subject-independent validation holds out complete participants. "
            "Subject-dependent validation splits trials within each participant."
        ),
    )
    data_directory = st.text_input(
        "Optional local MNE data directory",
        value="",
        help="Leave empty to use MNE's configured default data directory.",
    )
    include_eegnet = st.checkbox(
        "Include EEGNet (slower)", value=False, disabled=evaluation == "Subject-dependent"
    )
    run = st.button("Run evaluation", type="primary", use_container_width=True)

st.info(
    "The default task uses runs 4, 8, and 12 (left- vs right-hand imagery). "
    "Runs 6, 10, and 14 represent a different hands-vs-feet task."
)

if run:
    if not subjects:
        st.error("Select at least one participant.")
        st.stop()
    if evaluation == "Subject-independent" and len(set(subjects)) < 2:
        st.error("Subject-independent evaluation requires at least two participants.")
        st.stop()

    with st.spinner("Loading and preprocessing EEGBCI data..."):
        data, labels, groups, channel_names, sampling_frequency = cached_dataset(
            tuple(subjects),
            data_directory or None,
        )
    st.success(
        f"Loaded {len(labels):,} trials from {len(set(groups))} participants "
        f"({len(channel_names)} EEG channels at {sampling_frequency:g} Hz)."
    )

    with st.spinner("Training models and evaluating held-out trials..."):
        if evaluation == "Subject-independent":
            results = evaluate_subject_independent(
                data,
                labels,
                groups,
                channel_names,
                sampling_frequency,
                include_eegnet=include_eegnet,
            )
        else:
            results = evaluate_subject_dependent(
                data,
                labels,
                groups,
                channel_names,
                sampling_frequency,
            )

    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]
    summary = (
        results.groupby("model")[metrics]
        .agg(["mean", "std"])
        .round(3)
    )
    st.subheader(f"{evaluation} results")
    st.dataframe(summary, use_container_width=True)
    st.subheader("Fold-level results")
    st.dataframe(results, use_container_width=True)
    st.download_button(
        "Download fold metrics CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="eeg_motor_imagery_fold_metrics.csv",
        mime="text/csv",
    )

st.caption(
    "These results are not clinical evidence. Performance varies by participant "
    "and dataset; inspect the fold-level metrics rather than relying on one score."
)
