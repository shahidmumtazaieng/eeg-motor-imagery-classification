import pytest

from src.preprocessing import load_subject_epochs


def test_preprocessing_restricts_to_left_right_imagery_runs():
    with pytest.raises(ValueError, match="supports EEGBCI runs 4, 8, and 12"):
        load_subject_epochs(1, runs=(6, 10, 14))


def test_preprocessing_rejects_invalid_subject_without_downloading():
    with pytest.raises(ValueError, match="between 1 and 109"):
        load_subject_epochs(0)
