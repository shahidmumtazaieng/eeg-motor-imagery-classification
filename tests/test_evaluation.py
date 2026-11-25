import numpy as np
import pytest

from src.evaluation import subject_independent_predictions


def test_subject_independent_predictions_cover_each_trial_once():
    rng = np.random.default_rng(42)
    labels = np.tile([0, 1], 20)
    groups = np.repeat([1, 2], 20)
    data = rng.normal(scale=0.1, size=(40, 3, 640))
    data[labels == 0, 0, :] += 1.0
    data[labels == 1, 1, :] += 1.0

    truth, predictions, scores = subject_independent_predictions(
        data,
        labels,
        groups,
        ["C3", "C4", "Cz"],
        160.0,
        model_name="logistic_regression",
    )

    assert len(truth) == len(predictions) == len(scores) == len(labels)
    assert set(np.unique(truth)) == {0, 1}
    assert np.isfinite(scores).all()


def test_subject_independent_predictions_reject_one_participant():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(8, 3, 640))
    labels = np.tile([0, 1], 4)

    with pytest.raises(ValueError, match="at least two subjects"):
        subject_independent_predictions(
            data, labels, np.ones(8), ["C3", "C4", "Cz"], 160.0
        )
