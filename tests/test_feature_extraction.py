import numpy as np
import pytest

from src.feature_extraction import extract_band_power, motor_channel_indices


def test_band_power_extracts_finite_features_for_selected_motor_channels():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(8, 5, 640))

    features = extract_band_power(data, 160.0, channel_indices=[1, 2, 3])

    assert features.shape == (8, 15)
    assert np.isfinite(features).all()


def test_band_power_rejects_non_epoch_input():
    with pytest.raises(ValueError, match="epochs, channels, time"):
        extract_band_power(np.zeros((3, 10)), 160.0)


def test_motor_channels_reports_missing_channel_layout():
    assert motor_channel_indices(["Fp1", "C3", "C4", "Cz"]) == [1, 2, 3]
    with pytest.raises(ValueError, match="No C3, C4, or Cz"):
        motor_channel_indices(["Fp1", "Fp2"])
