"""Deterministic log band-power feature extraction."""

import numpy as np
from scipy.signal import welch


BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
MOTOR_CHANNELS = ("C3", "C4", "Cz")


def motor_channel_indices(channel_names: list[str]) -> list[int]:
    """Return indices for available central motor channels."""
    indices = [channel_names.index(name) for name in MOTOR_CHANNELS if name in channel_names]
    if not indices:
        raise ValueError("No C3, C4, or Cz motor channels are present.")
    return indices


def extract_band_power(
    data: np.ndarray,
    sampling_frequency: float,
    channel_indices: list[int] | None = None,
) -> np.ndarray:
    """Return log mean PSD by frequency band for each epoch and selected channel."""
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError("Expected EEG data with shape (epochs, channels, time).")
    if data.shape[0] == 0 or data.shape[2] < 2:
        raise ValueError("EEG data must contain epochs with at least two samples.")
    if not np.isfinite(data).all():
        raise ValueError("EEG data contains non-finite values.")
    if sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if channel_indices is not None:
        if not channel_indices or min(channel_indices) < 0 or max(channel_indices) >= data.shape[1]:
            raise ValueError("Channel indices are empty or outside the data shape.")
        data = data[:, channel_indices, :]

    n_per_segment = min(256, data.shape[-1])
    frequencies, powers = welch(
        data,
        fs=sampling_frequency,
        nperseg=n_per_segment,
        axis=-1,
    )
    features = []
    for lower, upper in BANDS.values():
        band_mask = (frequencies >= lower) & (frequencies < upper)
        if not band_mask.any():
            raise ValueError(
                f"No PSD bins fall in {lower:g}-{upper:g} Hz at this sampling rate."
            )
        features.append(np.mean(powers[..., band_mask], axis=-1))
    band_first = np.stack(features, axis=-1)
    return np.log(np.maximum(band_first, np.finfo(np.float64).tiny)).reshape(data.shape[0], -1)
