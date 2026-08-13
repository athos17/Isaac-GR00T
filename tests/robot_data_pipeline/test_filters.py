import numpy as np
from robot_data_pipeline.processing.filters import _spectral_metrics, butterworth_zero_phase


def _amplitude(signal: np.ndarray, times: np.ndarray, frequency: float) -> float:
    basis = np.exp(-2j * np.pi * frequency * times)
    return 2 * abs(np.sum(signal * basis)) / len(times)


def test_butterworth_preserves_low_and_attenuates_high_frequency() -> None:
    hz = 200
    times = np.arange(0, 4, 1 / hz)
    values = np.sin(2 * np.pi * 5 * times) + np.sin(2 * np.pi * 30 * times)

    filtered, estimated_hz = butterworth_zero_phase(
        np.round(times * 1e9).astype(np.int64), values[:, None], cutoff_hz=10, order=4
    )

    assert abs(estimated_hz - hz) < 0.1
    interior = slice(hz, -hz)
    assert _amplitude(filtered[interior, 0], times[interior], 5) > 0.9
    assert _amplitude(filtered[interior, 0], times[interior], 30) < 0.1


def test_zero_phase_filter_preserves_impulse_peak_time() -> None:
    hz = 200
    times = np.arange(0, 3, 1 / hz)
    values = np.zeros(len(times))
    values[len(times) // 2] = 1

    filtered, _ = butterworth_zero_phase(
        np.round(times * 1e9).astype(np.int64), values[:, None], cutoff_hz=10, order=4
    )

    assert int(np.argmax(filtered[:, 0])) == len(times) // 2


def test_spectral_metrics_measure_expected_band_attenuation() -> None:
    hz = 200
    times = np.arange(0, 8, 1 / hz)
    values = np.sin(2 * np.pi * 5 * times) + 0.5 * np.sin(2 * np.pi * 30 * times)
    timestamps_ns = np.round(times * 1e9).astype(np.int64)
    filtered, _ = butterworth_zero_phase(timestamps_ns, values[:, None], cutoff_hz=10, order=4)

    metrics = _spectral_metrics(timestamps_ns, values[:, None], filtered)

    assert metrics["method"] == "welch_velocity_detrend_linear"
    assert 0.9 < metrics["low_band_retention"] < 1.1
    assert metrics["high_band_retention"] < 0.01
