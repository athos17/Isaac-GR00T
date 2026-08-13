from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from robot_data_pipeline.models import (
    CanonicalEpisode,
    JointPositionSeries,
    PoseSeries,
    RobotProfile,
)


class FilterError(ValueError):
    pass


FILTER_IMPLEMENTATION = "scipy_sosfiltfilt_regular_grid/v1"


def _spectral_metrics(
    timestamps_ns: np.ndarray, before: np.ndarray, after: np.ndarray
) -> dict[str, object]:
    intervals = np.diff(timestamps_ns).astype(np.float64) * 1e-9
    sample_hz = 1.0 / float(np.median(intervals))
    source_times = (timestamps_ns - timestamps_ns[0]).astype(np.float64) * 1e-9
    regular_times = np.arange(0.0, source_times[-1] + intervals.mean() / 2, 1.0 / sample_hz)

    def velocity_psd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        regular = np.column_stack(
            [
                np.interp(regular_times, source_times, values[:, axis])
                for axis in range(values.shape[1])
            ]
        )
        velocity = np.gradient(regular, 1.0 / sample_hz, axis=0)
        nperseg = min(len(velocity), 512)
        frequencies, density = welch(
            velocity,
            fs=sample_hz,
            axis=0,
            detrend="linear",
            nperseg=nperseg,
        )
        return frequencies, density

    frequencies, before_density = velocity_psd(before)
    after_frequencies, after_density = velocity_psd(after)
    if not np.array_equal(frequencies, after_frequencies):
        raise FilterError("filter spectral grids differ")

    def power(density: np.ndarray, mask: np.ndarray) -> float | None:
        if not np.any(mask):
            return None
        return float(np.mean(np.trapz(density[mask], frequencies[mask], axis=0)))

    low = (frequencies >= 0.5) & (frequencies <= 10.0)
    high = frequencies >= 15.0
    low_before = power(before_density, low)
    low_after = power(after_density, low)
    high_before = power(before_density, high)
    high_after = power(after_density, high)
    return {
        "method": "welch_velocity_detrend_linear",
        "low_band_hz": [0.5, 10.0],
        "high_band_hz": [15.0, sample_hz / 2.0],
        "low_band_power_before": low_before,
        "low_band_power_after": low_after,
        "low_band_retention": (
            low_after / low_before if low_before is not None and low_before > 0 else None
        ),
        "high_band_power_before": high_before,
        "high_band_power_after": high_after,
        "high_band_retention": (
            high_after / high_before if high_before is not None and high_before > 0 else None
        ),
    }


def butterworth_zero_phase(
    timestamps_ns: np.ndarray, values: np.ndarray, *, cutoff_hz: float, order: int
) -> tuple[np.ndarray, float]:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if len(timestamps) != len(values) or len(timestamps) < 4:
        raise FilterError("filter input is too short")
    intervals = np.diff(timestamps).astype(np.float64) * 1e-9
    if np.any(intervals <= 0):
        raise FilterError("filter timestamps must be strictly increasing")
    sample_hz = 1.0 / float(np.median(intervals))
    if cutoff_hz >= sample_hz / 2:
        raise FilterError("filter cutoff must be below the input Nyquist frequency")
    origin = timestamps[0]
    source_times = (timestamps - origin).astype(np.float64) * 1e-9
    regular_times = np.arange(0.0, source_times[-1] + intervals.mean() / 2, 1.0 / sample_hz)
    regular_values = np.column_stack(
        [np.interp(regular_times, source_times, values[:, axis]) for axis in range(values.shape[1])]
    )
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sample_hz, output="sos")
    try:
        filtered_regular = sosfiltfilt(sos, regular_values, axis=0)
    except ValueError as exc:
        raise FilterError(f"filter input is too short for zero-phase padding: {exc}") from exc
    filtered = np.column_stack(
        [
            np.interp(source_times, regular_times, filtered_regular[:, axis])
            for axis in range(values.shape[1])
        ]
    )
    return filtered, sample_hz


def filter_state_streams(
    episode: CanonicalEpisode,
    profile: RobotProfile,
    *,
    padded_start_ns: int | None = None,
    padded_end_ns: int | None = None,
) -> tuple[CanonicalEpisode, dict[str, dict[str, object]]]:
    streams = dict(episode.streams)
    manifest = {}
    for key, stream_config in profile.streams.items():
        if not key.startswith("state.") or stream_config.smoothing.type == "none":
            continue
        series = streams.get(key)
        if not isinstance(series, (JointPositionSeries, PoseSeries)):
            continue
        start = padded_start_ns if padded_start_ns is not None else int(series.timestamps_ns[0])
        end = padded_end_ns if padded_end_ns is not None else int(series.timestamps_ns[-1])
        selected = (series.timestamps_ns >= start) & (series.timestamps_ns <= end)
        timestamps = series.timestamps_ns[selected]
        assert stream_config.smoothing.cutoff_hz is not None
        assert stream_config.smoothing.order is not None
        values = (
            series.values[selected]
            if isinstance(series, JointPositionSeries)
            else series.translations[selected]
        )
        filtered, input_hz = butterworth_zero_phase(
            timestamps,
            values,
            cutoff_hz=stream_config.smoothing.cutoff_hz,
            order=stream_config.smoothing.order,
        )
        if isinstance(series, JointPositionSeries):
            streams[key] = replace(series, timestamps_ns=timestamps, values=filtered)
        else:
            streams[key] = replace(
                series,
                timestamps_ns=timestamps,
                translations=filtered,
                quaternions_xyzw=series.quaternions_xyzw[selected],
            )
        manifest[key] = {
            "type": "butterworth",
            "implementation": FILTER_IMPLEMENTATION,
            "cutoff_hz": stream_config.smoothing.cutoff_hz,
            "order": stream_config.smoothing.order,
            "zero_phase": bool(stream_config.smoothing.zero_phase),
            "estimated_input_hz": input_hz,
            "spectral": _spectral_metrics(timestamps, values, filtered),
        }
    return CanonicalEpisode(streams), manifest
