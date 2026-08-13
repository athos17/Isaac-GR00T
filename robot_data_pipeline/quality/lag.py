from __future__ import annotations

import numpy as np

from robot_data_pipeline.models import AxisLagAudit, LagAudit


def _empty_window_metrics() -> dict[str, float | None]:
    return {
        "median": None,
        "mad": None,
        "range": None,
        "trend_sec_per_sec": None,
        "max_step_sec": None,
    }


def _insufficient_axis(status: str) -> AxisLagAudit:
    return AxisLagAudit(
        status=status,
        best_lag_sec=None,
        peak_correlation=None,
        secondary_peak_margin=None,
        direction_agreement=None,
        tracking_error={"mean": None, "p95": None, "max": None},
        valid_duration_sec=0.0,
        windowed_lag_sec=_empty_window_metrics(),
    )


def _insufficient_audit(axis_count: int, status: str) -> LagAudit:
    return LagAudit(
        axes=tuple(_insufficient_axis(status) for _ in range(axis_count)),
        consensus_lag_sec=None,
        active_axis_count=0,
    )


def _zoh(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(times, grid, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("QA grid starts before the first action")
    return values[indices]


def _linear(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(grid, times, values[:, axis]) for axis in range(values.shape[1])]
    )


def audit_action_state_lag(
    action_timestamps_ns: np.ndarray,
    action_values: np.ndarray,
    state_timestamps_ns: np.ndarray,
    state_values: np.ndarray,
    *,
    max_lag_sec: float = 0.3,
    qa_hz: float = 200.0,
    excitation_std: float = 1e-3,
    window_sec: float = 3.0,
) -> LagAudit:
    action_timestamps_ns = np.asarray(action_timestamps_ns, dtype=np.int64)
    state_timestamps_ns = np.asarray(state_timestamps_ns, dtype=np.int64)
    action_values = np.asarray(action_values, dtype=np.float64)
    state_values = np.asarray(state_values, dtype=np.float64)
    if action_values.ndim != 2 or state_values.ndim != 2:
        raise ValueError("action and state values must be two-dimensional")
    if action_values.shape[1] != state_values.shape[1]:
        raise ValueError("action and state dimensions differ")
    if len(action_timestamps_ns) != len(action_values) or len(state_timestamps_ns) != len(
        state_values
    ):
        raise ValueError("lag timestamps and values have different lengths")
    axis_count = action_values.shape[1]
    if len(action_timestamps_ns) < 2 or len(state_timestamps_ns) < 2:
        return _insufficient_audit(axis_count, "insufficient_overlap")
    origin_ns = min(int(action_timestamps_ns[0]), int(state_timestamps_ns[0]))
    action_times = (action_timestamps_ns - origin_ns).astype(np.float64) * 1e-9
    state_times = (state_timestamps_ns - origin_ns).astype(np.float64) * 1e-9
    start = max(action_times[0], state_times[0])
    end = min(action_times[-1], state_times[-1])
    step = 1.0 / qa_hz
    grid = np.arange(start, end, step)
    if len(grid) < 4:
        return _insufficient_audit(axis_count, "insufficient_overlap")
    action = _zoh(action_times, action_values, grid)
    state = _linear(state_times, state_values, grid)
    action_velocity = np.gradient(action, step, axis=0)
    state_velocity = np.gradient(state, step, axis=0)
    max_shift = min(round(max_lag_sec * qa_hz), len(grid) // 3)
    axes = []
    valid_lags = []
    for axis in range(action.shape[1]):
        action_axis = action_velocity[:, axis]
        state_axis = state_velocity[:, axis]
        if np.std(action_axis) < excitation_std or np.std(state_axis) < excitation_std:
            axes.append(_insufficient_axis("insufficient_excitation"))
            continue
        correlations = []
        for shift in range(max_shift + 1):
            action_slice = action_axis[: len(action_axis) - shift or None]
            state_slice = state_axis[shift:]
            if np.std(action_slice) < excitation_std or np.std(state_slice) < excitation_std:
                correlations.append(float("-inf"))
            else:
                correlations.append(float(np.corrcoef(action_slice, state_slice)[0, 1]))
        best_shift = int(np.argmax(correlations))
        peak = correlations[best_shift]
        finite = sorted((value for value in correlations if np.isfinite(value)), reverse=True)
        margin = peak - finite[1] if len(finite) > 1 else None
        aligned_action_velocity = action_axis[: len(action_axis) - best_shift or None]
        aligned_state_velocity = state_axis[best_shift:]
        moving = (np.abs(aligned_action_velocity) + np.abs(aligned_state_velocity)) > excitation_std
        direction = (
            float(
                np.mean(
                    np.sign(aligned_action_velocity[moving])
                    == np.sign(aligned_state_velocity[moving])
                )
            )
            if np.any(moving)
            else None
        )
        state_position = state[best_shift:, axis]
        action_position = action[: len(action) - best_shift or None, axis]
        errors = np.abs(action_position - state_position)
        lag = best_shift / qa_hz
        window_samples = max(4, round(window_sec * qa_hz))
        stride = max(1, window_samples // 2)
        window_lags = []
        window_centers_sec = []
        for window_start in range(0, len(action_axis) - window_samples + 1, stride):
            action_window = action_axis[window_start : window_start + window_samples]
            state_window = state_axis[window_start : window_start + window_samples]
            if np.std(action_window) < excitation_std or np.std(state_window) < excitation_std:
                continue
            window_correlations = []
            for shift in range(min(max_shift, window_samples // 3) + 1):
                left_window = action_window[: len(action_window) - shift or None]
                right_window = state_window[shift:]
                if np.std(left_window) < excitation_std or np.std(right_window) < excitation_std:
                    window_correlations.append(float("-inf"))
                else:
                    correlation = float(np.corrcoef(left_window, right_window)[0, 1])
                    window_correlations.append(
                        correlation if np.isfinite(correlation) else float("-inf")
                    )
            if any(np.isfinite(value) for value in window_correlations):
                window_lags.append(int(np.argmax(window_correlations)) / qa_hz)
                window_centers_sec.append((window_start + window_samples / 2) / qa_hz)
        if window_lags:
            window_array = np.asarray(window_lags, dtype=np.float64)
            window_median = float(np.median(window_array))
            if len(window_array) > 1:
                trend = float(
                    np.polyfit(np.asarray(window_centers_sec, dtype=np.float64), window_array, 1)[0]
                )
                max_step = float(np.max(np.abs(np.diff(window_array))))
            else:
                trend = None
                max_step = None
            window_metrics = {
                "median": window_median,
                "mad": float(np.median(np.abs(window_array - window_median))),
                "range": float(np.ptp(window_array)),
                "trend_sec_per_sec": trend,
                "max_step_sec": max_step,
            }
        else:
            window_metrics = _empty_window_metrics()
        valid_lags.append(lag)
        axes.append(
            AxisLagAudit(
                status="ok",
                best_lag_sec=lag,
                peak_correlation=peak,
                secondary_peak_margin=margin,
                direction_agreement=direction,
                tracking_error={
                    "mean": float(np.mean(errors)),
                    "p95": float(np.percentile(errors, 95)),
                    "max": float(np.max(errors)),
                },
                valid_duration_sec=float(len(aligned_action_velocity) / qa_hz),
                windowed_lag_sec=window_metrics,
            )
        )
    return LagAudit(
        axes=tuple(axes),
        consensus_lag_sec=float(np.median(valid_lags)) if valid_lags else None,
        active_axis_count=len(valid_lags),
    )
