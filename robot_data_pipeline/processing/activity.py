from __future__ import annotations

import numpy as np

from robot_data_pipeline.models import (
    ActivityInterval,
    CanonicalEpisode,
    JointPositionSeries,
    PoseSeries,
    RobotProfile,
)


def _windowed_mean(times_sec: np.ndarray, values: np.ndarray, window_sec: float) -> np.ndarray:
    half = window_sec / 2
    left = np.searchsorted(times_sec, times_sec - half, side="left")
    right = np.searchsorted(times_sec, times_sec + half, side="right")
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    return (prefix[right] - prefix[left]) / np.maximum(1, right - left)


def _speed(series: JointPositionSeries | PoseSeries) -> tuple[np.ndarray, np.ndarray]:
    timestamps = (series.timestamps_ns - series.timestamps_ns[0]).astype(np.float64) * 1e-9
    dt = np.diff(timestamps)
    if len(dt) == 0 or np.any(dt <= 0):
        return np.empty(0), np.empty(0)
    if isinstance(series, PoseSeries):
        delta = np.linalg.norm(np.diff(series.translations, axis=0), axis=1)
    else:
        delta = np.max(np.abs(np.diff(series.values, axis=0)), axis=1)
    return (timestamps[:-1] + timestamps[1:]) / 2, delta / dt


def detect_activity(
    episode: CanonicalEpisode,
    profile: RobotProfile,
    *,
    padding_before_sec: float,
    padding_after_sec: float,
) -> ActivityInterval | None:
    active_times: list[np.ndarray] = []
    episode_starts = []
    episode_ends = []
    for key in profile.activity_detection.groups:
        series = episode.streams.get(key)
        if not isinstance(series, (JointPositionSeries, PoseSeries)):
            continue
        episode_starts.append(int(series.timestamps_ns[0]))
        episode_ends.append(int(series.timestamps_ns[-1]))
        times, speed = _speed(series)
        if not len(times):
            continue
        smoothed = _windowed_mean(times, speed, profile.activity_detection.window_sec)
        threshold = (
            profile.activity_detection.eef_velocity_threshold
            if isinstance(series, PoseSeries)
            else profile.activity_detection.joint_velocity_threshold
        )
        active_times.append(
            series.timestamps_ns[0] + np.round(times[smoothed >= threshold] * 1e9).astype(np.int64)
        )
    nonempty = [times for times in active_times if len(times)]
    if not nonempty or not episode_starts:
        return None
    active_start = min(int(times[0]) for times in nonempty)
    active_end = max(int(times[-1]) for times in nonempty)
    data_start_ns = max(episode_starts)
    data_end_ns = min(episode_ends)
    active_start_ns = max(data_start_ns, active_start)
    active_end_ns = min(data_end_ns, active_end)
    if active_start_ns >= active_end_ns:
        return None
    return ActivityInterval(
        active_start_ns=active_start_ns,
        active_end_ns=active_end_ns,
        padded_start_ns=max(data_start_ns, active_start_ns - round(padding_before_sec * 1e9)),
        padded_end_ns=min(data_end_ns, active_end_ns + round(padding_after_sec * 1e9)),
    )
