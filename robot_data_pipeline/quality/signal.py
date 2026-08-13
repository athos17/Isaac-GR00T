from __future__ import annotations

from typing import Any

import numpy as np

from robot_data_pipeline.models import ActivityInterval, CanonicalEpisode, RobotProfile


def audit_active_interval_gaps(
    episode: CanonicalEpisode, profile: RobotProfile, activity: ActivityInterval
) -> list[dict[str, Any]]:
    """Return hard gap violations that overlap the unpadded activity interval."""
    violations = []
    for key, config in profile.streams.items():
        if not config.required or config.max_gap_sec is None:
            continue
        series = episode.streams.get(key)
        if series is None or len(series.timestamps_ns) < 2:
            continue
        timestamps = np.asarray(series.timestamps_ns, dtype=np.int64)
        intervals = np.diff(timestamps)
        overlap = (timestamps[:-1] < activity.active_end_ns) & (
            timestamps[1:] > activity.active_start_ns
        )
        indices = np.flatnonzero(overlap & (intervals > round(config.max_gap_sec * 1e9)))
        for index in indices:
            violations.append(
                {
                    "stream": key,
                    "start_ns": int(timestamps[index]),
                    "end_ns": int(timestamps[index + 1]),
                    "interval_sec": float(intervals[index] * 1e-9),
                    "threshold_sec": config.max_gap_sec,
                }
            )
    return violations
