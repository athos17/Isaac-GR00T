from __future__ import annotations

import numpy as np

from robot_data_pipeline.models import (
    ActivityInterval,
    AlignedEpisodeData,
    CanonicalEpisode,
    ImageSeries,
    JointPositionSeries,
    PoseSeries,
    PositionCommandSeries,
    RobotProfile,
)
from robot_data_pipeline.processing.rotations import quaternion_to_rot6d, slerp


class SynchronizationError(ValueError):
    def __init__(
        self, reason: str, message: str, *, details: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details or {}


def _linear(
    series: JointPositionSeries,
    anchors: np.ndarray,
    max_gap_sec: float | None,
    *,
    stream: str,
):
    right = np.searchsorted(series.timestamps_ns, anchors, side="left")
    before = (right == 0) & (anchors != series.timestamps_ns[0])
    if np.any(before) or np.any(right == len(series.timestamps_ns)):
        invalid = np.flatnonzero(before | (right == len(series.timestamps_ns)))
        first = int(invalid[0])
        raise SynchronizationError(
            "state_interpolation_out_of_range",
            f"anchor outside state range for {stream}",
            details={
                "stream": stream,
                "anchor_timestamp_ns": int(anchors[first]),
                "stream_start_ns": int(series.timestamps_ns[0]),
                "stream_end_ns": int(series.timestamps_ns[-1]),
            },
        )
    left = right - 1
    exact = series.timestamps_ns[right] == anchors
    left[exact] = right[exact]
    gaps_ns = series.timestamps_ns[right] - series.timestamps_ns[left]
    threshold_ns = round(max_gap_sec * 1e9) if max_gap_sec is not None else None
    violations = np.flatnonzero(gaps_ns > threshold_ns) if threshold_ns is not None else []
    if len(violations):
        first = int(violations[0])
        raise SynchronizationError(
            "state_interpolation_gap_exceeded",
            f"state interpolation gap exceeds profile threshold for {stream}: "
            f"{gaps_ns[first] * 1e-9:.6f}s > {max_gap_sec:.6f}s",
            details={
                "stream": stream,
                "anchor_timestamp_ns": int(anchors[first]),
                "left_source_timestamp_ns": int(series.timestamps_ns[left[first]]),
                "right_source_timestamp_ns": int(series.timestamps_ns[right[first]]),
                "bracket_gap_sec": float(gaps_ns[first] * 1e-9),
                "threshold_sec": max_gap_sec,
                "violation_count": len(violations),
            },
        )
    denominator = np.maximum(1, gaps_ns)
    fraction = ((anchors - series.timestamps_ns[left]) / denominator)[:, None]
    values = series.values[left] + fraction * (series.values[right] - series.values[left])
    values[exact] = series.values[right[exact]]
    source = np.column_stack((series.timestamps_ns[left], series.timestamps_ns[right]))
    return values, source, gaps_ns


def _pose_linear_slerp(
    series: PoseSeries,
    anchors: np.ndarray,
    max_gap_sec: float | None,
    *,
    stream: str,
):
    right = np.searchsorted(series.timestamps_ns, anchors, side="left")
    before = (right == 0) & (anchors != series.timestamps_ns[0])
    if np.any(before) or np.any(right == len(series.timestamps_ns)):
        invalid = np.flatnonzero(before | (right == len(series.timestamps_ns)))
        first = int(invalid[0])
        raise SynchronizationError(
            "state_interpolation_out_of_range",
            f"anchor outside pose range for {stream}",
            details={
                "stream": stream,
                "anchor_timestamp_ns": int(anchors[first]),
                "stream_start_ns": int(series.timestamps_ns[0]),
                "stream_end_ns": int(series.timestamps_ns[-1]),
            },
        )
    left = right - 1
    exact = series.timestamps_ns[right] == anchors
    left[exact] = right[exact]
    gaps_ns = series.timestamps_ns[right] - series.timestamps_ns[left]
    threshold_ns = round(max_gap_sec * 1e9) if max_gap_sec is not None else None
    violations = np.flatnonzero(gaps_ns > threshold_ns) if threshold_ns is not None else []
    if len(violations):
        first = int(violations[0])
        raise SynchronizationError(
            "state_interpolation_gap_exceeded",
            f"pose interpolation gap exceeds profile threshold for {stream}: "
            f"{gaps_ns[first] * 1e-9:.6f}s > {max_gap_sec:.6f}s",
            details={
                "stream": stream,
                "anchor_timestamp_ns": int(anchors[first]),
                "left_source_timestamp_ns": int(series.timestamps_ns[left[first]]),
                "right_source_timestamp_ns": int(series.timestamps_ns[right[first]]),
                "bracket_gap_sec": float(gaps_ns[first] * 1e-9),
                "threshold_sec": max_gap_sec,
                "violation_count": len(violations),
            },
        )
    fractions = (anchors - series.timestamps_ns[left]) / np.maximum(1, gaps_ns)
    translation = series.translations[left] + fractions[:, None] * (
        series.translations[right] - series.translations[left]
    )
    quaternions = np.stack(
        [
            slerp(
                series.quaternions_xyzw[left_index],
                series.quaternions_xyzw[right_index],
                float(fraction),
            )
            for left_index, right_index, fraction in zip(left, right, fractions)
        ]
    )
    translation[exact] = series.translations[right[exact]]
    quaternions[exact] = series.quaternions_xyzw[right[exact]]
    values = np.concatenate((translation, quaternion_to_rot6d(quaternions)), axis=1)
    source = np.column_stack((series.timestamps_ns[left], series.timestamps_ns[right]))
    return values, source, gaps_ns


def _previous(
    series: PositionCommandSeries | PoseSeries,
    anchors: np.ndarray,
    max_age_sec: float | None,
    *,
    stream: str,
):
    indices = np.searchsorted(series.timestamps_ns, anchors, side="right") - 1
    if np.any(indices < 0):
        first = int(np.flatnonzero(indices < 0)[0])
        raise SynchronizationError(
            "action_missing_history",
            f"no action at or before anchor for {stream}",
            details={
                "stream": stream,
                "anchor_timestamp_ns": int(anchors[first]),
                "first_action_timestamp_ns": int(series.timestamps_ns[0]),
            },
        )
    source_times = series.timestamps_ns[indices]
    age_ns = anchors - source_times
    if np.any(age_ns < 0):
        first = int(np.flatnonzero(age_ns < 0)[0])
        raise SynchronizationError(
            "future_action_violation",
            f"selected an action after anchor for {stream}",
            details={
                "stream": stream,
                "anchor_timestamp_ns": int(anchors[first]),
                "source_timestamp_ns": int(source_times[first]),
                "action_age_sec": float(age_ns[first] * 1e-9),
            },
        )
    threshold_ns = round(max_age_sec * 1e9) if max_age_sec is not None else None
    violations = np.flatnonzero(age_ns > threshold_ns) if threshold_ns is not None else []
    if len(violations):
        first = int(violations[0])
        raise SynchronizationError(
            "action_age_exceeded",
            f"action age exceeds profile threshold for {stream}: "
            f"{age_ns[first] * 1e-9:.6f}s > {max_age_sec:.6f}s",
            details={
                "stream": stream,
                "anchor_timestamp_ns": int(anchors[first]),
                "source_timestamp_ns": int(source_times[first]),
                "action_age_sec": float(age_ns[first] * 1e-9),
                "threshold_sec": max_age_sec,
                "violation_count": len(violations),
            },
        )
    if isinstance(series, PositionCommandSeries):
        values = series.values[indices]
    else:
        values = np.concatenate(
            (
                series.translations[indices],
                quaternion_to_rot6d(series.quaternions_xyzw[indices]),
            ),
            axis=1,
        )
    return values, source_times, age_ns


def _nearest(
    series: ImageSeries,
    anchors: np.ndarray,
):
    right = np.searchsorted(series.timestamps_ns, anchors, side="left")
    right = np.clip(right, 0, len(series.timestamps_ns) - 1)
    left = np.maximum(0, right - 1)
    left_distance = np.abs(anchors - series.timestamps_ns[left])
    right_distance = np.abs(series.timestamps_ns[right] - anchors)
    # Exact ties select the earlier frame for deterministic behavior.
    indices = np.where(left_distance <= right_distance, left, right)
    source_times = series.timestamps_ns[indices]
    skew_ns = source_times - anchors
    return indices, source_times, skew_ns


def _maximum_consecutive_true(values: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _audit_wrist_skew(
    anchors: np.ndarray,
    source_times: np.ndarray,
    skew_ns: np.ndarray,
    config,
    *,
    stream: str,
) -> tuple[np.ndarray, int, float]:
    assert config.max_skew_sec is not None
    assert config.hard_max_skew_sec is not None
    assert config.max_consecutive_skew_violations is not None
    assert config.max_skew_violation_ratio is not None
    soft = np.abs(skew_ns) > round(config.max_skew_sec * 1e9)
    hard = np.abs(skew_ns) > round(config.hard_max_skew_sec * 1e9)
    soft_count = int(np.count_nonzero(soft))
    soft_ratio = soft_count / len(anchors) if len(anchors) else 0.0
    consecutive = _maximum_consecutive_true(soft)
    if (
        np.any(hard)
        or consecutive > config.max_consecutive_skew_violations
        or (soft_ratio > config.max_skew_violation_ratio)
    ):
        first = int(np.flatnonzero(hard)[0]) if np.any(hard) else int(np.flatnonzero(soft)[0])
        maximum = int(np.argmax(np.abs(skew_ns)))
        raise SynchronizationError(
            "wrist_camera_skew_exceeded",
            f"wrist camera skew policy rejected {stream}: max "
            f"{abs(int(skew_ns[maximum])) * 1e-9:.6f}s, "
            f"{soft_count}/{len(anchors)} frames exceed {config.max_skew_sec:.6f}s",
            details={
                "stream": stream,
                "anchor_timestamp_ns": int(anchors[first]),
                "source_timestamp_ns": int(source_times[first]),
                "signed_skew_sec": float(skew_ns[first] * 1e-9),
                "absolute_skew_sec": float(abs(skew_ns[first]) * 1e-9),
                "max_absolute_skew_sec": float(abs(skew_ns[maximum]) * 1e-9),
                "threshold_sec": config.max_skew_sec,
                "hard_threshold_sec": config.hard_max_skew_sec,
                "violation_count": soft_count,
                "violation_ratio": soft_ratio,
                "maximum_consecutive_violations": consecutive,
                "maximum_allowed_consecutive_violations": (config.max_consecutive_skew_violations),
                "maximum_allowed_violation_ratio": config.max_skew_violation_ratio,
                "frame_count": len(anchors),
            },
        )
    return soft, consecutive, soft_ratio


def _head_anchor_indices(
    episode: CanonicalEpisode,
    head: ImageSeries,
    activity: ActivityInterval,
    space,
    profile: RobotProfile,
) -> tuple[np.ndarray, int, int]:
    start_ns = activity.active_start_ns
    end_ns = activity.active_end_ns
    required_keys = (
        *space.state_groups,
        *space.action_groups,
        "video.left_wrist",
        "video.right_wrist",
    )
    for key in required_keys:
        series = episode.streams.get(key)
        if series is None or not len(series.timestamps_ns):
            raise SynchronizationError("missing_required_stream", f"missing stream: {key}")
        if key.startswith("state."):
            start_ns = max(start_ns, int(series.timestamps_ns[0]))
            end_ns = min(end_ns, int(series.timestamps_ns[-1]))
        elif key.startswith("action."):
            start_ns = max(start_ns, int(series.timestamps_ns[0]))
        elif key.startswith("video."):
            # Nearest-neighbor cameras may legitimately lead or lag the head by half a frame.
            max_skew_sec = profile.streams[key].max_skew_sec
            assert max_skew_sec is not None
            tolerance_ns = round(max_skew_sec * 1e9)
            start_ns = max(start_ns, int(series.timestamps_ns[0]) - tolerance_ns)
            end_ns = min(end_ns, int(series.timestamps_ns[-1]) + tolerance_ns)
    selected = np.flatnonzero((head.timestamps_ns >= start_ns) & (head.timestamps_ns <= end_ns))
    original = np.flatnonzero(
        (head.timestamps_ns >= activity.active_start_ns)
        & (head.timestamps_ns <= activity.active_end_ns)
    )
    return (
        selected,
        int(selected[0] - original[0]) if len(selected) and len(original) else 0,
        int(original[-1] - selected[-1]) if len(selected) and len(original) else 0,
    )


def synchronize_episode(
    episode: CanonicalEpisode,
    profile: RobotProfile,
    activity: ActivityInterval,
    *,
    action_space: str,
    output_fps: float = 30.0,
    minimum_output_frames: int = 30,
) -> AlignedEpisodeData:
    try:
        space = profile.output_spaces[action_space]
        head = episode.streams["video.head"]
    except KeyError as exc:
        raise SynchronizationError(
            "missing_required_stream", f"missing stream: {exc.args[0]}"
        ) from exc
    if not isinstance(head, ImageSeries):
        raise SynchronizationError("missing_required_stream", "video.head is not an image series")
    selected_head, trimmed_before, trimmed_after = _head_anchor_indices(
        episode, head, activity, space, profile
    )
    if len(selected_head) < minimum_output_frames:
        raise SynchronizationError(
            "episode_too_short",
            "too few head frames in activity interval",
            details={
                "stream": "video.head",
                "frame_count": len(selected_head),
                "minimum_output_frames": minimum_output_frames,
                "active_start_ns": activity.active_start_ns,
                "active_end_ns": activity.active_end_ns,
            },
        )
    anchors = head.timestamps_ns[selected_head]
    state_groups = []
    action_groups = []
    diagnostics: dict[str, dict[str, np.ndarray]] = {
        "video.head": {
            "source_timestamp_ns": anchors.copy(),
            "signed_skew_ns": np.zeros_like(anchors),
            "boundary_trimmed_before": np.asarray([trimmed_before], dtype=np.int64),
            "boundary_trimmed_after": np.asarray([trimmed_after], dtype=np.int64),
        }
    }
    images = {"video.head": tuple(head.encoded_images[index] for index in selected_head)}

    for key in space.state_groups:
        config = profile.streams[key]
        series = episode.streams.get(key)
        if isinstance(series, JointPositionSeries):
            values, sources, gaps = _linear(series, anchors, config.max_gap_sec, stream=key)
        elif isinstance(series, PoseSeries):
            values, sources, gaps = _pose_linear_slerp(
                series, anchors, config.max_gap_sec, stream=key
            )
        else:
            raise SynchronizationError("missing_required_stream", f"missing state stream: {key}")
        if config.value_range is not None and np.any(
            (values < config.value_range[0]) | (values > config.value_range[1])
        ):
            frame, axis = np.argwhere(
                (values < config.value_range[0]) | (values > config.value_range[1])
            )[0]
            raise SynchronizationError(
                "value_range_exceeded",
                f"state range exceeded: {key}",
                details={
                    "stream": key,
                    "anchor_timestamp_ns": int(anchors[frame]),
                    "axis": int(axis),
                    "value": float(values[frame, axis]),
                    "range": list(config.value_range),
                },
            )
        state_groups.append(values)
        diagnostics[key] = {
            "left_source_timestamp_ns": sources[:, 0],
            "right_source_timestamp_ns": sources[:, 1],
            "bracket_gap_ns": gaps,
        }

    for key in space.action_groups:
        config = profile.streams[key]
        series = episode.streams.get(key)
        if not isinstance(series, (PositionCommandSeries, PoseSeries)):
            raise SynchronizationError("missing_required_stream", f"missing action stream: {key}")
        values, source_times, ages = _previous(
            series, anchors, config.max_action_age_sec, stream=key
        )
        if config.value_range is not None and np.any(
            (values < config.value_range[0]) | (values > config.value_range[1])
        ):
            frame, axis = np.argwhere(
                (values < config.value_range[0]) | (values > config.value_range[1])
            )[0]
            raise SynchronizationError(
                "value_range_exceeded",
                f"action range exceeded: {key}",
                details={
                    "stream": key,
                    "anchor_timestamp_ns": int(anchors[frame]),
                    "source_timestamp_ns": int(source_times[frame]),
                    "axis": int(axis),
                    "value": float(values[frame, axis]),
                    "range": list(config.value_range),
                },
            )
        action_groups.append(values)
        diagnostics[key] = {"source_timestamp_ns": source_times, "action_age_ns": ages}

    for key in ("video.left_wrist", "video.right_wrist"):
        config = profile.streams[key]
        series = episode.streams.get(key)
        if not isinstance(series, ImageSeries):
            raise SynchronizationError("missing_required_stream", f"missing image stream: {key}")
        indices, source_times, skew = _nearest(series, anchors)
        soft_violations, maximum_consecutive, violation_ratio = _audit_wrist_skew(
            anchors, source_times, skew, config, stream=key
        )
        images[key] = tuple(series.encoded_images[index] for index in indices)
        reused = np.concatenate(([False], indices[1:] == indices[:-1]))
        diagnostics[key] = {
            "source_timestamp_ns": source_times,
            "signed_skew_ns": skew,
            "frame_reused": reused,
            "soft_skew_violation": soft_violations,
            "maximum_consecutive_soft_skew_violations": np.asarray(
                [maximum_consecutive], dtype=np.int64
            ),
            "soft_skew_violation_ratio": np.asarray([violation_ratio], dtype=np.float64),
        }

    state = np.concatenate(state_groups, axis=1)
    action = np.concatenate(action_groups, axis=1)
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise SynchronizationError("non_finite_payload", "aligned state/action is non-finite")
    return AlignedEpisodeData(
        action_space=action_space,
        state=state,
        action=action,
        timestamps=np.arange(len(anchors), dtype=np.float64) / output_fps,
        head_timestamps_ns=anchors,
        images=images,
        diagnostics=diagnostics,
    )
