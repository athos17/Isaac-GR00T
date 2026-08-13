from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from robot_data_pipeline.config import load_robot_profile
from robot_data_pipeline.models import (
    ActivityInterval,
    CanonicalEpisode,
    ImageSeries,
    JointPositionSeries,
    PositionCommandSeries,
)
from robot_data_pipeline.processing.synchronize import (
    SynchronizationError,
    _nearest,
    synchronize_episode,
)
from robot_data_pipeline.quality.aligned import audit_aligned_episode


REPO_ROOT = Path(__file__).parents[2]
PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml"


def _series(profile):
    state_times = np.arange(0, 1.21, 0.01)
    action_times = np.arange(0, 1.21, 0.05)
    streams = {}
    for key in profile.output_spaces["joint_absolute"].state_groups:
        config = profile.streams[key]
        values = np.tile(state_times[:, None], (1, len(config.names)))
        streams[key] = JointPositionSeries(
            np.round(state_times * 1e9).astype(np.int64), values, config.names
        )
    for key in profile.output_spaces["joint_absolute"].action_groups:
        config = profile.streams[key]
        values = np.tile(action_times[:, None], (1, len(config.names)))
        streams[key] = PositionCommandSeries(
            np.round(action_times * 1e9).astype(np.int64), values, config.names
        )
    head_times = np.arange(0.1, 1.101, 1 / 30)
    for key, offset in (
        ("video.head", 0.0),
        ("video.left_wrist", 0.005),
        ("video.right_wrist", -0.005),
    ):
        timestamps = np.round((head_times + offset) * 1e9).astype(np.int64)
        streams[key] = ImageSeries(
            timestamps,
            timestamps + 1_000,
            tuple(f"{key}-{index}".encode() for index in range(len(timestamps))),
            tuple("jpeg" for _ in timestamps),
        )
    return CanonicalEpisode(streams), head_times


def test_sync_uses_head_anchor_bounded_linear_and_causal_zoh() -> None:
    profile = load_robot_profile(PROFILE)
    episode, head_times = _series(profile)
    activity = ActivityInterval(
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
    )

    aligned = synchronize_episode(
        episode,
        profile,
        activity,
        action_space="joint_absolute",
        minimum_output_frames=1,
    )

    assert np.array_equal(aligned.timestamps, np.arange(len(head_times)) / 30)
    assert aligned.state.shape == (len(head_times), 54)
    assert aligned.action.shape == (len(head_times), 54)
    for key in profile.output_spaces["joint_absolute"].action_groups:
        assert np.all(aligned.diagnostics[key]["source_timestamp_ns"] <= aligned.head_timestamps_ns)
        assert np.all(aligned.diagnostics[key]["action_age_ns"] >= 0)
    assert audit_aligned_episode(aligned, output_fps=30)["status"] == "PASS"


def test_sync_rejects_state_interpolation_across_large_gap() -> None:
    profile = load_robot_profile(PROFILE)
    episode, head_times = _series(profile)
    key = "state.left_arm_joint"
    series = episode.streams[key]
    keep = (series.timestamps_ns < 400_000_000) | (series.timestamps_ns > 600_000_000)
    episode.streams[key] = JointPositionSeries(
        series.timestamps_ns[keep], series.values[keep], series.names
    )
    activity = ActivityInterval(100_000_000, 1_100_000_000, 100_000_000, 1_100_000_000)

    with pytest.raises(SynchronizationError) as error:
        synchronize_episode(
            episode,
            profile,
            activity,
            action_space="joint_absolute",
            minimum_output_frames=1,
        )

    assert error.value.reason == "state_interpolation_gap_exceeded"
    assert error.value.details["stream"] == key
    assert error.value.details["bracket_gap_sec"] > error.value.details["threshold_sec"]


def test_sync_wrist_skew_rejection_identifies_stream_and_timestamp() -> None:
    profile = load_robot_profile(PROFILE)
    episode, head_times = _series(profile)
    key = "video.left_wrist"
    series = episode.streams[key]
    keep = np.ones(len(series.timestamps_ns), dtype=bool)
    keep[8:12] = False
    episode.streams[key] = ImageSeries(
        series.timestamps_ns[keep],
        series.bag_timestamps_ns[keep],
        tuple(value for value, selected in zip(series.encoded_images, keep) if selected),
        tuple(value for value, selected in zip(series.formats, keep) if selected),
    )
    activity = ActivityInterval(
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
    )

    with pytest.raises(SynchronizationError) as error:
        synchronize_episode(
            episode,
            profile,
            activity,
            action_space="joint_absolute",
            minimum_output_frames=1,
        )

    assert error.value.reason == "wrist_camera_skew_exceeded"
    assert error.value.details["stream"] == key
    assert error.value.details["violation_count"] >= 1
    assert error.value.details["max_absolute_skew_sec"] > error.value.details["hard_threshold_sec"]


def test_isolated_soft_wrist_skew_is_retained_with_warning() -> None:
    profile = load_robot_profile(PROFILE)
    episode, head_times = _series(profile)
    key = "video.left_wrist"
    profile = replace(
        profile,
        streams={
            **profile.streams,
            key: replace(profile.streams[key], max_skew_violation_ratio=0.1),
        },
    )
    series = episode.streams[key]
    keep = np.ones(len(series.timestamps_ns), dtype=bool)
    keep[10] = False
    episode.streams[key] = ImageSeries(
        series.timestamps_ns[keep],
        series.bag_timestamps_ns[keep],
        tuple(value for value, selected in zip(series.encoded_images, keep) if selected),
        tuple(value for value, selected in zip(series.formats, keep) if selected),
    )
    activity = ActivityInterval(
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
    )

    aligned = synchronize_episode(
        episode,
        profile,
        activity,
        action_space="joint_absolute",
        minimum_output_frames=1,
    )
    report = audit_aligned_episode(aligned, output_fps=30)

    assert report["status"] == "PASS_WITH_WARNING"
    assert report["warning_reasons"] == ["wrist_camera_skew_warning"]
    assert report["streams"][key]["soft_skew_violation_count"] == 1


def test_isolated_soft_wrist_skew_is_rejected_when_ratio_exceeds_limit() -> None:
    profile = load_robot_profile(PROFILE)
    episode, head_times = _series(profile)
    key = "video.left_wrist"
    series = episode.streams[key]
    keep = np.ones(len(series.timestamps_ns), dtype=bool)
    keep[10] = False
    episode.streams[key] = ImageSeries(
        series.timestamps_ns[keep],
        series.bag_timestamps_ns[keep],
        tuple(value for value, selected in zip(series.encoded_images, keep) if selected),
        tuple(value for value, selected in zip(series.formats, keep) if selected),
    )
    activity = ActivityInterval(
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
    )

    with pytest.raises(SynchronizationError) as error:
        synchronize_episode(
            episode,
            profile,
            activity,
            action_space="joint_absolute",
            minimum_output_frames=1,
        )

    assert error.value.reason == "wrist_camera_skew_exceeded"
    assert error.value.details["maximum_consecutive_violations"] == 1
    assert (
        error.value.details["violation_ratio"]
        > error.value.details["maximum_allowed_violation_ratio"]
    )


def test_consecutive_soft_wrist_skew_is_rejected() -> None:
    profile = load_robot_profile(PROFILE)
    episode, head_times = _series(profile)
    key = "video.left_wrist"
    profile = replace(
        profile,
        streams={
            **profile.streams,
            key: replace(profile.streams[key], max_skew_violation_ratio=0.5),
        },
    )
    series = episode.streams[key]
    keep = np.ones(len(series.timestamps_ns), dtype=bool)
    keep[9:11] = False
    episode.streams[key] = ImageSeries(
        series.timestamps_ns[keep],
        series.bag_timestamps_ns[keep],
        tuple(value for value, selected in zip(series.encoded_images, keep) if selected),
        tuple(value for value, selected in zip(series.formats, keep) if selected),
    )
    activity = ActivityInterval(
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
    )

    with pytest.raises(SynchronizationError) as error:
        synchronize_episode(
            episode,
            profile,
            activity,
            action_space="joint_absolute",
            minimum_output_frames=1,
        )

    assert error.value.reason == "wrist_camera_skew_exceeded"
    assert error.value.details["maximum_consecutive_violations"] > 1


def test_wrist_boundary_coverage_is_trimmed_without_warning() -> None:
    profile = load_robot_profile(PROFILE)
    episode, head_times = _series(profile)
    key = "video.left_wrist"
    series = episode.streams[key]
    shifted = series.timestamps_ns.copy()
    shifted[0] += 25_000_000
    episode.streams[key] = ImageSeries(
        shifted,
        series.bag_timestamps_ns,
        series.encoded_images,
        series.formats,
    )
    activity = ActivityInterval(
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
    )

    aligned = synchronize_episode(
        episode,
        profile,
        activity,
        action_space="joint_absolute",
        minimum_output_frames=1,
    )
    report = audit_aligned_episode(aligned, output_fps=30)

    assert len(aligned.timestamps) == len(head_times) - 1
    assert aligned.diagnostics["video.head"]["boundary_trimmed_before"].tolist() == [1]
    assert report["status"] == "PASS"


def test_camera_nearest_tie_selects_earlier_frame() -> None:
    series = ImageSeries(
        np.array([0, 20], dtype=np.int64),
        np.array([1, 21], dtype=np.int64),
        (b"earlier", b"later"),
        ("jpeg", "jpeg"),
    )

    indices, source_times, skew = _nearest(series, np.array([10], dtype=np.int64))

    assert indices.tolist() == [0]
    assert source_times.tolist() == [0]
    assert skew.tolist() == [-10]


def test_camera_frame_reuse_is_reported() -> None:
    profile = load_robot_profile(PROFILE)
    episode, head_times = _series(profile)
    key = "video.left_wrist"
    profile = replace(
        profile,
        streams={
            **profile.streams,
            key: replace(profile.streams[key], max_skew_sec=0.04),
        },
    )
    series = episode.streams[key]
    keep = np.arange(len(series.timestamps_ns)) % 2 == 0
    episode.streams[key] = ImageSeries(
        series.timestamps_ns[keep],
        series.bag_timestamps_ns[keep],
        tuple(value for value, selected in zip(series.encoded_images, keep) if selected),
        tuple(value for value, selected in zip(series.formats, keep) if selected),
    )
    activity = ActivityInterval(
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
    )

    aligned = synchronize_episode(
        episode,
        profile,
        activity,
        action_space="joint_absolute",
        minimum_output_frames=1,
    )
    report = audit_aligned_episode(aligned, output_fps=30)

    assert report["streams"][key]["reused_frame_count"] > 0
    assert report["streams"][key]["reused_frame_ratio"] > 0


def test_action_age_rejection_identifies_stream_timestamp_and_threshold() -> None:
    profile = load_robot_profile(PROFILE)
    episode, head_times = _series(profile)
    key = "action.left_arm_joint"
    series = episode.streams[key]
    keep = (series.timestamps_ns < 300_000_000) | (series.timestamps_ns > 600_000_000)
    episode.streams[key] = PositionCommandSeries(
        series.timestamps_ns[keep], series.values[keep], series.names
    )
    activity = ActivityInterval(
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
    )

    with pytest.raises(SynchronizationError) as error:
        synchronize_episode(
            episode,
            profile,
            activity,
            action_space="joint_absolute",
            minimum_output_frames=1,
        )

    assert error.value.reason == "action_age_exceeded"
    assert error.value.details["stream"] == key
    assert error.value.details["action_age_sec"] > error.value.details["threshold_sec"]
    assert error.value.details["source_timestamp_ns"] <= error.value.details["anchor_timestamp_ns"]


def test_short_activity_interval_reports_frame_count_and_minimum() -> None:
    profile = load_robot_profile(PROFILE)
    episode, _ = _series(profile)
    activity = ActivityInterval(100_000_000, 500_000_000, 100_000_000, 500_000_000)

    with pytest.raises(SynchronizationError) as error:
        synchronize_episode(
            episode,
            profile,
            activity,
            action_space="joint_absolute",
            minimum_output_frames=30,
        )

    assert error.value.reason == "episode_too_short"
    assert error.value.details["stream"] == "video.head"
    assert error.value.details["frame_count"] < error.value.details["minimum_output_frames"]


def test_aligned_value_range_rejection_identifies_stream_axis_and_value() -> None:
    profile = load_robot_profile(PROFILE)
    episode, head_times = _series(profile)
    key = "state.left_arm_joint"
    series = episode.streams[key]
    values = series.values.copy()
    values[:, 2] = 7.0
    episode.streams[key] = JointPositionSeries(series.timestamps_ns, values, series.names)
    activity = ActivityInterval(
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
        round(head_times[0] * 1e9),
        round(head_times[-1] * 1e9),
    )

    with pytest.raises(SynchronizationError) as error:
        synchronize_episode(
            episode,
            profile,
            activity,
            action_space="joint_absolute",
            minimum_output_frames=1,
        )

    assert error.value.reason == "value_range_exceeded"
    assert error.value.details["stream"] == key
    assert error.value.details["axis"] == 2
    assert error.value.details["value"] == 7.0
    assert error.value.details["range"] == list(profile.streams[key].value_range)
