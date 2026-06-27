import json
from types import SimpleNamespace

from data_preprocess.motion_detection import MotionDetectionConfig
from data_preprocess.wuji_rosbag_to_gr00t import (
    CUSTOM_MSG_DEFINITIONS,
    DEFAULT_TOPICS,
    AlignedEpisode,
    EpisodeConversionTask,
    LowPassFilterConfig,
    TimedSample,
    _align_episode,
    _apply_low_pass_filter,
    _build_rosbags_typestore,
    _clamp_timestamps_for_video_indexing,
    _joint_sample,
    _low_pass_filter_samples,
    _make_low_pass_filter_config,
    _message_timestamp,
    _metadata_feature_names,
    _prepare_video_frame,
    _required_topic_keys,
    _rewrite_parquet_global_index,
    _write_episode_outputs,
    _write_metadata,
    _write_video,
    parse_args,
)
import numpy as np
import pytest


def _samples_from_timestamps(timestamps, value_factory):
    return [
        TimedSample(float(timestamp), value_factory(float(timestamp))) for timestamp in timestamps
    ]


def test_align_episode_uses_common_time_window_across_all_target_streams(monkeypatch, tmp_path):
    def eef_value(timestamp):
        return np.full(6, timestamp, dtype=np.float32)

    def hand_value(timestamp):
        return np.full(4, timestamp, dtype=np.float32)

    def image_value(timestamp):
        return np.full((2, 2, 3), int(timestamp * 10), dtype=np.uint8)

    streams = {
        "left_eef_state": _samples_from_timestamps([1, 2, 3], eef_value),
        "right_eef_state": _samples_from_timestamps([1, 2, 3], eef_value),
        "left_eef_action": _samples_from_timestamps([1, 2, 3], eef_value),
        "right_eef_action": _samples_from_timestamps([1, 2, 3], eef_value),
        "left_hand_state": _samples_from_timestamps([1, 2, 3], hand_value),
        "right_hand_state": _samples_from_timestamps([1, 2, 3], hand_value),
        "left_hand_action": _samples_from_timestamps([1, 2, 3], hand_value),
        "right_hand_action": _samples_from_timestamps([1, 2, 3], hand_value),
        "head_rgb": _samples_from_timestamps([0, 1, 2, 3, 4], image_value),
        "left_wrist_rgb": _samples_from_timestamps([0, 1, 2, 3, 4], image_value),
        "right_wrist_rgb": _samples_from_timestamps([0, 1, 2, 3, 4], image_value),
    }
    hand_feature_names = {
        "left_hand_joints": [f"left_hand_joints.raw_joint_{index}" for index in range(4)],
        "right_hand_joints": [f"right_hand_joints.raw_joint_{index}" for index in range(4)],
    }

    monkeypatch.setattr(
        "data_preprocess.wuji_rosbag_to_gr00t._build_streams",
        lambda **kwargs: (streams, hand_feature_names),
    )

    episode = _align_episode(
        bag_dir=tmp_path,
        rotation_format="rotvec",
        max_time_skew=0.06,
        topics=DEFAULT_TOPICS,
        work_dir=None,
        bag_backend="rosbags",
        timestamp_source="header",
    )

    assert len(episode.state) == 3
    np.testing.assert_array_equal(episode.timestamps, np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert [int(frame[0, 0, 0]) for frame in episode.videos["head_rgb"]] == [10, 20, 30]


def test_align_episode_can_use_joint_action_space(monkeypatch, tmp_path):
    def left_joint_state_value(timestamp):
        return np.full(7, timestamp, dtype=np.float32)

    def right_joint_state_value(timestamp):
        return np.full(7, timestamp + 100, dtype=np.float32)

    def left_joint_action_value(timestamp):
        return np.full(7, timestamp + 10, dtype=np.float32)

    def right_joint_action_value(timestamp):
        return np.full(7, timestamp + 110, dtype=np.float32)

    def left_hand_value(timestamp):
        return np.full(4, timestamp + 200, dtype=np.float32)

    def right_hand_value(timestamp):
        return np.full(4, timestamp + 300, dtype=np.float32)

    def left_hand_action_value(timestamp):
        return np.full(4, timestamp + 210, dtype=np.float32)

    def right_hand_action_value(timestamp):
        return np.full(4, timestamp + 310, dtype=np.float32)

    def image_value(timestamp):
        return np.full((2, 2, 3), int(timestamp * 10), dtype=np.uint8)

    streams = {
        "left_joint_space_state": _samples_from_timestamps([1, 2], left_joint_state_value),
        "right_joint_space_state": _samples_from_timestamps([1, 2], right_joint_state_value),
        "left_joint_space_action": _samples_from_timestamps([1, 2], left_joint_action_value),
        "right_joint_space_action": _samples_from_timestamps([1, 2], right_joint_action_value),
        "left_hand_state": _samples_from_timestamps([1, 2], left_hand_value),
        "right_hand_state": _samples_from_timestamps([1, 2], right_hand_value),
        "left_hand_action": _samples_from_timestamps([1, 2], left_hand_action_value),
        "right_hand_action": _samples_from_timestamps([1, 2], right_hand_action_value),
        "head_rgb": _samples_from_timestamps([1, 2], image_value),
        "left_wrist_rgb": _samples_from_timestamps([1, 2], image_value),
        "right_wrist_rgb": _samples_from_timestamps([1, 2], image_value),
    }
    feature_names = {
        "left_joint_space": [f"left_joint_space.j{index}" for index in range(7)],
        "right_joint_space": [f"right_joint_space.j{index}" for index in range(7)],
        "left_hand_joints": [f"left_hand_joints.j{index}" for index in range(4)],
        "right_hand_joints": [f"right_hand_joints.j{index}" for index in range(4)],
    }

    monkeypatch.setattr(
        "data_preprocess.wuji_rosbag_to_gr00t._build_streams",
        lambda **kwargs: (streams, feature_names),
    )

    episode = _align_episode(
        bag_dir=tmp_path,
        rotation_format="rotvec",
        max_time_skew=0.06,
        topics=DEFAULT_TOPICS,
        work_dir=None,
        bag_backend="rosbags",
        timestamp_source="header",
        action_space="joint",
    )

    assert episode.state.shape == (2, 22)
    assert episode.action.shape == (2, 22)
    np.testing.assert_array_equal(
        episode.state[0],
        np.concatenate(
            [
                np.full(7, 1, dtype=np.float32),
                np.full(7, 101, dtype=np.float32),
                np.full(4, 201, dtype=np.float32),
                np.full(4, 301, dtype=np.float32),
            ]
        ),
    )
    np.testing.assert_array_equal(
        episode.action[0],
        np.concatenate(
            [
                np.full(7, 11, dtype=np.float32),
                np.full(7, 111, dtype=np.float32),
                np.full(4, 211, dtype=np.float32),
                np.full(4, 311, dtype=np.float32),
            ]
        ),
    )


def test_joint_action_space_requires_arm_joint_and_hand_topics():
    assert _required_topic_keys("joint") == [
        "left_joint_space_state",
        "right_joint_space_state",
        "left_joint_space_action",
        "right_joint_space_action",
        "left_hand_state",
        "right_hand_state",
        "left_hand_action",
        "right_hand_action",
        "head_rgb",
        "left_wrist_rgb",
        "right_wrist_rgb",
    ]


def test_rosbags_typestore_registers_astribot_joint_state():
    pytest.importorskip("rosbags")

    typestore = _build_rosbags_typestore()

    assert "astribot_msgs/msg/RobotJointState" in typestore.fielddefs


def test_custom_message_definitions_include_astribot_joint_state():
    assert "astribot_msgs/msg/RobotJointState" in CUSTOM_MSG_DEFINITIONS


def test_align_episode_trims_idle_frames_with_joint_action_space(monkeypatch, tmp_path):
    joint_values = {
        0.0: 0.0,
        1.0: 0.0,
        2.0: 1.0,
        3.0: 2.0,
        4.0: 2.0,
    }

    def joint_state_value(timestamp):
        value = np.zeros(7, dtype=np.float32)
        value[0] = joint_values[timestamp]
        return value

    def joint_action_value(timestamp):
        value = np.zeros(7, dtype=np.float32)
        value[0] = joint_values[timestamp]
        return value

    def eef_value(timestamp):
        value = np.zeros(6, dtype=np.float32)
        value[0] = joint_values[timestamp]
        return value

    def image_value(timestamp):
        return np.full((2, 2, 3), int(timestamp), dtype=np.uint8)

    timestamps = [0, 1, 2, 3, 4]
    streams = {
        "left_eef_state": _samples_from_timestamps(timestamps, eef_value),
        "right_eef_state": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(6, dtype=np.float32)
        ),
        "left_eef_action": _samples_from_timestamps(timestamps, eef_value),
        "right_eef_action": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(6, dtype=np.float32)
        ),
        "left_joint_space_state": _samples_from_timestamps(timestamps, joint_state_value),
        "right_joint_space_state": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(7, dtype=np.float32)
        ),
        "left_joint_space_action": _samples_from_timestamps(timestamps, joint_action_value),
        "right_joint_space_action": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(7, dtype=np.float32)
        ),
        "left_hand_state": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(4, dtype=np.float32)
        ),
        "right_hand_state": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(4, dtype=np.float32)
        ),
        "left_hand_action": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(4, dtype=np.float32)
        ),
        "right_hand_action": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(4, dtype=np.float32)
        ),
        "head_rgb": _samples_from_timestamps(timestamps, image_value),
        "left_wrist_rgb": _samples_from_timestamps(timestamps, image_value),
        "right_wrist_rgb": _samples_from_timestamps(timestamps, image_value),
    }
    feature_names = {
        "left_joint_space": [f"left_joint_space.j{index}" for index in range(7)],
        "right_joint_space": [f"right_joint_space.j{index}" for index in range(7)],
        "left_hand_joints": [f"left_hand_joints.j{index}" for index in range(4)],
        "right_hand_joints": [f"right_hand_joints.j{index}" for index in range(4)],
    }

    monkeypatch.setattr(
        "data_preprocess.wuji_rosbag_to_gr00t._build_streams",
        lambda **kwargs: (streams, feature_names),
    )

    episode = _align_episode(
        bag_dir=tmp_path,
        rotation_format="rotvec",
        max_time_skew=0.06,
        topics=DEFAULT_TOPICS,
        work_dir=None,
        bag_backend="rosbags",
        timestamp_source="header",
        action_space="joint",
        motion_detection_config=MotionDetectionConfig(
            velocity_threshold=0.1,
            hand_velocity_threshold=0.1,
            action_state_diff_threshold=0.1,
            window_duration_sec=1.0,
            min_motion_frames=1,
            fps=1.0,
        ),
    )

    assert episode.action.shape == (2, 22)
    np.testing.assert_array_equal(episode.timestamps, np.array([0.0, 1.0], dtype=np.float32))
    assert episode.motion_detection_result.idle_prefix_frames == 1
    assert episode.motion_detection_result.idle_suffix_frames == 2


def test_joint_action_space_motion_detection_uses_eef_streams(monkeypatch, tmp_path):
    joint_values = {
        0.0: 0.0,
        1.0: 0.0,
        2.0: 1.0,
        3.0: 2.0,
        4.0: 2.0,
    }

    def joint_state_value(timestamp):
        value = np.zeros(7, dtype=np.float32)
        value[0] = joint_values[timestamp]
        return value

    def joint_action_value(timestamp):
        value = np.zeros(7, dtype=np.float32)
        value[0] = joint_values[timestamp]
        return value

    def eef_value(timestamp):
        return np.zeros(6, dtype=np.float32)

    def image_value(timestamp):
        return np.full((2, 2, 3), int(timestamp), dtype=np.uint8)

    timestamps = [0, 1, 2, 3, 4]
    streams = {
        "left_eef_state": _samples_from_timestamps(timestamps, eef_value),
        "right_eef_state": _samples_from_timestamps(timestamps, eef_value),
        "left_eef_action": _samples_from_timestamps(timestamps, eef_value),
        "right_eef_action": _samples_from_timestamps(timestamps, eef_value),
        "left_joint_space_state": _samples_from_timestamps(timestamps, joint_state_value),
        "right_joint_space_state": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(7, dtype=np.float32)
        ),
        "left_joint_space_action": _samples_from_timestamps(timestamps, joint_action_value),
        "right_joint_space_action": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(7, dtype=np.float32)
        ),
        "left_hand_state": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(4, dtype=np.float32)
        ),
        "right_hand_state": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(4, dtype=np.float32)
        ),
        "left_hand_action": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(4, dtype=np.float32)
        ),
        "right_hand_action": _samples_from_timestamps(
            timestamps, lambda timestamp: np.zeros(4, dtype=np.float32)
        ),
        "head_rgb": _samples_from_timestamps(timestamps, image_value),
        "left_wrist_rgb": _samples_from_timestamps(timestamps, image_value),
        "right_wrist_rgb": _samples_from_timestamps(timestamps, image_value),
    }
    feature_names = {
        "left_joint_space": [f"left_joint_space.j{index}" for index in range(7)],
        "right_joint_space": [f"right_joint_space.j{index}" for index in range(7)],
        "left_hand_joints": [f"left_hand_joints.j{index}" for index in range(4)],
        "right_hand_joints": [f"right_hand_joints.j{index}" for index in range(4)],
    }

    def fake_build_streams(**kwargs):
        assert kwargs["include_eef_for_motion_detection"] is True
        return streams, feature_names

    monkeypatch.setattr(
        "data_preprocess.wuji_rosbag_to_gr00t._build_streams",
        fake_build_streams,
    )

    episode = _align_episode(
        bag_dir=tmp_path,
        rotation_format="rotvec",
        max_time_skew=0.06,
        topics=DEFAULT_TOPICS,
        work_dir=None,
        bag_backend="rosbags",
        timestamp_source="header",
        action_space="joint",
        motion_detection_config=MotionDetectionConfig(
            velocity_threshold=0.1,
            hand_velocity_threshold=0.1,
            action_state_diff_threshold=0.1,
            window_duration_sec=1.0,
            min_motion_frames=1,
            fps=1.0,
        ),
    )

    assert episode.action.shape == (5, 22)
    np.testing.assert_array_equal(
        episode.timestamps,
        np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
    assert episode.motion_detection_result.idle_prefix_frames == 0
    assert episode.motion_detection_result.idle_suffix_frames == 0


def test_metadata_feature_names_split_state_and_joint_action_names():
    feature_names = {
        "left_joint_space": [f"left_joint_space.j{index}" for index in range(7)],
        "right_joint_space": [f"right_joint_space.j{index}" for index in range(7)],
        "left_hand_joints": [f"left_hand_joints.j{index}" for index in range(4)],
        "right_hand_joints": [f"right_hand_joints.j{index}" for index in range(4)],
    }

    state_names, state_dims, action_names, action_dims = _metadata_feature_names(
        "rotvec",
        "joint",
        feature_names,
    )

    assert state_dims == [7, 7, 4, 4]
    assert action_dims == [7, 7, 4, 4]
    assert state_names[:2] == ["left_joint_space.j0", "left_joint_space.j1"]
    assert state_names[7:9] == ["right_joint_space.j0", "right_joint_space.j1"]
    assert state_names[14:16] == ["left_hand_joints.j0", "left_hand_joints.j1"]
    assert state_names[18:20] == ["right_hand_joints.j0", "right_hand_joints.j1"]
    assert action_names == state_names


def test_write_metadata_records_joint_action_shape_and_modalities(tmp_path):
    feature_names = {
        "left_joint_space": [f"left_joint_space.j{index}" for index in range(7)],
        "right_joint_space": [f"right_joint_space.j{index}" for index in range(7)],
        "left_hand_joints": [f"left_hand_joints.j{index}" for index in range(4)],
        "right_hand_joints": [f"right_hand_joints.j{index}" for index in range(4)],
    }

    _write_metadata(
        output_dir=tmp_path,
        episodes=[],
        task="",
        total_frames=0,
        fps=30.0,
        chunks_size=1000,
        rotation_format="rotvec",
        action_space="joint",
        video_shapes={
            "head_rgb": (2, 2, 3),
            "left_wrist_rgb": (2, 2, 3),
            "right_wrist_rgb": (2, 2, 3),
        },
        feature_names=feature_names,
    )

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    modality = json.loads((tmp_path / "meta" / "modality.json").read_text())

    assert info["features"]["observation.state"]["shape"] == [22]
    assert info["features"]["action"]["shape"] == [22]
    assert info["features"]["action"]["names"][:2] == [
        "left_joint_space.j0",
        "left_joint_space.j1",
    ]
    assert modality["state"]["left_joint_space"] == {"start": 0, "end": 7}
    assert modality["state"]["right_joint_space"] == {"start": 7, "end": 14}
    assert modality["state"]["left_hand_joints"] == {"start": 14, "end": 18}
    assert modality["state"]["right_hand_joints"] == {"start": 18, "end": 22}
    assert modality["action"]["left_joint_space"] == {"start": 0, "end": 7}
    assert modality["action"]["right_joint_space"] == {"start": 7, "end": 14}
    assert modality["action"]["left_hand_joints"] == {"start": 14, "end": 18}
    assert modality["action"]["right_hand_joints"] == {"start": 18, "end": 22}


def test_joint_sample_keeps_raw_left_hand_position_order_and_dimension():
    msg = SimpleNamespace(
        name=[
            f"left_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)
        ],
        position=np.arange(1, 21, dtype=np.float32),
    )

    sample = _joint_sample(msg)

    np.testing.assert_array_equal(sample, np.arange(1, 21, dtype=np.float32))


def test_joint_sample_keeps_raw_right_hand_position_order_and_dimension():
    msg = SimpleNamespace(
        name=[
            f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)
        ],
        position=np.arange(101, 121, dtype=np.float32),
    )

    sample = _joint_sample(msg)

    np.testing.assert_array_equal(sample, np.arange(101, 121, dtype=np.float32))


def test_prepare_video_frame_keeps_original_resolution_when_size_is_none():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    prepared = _prepare_video_frame(frame, None)

    assert prepared.shape == (480, 640, 3)


def test_prepare_video_frame_resizes_only_when_size_is_explicit():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    prepared = _prepare_video_frame(frame, (320, 240))

    assert prepared.shape == (240, 320, 3)


def test_message_timestamp_uses_header_stamp_by_default():
    msg = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=345_000_000))
    )

    assert _message_timestamp(msg, fallback_ns=99_000_000_000) == 12.345


def test_message_timestamp_can_use_rosbag_timestamp_even_when_header_exists():
    msg = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=345_000_000))
    )

    assert _message_timestamp(msg, fallback_ns=99_000_000_000, timestamp_source="rosbag") == 99.0


def test_message_timestamp_header_mode_falls_back_to_rosbag_when_header_is_zero():
    msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)))

    assert _message_timestamp(msg, fallback_ns=99_000_000_000, timestamp_source="header") == 99.0


def test_parse_args_accepts_num_workers(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["wuji_rosbag_to_gr00t.py", "--num-workers", "3"],
    )

    args = parse_args()

    assert args.num_workers == 3


def test_low_pass_filter_samples_uses_ema_formula():
    samples = [
        TimedSample(0.0, np.array([0.0, 10.0], dtype=np.float32)),
        TimedSample(1.0, np.array([10.0, 20.0], dtype=np.float32)),
        TimedSample(2.0, np.array([20.0, 30.0], dtype=np.float32)),
    ]

    filtered = _low_pass_filter_samples(samples, filter_scale=0.3)

    np.testing.assert_allclose(filtered[0].value, np.array([0.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(filtered[1].value, np.array([3.0, 13.0], dtype=np.float32))
    np.testing.assert_allclose(
        filtered[2].value,
        np.array([8.1, 18.1], dtype=np.float32),
        atol=1e-6,
    )
    assert [sample.timestamp for sample in filtered] == [0.0, 1.0, 2.0]


def test_apply_low_pass_filter_only_filters_configured_streams():
    streams = {
        "left_eef_action": [
            TimedSample(0.0, np.array([0.0], dtype=np.float32)),
            TimedSample(1.0, np.array([10.0], dtype=np.float32)),
        ],
        "left_eef_state": [
            TimedSample(0.0, np.array([0.0], dtype=np.float32)),
            TimedSample(1.0, np.array([10.0], dtype=np.float32)),
        ],
    }

    filtered = _apply_low_pass_filter(
        streams,
        LowPassFilterConfig(filter_scale=0.5, streams=("left_eef_action",)),
    )

    np.testing.assert_allclose(filtered["left_eef_action"][1].value, np.array([5.0]))
    np.testing.assert_allclose(filtered["left_eef_state"][1].value, np.array([10.0]))


def test_parse_args_accepts_low_pass_filter_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "wuji_rosbag_to_gr00t.py",
            "--enable-low-pass-filter",
            "--filter-scale",
            "0.1",
            "--low-pass-filter-stream",
            "left_hand_action",
        ],
    )

    config = _make_low_pass_filter_config(parse_args())

    assert config == LowPassFilterConfig(filter_scale=0.1, streams=("left_hand_action",))


def test_rewrite_parquet_global_index_updates_index_column(tmp_path):
    pd = pytest.importorskip("pandas")
    parquet_path = tmp_path / "episode_000001.parquet"
    df = pd.DataFrame(
        {
            "frame_index": np.arange(3, dtype=np.int64),
            "index": np.arange(3, dtype=np.int64),
        }
    )
    df.to_parquet(parquet_path, index=False)

    _rewrite_parquet_global_index(parquet_path, global_start_index=10, length=3)

    rewritten = pd.read_parquet(parquet_path)
    np.testing.assert_array_equal(rewritten["index"].to_numpy(), np.array([10, 11, 12]))


def test_clamp_timestamps_for_video_indexing_prevents_tail_rounding_oob():
    timestamps = np.array([25.156320571899414, 25.19158935546875], dtype=np.float32)
    actual_video_fps = 146785 / 4881

    clamped = _clamp_timestamps_for_video_indexing(
        timestamps,
        video_streams=[(758, actual_video_fps)],
    )

    assert clamped.dtype == np.float32
    assert clamped[0] == timestamps[0]
    assert clamped[-1] < timestamps[-1]
    assert round(float(clamped[-1]) * actual_video_fps) < 758


def test_clamp_timestamps_for_video_indexing_keeps_safe_timestamps_unchanged():
    timestamps = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    clamped = _clamp_timestamps_for_video_indexing(
        timestamps,
        video_streams=[(3, 1.0), (3, 1.0)],
    )

    np.testing.assert_array_equal(clamped, timestamps)


def test_write_episode_outputs_rewrites_timestamps_safe_for_encoded_video(monkeypatch, tmp_path):
    pd = pytest.importorskip("pandas")
    episode = AlignedEpisode(
        state=np.zeros((2, 1), dtype=np.float32),
        action=np.zeros((2, 1), dtype=np.float32),
        videos={
            "head_rgb": [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
            "left_wrist_rgb": [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
            "right_wrist_rgb": [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
        },
        timestamps=np.array([0.0, 1.6], dtype=np.float32),
        max_skew=0.0,
        camera_frame_count=2,
        source="bag",
        hand_feature_names={"left_hand_joints": [], "right_hand_joints": []},
        video_shapes={
            "head_rgb": (2, 2, 3),
            "left_wrist_rgb": (2, 2, 3),
            "right_wrist_rgb": (2, 2, 3),
        },
    )
    monkeypatch.setattr(
        "data_preprocess.wuji_rosbag_to_gr00t._align_episode",
        lambda **kwargs: episode,
    )
    monkeypatch.setattr(
        "data_preprocess.wuji_rosbag_to_gr00t._write_video",
        lambda *args, **kwargs: (2, 2, 3),
    )
    monkeypatch.setattr(
        "data_preprocess.wuji_rosbag_to_gr00t._probe_video_timing",
        lambda path: (2, 1.0),
    )

    result = _write_episode_outputs(
        EpisodeConversionTask(
            episode_index=0,
            bag_dir=tmp_path / "bag",
            output_dir=tmp_path,
            rotation_format="rotvec",
            max_time_skew=0.06,
            topics=DEFAULT_TOPICS,
            work_dir=None,
            bag_backend="rosbags",
            timestamp_source="header",
            output_fps=None,
            image_size=None,
            chunks_size=1000,
            global_start_index=0,
        )
    )

    rewritten = pd.read_parquet(result.parquet_path)
    assert len(rewritten) == 2
    np.testing.assert_array_equal(rewritten["frame_index"].to_numpy(), np.array([0, 1]))
    assert rewritten["timestamp"].iloc[-1] < 1.6
    assert round(float(rewritten["timestamp"].iloc[-1])) < 2


def test_write_video_streams_rgb_frames_to_ffmpeg(monkeypatch, tmp_path):
    calls = []

    class FakeStdin:
        def __init__(self):
            self.writes = []
            self.closed = False

        def write(self, data):
            self.writes.append(data)

        def close(self):
            self.closed = True

    class FakeStderr:
        def read(self):
            return b""

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.kwargs = kwargs
            self.stdin = FakeStdin()
            self.stderr = FakeStderr()
            calls.append(self)

        def communicate(self):
            raise AssertionError("communicate must not be used after stdin is closed")

        def wait(self):
            return 0

        @property
        def returncode(self):
            return 0

    monkeypatch.setattr("data_preprocess.wuji_rosbag_to_gr00t.subprocess.Popen", FakeProcess)
    frame0 = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    frame1 = np.full((2, 2, 3), 7, dtype=np.uint8)

    shape = _write_video(tmp_path / "episode_000000.mp4", [frame0, frame1], 15.0, None)

    assert shape == (2, 2, 3)
    assert len(calls) == 1
    process = calls[0]
    assert process.cmd[:4] == ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert process.cmd[process.cmd.index("-f") + 1] == "rawvideo"
    assert process.cmd[process.cmd.index("-pix_fmt") + 1] == "rgb24"
    assert process.cmd[process.cmd.index("-s") + 1] == "2x2"
    assert process.cmd[process.cmd.index("-r") + 1] == "15.000000"
    assert process.cmd[process.cmd.index("-c:v") + 1] == "libx264"
    assert process.cmd[process.cmd.index("-pix_fmt", process.cmd.index("-c:v")) + 1] == "yuv420p"
    assert process.stdin.writes == [frame0.tobytes(), frame1.tobytes()]
    assert process.stdin.closed


def test_write_video_reports_ffmpeg_failure(monkeypatch, tmp_path):
    class FakeStdin:
        def write(self, data):
            pass

        def close(self):
            pass

    class FakeStderr:
        def read(self):
            return b"encoder failed"

    class FakeProcess:
        stdin = FakeStdin()
        stderr = FakeStderr()
        returncode = 1

        def __init__(self, cmd, **kwargs):
            pass

        def communicate(self):
            raise AssertionError("communicate must not be used after stdin is closed")

        def wait(self):
            return 1

    monkeypatch.setattr("data_preprocess.wuji_rosbag_to_gr00t.subprocess.Popen", FakeProcess)

    with pytest.raises(RuntimeError, match="encoder failed"):
        _write_video(
            tmp_path / "episode_000000.mp4",
            [np.zeros((2, 2, 3), dtype=np.uint8)],
            30.0,
            None,
        )
