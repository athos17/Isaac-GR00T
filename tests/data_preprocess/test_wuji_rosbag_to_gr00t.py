from types import SimpleNamespace

from data_preprocess.wuji_rosbag_to_gr00t import (
    DEFAULT_TOPICS,
    LowPassFilterConfig,
    TimedSample,
    _apply_low_pass_filter,
    _align_episode,
    _joint_sample,
    _low_pass_filter_samples,
    _make_low_pass_filter_config,
    _message_timestamp,
    _prepare_video_frame,
    _rewrite_parquet_global_index,
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
    np.testing.assert_allclose(filtered[2].value, np.array([8.1, 18.1], dtype=np.float32))
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
