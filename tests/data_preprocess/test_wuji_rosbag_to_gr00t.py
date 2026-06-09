from types import SimpleNamespace

from data_preprocess.wuji_rosbag_to_gr00t import (
    _joint_sample,
    _message_timestamp,
    _prepare_video_frame,
    _write_video,
)
import numpy as np
import pytest


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
