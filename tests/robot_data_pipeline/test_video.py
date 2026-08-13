import cv2
import numpy as np
import pytest
from robot_data_pipeline.export.video import VideoWriteError, decode_image, probe_video, write_video


def test_opencv_decode_exception_becomes_video_write_error(monkeypatch) -> None:
    def fail_decode(*args, **kwargs):
        raise cv2.error("forced decode failure")

    monkeypatch.setattr(cv2, "imdecode", fail_decode)

    with pytest.raises(VideoWriteError, match="failed to decode"):
        decode_image(b"malformed")


def test_write_video_accepts_jpeg_image_pipe(tmp_path) -> None:
    images = []
    for value in (0, 80, 160, 240):
        image = np.full((24, 32, 3), value, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        images.append(encoded.tobytes())

    path = tmp_path / "episode.mp4"
    shape = write_video(
        path,
        tuple(images),
        fps=30,
        preset="ultrafast",
        encoder_threads=1,
    )

    assert shape == (24, 32, 3)
    assert probe_video(path) == (4, 30.0)
