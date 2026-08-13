from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


class VideoWriteError(RuntimeError):
    pass


def decode_image(encoded: bytes) -> np.ndarray:
    """Decode one image for callers that need eager validation or pixel access."""
    try:
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise VideoWriteError("failed to decode an aligned image") from exc
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise VideoWriteError("failed to decode an aligned image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_video(
    path: Path,
    encoded_images: tuple[bytes, ...],
    *,
    fps: float,
    preset: str = "veryfast",
    encoder_threads: int = 0,
) -> tuple[int, int, int]:
    if not encoded_images:
        raise VideoWriteError("cannot write an empty video")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-r",
        f"{fps:.9f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-threads",
        str(encoder_threads),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise VideoWriteError("ffmpeg is required for video export") from exc
    assert process.stdin is not None
    try:
        for encoded in encoded_images:
            process.stdin.write(encoded)
        process.stdin.close()
    except BrokenPipeError as exc:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        process.wait()
        raise VideoWriteError(f"ffmpeg stopped while writing {path}: {stderr.strip()}") from exc
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    process.wait()
    if process.returncode:
        raise VideoWriteError(f"ffmpeg failed while writing {path}: {stderr.strip()}")
    return probe_video_shape(path)


def probe_video_shape(path: Path) -> tuple[int, int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height,width,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    try:
        stream = json.loads(subprocess.check_output(command, text=True))["streams"][0]
        height = int(stream["height"])
        width = int(stream["width"])
        pix_fmt = stream["pix_fmt"]
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        KeyError,
        ValueError,
        IndexError,
    ) as exc:
        raise VideoWriteError(f"failed to inspect video {path}: {exc}") from exc
    if pix_fmt != "yuv420p":
        raise VideoWriteError(f"unexpected pixel format for {path}: {pix_fmt}")
    return height, width, 3


def probe_video(path: Path) -> tuple[int, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,nb_read_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        output = subprocess.check_output(command, text=True)
        stream = json.loads(output)["streams"][0]
        frame_count = int(stream.get("nb_frames") or stream["nb_read_frames"])
        fps = float(Fraction(stream["avg_frame_rate"]))
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        KeyError,
        ValueError,
        IndexError,
    ) as exc:
        raise VideoWriteError(f"failed to validate video {path}: {exc}") from exc
    return frame_count, fps
