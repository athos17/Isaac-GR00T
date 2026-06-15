#!/usr/bin/env python
"""Convert Wuji/Astribot ROS2 bags to GR00T LeRobot v2 EEF + hand datasets.

The converter uses the head RGB camera timestamps as anchors, aligns proprioception,
actions, and the wrist RGB cameras to those anchors, and writes one MP4 per camera view.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd
import yaml


DEFAULT_TOPICS = {
    "left_eef_state": "/astribot_arm_left/endpoint_current_states",
    "right_eef_state": "/astribot_arm_right/endpoint_current_states",
    "left_eef_action": "/astribot_arm_left/endpoint_desired_states",
    "right_eef_action": "/astribot_arm_right/endpoint_desired_states",
    "left_hand_state": "/left_hand/joint_states",
    "right_hand_state": "/right_hand/joint_states",
    "left_hand_action": "/left_hand/joint_commands",
    "right_hand_action": "/right_hand/joint_commands",
    "head_rgb": "/astribot_camera/head_rgbd/color_compress/compressed",
    "left_wrist_rgb": "/astribot_camera/left_wrist_rgbd/color_compress/compressed",
    "right_wrist_rgb": "/astribot_camera/right_wrist_rgbd/color_compress/compressed",
}

VIDEO_TOPIC_KEYS = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
VIDEO_MODALITY_KEYS = {
    "head_rgb": "head_view",
    "left_wrist_rgb": "left_wrist_view",
    "right_wrist_rgb": "right_wrist_view",
}

STATE_KEYS = ["left_eef", "right_eef", "left_hand_joints", "right_hand_joints"]
TimestampSource = Literal["header", "rosbag"]

CUSTOM_MSG_DEFINITIONS = {
    "astribot_msgs/msg/RobotCartesianState": """\
std_msgs/Header header
geometry_msgs/Pose pose
geometry_msgs/Twist twist
geometry_msgs/Wrench wrench
""",
    "astribot_msgs/msg/RobotCartesianStates": """\
std_msgs/Header header
astribot_msgs/RobotCartesianState[] states
""",
}


@dataclass(frozen=True)
class TimedSample:
    timestamp: float
    value: np.ndarray


@dataclass
class AlignedEpisode:
    state: np.ndarray
    action: np.ndarray
    videos: dict[str, list[np.ndarray]]
    timestamps: np.ndarray
    max_skew: float
    camera_frame_count: int
    source: str
    hand_feature_names: dict[str, list[str]]
    video_shapes: dict[str, tuple[int, int, int]]


@dataclass(frozen=True)
class EpisodeConversionTask:
    episode_index: int
    bag_dir: Path
    output_dir: Path
    rotation_format: str
    max_time_skew: float
    topics: dict[str, str]
    work_dir: Path | None
    bag_backend: str
    timestamp_source: TimestampSource
    output_fps: float | None
    image_size: tuple[int, int] | None
    chunks_size: int
    global_start_index: int


@dataclass(frozen=True)
class EpisodeConversionResult:
    episode_index: int
    length: int
    source: str
    fps: float
    max_skew: float
    camera_frame_count: int
    hand_feature_names: dict[str, list[str]]
    video_shapes: dict[str, tuple[int, int, int]]
    parquet_path: Path
    written_global_start_index: int


def _load_metadata(bag_dir: Path) -> dict[str, Any]:
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing rosbag metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _topic_type_map(metadata: dict[str, Any]) -> dict[str, str]:
    bag_info = metadata["rosbag2_bagfile_information"]
    topics = bag_info.get("topics_with_message_count", [])
    return {item["topic_metadata"]["name"]: item["topic_metadata"]["type"] for item in topics}


def _topic_counts(metadata: dict[str, Any]) -> dict[str, int]:
    bag_info = metadata["rosbag2_bagfile_information"]
    topics = bag_info.get("topics_with_message_count", [])
    return {item["topic_metadata"]["name"]: int(item.get("message_count", 0)) for item in topics}


def _maybe_decompress_bag(
    bag_dir: Path, work_dir: Path | None
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Return an uncompressed bag directory readable by the selected backend.

    When only *.db3.zstd exists, this helper materializes a temporary uncompressed copy
    so sqlite-based readers can open it.
    """
    if any(bag_dir.glob("*.db3")):
        return bag_dir, None
    zstd_files = sorted(bag_dir.glob("*.db3.zstd"))
    if not zstd_files:
        return bag_dir, None

    temp_ctx = None
    if work_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="wuji_bag_")
        target_dir = Path(temp_ctx.name)
    else:
        target_dir = work_dir / bag_dir.name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("Bag is zstd-compressed and `zstd` is not available in PATH.")

    decompressed_names = []
    for compressed in zstd_files:
        output = target_dir / compressed.name.removesuffix(".zstd")
        subprocess.run([zstd, "-d", "-f", str(compressed), "-o", str(output)], check=True)
        decompressed_names.append(output.name)

    metadata = _load_metadata(bag_dir)
    bag_info = metadata["rosbag2_bagfile_information"]
    bag_info["compression_format"] = ""
    bag_info["compression_mode"] = ""
    bag_info["relative_file_paths"] = decompressed_names
    for file_info in bag_info.get("files", []):
        path = Path(file_info.get("path", ""))
        if path.name.endswith(".zstd"):
            file_info["path"] = path.name.removesuffix(".zstd")
    with (target_dir / "metadata.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    return target_dir, temp_ctx


def _import_rosbag2_reader():
    try:
        from rclpy.serialization import deserialize_message  # type: ignore
        import rosbag2_py  # type: ignore
        from rosidl_runtime_py.utilities import get_message  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 Python packages are required to deserialize bags. "
            "Run this script inside the robot ROS2 environment, or install rosbag2_py, "
            "rclpy, and the custom astribot/wujihand message packages."
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def _message_timestamp(
    msg: Any,
    fallback_ns: int,
    timestamp_source: TimestampSource = "header",
) -> float:
    if timestamp_source == "rosbag":
        return fallback_ns * 1e-9
    if timestamp_source != "header":
        raise ValueError(f"Unknown timestamp source: {timestamp_source!r}")

    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    sec = int(getattr(stamp, "sec", 0) or 0)
    nanosec = int(getattr(stamp, "nanosec", 0) or 0)
    if sec != 0 or nanosec != 0:
        return sec + nanosec * 1e-9
    return fallback_ns * 1e-9


def _read_bag_messages_ros2(
    bag_dir: Path,
    topics: set[str],
    work_dir: Path | None,
    timestamp_source: TimestampSource,
) -> dict[str, list[tuple[float, Any]]]:
    rosbag2_py, deserialize_message, get_message = _import_rosbag2_reader()
    readable_dir, temp_ctx = _maybe_decompress_bag(bag_dir, work_dir)
    try:
        metadata = _load_metadata(readable_dir)
        topic_types = _topic_type_map(metadata)
        missing = sorted(topic for topic in topics if topic not in topic_types)
        if missing:
            raise ValueError(f"{bag_dir} is missing required topics: {missing}")

        type_classes = {topic: get_message(topic_types[topic]) for topic in topics}

        storage_options = rosbag2_py.StorageOptions(uri=str(readable_dir), storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        messages = {topic: [] for topic in topics}
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            if topic not in topics:
                continue
            msg = deserialize_message(data, type_classes[topic])
            messages[topic].append((_message_timestamp(msg, timestamp_ns, timestamp_source), msg))
        return messages
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def _build_rosbags_typestore():
    try:
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Pure-Python backend requires `rosbags`. Install it with `pip install rosbags`."
        ) from exc

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    custom_types = {}
    for msgtype, definition in CUSTOM_MSG_DEFINITIONS.items():
        custom_types.update(get_types_from_msg(definition, msgtype))
    typestore.register(custom_types)
    return typestore


def _read_bag_messages_rosbags(
    bag_dir: Path,
    topics: set[str],
    work_dir: Path | None,
    timestamp_source: TimestampSource,
) -> dict[str, list[tuple[float, Any]]]:
    try:
        from rosbags.highlevel import AnyReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Pure-Python backend requires `rosbags`. Install it with `pip install rosbags`."
        ) from exc

    readable_dir, temp_ctx = _maybe_decompress_bag(bag_dir, work_dir)
    try:
        typestore = _build_rosbags_typestore()
        metadata = _load_metadata(readable_dir)
        topic_types = _topic_type_map(metadata)
        missing = sorted(topic for topic in topics if topic not in topic_types)
        if missing:
            raise ValueError(f"{bag_dir} is missing required topics: {missing}")

        messages = {topic: [] for topic in topics}
        with AnyReader([readable_dir], default_typestore=typestore) as reader:
            connections = [conn for conn in reader.connections if conn.topic in topics]
            for conn, timestamp_ns, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                messages[conn.topic].append(
                    (_message_timestamp(msg, timestamp_ns, timestamp_source), msg)
                )
        return messages
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def _read_bag_messages(
    bag_dir: Path,
    topics: set[str],
    work_dir: Path | None,
    backend: str,
    timestamp_source: TimestampSource,
) -> dict[str, list[tuple[float, Any]]]:
    if backend == "rosbags":
        return _read_bag_messages_rosbags(bag_dir, topics, work_dir, timestamp_source)
    if backend == "ros2":
        return _read_bag_messages_ros2(bag_dir, topics, work_dir, timestamp_source)
    if backend != "auto":
        raise ValueError(f"Unknown bag backend: {backend}")

    try:
        return _read_bag_messages_rosbags(bag_dir, topics, work_dir, timestamp_source)
    except RuntimeError as rosbags_error:
        try:
            return _read_bag_messages_ros2(bag_dir, topics, work_dir, timestamp_source)
        except RuntimeError as ros2_error:
            raise RuntimeError(
                "Could not read rosbag with either backend. Install pure-Python `rosbags`, "
                "or run inside a ROS2 Python environment.\n"
                f"rosbags backend error: {rosbags_error}\n"
                f"ros2 backend error: {ros2_error}"
            ) from ros2_error


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat


def _quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat)
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:4]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half <= 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(sin_half, w)
    return (xyz * (angle / sin_half)).astype(np.float32)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _quat_wxyz_to_rot6d(quat: np.ndarray) -> np.ndarray:
    rotation_matrix = _quat_wxyz_to_matrix(quat)
    return rotation_matrix[:2, :].reshape(-1).astype(np.float32)


def _pose_message(msg: Any) -> Any:
    if hasattr(msg, "pose"):
        return msg.pose
    states = getattr(msg, "states", None)
    if states:
        return states[0].pose
    raise ValueError(f"Unsupported Cartesian message without pose/states: {type(msg)!r}")


def _eef_sample(msg: Any, rotation_format: str) -> np.ndarray:
    pose = _pose_message(msg)
    position = pose.position
    orientation = pose.orientation
    xyz = np.array([position.x, position.y, position.z], dtype=np.float32)
    quat_wxyz = np.array(
        [orientation.w, orientation.x, orientation.y, orientation.z],
        dtype=np.float64,
    )
    if rotation_format == "rotvec":
        rotation = _quat_wxyz_to_rotvec(quat_wxyz)
    elif rotation_format == "rot6d":
        rotation = _quat_wxyz_to_rot6d(quat_wxyz)
    else:
        raise ValueError(f"Unsupported rotation format: {rotation_format}")
    return np.concatenate([xyz, rotation]).astype(np.float32)


def _joint_sample(msg: Any) -> np.ndarray:
    positions = np.asarray(getattr(msg, "position", []), dtype=np.float32)
    if positions.size == 0:
        raise ValueError("JointState has no position values")
    return positions.astype(np.float32)


def _joint_feature_names(msg: Any, prefix: str) -> list[str]:
    positions = np.asarray(getattr(msg, "position", []), dtype=np.float32)
    names = [str(name) for name in getattr(msg, "name", [])]
    if len(names) == len(positions):
        return [f"{prefix}.{name}" for name in names]
    return [f"{prefix}.raw_joint_{index}" for index in range(len(positions))]


def _decode_image_sample(msg: Any) -> np.ndarray:
    """Decode common ROS image messages to RGB uint8 HWC arrays."""
    nested_image = getattr(msg, "image", None)
    if nested_image is not None:
        return _decode_image_sample(nested_image)

    data = getattr(msg, "data", None)
    if data is None:
        raise ValueError(f"Image message has no data field: {type(msg)!r}")
    data_array = np.frombuffer(bytes(data), dtype=np.uint8)

    # sensor_msgs/msg/CompressedImage exposes a format string and encoded bytes.
    if hasattr(msg, "format"):
        bgr = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode compressed image")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    height = int(getattr(msg, "height", 0) or 0)
    width = int(getattr(msg, "width", 0) or 0)
    encoding = str(getattr(msg, "encoding", "rgb8") or "rgb8").lower()
    if height <= 0 or width <= 0:
        raise ValueError(f"Raw image message missing height/width: {type(msg)!r}")

    channels = 1 if encoding in {"mono8", "8uc1"} else 3
    expected = height * width * channels
    if data_array.size < expected:
        raise ValueError(
            f"Raw image data too small: got {data_array.size} bytes, expected at least {expected}"
        )
    image = data_array[:expected].reshape(height, width, channels)
    if channels == 1:
        return np.repeat(image, 3, axis=2)
    if encoding in {"bgr8", "bgr"}:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if encoding in {"rgb8", "rgb"}:
        return image
    raise ValueError(f"Unsupported raw image encoding: {encoding!r}")


def _to_timed_samples(
    raw_messages: list[tuple[float, Any]],
    extractor,
) -> list[TimedSample]:
    samples = [TimedSample(timestamp=t, value=extractor(msg)) for t, msg in raw_messages]
    samples.sort(key=lambda sample: sample.timestamp)
    return samples


def _nearest_sample(
    samples: list[TimedSample], timestamp: float, cursor: int
) -> tuple[TimedSample, int, float]:
    while cursor + 1 < len(samples) and samples[cursor + 1].timestamp <= timestamp:
        cursor += 1
    best = samples[cursor]
    if cursor + 1 < len(samples):
        next_sample = samples[cursor + 1]
        if abs(next_sample.timestamp - timestamp) < abs(best.timestamp - timestamp):
            best = next_sample
    return best, cursor, abs(best.timestamp - timestamp)


def _common_time_window(streams: dict[str, list[TimedSample]]) -> tuple[float, float]:
    starts = {name: samples[0].timestamp for name, samples in streams.items()}
    ends = {name: samples[-1].timestamp for name, samples in streams.items()}
    common_start = max(starts.values())
    common_end = min(ends.values())
    if common_start > common_end:
        latest_start_stream = max(starts, key=starts.__getitem__)
        earliest_end_stream = min(ends, key=ends.__getitem__)
        raise ValueError(
            "Target topics do not share a common time window: "
            f"latest start is {latest_start_stream} at {common_start:.6f}s, "
            f"earliest end is {earliest_end_stream} at {common_end:.6f}s"
        )
    return common_start, common_end


def _build_streams(
    bag_dir: Path,
    rotation_format: str,
    topics: dict[str, str],
    work_dir: Path | None,
    bag_backend: str,
    timestamp_source: TimestampSource,
) -> tuple[dict[str, list[TimedSample]], dict[str, list[str]]]:
    required_topics = set(topics.values())
    raw = _read_bag_messages(
        bag_dir,
        required_topics,
        work_dir,
        bag_backend,
        timestamp_source,
    )
    empty_raw = [key for key, topic in topics.items() if not raw[topic]]
    if empty_raw:
        raise ValueError(f"{bag_dir} has no readable raw messages for streams: {empty_raw}")

    hand_feature_names = {
        "left_hand_joints": _joint_feature_names(
            raw[topics["left_hand_state"]][0][1],
            "left_hand_joints",
        ),
        "right_hand_joints": _joint_feature_names(
            raw[topics["right_hand_state"]][0][1],
            "right_hand_joints",
        ),
    }
    streams = {
        "left_eef_state": _to_timed_samples(
            raw[topics["left_eef_state"]],
            lambda msg: _eef_sample(msg, rotation_format),
        ),
        "right_eef_state": _to_timed_samples(
            raw[topics["right_eef_state"]],
            lambda msg: _eef_sample(msg, rotation_format),
        ),
        "left_eef_action": _to_timed_samples(
            raw[topics["left_eef_action"]],
            lambda msg: _eef_sample(msg, rotation_format),
        ),
        "right_eef_action": _to_timed_samples(
            raw[topics["right_eef_action"]],
            lambda msg: _eef_sample(msg, rotation_format),
        ),
        "left_hand_state": _to_timed_samples(
            raw[topics["left_hand_state"]],
            _joint_sample,
        ),
        "right_hand_state": _to_timed_samples(
            raw[topics["right_hand_state"]],
            _joint_sample,
        ),
        "left_hand_action": _to_timed_samples(
            raw[topics["left_hand_action"]],
            _joint_sample,
        ),
        "right_hand_action": _to_timed_samples(
            raw[topics["right_hand_action"]],
            _joint_sample,
        ),
        "head_rgb": _to_timed_samples(
            raw[topics["head_rgb"]],
            _decode_image_sample,
        ),
        "left_wrist_rgb": _to_timed_samples(
            raw[topics["left_wrist_rgb"]],
            _decode_image_sample,
        ),
        "right_wrist_rgb": _to_timed_samples(
            raw[topics["right_wrist_rgb"]],
            _decode_image_sample,
        ),
    }
    empty = [name for name, samples in streams.items() if not samples]
    if empty:
        raise ValueError(f"{bag_dir} has no readable samples for streams: {empty}")
    hand_dim_pairs = {
        "left_hand_joints": (
            len(streams["left_hand_state"][0].value),
            len(streams["left_hand_action"][0].value),
        ),
        "right_hand_joints": (
            len(streams["right_hand_state"][0].value),
            len(streams["right_hand_action"][0].value),
        ),
    }
    mismatched = {
        key: {"state": state_dim, "action": action_dim}
        for key, (state_dim, action_dim) in hand_dim_pairs.items()
        if state_dim != action_dim
    }
    if mismatched:
        raise ValueError(f"Hand state/action dimensions must match for metadata: {mismatched}")
    for key, names in hand_feature_names.items():
        state_dim = hand_dim_pairs[key][0]
        if len(names) != state_dim:
            raise ValueError(
                f"{key} metadata names length {len(names)} does not match dim {state_dim}"
            )
    return streams, hand_feature_names


def _align_episode(
    bag_dir: Path,
    rotation_format: str,
    max_time_skew: float,
    topics: dict[str, str],
    work_dir: Path | None,
    bag_backend: str,
    timestamp_source: TimestampSource,
) -> AlignedEpisode:
    streams, hand_feature_names = _build_streams(
        bag_dir=bag_dir,
        rotation_format=rotation_format,
        topics=topics,
        work_dir=work_dir,
        bag_backend=bag_backend,
        timestamp_source=timestamp_source,
    )

    common_start, common_end = _common_time_window(streams)

    # The head camera is the clock, but only after trimming to the time range
    # shared by every target topic so boundary frames cannot align to stale data.
    anchor_samples = [
        sample for sample in streams["head_rgb"] if common_start <= sample.timestamp <= common_end
    ]
    camera_frame_count = len(anchor_samples)
    if camera_frame_count <= 0:
        raise ValueError(
            f"{bag_dir} has no head RGB frames inside the common target-topic time window "
            f"[{common_start:.6f}, {common_end:.6f}]"
        )
    start = anchor_samples[0].timestamp

    cursors = {name: 0 for name in streams}
    states = []
    actions = []
    videos = {key: [] for key in VIDEO_TOPIC_KEYS}
    kept_timestamps = []
    max_seen_skew = 0.0

    for anchor_sample in anchor_samples:
        timestamp = anchor_sample.timestamp
        selected = {}
        skews = []
        for name, samples in streams.items():
            sample, cursor, skew = _nearest_sample(samples, float(timestamp), cursors[name])
            cursors[name] = cursor
            selected[name] = sample.value
            skews.append(skew)

        frame_max_skew = max(skews)
        max_seen_skew = max(max_seen_skew, frame_max_skew)

        states.append(
            np.concatenate(
                [
                    selected["left_eef_state"],
                    selected["right_eef_state"],
                    selected["left_hand_state"],
                    selected["right_hand_state"],
                ]
            )
        )
        actions.append(
            np.concatenate(
                [
                    selected["left_eef_action"],
                    selected["right_eef_action"],
                    selected["left_hand_action"],
                    selected["right_hand_action"],
                ]
            )
        )
        for key in VIDEO_TOPIC_KEYS:
            videos[key].append(selected[key])
        kept_timestamps.append(float(timestamp - start))

    if not states:
        raise ValueError(f"No frames produced for {bag_dir}. Inspect RGB topic timestamps.")

    return AlignedEpisode(
        state=np.stack(states).astype(np.float32),
        action=np.stack(actions).astype(np.float32),
        videos=videos,
        timestamps=np.asarray(kept_timestamps, dtype=np.float32),
        max_skew=max_seen_skew,
        camera_frame_count=camera_frame_count,
        source=str(bag_dir),
        hand_feature_names=hand_feature_names,
        video_shapes={key: tuple(videos[key][0].shape) for key in VIDEO_TOPIC_KEYS},
    )


def _prepare_video_frame(
    frame_rgb: np.ndarray,
    image_size: tuple[int, int] | None,
) -> np.ndarray:
    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if image_size is None:
        return frame
    width, height = image_size
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def _write_video(
    video_path: Path,
    frames_rgb: list[np.ndarray],
    fps: float,
    image_size: tuple[int, int] | None,
) -> tuple[int, int, int]:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_rgb:
        raise ValueError(f"No frames to write for {video_path}")
    first_frame = _prepare_video_frame(frames_rgb[0], image_size)
    height, width = first_frame.shape[:2]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable not found; it is required for video writing") from exc

    assert process.stdin is not None
    try:
        for index, frame_rgb in enumerate(frames_rgb):
            frame_rgb = first_frame if index == 0 else _prepare_video_frame(frame_rgb, image_size)
            if frame_rgb.shape[:2] != (height, width):
                raise ValueError(
                    f"Video frames for {video_path} have inconsistent shapes: "
                    f"first={(height, width)}, frame_{index}={frame_rgb.shape[:2]}. "
                    "Pass --image-width and --image-height to resize during conversion."
                )
            process.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())
        process.stdin.close()
    except BrokenPipeError as exc:
        assert process.stderr is not None
        stderr = process.stderr.read()
        process.wait()
        error = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed while writing video '{video_path}': {error}") from exc

    assert process.stderr is not None
    stderr = process.stderr.read()
    process.wait()
    if process.returncode != 0:
        error = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed while writing video '{video_path}': {error}")
    return height, width, 3


def _estimate_fps(timestamps: np.ndarray, fallback: float = 30.0) -> float:
    if len(timestamps) < 2:
        return fallback
    deltas = np.diff(timestamps.astype(np.float64))
    deltas = deltas[deltas > 1e-6]
    if len(deltas) == 0:
        return fallback
    return float(1.0 / np.median(deltas))


def _rows_to_dataframe(
    episode: AlignedEpisode,
    episode_index: int,
    task_index: int,
    global_start_index: int,
) -> pd.DataFrame:
    length = len(episode.state)
    return pd.DataFrame(
        {
            "observation.state": list(episode.state),
            "action": list(episode.action),
            "timestamp": episode.timestamps,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(global_start_index, global_start_index + length, dtype=np.int64),
            "task_index": np.full(length, task_index, dtype=np.int64),
            "annotation.human.action.task_description": np.full(length, task_index, dtype=np.int64),
        }
    )


def _modality_ranges(keys: list[str], dims: list[int]) -> dict[str, dict[str, int]]:
    ranges = {}
    start = 0
    for key, dim in zip(keys, dims):
        ranges[key] = {"start": start, "end": start + dim}
        start += dim
    return ranges


def _feature_names(
    rotation_format: str,
    hand_feature_names: dict[str, list[str]],
) -> tuple[list[str], list[int]]:
    if rotation_format == "rotvec":
        eef_suffixes = ["x", "y", "z", "rotvec_x", "rotvec_y", "rotvec_z"]
        eef_dim = 6
    elif rotation_format == "rot6d":
        eef_suffixes = [
            "x",
            "y",
            "z",
            "rot6d_r0c0",
            "rot6d_r0c1",
            "rot6d_r0c2",
            "rot6d_r1c0",
            "rot6d_r1c1",
            "rot6d_r1c2",
        ]
        eef_dim = 9
    else:
        raise ValueError(rotation_format)

    names = []
    for prefix in ["left_eef", "right_eef"]:
        names.extend([f"{prefix}.{suffix}" for suffix in eef_suffixes])
    left_hand_names = hand_feature_names["left_hand_joints"]
    right_hand_names = hand_feature_names["right_hand_joints"]
    names.extend(left_hand_names)
    names.extend(right_hand_names)
    return names, [eef_dim, eef_dim, len(left_hand_names), len(right_hand_names)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_metadata(
    output_dir: Path,
    episodes: list[dict[str, Any]],
    task: str,
    total_frames: int,
    fps: float,
    chunks_size: int,
    rotation_format: str,
    video_shapes: dict[str, tuple[int, int, int]],
    hand_feature_names: dict[str, list[str]],
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    tasks = [{"task_index": 0, "task": task}]
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _write_jsonl(meta_dir / "tasks.jsonl", tasks)

    names, dims = _feature_names(rotation_format, hand_feature_names)
    state_dim = sum(dims)
    video_modality = {
        modality_key: {"original_key": f"observation.images.{modality_key}"}
        for modality_key in VIDEO_MODALITY_KEYS.values()
    }
    modality = {
        "state": _modality_ranges(STATE_KEYS, dims),
        "action": _modality_ranges(STATE_KEYS, dims),
        "video": video_modality,
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }
    with (meta_dir / "modality.json").open("w", encoding="utf-8") as f:
        json.dump(modality, f, indent=2)

    info = {
        "codebase_version": "v2.0",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "fps": fps,
        "chunks_size": chunks_size,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes) * len(VIDEO_MODALITY_KEYS),
        "total_chunks": max(1, (len(episodes) + chunks_size - 1) // chunks_size),
        "splits": {"train": f"0:{len(episodes)}"},
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": names,
            },
            "action": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": names,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
        "robot_type": f"WUJI_ASTRIBOT_EEF_HAND_{rotation_format.upper()}",
    }
    for topic_key, modality_key in VIDEO_MODALITY_KEYS.items():
        height, width, channels = video_shapes[topic_key]
        info["features"][f"observation.images.{modality_key}"] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    with (meta_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _find_bag_dirs(input_root: Path) -> list[Path]:
    if (input_root / "metadata.yaml").is_file():
        return [input_root]
    return sorted(path.parent for path in input_root.glob("*/metadata.yaml"))


def _parse_topic_overrides(values: list[str] | None) -> dict[str, str]:
    topics = dict(DEFAULT_TOPICS)
    if not values:
        return topics
    for item in values:
        if "=" not in item:
            raise ValueError(f"Topic override must be KEY=/topic, got {item!r}")
        key, topic = item.split("=", 1)
        if key not in topics:
            raise ValueError(f"Unknown topic key {key!r}; valid keys: {sorted(topics)}")
        topics[key] = topic
    return topics


def _validate_metadata_topics(bag_dir: Path, topics: dict[str, str]) -> None:
    metadata = _load_metadata(bag_dir)
    counts = _topic_counts(metadata)
    missing = [topic for topic in topics.values() if topic not in counts]
    empty = [topic for topic in topics.values() if counts.get(topic, 0) == 0]
    if missing:
        raise ValueError(f"{bag_dir} metadata is missing required topics: {missing}")
    if empty:
        raise ValueError(f"{bag_dir} has required topics with zero messages: {empty}")


def _episode_data_dir(output_dir: Path, episode_index: int, chunks_size: int) -> Path:
    episode_chunk = episode_index // chunks_size
    return output_dir / "data" / f"chunk-{episode_chunk:03d}"


def _episode_video_base_dir(output_dir: Path, episode_index: int, chunks_size: int) -> Path:
    episode_chunk = episode_index // chunks_size
    return output_dir / "videos" / f"chunk-{episode_chunk:03d}"


def _parallel_episode_work_dir(work_dir: Path | None, episode_index: int) -> Path | None:
    if work_dir is None:
        return None
    return work_dir / f"episode-{episode_index:06d}"


def _write_episode_outputs(task: EpisodeConversionTask) -> EpisodeConversionResult:
    episode = _align_episode(
        bag_dir=task.bag_dir,
        rotation_format=task.rotation_format,
        max_time_skew=task.max_time_skew,
        topics=task.topics,
        work_dir=task.work_dir,
        bag_backend=task.bag_backend,
        timestamp_source=task.timestamp_source,
    )
    episode_fps = task.output_fps or _estimate_fps(episode.timestamps)

    data_dir = _episode_data_dir(task.output_dir, task.episode_index, task.chunks_size)
    video_base_dir = _episode_video_base_dir(task.output_dir, task.episode_index, task.chunks_size)
    data_dir.mkdir(parents=True, exist_ok=True)

    df = _rows_to_dataframe(
        episode=episode,
        episode_index=task.episode_index,
        task_index=0,
        global_start_index=task.global_start_index,
    )
    parquet_path = data_dir / f"episode_{task.episode_index:06d}.parquet"
    df.to_parquet(parquet_path, index=False)

    for topic_key, modality_key in VIDEO_MODALITY_KEYS.items():
        video_path = (
            video_base_dir
            / f"observation.images.{modality_key}"
            / f"episode_{task.episode_index:06d}.mp4"
        )
        written_shape = _write_video(
            video_path,
            episode.videos[topic_key],
            episode_fps,
            task.image_size,
        )
        episode.video_shapes[topic_key] = written_shape

    return EpisodeConversionResult(
        episode_index=task.episode_index,
        length=len(df),
        source=episode.source,
        fps=episode_fps,
        max_skew=episode.max_skew,
        camera_frame_count=episode.camera_frame_count,
        hand_feature_names=episode.hand_feature_names,
        video_shapes=episode.video_shapes,
        parquet_path=parquet_path,
        written_global_start_index=task.global_start_index,
    )


def _rewrite_parquet_global_index(
    parquet_path: Path,
    global_start_index: int,
    length: int,
) -> None:
    df = pd.read_parquet(parquet_path)
    df["index"] = np.arange(global_start_index, global_start_index + length, dtype=np.int64)
    df.to_parquet(parquet_path, index=False)


def _episode_metadata(
    result: EpisodeConversionResult,
    task_description: str,
    timestamp_source: TimestampSource,
    max_time_skew: float,
    topics: dict[str, str],
) -> dict[str, Any]:
    return {
        "episode_index": result.episode_index,
        "tasks": [task_description],
        "length": result.length,
        "source_file": result.source,
        "alignment": {
            "anchor": "head_rgb",
            "timestamp_source": timestamp_source,
            "output_fps": result.fps,
            "camera_frame_count": result.camera_frame_count,
            "max_time_skew_sec": result.max_skew,
            "max_time_skew_warning_threshold_sec": max_time_skew,
            "video_topics": {key: topics[key] for key in VIDEO_TOPIC_KEYS},
        },
    }


def _print_episode_summary(
    result: EpisodeConversionResult, bag_name: str, max_time_skew: float
) -> None:
    warning = ""
    if result.max_skew > max_time_skew:
        warning = f", warning: max_skew>{max_time_skew:.4f}s"
    print(
        f"[{result.episode_index:04d}] {bag_name}: {result.length} frames, "
        f"head-anchored, output_fps={result.fps:.3f}, "
        f"max_skew={result.max_skew:.4f}s{warning}"
    )


def _validate_common_episode_metadata(
    result: EpisodeConversionResult,
    hand_feature_names: dict[str, list[str]] | None,
    video_shapes: dict[str, tuple[int, int, int]] | None,
) -> tuple[dict[str, list[str]], dict[str, tuple[int, int, int]]]:
    if hand_feature_names is None:
        hand_feature_names = result.hand_feature_names
    elif hand_feature_names != result.hand_feature_names:
        raise ValueError(
            "All episodes must use the same raw hand joint names and dimensions for one dataset."
        )

    if video_shapes is None:
        video_shapes = result.video_shapes
    elif video_shapes != result.video_shapes:
        raise ValueError(
            "All episodes must use the same video shapes for one dataset. "
            "Pass --image-width and --image-height to resize during conversion."
        )
    return hand_feature_names, video_shapes


def _make_episode_task(
    *,
    episode_index: int,
    bag_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
    topics: dict[str, str],
    work_dir: Path | None,
    image_size: tuple[int, int] | None,
    global_start_index: int,
) -> EpisodeConversionTask:
    return EpisodeConversionTask(
        episode_index=episode_index,
        bag_dir=bag_dir,
        output_dir=output_dir,
        rotation_format=args.eef_rotation_format,
        max_time_skew=args.max_time_skew,
        topics=topics,
        work_dir=work_dir,
        bag_backend=args.bag_backend,
        timestamp_source=args.timestamp_source,
        output_fps=args.output_fps,
        image_size=image_size,
        chunks_size=args.chunks_size,
        global_start_index=global_start_index,
    )


def _convert_episodes_sequential(
    bag_dirs: list[Path],
    args: argparse.Namespace,
    output_dir: Path,
    topics: dict[str, str],
    work_dir: Path | None,
    image_size: tuple[int, int] | None,
) -> tuple[
    list[dict[str, Any]],
    list[float],
    int,
    dict[str, list[str]] | None,
    dict[str, tuple[int, int, int]] | None,
]:
    global_frame_index = 0
    episodes_meta = []
    episode_fps_values = []
    hand_feature_names = None
    video_shapes = None

    for episode_index, bag_dir in enumerate(bag_dirs):
        task = _make_episode_task(
            episode_index=episode_index,
            bag_dir=bag_dir,
            args=args,
            output_dir=output_dir,
            topics=topics,
            work_dir=work_dir,
            image_size=image_size,
            global_start_index=global_frame_index,
        )
        result = _write_episode_outputs(task)
        hand_feature_names, video_shapes = _validate_common_episode_metadata(
            result,
            hand_feature_names,
            video_shapes,
        )
        episodes_meta.append(
            _episode_metadata(
                result,
                args.task_description,
                args.timestamp_source,
                args.max_time_skew,
                topics,
            )
        )
        episode_fps_values.append(result.fps)
        _print_episode_summary(result, bag_dir.name, args.max_time_skew)
        global_frame_index += result.length

    return episodes_meta, episode_fps_values, global_frame_index, hand_feature_names, video_shapes


def _convert_episodes_parallel(
    bag_dirs: list[Path],
    args: argparse.Namespace,
    output_dir: Path,
    topics: dict[str, str],
    work_dir: Path | None,
    image_size: tuple[int, int] | None,
) -> tuple[
    list[dict[str, Any]],
    list[float],
    int,
    dict[str, list[str]] | None,
    dict[str, tuple[int, int, int]] | None,
]:
    tasks = [
        _make_episode_task(
            episode_index=episode_index,
            bag_dir=bag_dir,
            args=args,
            output_dir=output_dir,
            topics=topics,
            work_dir=_parallel_episode_work_dir(work_dir, episode_index),
            image_size=image_size,
            global_start_index=0,
        )
        for episode_index, bag_dir in enumerate(bag_dirs)
    ]

    pending_results: dict[int, EpisodeConversionResult] = {}
    next_to_finalize = 0
    global_frame_index = 0
    episodes_meta = []
    episode_fps_values = []
    hand_feature_names = None
    video_shapes = None

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_write_episode_outputs, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            pending_results[task.episode_index] = future.result()

            while next_to_finalize in pending_results:
                result = pending_results.pop(next_to_finalize)
                if result.written_global_start_index != global_frame_index:
                    _rewrite_parquet_global_index(
                        result.parquet_path,
                        global_frame_index,
                        result.length,
                    )
                hand_feature_names, video_shapes = _validate_common_episode_metadata(
                    result,
                    hand_feature_names,
                    video_shapes,
                )
                episodes_meta.append(
                    _episode_metadata(
                        result,
                        args.task_description,
                        args.timestamp_source,
                        args.max_time_skew,
                        topics,
                    )
                )
                episode_fps_values.append(result.fps)
                _print_episode_summary(
                    result, bag_dirs[result.episode_index].name, args.max_time_skew
                )
                global_frame_index += result.length
                next_to_finalize += 1

    return episodes_meta, episode_fps_values, global_frame_index, hand_feature_names, video_shapes


def convert(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else None
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)

    bag_dirs = _find_bag_dirs(input_root)
    if args.max_episodes is not None:
        bag_dirs = bag_dirs[: args.max_episodes]
    if not bag_dirs:
        raise FileNotFoundError(f"No ROS2 bags found under {input_root}")

    topics = _parse_topic_overrides(args.topic)
    for bag_dir in bag_dirs:
        _validate_metadata_topics(bag_dir, topics)

    if args.dry_run:
        for bag_dir in bag_dirs:
            counts = _topic_counts(_load_metadata(bag_dir))
            print(bag_dir)
            for key, topic in topics.items():
                print(f"  {key}: {topic} ({counts.get(topic, 0)} messages)")
        return

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    if args.num_workers == 1:
        (
            episodes_meta,
            episode_fps_values,
            global_frame_index,
            hand_feature_names,
            video_shapes,
        ) = _convert_episodes_sequential(
            bag_dirs=bag_dirs,
            args=args,
            output_dir=output_dir,
            topics=topics,
            work_dir=work_dir,
            image_size=image_size,
        )
    else:
        print(f"Converting {len(bag_dirs)} episodes with {args.num_workers} worker processes")
        (
            episodes_meta,
            episode_fps_values,
            global_frame_index,
            hand_feature_names,
            video_shapes,
        ) = _convert_episodes_parallel(
            bag_dirs=bag_dirs,
            args=args,
            output_dir=output_dir,
            topics=topics,
            work_dir=work_dir,
            image_size=image_size,
        )

    metadata_fps = args.output_fps or float(np.median(np.asarray(episode_fps_values)))
    _write_metadata(
        output_dir=output_dir,
        episodes=episodes_meta,
        task=args.task_description,
        total_frames=global_frame_index,
        fps=metadata_fps,
        chunks_size=args.chunks_size,
        rotation_format=args.eef_rotation_format,
        video_shapes=video_shapes or {},
        hand_feature_names=hand_feature_names
        or {
            "left_hand_joints": [],
            "right_hand_joints": [],
        },
    )
    print(f"Wrote {len(episodes_meta)} episodes / {global_frame_index} frames to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default="/data_all/share/datasets/teleop/data_example",
        help="A ROS2 bag directory or a directory containing bag subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_preprocess/output/wuji_gr00t_eef_hand",
        help="Output GR00T LeRobot dataset directory.",
    )
    parser.add_argument(
        "--eef-rotation-format",
        choices=["rotvec", "rot6d"],
        default="rotvec",
        help="EEF pose layout in observation.state/action.",
    )
    parser.add_argument(
        "--fps",
        "--output-fps",
        dest="output_fps",
        type=float,
        default=None,
        help=(
            "MP4/metadata FPS. If omitted, estimate it from the head camera timestamps. "
            "The converter does not resample to this FPS."
        ),
    )
    parser.add_argument(
        "--max-time-skew",
        type=float,
        default=0.06,
        help=(
            "Warning threshold for nearest-neighbor alignment skew. "
            "Frames are not dropped; the final frame count is the minimum RGB camera count."
        ),
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Optional output video width. If omitted, keep original camera resolution.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Optional output video height. If omitted, keep original camera resolution.",
    )
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of episode conversion worker processes. The default 1 keeps the "
            "previous sequential behavior; use >1 to process multiple bags in parallel."
        ),
    )
    parser.add_argument(
        "--task-description",
        default="",
        help="Task text written to meta/tasks.jsonl and episodes.jsonl. Defaults to empty.",
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=None,
        help=(
            "Override a topic mapping as KEY=/topic. Valid keys are "
            f"{', '.join(sorted(DEFAULT_TOPICS))}."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional writable directory for temporary decompressed .db3 files.",
    )
    parser.add_argument(
        "--bag-backend",
        choices=["rosbags", "ros2", "auto"],
        default="rosbags",
        help=(
            "Rosbag reader backend. `rosbags` is pure Python and does not require ROS. "
            "`ros2` uses rosbag2_py/rclpy. `auto` tries rosbags first, then ros2."
        ),
    )
    parser.add_argument(
        "--timestamp-source",
        choices=["header", "rosbag"],
        default="header",
        help=(
            "Timestamp source for alignment. `header` uses nonzero message header.stamp "
            "and falls back to the rosbag record timestamp; `rosbag` always uses the "
            "rosbag record timestamp."
        ),
    )
    parser.add_argument(
        "--allow-prefixless-joint-names",
        action="store_true",
        help="Deprecated no-op. Hand JointState positions are now written in raw order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate bag directories and required topic counts; do not deserialize messages.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
