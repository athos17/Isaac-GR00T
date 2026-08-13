from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from robot_data_pipeline.io.rosbag2 import build_typestore
from robot_data_pipeline.models import RobotProfile, StreamConfig
from rosbags.rosbag2 import Writer


def _header(types, timestamp_ns: int):
    stamp = types["builtin_interfaces/msg/Time"](
        sec=timestamp_ns // 1_000_000_000,
        nanosec=timestamp_ns % 1_000_000_000,
    )
    return types["std_msgs/msg/Header"](stamp=stamp, frame_id="synthetic")


def _joint_message(types, stream: StreamConfig, timestamp_ns: int, value: float):
    header = _header(types, timestamp_ns)
    values = np.full(len(stream.names), value, dtype=np.float64)
    empty = np.empty(0, dtype=np.float64)
    if stream.message_type == "sensor_msgs/msg/JointState":
        return types[stream.message_type](
            header=header,
            name=list(stream.names),
            position=values,
            velocity=empty,
            effort=empty,
        )
    names = [] if stream.allow_unnamed else list(stream.names)
    return types[stream.message_type](
        header=header,
        mode=0,
        name=names,
        position=values,
        velocity=empty,
        acceleration=empty,
        torque=empty,
    )


def _pose_message(types, stream: StreamConfig, timestamp_ns: int, value: float):
    header = _header(types, timestamp_ns)
    vector = types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.0)
    return types[stream.message_type](
        header=header,
        pose=types["geometry_msgs/msg/Pose"](
            position=types["geometry_msgs/msg/Point"](x=0.2 + value * 0.1, y=0.0, z=0.8),
            orientation=types["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        twist=types["geometry_msgs/msg/Twist"](linear=vector, angular=vector),
        wrench=types["geometry_msgs/msg/Wrench"](force=vector, torque=vector),
    )


def write_synthetic_bag(
    bag_path: Path,
    profile: RobotProfile,
    *,
    duration_sec: float = 1.6,
    epoch_ns: int = 1_700_000_000_000_000_000,
    fault: Literal[
        "missing_topic",
        "zero_header",
        "duplicate_header",
        "corrupt_jpeg",
        "active_gap",
        "stationary",
    ]
    | None = None,
) -> Path:
    typestore = build_typestore()
    types = typestore.types
    ok, encoded = cv2.imencode(".jpg", np.zeros((16, 24, 3), dtype=np.uint8))
    assert ok
    image_bytes = np.asarray(encoded, dtype=np.uint8)
    events = []
    with Writer(bag_path, version=9) as writer:
        connections = {
            key: writer.add_connection(stream.topic, stream.message_type, typestore=typestore)
            for key, stream in profile.streams.items()
        }
        for key, stream in profile.streams.items():
            count = int(duration_sec * stream.expected_hz) + 1
            for index in range(count):
                relative_ns = round(index / stream.expected_hz * 1e9)
                nominal_header_ns = epoch_ns + relative_ns
                if fault == "missing_topic" and key == "video.right_wrist":
                    continue
                if (
                    fault == "active_gap"
                    and key == "state.left_arm_joint"
                    and 650_000_000 <= relative_ns <= 850_000_000
                ):
                    continue
                header_ns = nominal_header_ns
                if fault == "zero_header" and key == "state.left_hand_joint" and index == 1:
                    header_ns = 0
                if fault == "duplicate_header" and key == "state.left_hand_joint" and index == 2:
                    header_ns = epoch_ns + round((index - 1) / stream.expected_hz * 1e9)
                value = (
                    0.0 if fault == "stationary" else float(np.sin(2 * np.pi * relative_ns * 1e-9))
                )
                if stream.semantic == "rgb_image":
                    data = image_bytes
                    if fault == "corrupt_jpeg" and key == "video.head" and index == 5:
                        data = np.frombuffer(b"not-a-jpeg", dtype=np.uint8)
                    message = types[stream.message_type](
                        header=_header(types, header_ns),
                        format="jpeg",
                        data=data,
                    )
                elif stream.semantic.startswith("joint_position"):
                    message = _joint_message(types, stream, header_ns, value)
                else:
                    message = _pose_message(types, stream, header_ns, value)
                events.append(
                    (
                        nominal_header_ns + 1_000_000,
                        connections[key],
                        typestore.serialize_cdr(message, stream.message_type),
                    )
                )
        for bag_timestamp, connection, payload in sorted(events, key=lambda item: item[0]):
            writer.write(connection, bag_timestamp, payload)
    return bag_path
