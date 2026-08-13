from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from robot_data_pipeline.models import BagReader, EpisodeSpec, RawMessage, StreamConfig


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
    "astribot_msgs/msg/RobotJointState": """\
std_msgs/Header header
int8 mode
string[] name
float64[] position
float64[] velocity
float64[] acceleration
float64[] torque
""",
}


class RosbagReadError(RuntimeError):
    pass


def build_typestore():
    try:
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as exc:
        raise RosbagReadError(
            "reading ROS bags requires the 'data-pipeline' extra (rosbags)"
        ) from exc
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    custom_types: dict[str, Any] = {}
    for message_type, definition in CUSTOM_MSG_DEFINITIONS.items():
        custom_types.update(get_types_from_msg(definition, message_type))
    typestore.register(custom_types)
    return typestore


def header_timestamp_ns(message: Any) -> int:
    try:
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RosbagReadError(f"message does not contain a valid header.stamp: {exc}") from exc


class RosbagsReader(BagReader):
    def __init__(self) -> None:
        self._typestore = build_typestore()

    def messages(
        self, episode: EpisodeSpec, streams: dict[str, StreamConfig]
    ) -> Iterator[RawMessage]:
        try:
            from rosbags.highlevel import AnyReader
        except ImportError as exc:
            raise RosbagReadError(
                "reading ROS bags requires the 'data-pipeline' extra (rosbags)"
            ) from exc

        topics = {stream.topic: stream for stream in streams.values()}
        bag_path = Path(episode.bag_path)
        sequence_by_topic = {topic: 0 for topic in topics}
        try:
            with AnyReader([bag_path], default_typestore=self._typestore) as reader:
                connections = [
                    connection for connection in reader.connections if connection.topic in topics
                ]
                for connection, bag_time_ns, rawdata in reader.messages(connections=connections):
                    stream = topics[connection.topic]
                    sequence = sequence_by_topic[connection.topic]
                    sequence_by_topic[connection.topic] += 1
                    try:
                        message = self._typestore.deserialize_cdr(rawdata, connection.msgtype)
                        header_time_ns = header_timestamp_ns(message)
                    except Exception as exc:
                        raise RosbagReadError(
                            f"failed to decode {connection.topic} message {sequence}: {exc}"
                        ) from exc
                    yield RawMessage(
                        stream_key=stream.key,
                        topic=connection.topic,
                        message_type=connection.msgtype,
                        header_time_ns=header_time_ns,
                        bag_time_ns=int(bag_time_ns),
                        sequence=sequence,
                        message=message,
                    )
        except RosbagReadError:
            raise
        except Exception as exc:
            raise RosbagReadError(f"failed to read rosbag {bag_path}: {exc}") from exc
