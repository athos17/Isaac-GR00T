from __future__ import annotations

from typing import Any

from robot_data_pipeline.adapters.base import MessageAdapter
from robot_data_pipeline.adapters.registry import AdapterError
from robot_data_pipeline.models import AdaptedPayload, StreamConfig


class RobotJointStatePositionAdapter(MessageAdapter):
    def adapt(self, message: Any, stream: StreamConfig) -> AdaptedPayload:
        try:
            values = tuple(float(value) for value in message.position)
            names = tuple(str(value) for value in message.name)
        except (AttributeError, TypeError, ValueError) as exc:
            raise AdapterError(f"invalid RobotJointState message: {exc}") from exc
        if not values:
            raise AdapterError("joint position payload is empty")
        if not names and stream.allow_unnamed:
            if len(values) != len(stream.names):
                raise AdapterError("unnamed joint count differs from configured names")
            names = stream.names
        if len(names) != len(values):
            raise AdapterError("joint name and position lengths differ")
        if len(set(names)) != len(names):
            raise AdapterError("joint message contains duplicate names")
        if stream.names:
            missing = sorted(set(stream.names) - set(names))
            extra = sorted(set(names) - set(stream.names))
            if missing or extra:
                raise AdapterError(f"joint names differ; missing={missing}, extra={extra}")
        return AdaptedPayload(values=values, names=names)


class RobotCartesianPoseAdapter(MessageAdapter):
    def adapt(self, message: Any, stream: StreamConfig) -> AdaptedPayload:
        try:
            pose = message.pose
            values = (
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise AdapterError(f"invalid RobotCartesianState message: {exc}") from exc
        return AdaptedPayload(values=values)
