from __future__ import annotations

from robot_data_pipeline.adapters.base import MessageAdapter


class AdapterError(ValueError):
    pass


def _registry() -> dict[str, MessageAdapter]:
    # Imports are local to avoid a module cycle while adapter modules import AdapterError.
    from robot_data_pipeline.adapters.astribot_msgs import (
        RobotCartesianPoseAdapter,
        RobotJointStatePositionAdapter,
    )
    from robot_data_pipeline.adapters.sensor_msgs import (
        CompressedImageAdapter,
        JointStatePositionAdapter,
    )

    return {
        "sensor_msgs.compressed_image": CompressedImageAdapter(),
        "sensor_msgs.joint_state_position": JointStatePositionAdapter(),
        "astribot_msgs.robot_joint_state_position": RobotJointStatePositionAdapter(),
        "astribot_msgs.robot_cartesian_pose": RobotCartesianPoseAdapter(),
    }


_ADAPTERS: dict[str, MessageAdapter] | None = None


def get_adapter(name: str) -> MessageAdapter:
    global _ADAPTERS
    if _ADAPTERS is None:
        _ADAPTERS = _registry()
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise AdapterError(f"unknown adapter: {name!r}") from exc


def adapter_names() -> tuple[str, ...]:
    global _ADAPTERS
    if _ADAPTERS is None:
        _ADAPTERS = _registry()
    return tuple(sorted(_ADAPTERS))
