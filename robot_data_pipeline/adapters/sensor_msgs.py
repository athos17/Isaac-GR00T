from __future__ import annotations

from typing import Any

from robot_data_pipeline.adapters.base import MessageAdapter
from robot_data_pipeline.adapters.registry import AdapterError
from robot_data_pipeline.models import AdaptedPayload, StreamConfig


class JointStatePositionAdapter(MessageAdapter):
    def adapt(self, message: Any, stream: StreamConfig) -> AdaptedPayload:
        try:
            values = tuple(float(value) for value in message.position)
            names = tuple(str(value) for value in message.name)
        except (AttributeError, TypeError, ValueError) as exc:
            raise AdapterError(f"invalid joint position message: {exc}") from exc
        if not values:
            raise AdapterError("joint position payload is empty")
        if names and len(names) != len(values):
            raise AdapterError("joint name and position lengths differ")
        if stream.names:
            if not names and stream.allow_unnamed:
                if len(values) != len(stream.names):
                    raise AdapterError("unnamed joint count differs from configured names")
                names = stream.names
            elif not names:
                raise AdapterError("joint names are required by the profile")
            if len(set(names)) != len(names):
                raise AdapterError("joint message contains duplicate names")
            missing = sorted(set(stream.names) - set(names))
            extra = sorted(set(names) - set(stream.names))
            if missing or extra:
                raise AdapterError(f"joint names differ; missing={missing}, extra={extra}")
        return AdaptedPayload(values=values, names=names)


class CompressedImageAdapter(MessageAdapter):
    def adapt(self, message: Any, stream: StreamConfig) -> AdaptedPayload:
        try:
            data = bytes(message.data)
            image_format = str(message.format)
        except (AttributeError, TypeError, ValueError) as exc:
            raise AdapterError(f"invalid compressed image message: {exc}") from exc
        if not data:
            raise AdapterError("compressed image payload is empty")
        return AdaptedPayload(encoded_image=data, image_format=image_format)
