from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from robot_data_pipeline.adapters import get_adapter
from robot_data_pipeline.models import (
    CanonicalEpisode,
    ImageSeries,
    JointPositionSeries,
    PoseSeries,
    PositionCommandSeries,
    RawMessage,
    RobotProfile,
)
from robot_data_pipeline.processing.rotations import make_quaternion_signs_continuous
from robot_data_pipeline.quality.decisions import QUATERNION_NORM_INVALID


class CanonicalizationError(ValueError):
    def __init__(
        self, reason: str, message: str, *, details: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details or {}


def _reorder(
    values: tuple[float, ...], names: tuple[str, ...], target: tuple[str, ...]
) -> np.ndarray:
    if not target:
        return np.asarray(values, dtype=np.float64)
    indices = {name: index for index, name in enumerate(names)}
    try:
        return np.asarray([values[indices[name]] for name in target], dtype=np.float64)
    except KeyError as exc:
        raise CanonicalizationError(
            "joint_schema_mismatch", f"missing configured joint: {exc.args[0]}"
        ) from exc


def canonicalize_messages(
    messages: Iterable[RawMessage], profile: RobotProfile, *, include_images: bool = True
) -> CanonicalEpisode:
    grouped: dict[str, list[RawMessage]] = defaultdict(list)
    for message in messages:
        grouped[message.stream_key].append(message)
    result = {}
    for key, raw_messages in grouped.items():
        stream = profile.streams[key]
        timestamps = np.asarray(
            [message.header_time_ns for message in raw_messages], dtype=np.int64
        )
        if np.any(np.diff(timestamps) <= 0):
            raise CanonicalizationError(
                "non_monotonic_header_timestamp",
                f"{key} timestamps are not strictly increasing",
                details={"stream": key},
            )
        adapter = get_adapter(stream.adapter)
        payloads = [adapter.adapt(message.message, stream) for message in raw_messages]
        if stream.semantic == "rgb_image":
            if not include_images:
                continue
            result[key] = ImageSeries(
                timestamps_ns=timestamps,
                bag_timestamps_ns=np.asarray(
                    [message.bag_time_ns for message in raw_messages], dtype=np.int64
                ),
                encoded_images=tuple(payload.encoded_image or b"" for payload in payloads),
                formats=tuple(payload.image_format or "" for payload in payloads),
            )
            continue
        if stream.semantic.startswith("joint_position"):
            values = np.stack(
                [_reorder(payload.values, payload.names, stream.names) for payload in payloads]
            )
            for name in stream.continuous_joints:
                values[:, stream.names.index(name)] = np.unwrap(values[:, stream.names.index(name)])
            series_class = (
                PositionCommandSeries if key.startswith("action.") else JointPositionSeries
            )
            result[key] = series_class(timestamps, values, stream.names)
            continue
        if stream.semantic.startswith("eef_pose"):
            values = np.asarray([payload.values for payload in payloads], dtype=np.float64)
            quaternion_norms = np.linalg.norm(values[:, 3:], axis=1)
            invalid = np.flatnonzero((quaternion_norms < 0.5) | (quaternion_norms > 1.5))
            if len(invalid):
                first = int(invalid[0])
                raise CanonicalizationError(
                    QUATERNION_NORM_INVALID,
                    f"quaternion norm outside [0.5, 1.5] for {key}: {quaternion_norms[first]:.6f}",
                    details={
                        "stream": key,
                        "timestamp_ns": int(timestamps[first]),
                        "norm": float(quaternion_norms[first]),
                        "minimum_norm": 0.5,
                        "maximum_norm": 1.5,
                    },
                )
            result[key] = PoseSeries(
                timestamps_ns=timestamps,
                translations=values[:, :3],
                quaternions_xyzw=make_quaternion_signs_continuous(values[:, 3:]),
            )
            continue
        raise CanonicalizationError(
            "unsupported_stream_semantic", f"unsupported stream semantic: {stream.semantic}"
        )
    return CanonicalEpisode(streams=result)
