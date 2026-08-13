from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from typing import Any, Iterable

import cv2
import numpy as np

from robot_data_pipeline.adapters import AdapterError, get_adapter
from robot_data_pipeline.io.rosbag2 import RosbagReadError, RosbagsReader
from robot_data_pipeline.models import (
    BagReader,
    EpisodeAudit,
    EpisodeSpec,
    ProcessingRoster,
    RobotProfile,
    StreamAudit,
)
from robot_data_pipeline.quality.decisions import (
    CAMERA_DECODE_FAILURE,
    CAMERA_SCHEMA_MISMATCH,
    JOINT_SCHEMA_MISMATCH,
    MESSAGE_TYPE_MISMATCH,
    MISSING_REQUIRED_TOPIC,
    NON_FINITE_PAYLOAD,
    NON_MONOTONIC_HEADER_TIMESTAMP,
    RAW_GAP_EXCEEDED,
    ROSBAG_READ_FAILURE,
    ZERO_HEADER_TIMESTAMP,
)


def _percentiles(
    values: list[float], names: tuple[tuple[str, float], ...]
) -> dict[str, float | None]:
    if not values:
        return {name: None for name, _ in names}
    array = np.asarray(values, dtype=np.float64)
    return {name: float(np.percentile(array, percentile)) for name, percentile in names}


def _add_reason(report: StreamAudit, reason: str, **detail: Any) -> None:
    if reason not in report.reject_reasons:
        report.reject_reasons.append(reason)
    if detail and len(report.details) < 20:
        report.details.append({"reason": reason, **detail})


def _decode_image(encoded: bytes) -> tuple[int, int, int] | None:
    array = np.frombuffer(encoded, dtype=np.uint8)
    try:
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if image is None or image.ndim != 3:
        return None
    return tuple(int(value) for value in image.shape)


def audit_episode(
    episode: EpisodeSpec, profile: RobotProfile, reader: BagReader | None = None
) -> EpisodeAudit:
    reader = reader or RosbagsReader()
    reports = {
        key: StreamAudit(
            stream_key=key,
            topic=stream.topic,
            message_type=stream.message_type,
            expected_hz=stream.expected_hz,
        )
        for key, stream in profile.streams.items()
    }
    header_times: dict[str, list[int]] = defaultdict(list)
    offsets: dict[str, list[float]] = defaultdict(list)
    initial_joint_schema: dict[str, tuple[str, ...]] = {}
    previous_image_digest: dict[str, bytes] = {}
    image_digests: dict[str, set[bytes]] = defaultdict(set)
    consecutive_frozen: dict[str, int] = defaultdict(int)

    for key, stream in profile.streams.items():
        metadata_topic = episode.metadata.topics.get(stream.topic)
        if metadata_topic is None or metadata_topic[1] == 0:
            if stream.required:
                _add_reason(reports[key], MISSING_REQUIRED_TOPIC, topic=stream.topic)
            continue
        if metadata_topic[0] != stream.message_type:
            _add_reason(
                reports[key],
                MESSAGE_TYPE_MISMATCH,
                expected=stream.message_type,
                actual=metadata_topic[0],
            )

    try:
        messages = reader.messages(episode, profile.streams)
        for raw in messages:
            stream = profile.streams[raw.stream_key]
            report = reports[raw.stream_key]
            report.message_count += 1
            if raw.message_type != stream.message_type:
                _add_reason(
                    report,
                    MESSAGE_TYPE_MISMATCH,
                    sequence=raw.sequence,
                    expected=stream.message_type,
                    actual=raw.message_type,
                )
            if raw.header_time_ns == 0:
                report.zero_header_count += 1
                if profile.clock.require_nonzero:
                    _add_reason(report, ZERO_HEADER_TIMESTAMP, sequence=raw.sequence)
            times = header_times[raw.stream_key]
            if times:
                if raw.header_time_ns == times[-1]:
                    report.duplicate_timestamp_count += 1
                    _add_reason(
                        report,
                        NON_MONOTONIC_HEADER_TIMESTAMP,
                        sequence=raw.sequence,
                        kind="duplicate",
                        timestamp_ns=raw.header_time_ns,
                    )
                elif raw.header_time_ns < times[-1]:
                    report.backward_timestamp_count += 1
                    _add_reason(
                        report,
                        NON_MONOTONIC_HEADER_TIMESTAMP,
                        sequence=raw.sequence,
                        kind="backward",
                        previous_ns=times[-1],
                        timestamp_ns=raw.header_time_ns,
                    )
            times.append(raw.header_time_ns)
            offsets[raw.stream_key].append((raw.bag_time_ns - raw.header_time_ns) * 1e-9)
            try:
                payload = get_adapter(stream.adapter).adapt(raw.message, stream)
            except AdapterError as exc:
                reason = (
                    CAMERA_DECODE_FAILURE
                    if stream.semantic == "rgb_image"
                    else JOINT_SCHEMA_MISMATCH
                )
                if not stream.semantic.startswith("joint_position"):
                    reason = (
                        CAMERA_DECODE_FAILURE
                        if stream.semantic == "rgb_image"
                        else NON_FINITE_PAYLOAD
                    )
                if reason == CAMERA_DECODE_FAILURE:
                    report.camera_decode_failure_count += 1
                else:
                    report.schema_mismatch_count += 1
                _add_reason(report, reason, sequence=raw.sequence, error=str(exc))
                continue
            if payload.values and not all(math.isfinite(value) for value in payload.values):
                report.non_finite_payload_count += 1
                _add_reason(report, NON_FINITE_PAYLOAD, sequence=raw.sequence)
            if stream.semantic.startswith("joint_position"):
                initial = initial_joint_schema.setdefault(raw.stream_key, payload.names)
                if payload.names != initial:
                    report.schema_mismatch_count += 1
                    _add_reason(
                        report,
                        JOINT_SCHEMA_MISMATCH,
                        sequence=raw.sequence,
                        expected=list(initial),
                        actual=list(payload.names),
                    )
            if payload.encoded_image is not None:
                if report.image_format is None:
                    report.image_format = payload.image_format
                elif payload.image_format != report.image_format:
                    report.schema_mismatch_count += 1
                    _add_reason(
                        report,
                        CAMERA_SCHEMA_MISMATCH,
                        sequence=raw.sequence,
                        expected_format=report.image_format,
                        actual_format=payload.image_format,
                    )
                shape = _decode_image(payload.encoded_image)
                if shape is None:
                    report.camera_decode_failure_count += 1
                    _add_reason(report, CAMERA_DECODE_FAILURE, sequence=raw.sequence)
                    continue
                if report.image_shape is None:
                    report.image_shape = shape
                elif shape != report.image_shape:
                    report.schema_mismatch_count += 1
                    _add_reason(
                        report,
                        CAMERA_SCHEMA_MISMATCH,
                        sequence=raw.sequence,
                        expected=report.image_shape,
                        actual=shape,
                    )
                digest = hashlib.blake2b(payload.encoded_image, digest_size=16).digest()
                if digest in image_digests[raw.stream_key]:
                    report.duplicate_image_payload_count += 1
                image_digests[raw.stream_key].add(digest)
                if previous_image_digest.get(raw.stream_key) == digest:
                    consecutive_frozen[raw.stream_key] += 1
                    report.max_consecutive_frozen_frames = max(
                        report.max_consecutive_frozen_frames,
                        consecutive_frozen[raw.stream_key] + 1,
                    )
                else:
                    consecutive_frozen[raw.stream_key] = 0
                previous_image_digest[raw.stream_key] = digest
    except RosbagReadError as exc:
        first = next(iter(reports.values()))
        _add_reason(first, ROSBAG_READ_FAILURE, error=str(exc))

    for key, stream in profile.streams.items():
        report = reports[key]
        times = header_times[key]
        if report.message_count == 0 and stream.required:
            _add_reason(report, MISSING_REQUIRED_TOPIC, topic=stream.topic)
        if times:
            report.header_start_ns = times[0]
            report.header_end_ns = times[-1]
        positive_intervals = [
            (current - previous) * 1e-9
            for previous, current in zip(times, times[1:])
            if current > previous
        ]
        interval_metrics = _percentiles(
            positive_intervals, (("median", 50), ("p95", 95), ("max", 100))
        )
        report.interval_sec = interval_metrics
        median_interval = interval_metrics["median"]
        if median_interval is not None and median_interval > 0:
            report.frequency_hz = 1.0 / median_interval
        expected_period = 1.0 / stream.expected_hz
        report.estimated_dropped_messages = sum(
            max(0, round(interval / expected_period) - 1) for interval in positive_intervals
        )
        if stream.max_gap_sec is not None:
            threshold_ns = round(stream.max_gap_sec * 1e9)
            gap_indices = [
                index
                for index, (previous, current) in enumerate(zip(times, times[1:]))
                if current > previous and current - previous > threshold_ns
            ]
            report.large_gap_count = len(gap_indices)
            if gap_indices:
                largest_index = max(gap_indices, key=lambda index: times[index + 1] - times[index])
                report.largest_gap_start_ns = times[largest_index]
                report.largest_gap_end_ns = times[largest_index + 1]
                report.warning_reasons.append(RAW_GAP_EXCEEDED)
                if len(report.details) < 20:
                    report.details.append(
                        {
                            "reason": RAW_GAP_EXCEEDED,
                            "status": "deferred_until_activity_detection",
                            "start_ns": report.largest_gap_start_ns,
                            "end_ns": report.largest_gap_end_ns,
                            "max_interval_sec": interval_metrics["max"],
                            "threshold_sec": stream.max_gap_sec,
                        }
                    )
        offset_values = offsets[key]
        report.bag_header_offset_sec = _percentiles(
            offset_values, (("p01", 1), ("p50", 50), ("p99", 99))
        )
        if len(offset_values) > 1:
            report.offset_drift_sec = float(offset_values[-1] - offset_values[0])

    reject_reasons = tuple(
        sorted({reason for report in reports.values() for reason in report.reject_reasons})
    )
    return EpisodeAudit(
        roster_index=episode.roster_index,
        task_id=episode.task_id,
        bag_path=str(episode.bag_path),
        status="REJECT" if reject_reasons else "PASS",
        reject_reasons=reject_reasons,
        streams=reports,
    )


def audit_roster(
    roster: ProcessingRoster,
    profile: RobotProfile,
    *,
    reader: BagReader | None = None,
    episode_indices: Iterable[int] | None = None,
    num_workers: int = 1,
) -> list[EpisodeAudit]:
    selected = set(episode_indices) if episode_indices is not None else None
    episodes = [
        episode
        for episode in roster.episodes
        if selected is None or episode.roster_index in selected
    ]
    if reader is not None or num_workers == 1:
        shared_reader = reader or RosbagsReader()
        return [audit_episode(episode, profile, shared_reader) for episode in episodes]
    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="rosbag-audit") as executor:
        return list(executor.map(lambda episode: audit_episode(episode, profile), episodes))
