from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from robot_data_pipeline.config import load_robot_profile
from robot_data_pipeline.models import BagMetadata, BagReader, EpisodeSpec, RawMessage
from robot_data_pipeline.quality.decisions import (
    CAMERA_DECODE_FAILURE,
    CAMERA_SCHEMA_MISMATCH,
    NON_FINITE_PAYLOAD,
    NON_MONOTONIC_HEADER_TIMESTAMP,
    RAW_GAP_EXCEEDED,
    ZERO_HEADER_TIMESTAMP,
)
from robot_data_pipeline.quality.raw import audit_episode


REPO_ROOT = Path(__file__).parents[2]
PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml"


class FakeReader(BagReader):
    def __init__(self, messages: list[RawMessage]) -> None:
        self._messages = messages

    def messages(self, episode, streams):
        yield from self._messages


def _episode(profile, keys: list[str]) -> EpisodeSpec:
    topics = {profile.streams[key].topic: (profile.streams[key].message_type, 1) for key in keys}
    return EpisodeSpec(
        roster_index=0,
        task_index=0,
        task_id="task",
        instruction="Do the task",
        root=Path("/input"),
        bag_path=Path("/input/bag"),
        metadata_hash="hash",
        metadata=BagMetadata("sqlite3", ("data.db3",), 0, 1, len(keys), topics),
    )


def _raw(
    stream, timestamp: int, sequence: int, message, *, bag_time_ns: int | None = None
) -> RawMessage:
    return RawMessage(
        stream_key=stream.key,
        topic=stream.topic,
        message_type=stream.message_type,
        header_time_ns=timestamp,
        bag_time_ns=timestamp + 1_000 if bag_time_ns is None else bag_time_ns,
        sequence=sequence,
        message=message,
    )


def _only_required(profile, key: str):
    streams = {
        name: replace(stream, required=name == key) for name, stream in profile.streams.items()
    }
    return replace(profile, streams=streams)


def test_zero_duplicate_and_non_finite_are_stable_reject_reasons() -> None:
    profile = load_robot_profile(PROFILE)
    key = "state.left_hand_joint"
    profile = _only_required(profile, key)
    stream = profile.streams[key]
    names = stream.names
    values = [0.0] * len(names)
    message = SimpleNamespace(name=names, position=values)
    bad_message = SimpleNamespace(name=names, position=[float("nan"), *values[1:]])
    messages = [
        _raw(stream, 0, 0, message),
        _raw(stream, 10_000_000, 1, message),
        _raw(stream, 10_000_000, 2, bad_message),
    ]

    report = audit_episode(_episode(profile, [key]), profile, FakeReader(messages))

    assert report.status == "REJECT"
    assert ZERO_HEADER_TIMESTAMP in report.reject_reasons
    assert NON_MONOTONIC_HEADER_TIMESTAMP in report.reject_reasons
    assert NON_FINITE_PAYLOAD in report.reject_reasons


def test_corrupt_compressed_image_is_rejected() -> None:
    profile = load_robot_profile(PROFILE)
    key = "video.head"
    profile = _only_required(profile, key)
    stream = profile.streams[key]
    message = SimpleNamespace(data=b"not-an-image", format="jpeg")

    report = audit_episode(
        _episode(profile, [key]), profile, FakeReader([_raw(stream, 1, 0, message)])
    )

    assert CAMERA_DECODE_FAILURE in report.reject_reasons


def test_opencv_decode_exception_is_an_episode_rejection(monkeypatch) -> None:
    profile = load_robot_profile(PROFILE)
    key = "video.head"
    profile = _only_required(profile, key)
    stream = profile.streams[key]
    message = SimpleNamespace(data=b"malformed", format="jpeg")

    def fail_decode(*args, **kwargs):
        raise cv2.error("forced decode failure")

    monkeypatch.setattr(cv2, "imdecode", fail_decode)
    report = audit_episode(
        _episode(profile, [key]), profile, FakeReader([_raw(stream, 1, 0, message)])
    )

    assert report.status == "REJECT"
    assert CAMERA_DECODE_FAILURE in report.reject_reasons


def test_valid_image_reports_shape() -> None:
    profile = load_robot_profile(PROFILE)
    key = "video.head"
    profile = _only_required(profile, key)
    stream = profile.streams[key]
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 12, 3), dtype=np.uint8))
    assert ok
    message = SimpleNamespace(data=encoded.tobytes(), format="jpeg")

    report = audit_episode(
        _episode(profile, [key]), profile, FakeReader([_raw(stream, 1, 0, message)])
    )

    assert report.streams[key].image_shape == (8, 12, 3)


def test_camera_shape_format_and_frozen_payload_metrics_are_reported() -> None:
    profile = load_robot_profile(PROFILE)
    key = "video.head"
    profile = _only_required(profile, key)
    stream = profile.streams[key]
    ok, first = cv2.imencode(".jpg", np.zeros((8, 12, 3), dtype=np.uint8))
    assert ok
    ok, changed = cv2.imencode(".png", np.zeros((9, 12, 3), dtype=np.uint8))
    assert ok
    messages = [
        _raw(stream, 1_000_000_000, 0, SimpleNamespace(data=first, format="jpeg")),
        _raw(stream, 1_033_333_333, 1, SimpleNamespace(data=first, format="jpeg")),
        _raw(stream, 1_066_666_667, 2, SimpleNamespace(data=first, format="jpeg")),
        _raw(stream, 1_100_000_000, 3, SimpleNamespace(data=changed, format="png")),
    ]

    report = audit_episode(_episode(profile, [key]), profile, FakeReader(messages))
    stream_report = report.streams[key]

    assert CAMERA_SCHEMA_MISMATCH in report.reject_reasons
    assert stream_report.schema_mismatch_count == 2
    assert stream_report.duplicate_image_payload_count == 2
    assert stream_report.max_consecutive_frozen_frames == 3


def test_bag_header_offset_distribution_and_drift_are_reported() -> None:
    profile = load_robot_profile(PROFILE)
    key = "state.left_hand_joint"
    profile = _only_required(profile, key)
    stream = profile.streams[key]
    message = SimpleNamespace(name=stream.names, position=[0.0] * len(stream.names))
    first_header = 1_000_000_000
    second_header = 1_005_000_000
    messages = [
        _raw(
            stream,
            first_header,
            0,
            message,
            bag_time_ns=first_header + 1_000_000,
        ),
        _raw(
            stream,
            second_header,
            1,
            message,
            bag_time_ns=second_header + 3_000_000,
        ),
    ]

    report = audit_episode(_episode(profile, [key]), profile, FakeReader(messages))
    stream_report = report.streams[key]

    assert stream_report.bag_header_offset_sec["p50"] == pytest.approx(0.002)
    assert stream_report.offset_drift_sec == pytest.approx(0.002)


def test_raw_gap_is_reported_but_deferred_until_activity_is_known() -> None:
    profile = load_robot_profile(PROFILE)
    key = "state.left_hand_joint"
    profile = _only_required(profile, key)
    stream = profile.streams[key]
    message = SimpleNamespace(name=stream.names, position=[0.0] * len(stream.names))
    messages = [
        _raw(stream, 1_000_000_000, 0, message),
        _raw(stream, 1_200_000_000, 1, message),
    ]

    report = audit_episode(_episode(profile, [key]), profile, FakeReader(messages))

    assert report.status == "PASS"
    assert RAW_GAP_EXCEEDED not in report.reject_reasons
    assert RAW_GAP_EXCEEDED in report.streams[key].warning_reasons
    assert report.streams[key].large_gap_count == 1
