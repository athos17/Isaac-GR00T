from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from robot_data_pipeline.adapters import AdapterError
from robot_data_pipeline.config import load_robot_profile
from robot_data_pipeline.models import JointPositionSeries, RawMessage
from robot_data_pipeline.processing.canonicalize import CanonicalizationError, canonicalize_messages
from robot_data_pipeline.quality.decisions import QUATERNION_NORM_INVALID


REPO_ROOT = Path(__file__).parents[2]
PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml"


def test_named_joints_are_reordered_to_profile_order() -> None:
    profile = load_robot_profile(PROFILE)
    original = profile.streams["state.left_hand_joint"]
    stream = replace(original, names=("joint_a", "joint_b"))
    profile = replace(profile, streams={stream.key: stream})
    message = SimpleNamespace(name=("joint_b", "joint_a"), position=(2.0, 1.0))
    raw = [
        RawMessage(stream.key, stream.topic, stream.message_type, 1, 2, 0, message),
        RawMessage(stream.key, stream.topic, stream.message_type, 2, 3, 1, message),
    ]

    episode = canonicalize_messages(raw, profile)

    series = episode.streams[stream.key]
    assert isinstance(series, JointPositionSeries)
    assert np.array_equal(series.values, [[1.0, 2.0], [1.0, 2.0]])


@pytest.mark.parametrize(
    ("names", "match"),
    [
        (("joint_a", "joint_a"), "duplicate names"),
        (("joint_a", "joint_c"), "missing=\\['joint_b'\\]"),
    ],
)
def test_joint_schema_rejects_duplicate_and_missing_names(names, match) -> None:
    profile = load_robot_profile(PROFILE)
    original = profile.streams["state.left_hand_joint"]
    stream = replace(original, names=("joint_a", "joint_b"))
    profile = replace(profile, streams={stream.key: stream})
    message = SimpleNamespace(name=names, position=(1.0, 2.0))
    raw = [RawMessage(stream.key, stream.topic, stream.message_type, 1, 2, 0, message)]

    with pytest.raises(AdapterError, match=match):
        canonicalize_messages(raw, profile)


def test_invalid_quaternion_norm_reports_stream_and_timestamp() -> None:
    profile = load_robot_profile(PROFILE)
    stream = profile.streams["state.left_eef"]
    profile = replace(profile, streams={stream.key: stream})
    position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)
    message = SimpleNamespace(pose=SimpleNamespace(position=position, orientation=orientation))
    raw = [RawMessage(stream.key, stream.topic, stream.message_type, 123, 124, 0, message)]

    with pytest.raises(CanonicalizationError) as error:
        canonicalize_messages(raw, profile)

    assert error.value.reason == QUATERNION_NORM_INVALID
    assert error.value.details == {
        "stream": stream.key,
        "timestamp_ns": 123,
        "norm": 0.0,
        "minimum_norm": 0.5,
        "maximum_norm": 1.5,
    }
