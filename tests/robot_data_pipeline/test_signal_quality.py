from dataclasses import replace
from pathlib import Path

import numpy as np
from robot_data_pipeline.config import load_robot_profile
from robot_data_pipeline.models import ActivityInterval, CanonicalEpisode, JointPositionSeries
from robot_data_pipeline.quality.signal import audit_active_interval_gaps


REPO_ROOT = Path(__file__).parents[2]
PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml"


def test_only_gaps_overlapping_activity_are_hard_violations() -> None:
    profile = load_robot_profile(PROFILE)
    key = "state.left_hand_joint"
    streams = {
        name: replace(stream, required=name == key) for name, stream in profile.streams.items()
    }
    profile = replace(profile, streams=streams)
    names = profile.streams[key].names
    timestamps = np.array([0, 200_000_000, 210_000_000, 400_000_000, 410_000_000])
    episode = CanonicalEpisode(
        {key: JointPositionSeries(timestamps, np.zeros((len(timestamps), len(names))), names)}
    )

    outside = audit_active_interval_gaps(
        episode, profile, ActivityInterval(201_000_000, 209_000_000, 0, 410_000_000)
    )
    inside = audit_active_interval_gaps(
        episode, profile, ActivityInterval(205_000_000, 405_000_000, 0, 410_000_000)
    )

    assert outside == []
    assert len(inside) == 1
    assert inside[0]["stream"] == key
