from dataclasses import replace
from pathlib import Path

import numpy as np
from robot_data_pipeline.config import load_robot_profile
from robot_data_pipeline.models import CanonicalEpisode, JointPositionSeries
from robot_data_pipeline.processing.activity import detect_activity


REPO_ROOT = Path(__file__).parents[2]
PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml"


def _episode(hz: int) -> CanonicalEpisode:
    times = np.arange(0, 5, 1 / hz)
    values = np.zeros((len(times), 1))
    moving = (times >= 2.0) & (times <= 3.0)
    values[moving, 0] = (times[moving] - 2.0) * 2.0
    values[times > 3.0, 0] = 2.0
    return CanonicalEpisode(
        {"state.test": JointPositionSeries(np.round(times * 1e9).astype(np.int64), values, ("j",))}
    )


def test_activity_crop_uses_seconds_at_different_input_rates() -> None:
    base = load_robot_profile(PROFILE)
    config = replace(
        base.activity_detection,
        groups=("state.test",),
        joint_velocity_threshold=0.5,
        window_sec=0.2,
    )
    profile = replace(base, activity_detection=config)

    slow = detect_activity(_episode(50), profile, padding_before_sec=0.5, padding_after_sec=0.5)
    fast = detect_activity(_episode(200), profile, padding_before_sec=0.5, padding_after_sec=0.5)

    assert slow is not None and fast is not None
    assert abs(slow.active_start_ns - fast.active_start_ns) < 30_000_000
    assert abs(slow.active_end_ns - fast.active_end_ns) < 30_000_000
    assert slow.padded_start_ns == slow.active_start_ns - 500_000_000


def test_stationary_episode_has_no_activity() -> None:
    base = load_robot_profile(PROFILE)
    config = replace(base.activity_detection, groups=("state.test",))
    profile = replace(base, activity_detection=config)
    times = np.arange(100, dtype=np.int64) * 10_000_000
    episode = CanonicalEpisode(
        {"state.test": JointPositionSeries(times, np.zeros((100, 1)), ("j",))}
    )

    assert detect_activity(episode, profile, padding_before_sec=0.5, padding_after_sec=0.5) is None
