import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "examples" / "wuji_rot6d" / "run_gr00t_client.py"
)
SPEC = importlib.util.spec_from_file_location("wuji_run_gr00t_client", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
wuji_client = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wuji_client
SPEC.loader.exec_module(wuji_client)

LatencyTracker = wuji_client.LatencyTracker
build_rtc_options = wuji_client.build_rtc_options
discard_action_prefix = wuji_client.discard_action_prefix
latency_seconds_to_delay_steps = wuji_client.latency_seconds_to_delay_steps
slice_leftover_actions = wuji_client.slice_leftover_actions


ACTION_KEYS = ("left_eef", "right_eef", "left_hand_joints", "right_hand_joints")


def _make_actions(horizon: int = 6) -> dict[str, np.ndarray]:
    return {
        key: np.arange(horizon, dtype=np.float32).reshape(1, horizon, 1) + index * 100.0
        for index, key in enumerate(ACTION_KEYS)
    }


def test_latency_seconds_to_delay_steps_ceilings_control_periods():
    assert latency_seconds_to_delay_steps(0.0, control_hz=30.0) == 0
    assert latency_seconds_to_delay_steps(1.0 / 30.0, control_hz=30.0) == 1
    assert latency_seconds_to_delay_steps(0.034, control_hz=30.0) == 2


def test_latency_tracker_returns_percentile_delay_steps():
    tracker = LatencyTracker(window_size=4, percentile=95.0)
    for latency_sec in [0.010, 0.020, 0.030, 0.100, 0.200]:
        tracker.add(latency_sec)

    assert tracker.delay_steps(control_hz=30.0) == 6


def test_slice_leftover_actions_preserves_all_action_keys():
    actions = _make_actions(horizon=6)

    leftover = slice_leftover_actions(actions, start_step=2)

    assert set(leftover) == set(ACTION_KEYS)
    for key in ACTION_KEYS:
        np.testing.assert_array_equal(leftover[key], actions[key][:, 2:, :])


def test_discard_action_prefix_drops_real_delay_steps():
    actions = _make_actions(horizon=6)

    replacement = discard_action_prefix(actions, delay_steps=2)

    for key in ACTION_KEYS:
        np.testing.assert_array_equal(replacement[key], actions[key][:, 2:, :])


def test_discard_action_prefix_rejects_empty_queue():
    actions = _make_actions(horizon=3)

    with pytest.raises(ValueError, match="exhausts"):
        discard_action_prefix(actions, delay_steps=3)


def test_build_rtc_options_uses_leftover_and_delay_defaults():
    leftover = slice_leftover_actions(_make_actions(horizon=8), start_step=4)

    options = build_rtc_options(
        prev_chunk_left_over=leftover,
        action_horizon=8,
        overlap_steps=None,
        frozen_steps=None,
        estimated_delay_steps=2,
        execute_horizon=4,
        ramp_rate=2.5,
    )

    assert options == {
        "rtc": {
            "enabled": True,
            "prev_chunk_left_over": leftover,
            "action_horizon": 8,
            "rtc_overlap_steps": 4,
            "rtc_frozen_steps": 2,
            "rtc_ramp_rate": 2.5,
        }
    }
