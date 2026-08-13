import warnings

import numpy as np
from robot_data_pipeline.quality.lag import audit_action_state_lag


def test_known_positive_response_lag_is_estimated() -> None:
    sample_hz = 200
    times = np.arange(0, 8, 1 / sample_hz)
    change_index = np.floor(times / 0.2).astype(int)
    rng = np.random.default_rng(7)
    levels = rng.normal(size=change_index.max() + 1)
    action = levels[change_index]
    lag_sec = 0.08
    delayed_index = np.floor(np.maximum(0, times - lag_sec) / 0.2).astype(int)
    state = levels[delayed_index]

    result = audit_action_state_lag(
        np.round(times * 1e9).astype(np.int64),
        action[:, None],
        np.round(times * 1e9).astype(np.int64),
        state[:, None],
        qa_hz=sample_hz,
    )

    assert result.active_axis_count == 1
    assert result.consensus_lag_sec is not None
    assert abs(result.consensus_lag_sec - lag_sec) <= 1 / sample_hz
    windowed = result.axes[0].windowed_lag_sec
    assert windowed["trend_sec_per_sec"] is not None
    assert windowed["max_step_sec"] is not None


def test_stationary_axis_is_insufficient_excitation() -> None:
    times = np.arange(100, dtype=np.int64) * 10_000_000
    values = np.ones((100, 1))

    result = audit_action_state_lag(times, values, times, values, qa_hz=100)

    assert result.active_axis_count == 0
    assert result.axes[0].status == "insufficient_excitation"
    assert result.axes[0].windowed_lag_sec["trend_sec_per_sec"] is None
    assert result.axes[0].windowed_lag_sec["max_step_sec"] is None


def test_insufficient_overlap_is_reported_without_rejecting() -> None:
    action_times = np.array([0], dtype=np.int64)
    state_times = np.array([1_000_000_000, 1_010_000_000], dtype=np.int64)
    action = np.zeros((1, 2))
    state = np.zeros((2, 2))

    result = audit_action_state_lag(action_times, action, state_times, state)

    assert result.active_axis_count == 0
    assert result.consensus_lag_sec is None
    assert [axis.status for axis in result.axes] == [
        "insufficient_overlap",
        "insufficient_overlap",
    ]


def test_windowed_lag_skips_constant_shifted_slices_without_warning() -> None:
    sample_hz = 20
    times = np.arange(0, 4, 1 / sample_hz)
    action = np.concatenate((np.sin(2 * np.pi * times[:40]), np.zeros(40)))
    state = np.concatenate((np.zeros(40), np.sin(2 * np.pi * times[:40])))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = audit_action_state_lag(
            np.round(times * 1e9).astype(np.int64),
            action[:, None],
            np.round(times * 1e9).astype(np.int64),
            state[:, None],
            qa_hz=sample_hz,
            max_lag_sec=1.0,
            window_sec=2.0,
        )

    assert not caught
    assert result.active_axis_count == 1
