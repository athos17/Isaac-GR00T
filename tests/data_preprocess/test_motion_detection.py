from data_preprocess.motion_detection import MotionDetectionConfig, detect_motion_window
import numpy as np


def _eef_trajectory(left_x: list[float]) -> np.ndarray:
    state = np.zeros((len(left_x), 16), dtype=np.float32)
    state[:, 0] = left_x
    return state


def test_motion_window_includes_frame_reached_by_final_transition():
    state = _eef_trajectory([0.0, 0.0, 1.0, 2.0])
    result = detect_motion_window(
        state,
        state.copy(),
        MotionDetectionConfig(
            velocity_threshold=0.1,
            hand_velocity_threshold=0.1,
            use_action_state_diff=False,
            window_duration_sec=1.0,
            min_motion_frames=1,
            fps=1.0,
        ),
        eef_dim=6,
    )

    assert result.motion_start_index == 1
    assert result.motion_end_index == len(state)
    assert result.idle_suffix_frames == 0


def test_absolute_action_offset_can_be_excluded_from_motion_signal():
    state = _eef_trajectory([0.0, 0.0, 1.0, 2.0, 2.0])
    action = state + 0.5
    base_config = dict(
        velocity_threshold=0.1,
        hand_velocity_threshold=0.1,
        action_state_diff_threshold=0.1,
        window_duration_sec=1.0,
        min_motion_frames=1,
        fps=1.0,
    )

    with_action_diff = detect_motion_window(
        state,
        action,
        MotionDetectionConfig(**base_config, use_action_state_diff=True),
        eef_dim=6,
    )
    without_action_diff = detect_motion_window(
        state,
        action,
        MotionDetectionConfig(**base_config, use_action_state_diff=False),
        eef_dim=6,
    )

    assert with_action_diff.motion_start_index == 0
    assert with_action_diff.motion_end_index == len(state)
    assert without_action_diff.motion_start_index == 1
    assert without_action_diff.motion_end_index == 4
    assert without_action_diff.idle_prefix_frames == 1
    assert without_action_diff.idle_suffix_frames == 1
