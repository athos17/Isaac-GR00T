#!/usr/bin/env python
"""Motion detection module for identifying valid movement in robot episodes.

Detects the start and end of meaningful robot motion by analyzing:
- End-effector (EEF) velocity in xyz space
- Hand joint command/state changes
- Action-state differences
- Combined sliding window statistics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotionDetectionConfig:
    """Configuration for motion detection parameters."""

    velocity_threshold: float = 0.01  # m/s, combined EEF velocity threshold
    hand_velocity_threshold: float = 0.05  # rad/s, hand joint velocity threshold
    action_state_diff_threshold: float = 0.02  # Threshold for action-state mismatch
    use_action_state_diff: bool = True  # Include action-state mismatch in the motion signal
    window_duration_sec: float = 0.5  # Duration of sliding window for smoothing
    min_motion_frames: int = 30  # Minimum frames to consider valid motion
    fps: float = 30.0  # Expected FPS for velocity computation


@dataclass
class MotionDetectionResult:
    """Result of motion detection on an episode."""

    motion_start_index: int  # Frame index where motion starts
    motion_end_index: int  # Frame index where motion ends
    idle_prefix_frames: int  # Number of frames trimmed from start
    idle_suffix_frames: int  # Number of frames trimmed from end
    mean_eef_velocity: float  # Mean EEF velocity during motion
    max_eef_velocity: float  # Max EEF velocity during motion
    mean_action_state_diff: float  # Mean action-state difference during motion


def _compute_eef_velocity(eef_xyz: np.ndarray, fps: float) -> np.ndarray:
    """Compute velocity from position trajectory.

    Args:
        eef_xyz: Shape (T, 3) position trajectory
        fps: Frames per second

    Returns:
        Shape (T-1,) velocity magnitudes
    """
    diff = np.diff(eef_xyz, axis=0)  # (T-1, 3)
    velocity = np.linalg.norm(diff, axis=1) * fps  # (T-1,)
    return velocity


def _compute_hand_velocity(hand_joints: np.ndarray, fps: float) -> np.ndarray:
    """Compute hand joint velocity.

    Args:
        hand_joints: Shape (T, N) joint positions
        fps: Frames per second

    Returns:
        Shape (T-1,) max joint velocity across all joints
    """
    diff = np.diff(hand_joints, axis=0)  # (T-1, N)
    velocity = np.abs(diff) * fps  # (T-1, N)
    max_velocity = np.max(velocity, axis=1)  # (T-1,) max across joints
    return max_velocity


def _sliding_window_mean(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Apply sliding window mean smoothing.

    Args:
        signal: Shape (T,) input signal
        window_size: Window size in frames

    Returns:
        Shape (T - window_size + 1,) smoothed signal
    """
    if len(signal) < window_size:
        return np.array([np.mean(signal)])
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode="valid")


def detect_motion_window(
    state: np.ndarray,
    action: np.ndarray,
    config: MotionDetectionConfig,
    eef_dim: int = 9,  # For rot6d: xyz (3) + rot6d (6)
    state_layout: str = "eef",
    joint_arm_dims: tuple[int, int] | None = None,
) -> MotionDetectionResult:
    """Detect the window of meaningful motion in an episode.

    Args:
        state: Shape (T, state_dim) state trajectory
            Layout: [left_eef (9), right_eef (9), left_hand_joints (N), right_hand_joints (M)]
        action: Shape (T, action_dim) action trajectory (same layout as state)
        config: Motion detection configuration
        eef_dim: Dimension of each EEF state (9 for rot6d, 6 for rotvec)
        state_layout: `eef` for [left_eef, right_eef, hands] or `joint` for joint_space
        joint_arm_dims: Optional left/right arm dimensions for joint layout with trailing hands.

    Returns:
        MotionDetectionResult with detected motion window and statistics
    """
    T = len(state)
    if config.velocity_threshold <= 0:
        raise ValueError("velocity_threshold must be > 0")
    if config.hand_velocity_threshold <= 0:
        raise ValueError("hand_velocity_threshold must be > 0")
    if config.use_action_state_diff and config.action_state_diff_threshold <= 0:
        raise ValueError("action_state_diff_threshold must be > 0 when enabled")

    window_frames = max(1, int(config.window_duration_sec * config.fps))

    if state_layout == "eef":
        # Extract left and right EEF xyz (first 3 dimensions of each EEF)
        left_eef_xyz = state[:, :3]  # (T, 3)
        right_eef_xyz = state[:, eef_dim : eef_dim + 3]  # (T, 3)

        # Compute EEF velocities
        left_vel = _compute_eef_velocity(left_eef_xyz, config.fps)  # (T-1,)
        right_vel = _compute_eef_velocity(right_eef_xyz, config.fps)  # (T-1,)
        combined_eef_vel = left_vel + right_vel  # (T-1,)

        # Extract hand joints (everything after the two EEFs)
        hand_start_idx = 2 * eef_dim
        joint_state = state[:, hand_start_idx:]  # Includes both hands
    elif state_layout == "joint":
        if joint_arm_dims is None:
            arm_state = state
            joint_state = state
        else:
            arm_dim = sum(joint_arm_dims)
            arm_state = state[:, :arm_dim]
            joint_state = state[:, arm_dim:]
            if joint_state.shape[1] == 0:
                joint_state = arm_state
        combined_eef_vel = _compute_hand_velocity(arm_state, config.fps)
    else:
        raise ValueError(f"Unsupported state_layout: {state_layout}")

    hand_vel = _compute_hand_velocity(joint_state, config.fps)  # (T-1,)

    # Compute action-state difference only when action and state share a layout.
    if action.shape == state.shape:
        action_state_diff = np.linalg.norm(action - state, axis=1)  # (T,)
    else:
        action_state_diff = np.zeros(T, dtype=np.float32)
    action_state_diff_vel = action_state_diff[:-1]  # Align with velocity (T-1,)

    # Combine enabled motion signals. Absolute desired actions can maintain a non-zero tracking
    # error while the robot is stationary, so callers may disable action-state mismatch and rely
    # only on observed EEF/hand velocities for idle trimming.
    motion_signals = [
        combined_eef_vel / config.velocity_threshold,
        hand_vel / config.hand_velocity_threshold,
    ]
    if config.use_action_state_diff:
        motion_signals.append(action_state_diff_vel / config.action_state_diff_threshold)
    motion_signal = np.maximum.reduce(motion_signals)  # (T-1,)

    # Apply sliding window smoothing
    if len(motion_signal) >= window_frames:
        smooth_signal = _sliding_window_mean(motion_signal, window_frames)
    else:
        smooth_signal = np.array([np.mean(motion_signal)])

    # Find motion start: first window where signal > 1.0
    motion_indices = np.where(smooth_signal > 1.0)[0]

    if len(motion_indices) == 0:
        # No motion detected, keep entire episode
        return MotionDetectionResult(
            motion_start_index=0,
            motion_end_index=T,
            idle_prefix_frames=0,
            idle_suffix_frames=0,
            mean_eef_velocity=float(np.mean(combined_eef_vel)),
            max_eef_velocity=float(np.max(combined_eef_vel)),
            mean_action_state_diff=float(np.mean(action_state_diff)),
        )

    motion_start = motion_indices[0]
    # motion_signal describes transitions between state frames and therefore has length T-1.
    # Include the state frame reached by the last transition; the returned end is exclusive.
    motion_end = motion_indices[-1] + window_frames + 1

    # Ensure minimum motion duration
    if motion_end - motion_start < config.min_motion_frames:
        # Motion too short, keep entire episode
        return MotionDetectionResult(
            motion_start_index=0,
            motion_end_index=T,
            idle_prefix_frames=0,
            idle_suffix_frames=0,
            mean_eef_velocity=float(np.mean(combined_eef_vel)),
            max_eef_velocity=float(np.max(combined_eef_vel)),
            mean_action_state_diff=float(np.mean(action_state_diff)),
        )

    # Compute statistics for the detected motion window
    motion_end = min(motion_end, T)
    motion_eef_vel = combined_eef_vel[motion_start : motion_end - 1]
    motion_action_state_diff = action_state_diff[motion_start:motion_end]

    return MotionDetectionResult(
        motion_start_index=motion_start,
        motion_end_index=motion_end,
        idle_prefix_frames=motion_start,
        idle_suffix_frames=max(0, T - motion_end),
        mean_eef_velocity=float(np.mean(motion_eef_vel)) if len(motion_eef_vel) > 0 else 0.0,
        max_eef_velocity=float(np.max(motion_eef_vel)) if len(motion_eef_vel) > 0 else 0.0,
        mean_action_state_diff=(
            float(np.mean(motion_action_state_diff)) if len(motion_action_state_diff) > 0 else 0.0
        ),
    )


def trim_episode_to_motion(
    state: np.ndarray,
    action: np.ndarray,
    videos: dict[str, list[np.ndarray]],
    timestamps: np.ndarray,
    motion_result: MotionDetectionResult,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[np.ndarray]], np.ndarray]:
    """Trim episode arrays to the detected motion window.

    Args:
        state: Shape (T, state_dim)
        action: Shape (T, action_dim)
        videos: Dict of video key -> list of frames
        timestamps: Shape (T,)
        motion_result: Motion detection result with start/end indices

    Returns:
        Tuple of (trimmed_state, trimmed_action, trimmed_videos, trimmed_timestamps)
    """
    start = motion_result.motion_start_index
    end = motion_result.motion_end_index

    trimmed_state = state[start:end]
    trimmed_action = action[start:end]
    trimmed_timestamps = timestamps[start:end] - timestamps[start]  # Reset to start at 0
    trimmed_videos = {key: frames[start:end] for key, frames in videos.items()}

    return trimmed_state, trimmed_action, trimmed_videos, trimmed_timestamps
