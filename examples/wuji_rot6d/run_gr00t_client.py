#!/usr/bin/env python3
"""Run a local Wuji/Astribot client against a GR00T policy server.

The policy output is expected to match ``wuji_eef_hand_rot6d_config.py``:

- left_eef/right_eef: xyz + row-major rot6d, shape (B, T, 9)
- left_hand_joints/right_hand_joints: hand joint positions, shape (B, T, 20)

examples/wuji_rot6d/run_gr00t_client_with_env.sh     --host 0.0.0.0     --port 5555     --task "grasp the rugby ball and place it into the basket"     --execute-horizon 16  --no-home     --eef-control-way direct     --arm-interp-hz 30     --hand-interp-hz 30  --no-wbc   --workspace-radius 2 2 2     --max-eef-step 1.0     --max-eef-rotation-step-deg 180     --actual-log-mode stream     --actual-stream-hz 30     --log-prefix client_model_local_30hz_stream  --save-observation-images

"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty
from typing import Any, Iterator

import numpy as np


ACTION_KEYS = ("left_eef", "right_eef", "left_hand_joints", "right_hand_joints")
VIDEO_KEYS = ("head_view", "left_wrist_view", "right_wrist_view")
LANGUAGE_KEY = "annotation.human.action.task_description"
ARM_NAMES = ("left", "right")
DEFAULT_DATASET_PATH = "/home/zxc/Desktop/GR00T/Isaac-GR00T/examples/wuji_rot6d/grasp_mango_rot6d_rosbag_ts"
DEFAULT_VIDEO_TOPICS = {
    "head_view": "/astribot_camera/head_rgbd/color_compress/compressed",
    "left_wrist_view": "/astribot_camera/left_wrist_rgbd/color_compress/compressed",
    "right_wrist_view": "/astribot_camera/right_wrist_rgbd/color_compress/compressed",
}


@dataclass(frozen=True)
class WorkspaceLimits:
    xyz_min: np.ndarray
    xyz_max: np.ndarray
    max_xyz_step: float
    max_rotation_step_rad: float = math.radians(10.0)


@dataclass(frozen=True)
class ActionStep:
    left_eef: np.ndarray
    right_eef: np.ndarray
    left_hand_joints: np.ndarray
    right_hand_joints: np.ndarray


@dataclass(frozen=True)
class ActionEvent:
    arm_step: ActionStep | None
    hand_step: ActionStep | None
    sleep_seconds: float


class LatencyTracker:
    def __init__(self, window_size: int, percentile: float) -> None:
        if window_size <= 0:
            raise ValueError("RTC latency window must be positive")
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError("RTC delay percentile must be in [0, 100]")
        self.window_size = int(window_size)
        self.percentile = float(percentile)
        self._values: list[float] = []

    def add(self, latency_sec: float) -> None:
        latency_sec = float(latency_sec)
        if latency_sec < 0.0:
            raise ValueError("Latency must be non-negative")
        self._values.append(latency_sec)
        if len(self._values) > self.window_size:
            del self._values[: len(self._values) - self.window_size]

    def latency_sec(self) -> float:
        if not self._values:
            return 0.0
        return float(np.percentile(np.asarray(self._values, dtype=np.float64), self.percentile))

    def delay_steps(self, control_hz: float) -> int:
        return latency_seconds_to_delay_steps(self.latency_sec(), control_hz=control_hz)


def latency_seconds_to_delay_steps(latency_sec: float, *, control_hz: float) -> int:
    if control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    if latency_sec < 0.0:
        raise ValueError("latency_sec must be non-negative")
    return int(math.ceil(float(latency_sec) * float(control_hz)))


def _action_horizon(actions: dict[str, np.ndarray]) -> int:
    for key in ACTION_KEYS:
        if key not in actions:
            raise KeyError(f"Policy action missing required key: {key}")
    return min(int(np.asarray(actions[key]).shape[1]) for key in ACTION_KEYS)


def slice_leftover_actions(actions: dict[str, np.ndarray], start_step: int) -> dict[str, np.ndarray]:
    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    horizon = _action_horizon(actions)
    if start_step > horizon:
        raise ValueError(f"start_step={start_step} exceeds action horizon {horizon}")
    return {
        key: np.asarray(actions[key], dtype=np.float32)[:, start_step:, :].copy()
        for key in ACTION_KEYS
    }


def discard_action_prefix(actions: dict[str, np.ndarray], delay_steps: int) -> dict[str, np.ndarray]:
    if delay_steps < 0:
        raise ValueError("delay_steps must be non-negative")
    horizon = _action_horizon(actions)
    if delay_steps >= horizon:
        raise ValueError(
            f"delay_steps={delay_steps} exhausts returned action horizon {horizon}"
        )
    return {
        key: np.asarray(actions[key], dtype=np.float32)[:, delay_steps:, :].copy()
        for key in ACTION_KEYS
    }


def build_rtc_options(
    *,
    prev_chunk_left_over: dict[str, np.ndarray],
    action_horizon: int,
    overlap_steps: int | None,
    frozen_steps: int | None,
    estimated_delay_steps: int,
    execute_horizon: int,
    ramp_rate: float,
) -> dict[str, Any]:
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if execute_horizon <= 0:
        raise ValueError("execute_horizon must be positive")
    if estimated_delay_steps < 0:
        raise ValueError("estimated_delay_steps must be non-negative")
    if ramp_rate <= 0.0:
        raise ValueError("rtc_ramp_rate must be positive")

    resolved_overlap_steps = (
        max(0, action_horizon - execute_horizon)
        if overlap_steps is None
        else int(overlap_steps)
    )
    if resolved_overlap_steps < 0:
        raise ValueError("rtc_overlap_steps must be non-negative")
    if resolved_overlap_steps > action_horizon:
        raise ValueError("rtc_overlap_steps must be <= action_horizon")

    resolved_frozen_steps = estimated_delay_steps if frozen_steps is None else int(frozen_steps)
    resolved_frozen_steps = min(max(0, resolved_frozen_steps), resolved_overlap_steps)

    return {
        "rtc": {
            "enabled": True,
            "prev_chunk_left_over": prev_chunk_left_over,
            "action_horizon": action_horizon,
            "rtc_overlap_steps": resolved_overlap_steps,
            "rtc_frozen_steps": resolved_frozen_steps,
            "rtc_ramp_rate": float(ramp_rate),
        }
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class CommandLogger:
    def __init__(
        self, log_dir: str | Path | None, *, prefix: str, run_timestamp: str | None = None
    ) -> None:
        self.enabled = bool(log_dir)
        self.prefix = prefix
        self.run_timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = None
        self._lock = threading.Lock()
        if log_dir:
            root = Path(log_dir).expanduser().resolve()
            self.log_dir = root / "client_log" / f"{prefix}_{self.run_timestamp}"
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, phase: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self.log_dir is None:
            return
        record = {
            "schema": "wuji_rot6d_command_log.v1",
            "phase": phase,
            "wall_time": time.time(),
            "monotonic_time": time.monotonic(),
            **payload,
        }
        path = self.log_dir / f"{self.prefix}_{phase}.jsonl"
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(record), separators=(",", ":")))
                f.write("\n")

    def write_config(
        self,
        args: argparse.Namespace,
        *,
        argv: list[str],
        client: str,
        extra: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.enabled or self.log_dir is None:
            return None
        payload = {
            "schema": "wuji_rot6d_run_config.v1",
            "phase": "config",
            "wall_time": time.time(),
            "monotonic_time": time.monotonic(),
            "client": client,
            "argv": list(argv),
            "args": vars(args),
            "log_prefix": self.prefix,
            "run_timestamp": self.run_timestamp,
            "log_dir": str(self.log_dir),
        }
        if extra is not None:
            payload["extra"] = extra
        path = self.log_dir / f"{self.prefix}_config.json"
        with self._lock:
            path.write_text(
                json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return path


def write_client_config(
    logger: CommandLogger,
    args: argparse.Namespace,
    *,
    argv: list[str] | None = None,
    client: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    return logger.write_config(
        args,
        argv=list(sys.argv if argv is None else argv),
        client=client or Path(sys.argv[0]).name,
        extra=extra,
    )


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        diag = np.diagonal(m)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    quat = quat / norm
    if quat[3] < 0.0:
        quat = -quat
    return quat.astype(np.float32)


def quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert SDK quaternion order xyzw to GR00T row-major rot6d."""
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return _quat_wxyz_to_matrix(quat_wxyz)[:2, :].reshape(-1).astype(np.float32)


def rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """Convert GR00T row-major rot6d to SDK quaternion order xyzw."""
    rows = np.asarray(rot6d, dtype=np.float32).reshape(2, 3)
    row0 = _normalize(rows[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    row1 = rows[1] - float(np.dot(rows[1], row0)) * row0
    row1 = _normalize(row1, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    row2 = np.cross(row0, row1).astype(np.float32)
    rotation = np.stack([row0, row1, row2], axis=0)
    return _matrix_to_quat_xyzw(rotation)


def eef9_to_astribot_pose(eef9: np.ndarray) -> np.ndarray:
    eef9 = np.asarray(eef9, dtype=np.float32)
    if eef9.shape != (9,):
        raise ValueError(f"EEF action must have shape (9,), got {eef9.shape}")
    quat_xyzw = rot6d_to_quat_xyzw(eef9[3:])
    return np.concatenate([eef9[:3], quat_xyzw]).astype(np.float32)


def astribot_pose_to_eef9(pose7: np.ndarray | list[float]) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float32)
    if pose7.shape != (7,):
        raise ValueError(f"AsFtribot pose must have shape (7,), got {pose7.shape}")
    rot6d = quat_xyzw_to_rot6d(pose7[3:])
    return np.concatenate([pose7[:3], rot6d]).astype(np.float32)


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    quat = quat / norm
    if quat[3] < 0.0:
        quat = -quat
    return quat.astype(np.float32)


def _slerp_quat_xyzw(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    start = _normalize_quat_xyzw(start)
    end = _normalize_quat_xyzw(end)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        return _normalize_quat_xyzw(start + fraction * (end - start))

    theta_0 = math.acos(dot)
    theta = theta_0 * fraction
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _normalize_quat_xyzw(s0 * start + s1 * end)


def interpolate_eef9(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    if start.shape != (9,) or end.shape != (9,):
        raise ValueError(f"EEF interpolation requires shape (9,), got {start.shape} and {end.shape}")
    fraction = float(np.clip(fraction, 0.0, 1.0))
    xyz = start[:3] + fraction * (end[:3] - start[:3])
    quat = _slerp_quat_xyzw(rot6d_to_quat_xyzw(start[3:]), rot6d_to_quat_xyzw(end[3:]), fraction)
    return np.concatenate([xyz, quat_xyzw_to_rot6d(quat)]).astype(np.float32)


def interpolate_action_step(start: ActionStep, end: ActionStep, fraction: float) -> ActionStep:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return ActionStep(
        left_eef=interpolate_eef9(start.left_eef, end.left_eef, fraction),
        right_eef=interpolate_eef9(start.right_eef, end.right_eef, fraction),
        left_hand_joints=(
            start.left_hand_joints + fraction * (end.left_hand_joints - start.left_hand_joints)
        ).astype(np.float32),
        right_hand_joints=(
            start.right_hand_joints + fraction * (end.right_hand_joints - start.right_hand_joints)
        ).astype(np.float32),
    )


def _interp_steps(duration: float, target_hz: float) -> int:
    if target_hz <= 0.0:
        return 1
    return max(1, int(math.ceil(duration * target_hz)))


def build_interpolated_action_events(
    steps: list[ActionStep],
    *,
    previous_step: ActionStep | None,
    source_hz: float,
    arm_interp_hz: float,
    hand_interp_hz: float,
) -> list[ActionEvent]:
    if source_hz <= 0.0:
        raise ValueError("source_hz must be positive")
    if arm_interp_hz < 0.0 or hand_interp_hz < 0.0:
        raise ValueError("interpolation rates must be non-negative")
    if not steps:
        return []

    events: list[ActionEvent] = []
    duration = 1.0 / source_hz

    if previous_step is None:
        events.append(ActionEvent(arm_step=steps[0], hand_step=steps[0], sleep_seconds=0.0))
        segment_pairs = zip(steps[:-1], steps[1:])
    else:
        segment_pairs = zip([previous_step, *steps[:-1]], steps)

    for start, end in segment_pairs:
        arm_count = _interp_steps(duration, arm_interp_hz)
        hand_count = _interp_steps(duration, hand_interp_hz)
        arm_fractions = {
            round(step / arm_count, 12): step / arm_count
            for step in range(1, arm_count + 1)
        }
        hand_fractions = {
            round(step / hand_count, 12): step / hand_count
            for step in range(1, hand_count + 1)
        }

        previous_fraction = 0.0
        for fraction_key in sorted(set(arm_fractions) | set(hand_fractions)):
            fraction = float(fraction_key)
            sleep_seconds = max(0.0, (fraction - previous_fraction) * duration)
            arm_step = (
                interpolate_action_step(start, end, arm_fractions[fraction_key])
                if fraction_key in arm_fractions
                else None
            )
            hand_step = (
                interpolate_action_step(start, end, hand_fractions[fraction_key])
                if fraction_key in hand_fractions
                else None
            )
            events.append(
                ActionEvent(
                    arm_step=arm_step,
                    hand_step=hand_step,
                    sleep_seconds=sleep_seconds,
                )
            )
            previous_fraction = fraction

    return events


def iter_action_steps(actions: dict[str, np.ndarray], execute_horizon: int) -> Iterator[ActionStep]:
    if execute_horizon <= 0:
        raise ValueError("--execute-horizon must be positive")
    for key in ACTION_KEYS:
        if key not in actions:
            raise KeyError(f"Policy action missing required key: {key}")

    horizon = min(int(actions["left_eef"].shape[1]), execute_horizon)
    for step in range(horizon):
        yield ActionStep(
            left_eef=np.asarray(actions["left_eef"][0, step], dtype=np.float32),
            right_eef=np.asarray(actions["right_eef"][0, step], dtype=np.float32),
            left_hand_joints=np.asarray(actions["left_hand_joints"][0, step], dtype=np.float32),
            right_hand_joints=np.asarray(actions["right_hand_joints"][0, step], dtype=np.float32),
        )


def clip_hand_joints(hand: np.ndarray) -> np.ndarray:
    hand = np.asarray(hand, dtype=np.float32)
    if hand.shape != (20,):
        raise ValueError(f"Hand command must have shape (20,), got {hand.shape}")
    lower = np.full(20, -1.57, dtype=np.float32)
    upper = np.full(20, 1.57, dtype=np.float32)
    return np.clip(hand, lower, upper).astype(np.float32)


def limit_hand_joint_step(
    target: np.ndarray, previous: np.ndarray | None, *, max_step: float
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    if target.shape != (20,):
        raise ValueError(f"Hand command must have shape (20,), got {target.shape}")
    if previous is None or max_step <= 0.0:
        return target.astype(np.float32)

    previous = np.asarray(previous, dtype=np.float32)
    if previous.shape != (20,):
        raise ValueError(f"Previous hand command must have shape (20,), got {previous.shape}")
    delta = np.clip(target - previous, -max_step, max_step)
    return (previous + delta).astype(np.float32)


def default_workspace_limits() -> dict[str, WorkspaceLimits]:
    # Conservative defaults for startup. Override on the command line after measuring robot limits.
    return {
        "left": WorkspaceLimits(
            xyz_min=np.array([0.10, -0.80, 0.50], dtype=np.float32),
            xyz_max=np.array([0.90, 0.80, 1.60], dtype=np.float32),
            max_xyz_step=0.05,
        ),
        "right": WorkspaceLimits(
            xyz_min=np.array([0.10, -0.80, 0.50], dtype=np.float32),
            xyz_max=np.array([0.90, 0.80, 1.60], dtype=np.float32),
            max_xyz_step=0.05,
        ),
    }


def workspace_limits_from_current(
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    *,
    radius: np.ndarray,
    max_xyz_step: float,
    max_rotation_step_rad: float = math.radians(10.0),
) -> dict[str, WorkspaceLimits]:
    radius = np.asarray(radius, dtype=np.float32)
    if radius.shape != (3,):
        raise ValueError(f"Workspace radius must have shape (3,), got {radius.shape}")
    left_xyz = np.asarray(left_pose, dtype=np.float32)[:3]
    right_xyz = np.asarray(right_pose, dtype=np.float32)[:3]
    return {
        "left": WorkspaceLimits(
            xyz_min=left_xyz - radius,
            xyz_max=left_xyz + radius,
            max_xyz_step=max_xyz_step,
            max_rotation_step_rad=max_rotation_step_rad,
        ),
        "right": WorkspaceLimits(
            xyz_min=right_xyz - radius,
            xyz_max=right_xyz + radius,
            max_xyz_step=max_xyz_step,
            max_rotation_step_rad=max_rotation_step_rad,
        ),
    }


def clip_eef_pose(
    target_pose: np.ndarray, previous_pose: np.ndarray | None, limits: WorkspaceLimits
) -> np.ndarray:
    pose = np.asarray(target_pose, dtype=np.float32).copy()
    if pose.shape != (7,):
        raise ValueError(f"EEF pose must have shape (7,), got {pose.shape}")

    pose[:3] = np.clip(pose[:3], limits.xyz_min, limits.xyz_max)
    if previous_pose is not None:
        previous_pose = np.asarray(previous_pose, dtype=np.float32)
        delta = np.clip(
            pose[:3] - previous_pose[:3],
            -limits.max_xyz_step,
            limits.max_xyz_step,
        )
        pose[:3] = previous_pose[:3] + delta

    pose[3:] = _normalize_quat_xyzw(pose[3:])
    if previous_pose is not None:
        previous_quat = _normalize_quat_xyzw(previous_pose[3:])
        dot = float(np.clip(abs(np.dot(previous_quat, pose[3:])), -1.0, 1.0))
        angle = 2.0 * math.acos(dot)
        if angle > limits.max_rotation_step_rad > 0.0:
            pose[3:] = _slerp_quat_xyzw(
                previous_quat,
                pose[3:],
                limits.max_rotation_step_rad / angle,
            )
    return pose.astype(np.float32)


def command_cartesian_pose(
    astribot: Any,
    arm_names: list[str],
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    *,
    control_way: str,
    use_wbc: bool,
) -> None:
    poses = [
        np.asarray(left_pose, dtype=np.float32).tolist(),
        np.asarray(right_pose, dtype=np.float32).tolist(),
    ]
    requested_arm_names = left_right_arm_names(arm_names)
    try:
        astribot.set_cartesian_pose(
            requested_arm_names,
            poses,
            control_way=control_way,
            use_wbc=use_wbc,
            remote_wbc_option=use_wbc,
        )
    except TypeError:
        astribot.set_cartesian_pose(
            requested_arm_names,
            poses,
            control_way=control_way,
            use_wbc=use_wbc,
        )


def move_to_first_action(
    astribot: Any,
    arm_names: list[str],
    step: ActionStep,
    *,
    duration: float,
    use_wbc: bool,
) -> tuple[np.ndarray, np.ndarray]:
    requested_arm_names = left_right_arm_names(arm_names)
    left_pose = eef9_to_astribot_pose(step.left_eef)
    right_pose = eef9_to_astribot_pose(step.right_eef)
    astribot.move_cartesian_pose(
        requested_arm_names,
        [left_pose.tolist(), right_pose.tolist()],
        duration=duration,
        use_wbc=use_wbc,
    )
    return left_pose, right_pose


def configure_filter_parameters(
    astribot: Any,
    *,
    filter_scale: float | None,
    gripper_filter_scale: float | None,
) -> None:
    if filter_scale is None and gripper_filter_scale is None:
        return
    resolved_filter_scale = filter_scale if filter_scale is not None else gripper_filter_scale
    resolved_gripper_filter_scale = (
        gripper_filter_scale if gripper_filter_scale is not None else resolved_filter_scale
    )
    astribot.set_filter_parameters(resolved_filter_scale, resolved_gripper_filter_scale)


def action_step_payload(step: ActionStep | None) -> dict[str, Any] | None:
    if step is None:
        return None
    return {
        "left_eef9": step.left_eef,
        "right_eef9": step.right_eef,
        "left_pose7": eef9_to_astribot_pose(step.left_eef),
        "right_pose7": eef9_to_astribot_pose(step.right_eef),
        "left_hand_joints": step.left_hand_joints,
        "right_hand_joints": step.right_hand_joints,
    }


def sent_command_payload(
    *,
    arm_names: list[str],
    left_pose: np.ndarray | None,
    right_pose: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    control_way: str,
    use_wbc: bool,
) -> dict[str, Any]:
    return {
        "arm_names": arm_names,
        "left_pose7": left_pose,
        "right_pose7": right_pose,
        "left_hand_joints": left_hand,
        "right_hand_joints": right_hand,
        "control_way": control_way,
        "use_wbc": use_wbc,
        "has_arm_command": left_pose is not None and right_pose is not None,
        "has_hand_command": left_hand is not None or right_hand is not None,
    }


class EmergencyStop:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_settings: list[Any] | None = None
        self.triggered = False

    def __enter__(self) -> "EmergencyStop":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def poll(self) -> bool:
        if self.triggered:
            return True
        if self._fd is None:
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if readable and sys.stdin.read(1) == " ":
            self.triggered = True
        return self.triggered


class LatestJointState:
    def __init__(self, expected_dim: int = 20) -> None:
        self.expected_dim = expected_dim
        self.position: np.ndarray | None = None

    def update(self, msg: Any) -> None:
        position = np.asarray(msg.position, dtype=np.float32)
        if position.shape != (self.expected_dim,):
            raise ValueError(f"Expected {self.expected_dim} hand joints, got {position.shape}")
        self.position = position

    def get(self) -> np.ndarray:
        if self.position is None:
            raise RuntimeError("Hand joint state has not been received yet")
        return self.position.copy()


class LatestCompressedImage:
    def __init__(self) -> None:
        self.image: np.ndarray | None = None
        self.update_count = 0

    def update(self, msg: Any) -> None:
        import cv2

        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode compressed image")
        self.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.update_count += 1

    def get(self) -> np.ndarray:
        if self.image is None:
            raise RuntimeError("Image has not been received yet")
        return self.image.copy()


def readiness_missing_items(
    *,
    left_hand: LatestJointState,
    right_hand: LatestJointState,
    head_image: LatestCompressedImage,
    left_wrist_image: LatestCompressedImage,
    right_wrist_image: LatestCompressedImage,
) -> list[str]:
    missing = []
    if left_hand.position is None:
        missing.append("left_hand joint_states")
    if right_hand.position is None:
        missing.append("right_hand joint_states")
    if head_image.image is None:
        missing.append("head_view image")
    if left_wrist_image.image is None:
        missing.append("left_wrist_view image")
    if right_wrist_image.image is None:
        missing.append("right_wrist_view image")
    return missing


class WujiRosIO:
    def __init__(
        self,
        node: Any,
        *,
        left_hand_ns: str,
        right_hand_ns: str,
        video_topics: dict[str, str] | None = None,
    ) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, JointState

        self.node = node
        self.video_topics = dict(video_topics or DEFAULT_VIDEO_TOPICS)
        self.joint_state_msg_type = JointState
        self.left_hand = LatestJointState()
        self.right_hand = LatestJointState()
        self.head_image = LatestCompressedImage()
        self.left_wrist_image = LatestCompressedImage()
        self.right_wrist_image = LatestCompressedImage()
        self._spin_lock = threading.Lock()
        self._background_spin_thread: threading.Thread | None = None
        self._executor: Any | None = None
        self._background_spin_error: BaseException | None = None

        self.left_hand_pub = node.create_publisher(
            JointState, f"/{left_hand_ns}/joint_commands", qos_profile_sensor_data
        )
        self.right_hand_pub = node.create_publisher(
            JointState, f"/{right_hand_ns}/joint_commands", qos_profile_sensor_data
        )

        node.create_subscription(
            JointState,
            f"/{left_hand_ns}/joint_states",
            self.left_hand.update,
            qos_profile_sensor_data,
        )
        node.create_subscription(
            JointState,
            f"/{right_hand_ns}/joint_states",
            self.right_hand.update,
            qos_profile_sensor_data,
        )
        node.create_subscription(
            CompressedImage,
            self.video_topics["head_view"],
            self.head_image.update,
            qos_profile_sensor_data,
        )
        node.create_subscription(
            CompressedImage,
            self.video_topics["left_wrist_view"],
            self.left_wrist_image.update,
            qos_profile_sensor_data,
        )
        node.create_subscription(
            CompressedImage,
            self.video_topics["right_wrist_view"],
            self.right_wrist_image.update,
            qos_profile_sensor_data,
        )

    def spin_once(self, timeout_sec: float = 0.01) -> None:
        if self._background_spin_thread is not None and self._background_spin_thread.is_alive():
            if timeout_sec > 0.0:
                time.sleep(timeout_sec)
            return

        import rclpy

        with self._spin_lock:
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def start_background_spin(self, timeout_sec: float = 0.005) -> None:
        if self._background_spin_thread is not None:
            return
        from rclpy.executors import MultiThreadedExecutor

        self._background_spin_error = None
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self.node)
        self._background_spin_thread = threading.Thread(
            target=self._background_spin,
            name="wuji_ros_spin",
            daemon=True,
        )
        self._background_spin_thread.start()

    def stop_background_spin(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:
                pass
        if self._background_spin_thread is not None:
            self._background_spin_thread.join(timeout=1.0)
            self._background_spin_thread = None
            try:
                if self._executor is not None:
                    self._executor.remove_node(self.node)
            except Exception:
                pass
        self._executor = None

    def _background_spin(self) -> None:
        try:
            if self._executor is not None:
                self._executor.spin()
        except BaseException as exc:
            self._background_spin_error = exc
            print(f"[ERROR] ROS background spinner stopped: {exc!r}", file=sys.stderr, flush=True)

    def background_spin_status(self) -> dict[str, Any]:
        return {
            "alive": bool(
                self._background_spin_thread is not None
                and self._background_spin_thread.is_alive()
            ),
            "error": repr(self._background_spin_error)
            if self._background_spin_error is not None
            else None,
        }

    def wait_until_ready(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.spin_once(0.05)
            missing = readiness_missing_items(
                left_hand=self.left_hand,
                right_hand=self.right_hand,
                head_image=self.head_image,
                left_wrist_image=self.left_wrist_image,
                right_wrist_image=self.right_wrist_image,
            )
            if not missing:
                return
        missing = readiness_missing_items(
            left_hand=self.left_hand,
            right_hand=self.right_hand,
            head_image=self.head_image,
            left_wrist_image=self.left_wrist_image,
            right_wrist_image=self.right_wrist_image,
        )
        raise TimeoutError(
            "Timed out waiting for hand joint states and camera images; "
            f"missing: {', '.join(missing)}"
        )

    def publish_hand(self, publisher: Any, joints: np.ndarray) -> None:
        msg = self.joint_state_msg_type()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.position = np.asarray(joints, dtype=np.float32).tolist()
        publisher.publish(msg)


def left_right_arm_names(arm_names: list[str]) -> list[str]:
    requested_arm_names = list(arm_names[:2])
    if len(requested_arm_names) != 2:
        raise ValueError(f"Expected at least two arm names, got {arm_names!r}")
    return requested_arm_names


def get_left_right_cartesian_pose(
    astribot: Any,
    arm_names: list[str],
    *,
    frame: str,
) -> tuple[Any, Any]:
    requested_arm_names = left_right_arm_names(arm_names)
    poses = astribot.get_current_cartesian_pose(requested_arm_names, frame=frame)
    if isinstance(poses, dict):
        missing = [name for name in requested_arm_names if name not in poses]
        if missing:
            raise ValueError(
                "Astribot returned cartesian pose dict missing requested arms: "
                f"{missing}; available: {list(poses)}"
            )
        left_pose, right_pose = poses[requested_arm_names[0]], poses[requested_arm_names[1]]
    else:
        pose_list = list(poses)
        if len(pose_list) < 2:
            raise ValueError(
                "Astribot returned fewer than two cartesian poses for "
                f"{requested_arm_names!r}: {poses!r}"
            )
        left_pose, right_pose = pose_list[:2]

    for label, pose in (("left", left_pose), ("right", right_pose)):
        if np.asarray(pose, dtype=np.float32).shape != (7,):
            raise ValueError(
                f"Astribot returned invalid {label} cartesian pose shape "
                f"{np.asarray(pose).shape}; expected (7,)"
            )
    return left_pose, right_pose


def build_observation(
    *,
    io: WujiRosIO,
    astribot: Any,
    arm_names: list[str],
    eef_frame: str,
    task: str,
) -> dict[str, Any]:
    left_pose, right_pose = get_left_right_cartesian_pose(
        astribot, arm_names, frame=eef_frame
    )
    return {
        "video": {
            "head_view": io.head_image.get()[None, None].astype(np.uint8),
            "left_wrist_view": io.left_wrist_image.get()[None, None].astype(np.uint8),
            "right_wrist_view": io.right_wrist_image.get()[None, None].astype(np.uint8),
        },
        "state": {
            "left_eef": astribot_pose_to_eef9(left_pose)[None, None].astype(np.float32),
            "right_eef": astribot_pose_to_eef9(right_pose)[None, None].astype(np.float32),
            "left_hand_joints": io.left_hand.get()[None, None].astype(np.float32),
            "right_hand_joints": io.right_hand.get()[None, None].astype(np.float32),
        },
        "language": {LANGUAGE_KEY: [[task]]},
    }


def read_actual_payload(
    *,
    io: WujiRosIO,
    astribot: Any,
    arm_names: list[str],
    eef_frame: str,
    delay_sec: float,
) -> dict[str, Any]:
    if delay_sec > 0.0:
        time.sleep(delay_sec)
    io.spin_once(0.0)
    left_pose, right_pose = get_left_right_cartesian_pose(
        astribot, arm_names, frame=eef_frame
    )
    return {
        "arm_names": arm_names,
        "eef_frame": eef_frame,
        "left_pose7": np.asarray(left_pose, dtype=np.float32),
        "right_pose7": np.asarray(right_pose, dtype=np.float32),
        "left_eef9": astribot_pose_to_eef9(left_pose),
        "right_eef9": astribot_pose_to_eef9(right_pose),
        "left_hand_joints": io.left_hand.get(),
        "right_hand_joints": io.right_hand.get(),
    }


class ActualStateStreamer:
    def __init__(
        self,
        *,
        io: WujiRosIO,
        astribot: Any,
        arm_names: list[str],
        eef_frame: str,
        logger: CommandLogger,
        hz: float,
        enabled: bool,
    ) -> None:
        self.io = io
        self.astribot = astribot
        self.arm_names = arm_names
        self.eef_frame = eef_frame
        self.logger = logger
        self.hz = hz
        self.enabled = enabled and logger.enabled and hz > 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ActualStateStreamer":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="wuji_actual_state_stream",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        period = 1.0 / self.hz
        next_sample = time.monotonic()
        sample_index = 0
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_sample:
                self._stop.wait(next_sample - now)
                continue
            try:
                payload = read_actual_payload(
                    io=self.io,
                    astribot=self.astribot,
                    arm_names=self.arm_names,
                    eef_frame=self.eef_frame,
                    delay_sec=0.0,
                )
                self.logger.log(
                    "actual_stream",
                    {
                        "sample_index": sample_index,
                        "actual_stream_hz": self.hz,
                        **payload,
                    },
                )
            except Exception as exc:  # pragma: no cover - depends on live robot/ROS state.
                self.logger.log(
                    "actual_stream",
                    {
                        "sample_index": sample_index,
                        "actual_stream_hz": self.hz,
                        "error": repr(exc),
                    },
                )
            sample_index += 1
            next_sample += period
            if next_sample < time.monotonic() - period:
                next_sample = time.monotonic() + period


def should_log_inline_actual(args: argparse.Namespace) -> bool:
    return args.actual_log_mode in ("inline", "both")


def send_direct_action_step(
    *,
    astribot: Any,
    io: Any,
    arm_names: list[str],
    step: ActionStep,
    previous_left_pose: np.ndarray,
    previous_right_pose: np.ndarray,
    workspace: dict[str, WorkspaceLimits],
    args: argparse.Namespace,
    logger: CommandLogger,
    loop_index: int,
    policy_step: Any,
    chunk_step_index: int,
    rtc_log: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    left_pose = clip_eef_pose(
        eef9_to_astribot_pose(step.left_eef),
        previous_left_pose,
        workspace["left"],
    )
    right_pose = clip_eef_pose(
        eef9_to_astribot_pose(step.right_eef),
        previous_right_pose,
        workspace["right"],
    )
    left_hand = clip_hand_joints(step.left_hand_joints)
    right_hand = clip_hand_joints(step.right_hand_joints)

    command_cartesian_pose(
        astribot,
        arm_names,
        left_pose,
        right_pose,
        control_way=args.eef_control_way,
        use_wbc=args.use_wbc,
    )
    io.publish_hand(io.left_hand_pub, left_hand)
    io.publish_hand(io.right_hand_pub, right_hand)

    log_payload = {
        "loop_index": loop_index,
        "policy_step": policy_step,
        "chunk_step_index": chunk_step_index,
        **sent_command_payload(
            arm_names=arm_names,
            left_pose=left_pose,
            right_pose=right_pose,
            left_hand=left_hand,
            right_hand=right_hand,
            control_way=args.eef_control_way,
            use_wbc=args.use_wbc,
        ),
    }
    if rtc_log is not None:
        log_payload.update(rtc_log)
    logger.log("sent", log_payload)

    if logger.enabled and should_log_inline_actual(args):
        actual_payload = {
            "loop_index": loop_index,
            "policy_step": policy_step,
            "chunk_step_index": chunk_step_index,
            **read_actual_payload(
                io=io,
                astribot=astribot,
                arm_names=arm_names,
                eef_frame=args.eef_frame,
                delay_sec=args.actual_readback_delay_sec,
            ),
        }
        if rtc_log is not None:
            actual_payload.update(rtc_log)
        logger.log("actual", actual_payload)

    return left_pose, right_pose


def image_update_counts(io: Any) -> dict[str, int]:
    return {
        "head_view": int(io.head_image.update_count),
        "left_wrist_view": int(io.left_wrist_image.update_count),
        "right_wrist_view": int(io.right_wrist_image.update_count),
    }


def wait_for_fresh_images(
    io: Any,
    previous_counts: dict[str, int] | None,
    *,
    timeout_sec: float,
) -> tuple[bool, dict[str, int]]:
    if previous_counts is None:
        io.spin_once(0.0)
        return True, image_update_counts(io)

    deadline = time.monotonic() + timeout_sec
    while True:
        io.spin_once(0.01)
        current_counts = image_update_counts(io)
        if all(current_counts[view] > previous_counts.get(view, -1) for view in VIDEO_KEYS):
            return True, current_counts
        if time.monotonic() >= deadline:
            return False, current_counts


def measure_image_update_rates(
    io: Any,
    *,
    duration_sec: float,
) -> dict[str, float]:
    if duration_sec <= 0.0:
        return {view: 0.0 for view in VIDEO_KEYS}

    start_counts = image_update_counts(io)
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        io.spin_once(0.01)
    end_counts = image_update_counts(io)
    return {
        view: max(0, end_counts[view] - start_counts[view]) / duration_sec
        for view in VIDEO_KEYS
    }


def ensure_live_camera_streams(
    io: Any,
    *,
    duration_sec: float,
    min_hz: float,
) -> dict[str, float]:
    rates = measure_image_update_rates(io, duration_sec=duration_sec)
    stale_views = {view: hz for view, hz in rates.items() if hz < min_hz}
    if stale_views:
        topics = getattr(io, "video_topics", DEFAULT_VIDEO_TOPICS)
        topic_lines = ", ".join(f"{view}={topics.get(view, '<unknown>')}" for view in VIDEO_KEYS)
        raise RuntimeError(
            "Camera preflight failed: expected every view to publish at "
            f">= {min_hz:g} Hz over {duration_sec:g}s, got {rates}. Topics: {topic_lines}. "
            "Do not run policy inference until the failing camera topic is live; otherwise "
            "the client would reuse stale images."
        )
    return rates


def save_observation_images(
    observation: dict[str, Any],
    output_dir: str | Path,
    *,
    loop_index: int,
    prefix: str = "loop",
) -> list[Path]:
    import cv2

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    video = observation.get("video")
    if not isinstance(video, dict):
        raise ValueError("Observation is missing video data")

    for view in VIDEO_KEYS:
        if view not in video:
            raise KeyError(f"Observation video missing required view: {view}")
        array = np.asarray(video[view])
        if array.ndim != 5 or array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError(f"Observation video.{view} must have shape (B, T, H, W, C)")
        image = array[0, 0]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Observation video.{view} frame must have shape (H, W, 3)")
        filename = f"{prefix}_{loop_index:06d}_{view}.png"
        path = output_path / filename
        bgr = cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"Failed to write observation image: {path}")
        saved_paths.append(path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="GR00T policy server host/IP")
    parser.add_argument("--port", type=int, default=5555, help="GR00T policy server port")
    parser.add_argument("--task", required=True, help="Language task instruction")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Robot command rate")
    parser.add_argument("--execute-horizon", type=int, default=16, help="Steps to execute per call")
    parser.add_argument("--rtc-enable", action="store_true", help="Enable async RTC scheduling")
    parser.add_argument(
        "--rtc-overlap-steps",
        type=int,
        default=None,
        help="RTC overlap prefix length; defaults to action_horizon - execute_horizon",
    )
    parser.add_argument(
        "--rtc-frozen-steps",
        type=int,
        default=None,
        help="RTC frozen prefix length; defaults to latency percentile converted to control steps",
    )
    parser.add_argument("--rtc-ramp-rate", type=float, default=2.5, help="RTC exponential ramp rate")
    parser.add_argument(
        "--rtc-latency-window",
        type=int,
        default=20,
        help="Number of recent RTC latencies used for delay estimation",
    )
    parser.add_argument(
        "--rtc-delay-percentile",
        type=float,
        default=95.0,
        help="Latency percentile converted to estimated RTC delay steps",
    )
    parser.add_argument(
        "--rtc-min-leftover",
        type=int,
        default=1,
        help="Minimum leftover steps required to send RTC guidance",
    )
    parser.add_argument(
        "--rtc-fallback",
        choices=["hold-last", "stop"],
        default="hold-last",
        help="Fallback action when async inference is late and the queue is exhausted",
    )
    parser.add_argument(
        "--rtc-server-unsupported",
        choices=["fail", "disable"],
        default="fail",
        help="How to handle server errors that indicate RTC options are unsupported",
    )
    parser.add_argument(
        "--arm-interp-hz",
        type=float,
        default=0.0,
        help="Interpolate EEF commands to this rate; 0 disables arm interpolation",
    )
    parser.add_argument(
        "--hand-interp-hz",
        type=float,
        default=0.0,
        help="Interpolate hand joint commands to this rate; 0 disables hand interpolation",
    )
    parser.add_argument("--timeout-ms", type=int, default=15000, help="Policy client timeout")
    parser.add_argument("--eef-frame", choices=["chassis", "world"], default="chassis")
    parser.add_argument("--left-hand-ns", default="left_hand")
    parser.add_argument("--right-hand-ns", default="right_hand")
    parser.add_argument(
        "--head-image-topic",
        default=DEFAULT_VIDEO_TOPICS["head_view"],
        help="CompressedImage topic used for observation video.head_view",
    )
    parser.add_argument(
        "--left-wrist-image-topic",
        default=DEFAULT_VIDEO_TOPICS["left_wrist_view"],
        help="CompressedImage topic used for observation video.left_wrist_view",
    )
    parser.add_argument(
        "--right-wrist-image-topic",
        default=DEFAULT_VIDEO_TOPICS["right_wrist_view"],
        help="CompressedImage topic used for observation video.right_wrist_view",
    )
    parser.add_argument("--warmup-timeout-sec", type=float, default=10.0)
    parser.add_argument(
        "--camera-preflight-sec",
        type=float,
        default=1.0,
        help="Measure live camera update rates before inference; 0 disables the preflight",
    )
    parser.add_argument(
        "--min-camera-hz",
        type=float,
        default=5.0,
        help="Minimum per-view camera update rate required by the preflight",
    )
    parser.add_argument(
        "--workspace-radius",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.25, 0.25, 0.25),
        help="Per-arm xyz safety window around the startup EEF pose, in meters",
    )
    parser.add_argument(
        "--max-eef-step",
        type=float,
        default=0.03,
        help="Maximum xyz command change per executed step, in meters",
    )
    parser.add_argument(
        "--max-eef-rotation-step-deg",
        type=float,
        default=10.0,
        help="Maximum EEF quaternion command change per executed step, in degrees",
    )
    parser.add_argument(
        "--eef-control-way",
        choices=["direct", "filter"],
        default="direct",
        help="Astribot arm control mode passed to set_cartesian_pose",
    )
    parser.add_argument(
        "--no-wbc",
        dest="use_wbc",
        action="store_false",
        help="Disable whole-body control in Astribot set_cartesian_pose",
    )
    parser.set_defaults(use_wbc=True)
    parser.add_argument("--max-loops", type=int, default=0, help="0 means run until stopped")
    parser.add_argument("--no-home", action="store_true", help="Do not call Astribot.move_to_home()")
    parser.add_argument(
        "--move-to-first",
        action="store_true",
        help="Move both arms to the first policy action before starting online execution",
    )
    parser.add_argument(
        "--move-to-first-duration",
        type=float,
        default=3.0,
        help="Duration for --move-to-first cartesian move, in seconds",
    )
    parser.add_argument(
        "--filter-scale",
        type=float,
        default=None,
        help="Astribot SDK arm filter scale; higher tracks faster, lower is smoother",
    )
    parser.add_argument(
        "--gripper-filter-scale",
        type=float,
        default=None,
        help="Astribot SDK gripper filter scale; defaults to --filter-scale when omitted",
    )
    parser.add_argument("--log-dir", default=DEFAULT_DATASET_PATH, help="Directory for JSONL command logs")
    parser.add_argument("--log-prefix", default="client", help="Prefix for JSONL command log files")
    parser.add_argument(
        "--actual-readback-delay-sec",
        type=float,
        default=0.0,
        help="Delay before inline actual readback; only used when --actual-log-mode is inline/both",
    )
    parser.add_argument(
        "--actual-stream-hz",
        type=float,
        default=30.0,
        help="Background actual-state logging rate in Hz; 0 disables the stream",
    )
    parser.add_argument(
        "--actual-log-mode",
        choices=["stream", "inline", "both", "off"],
        default="stream",
        help=(
            "How to log actual robot feedback: stream logs independent timestamped feedback, "
            "inline logs the old per-command readback, both writes both, off disables actual logging"
        ),
    )
    parser.add_argument(
        "--save-observation-images",
        action="store_true",
        help="Save the exact observation images sent to the policy server before each inference",
    )
    parser.add_argument(
        "--save-observation-image-every",
        type=int,
        default=1,
        help="Save observation images every N inference loops when --save-observation-images is set",
    )
    parser.add_argument(
        "--fresh-image-timeout-sec",
        type=float,
        default=0.2,
        help="Maximum time to wait for all camera views to update before each policy request",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.control_hz <= 0.0:
        raise ValueError("--control-hz must be positive")
    if args.execute_horizon <= 0:
        raise ValueError("--execute-horizon must be positive")
    if args.arm_interp_hz < 0.0:
        raise ValueError("--arm-interp-hz must be non-negative")
    if args.hand_interp_hz < 0.0:
        raise ValueError("--hand-interp-hz must be non-negative")
    if args.actual_readback_delay_sec < 0.0:
        raise ValueError("--actual-readback-delay-sec must be non-negative")
    if args.actual_stream_hz < 0.0:
        raise ValueError("--actual-stream-hz must be non-negative")
    if args.save_observation_image_every <= 0:
        raise ValueError("--save-observation-image-every must be positive")
    if args.save_observation_images and not args.log_dir:
        raise ValueError("--save-observation-images requires --log-dir")
    if args.fresh_image_timeout_sec < 0.0:
        raise ValueError("--fresh-image-timeout-sec must be non-negative")
    if args.camera_preflight_sec < 0.0:
        raise ValueError("--camera-preflight-sec must be non-negative")
    if args.min_camera_hz < 0.0:
        raise ValueError("--min-camera-hz must be non-negative")
    if args.max_eef_rotation_step_deg < 0.0:
        raise ValueError("--max-eef-rotation-step-deg must be non-negative")
    if args.move_to_first_duration < 0.0:
        raise ValueError("--move-to-first-duration must be non-negative")
    if args.filter_scale is not None and args.filter_scale < 0.0:
        raise ValueError("--filter-scale must be non-negative")
    if args.gripper_filter_scale is not None and args.gripper_filter_scale < 0.0:
        raise ValueError("--gripper-filter-scale must be non-negative")
    if args.rtc_overlap_steps is not None and args.rtc_overlap_steps < 0:
        raise ValueError("--rtc-overlap-steps must be non-negative")
    if args.rtc_frozen_steps is not None and args.rtc_frozen_steps < 0:
        raise ValueError("--rtc-frozen-steps must be non-negative")
    if args.rtc_ramp_rate <= 0.0:
        raise ValueError("--rtc-ramp-rate must be positive")
    if args.rtc_latency_window <= 0:
        raise ValueError("--rtc-latency-window must be positive")
    if args.rtc_delay_percentile < 0.0 or args.rtc_delay_percentile > 100.0:
        raise ValueError("--rtc-delay-percentile must be in [0, 100]")
    if args.rtc_min_leftover < 0:
        raise ValueError("--rtc-min-leftover must be non-negative")
    if args.rtc_enable and (args.arm_interp_hz > 0.0 or args.hand_interp_hz > 0.0):
        raise ValueError("--rtc-enable currently requires --arm-interp-hz 0 and --hand-interp-hz 0")

    from astribot_sdk.core.astribot_api.astribot_client import Astribot
    from gr00t.policy.server_client import PolicyClient
    import rclpy

    rclpy.init()
    node = rclpy.create_node("gr00t_wuji_rot6d_client")
    video_topics = {
        "head_view": args.head_image_topic,
        "left_wrist_view": args.left_wrist_image_topic,
        "right_wrist_view": args.right_wrist_image_topic,
    }
    io = WujiRosIO(
        node,
        left_hand_ns=args.left_hand_ns,
        right_hand_ns=args.right_hand_ns,
        video_topics=video_topics,
    )
    io.start_background_spin()

    policy = PolicyClient(host=args.host, port=args.port, timeout_ms=args.timeout_ms, strict=False)
    if not policy.ping():
        raise ConnectionError(f"Failed to ping GR00T server at {args.host}:{args.port}")

    astribot = Astribot(
        freq=max(args.control_hz, args.arm_interp_hz, args.hand_interp_hz, 30.0),
        high_control_rights=True,
    )
    configure_filter_parameters(
        astribot,
        filter_scale=args.filter_scale,
        gripper_filter_scale=args.gripper_filter_scale,
    )
    arm_names = [astribot.arm_left_name, astribot.arm_right_name]

    if not args.no_home:
        print("Moving Astribot to home pose...")
        astribot.move_to_home()

    print("Waiting for hand joint states and camera images...")
    io.wait_until_ready(args.warmup_timeout_sec)
    if args.camera_preflight_sec > 0.0:
        print(
            "Checking live camera streams "
            f"for {args.camera_preflight_sec:g}s: {video_topics}"
        )
        camera_rates = ensure_live_camera_streams(
            io,
            duration_sec=args.camera_preflight_sec,
            min_hz=args.min_camera_hz,
        )
        print(f"Camera stream rates: {camera_rates}")
    previous_image_counts = image_update_counts(io)

    current_left_pose, current_right_pose = get_left_right_cartesian_pose(
        astribot, arm_names, frame=args.eef_frame
    )
    previous_left_pose = np.asarray(current_left_pose, dtype=np.float32)
    previous_right_pose = np.asarray(current_right_pose, dtype=np.float32)
    workspace = workspace_limits_from_current(
        previous_left_pose,
        previous_right_pose,
        radius=np.asarray(args.workspace_radius, dtype=np.float32),
        max_xyz_step=args.max_eef_step,
        max_rotation_step_rad=math.radians(args.max_eef_rotation_step_deg),
    )
    period = 1.0 / args.control_hz
    loop_count = 0
    skipped_policy_requests = 0
    previous_action_step: ActionStep | None = None
    interpolation_enabled = args.arm_interp_hz > 0.0 or args.hand_interp_hz > 0.0
    moved_to_first = False
    logger = CommandLogger(args.log_dir, prefix=args.log_prefix)
    write_client_config(logger, args)
    if logger.enabled:
        print(f"Writing command logs to {logger.log_dir}")
    observation_image_dir = (
        logger.log_dir / "observation_images"
        if args.save_observation_images and logger.log_dir is not None
        else None
    )
    if observation_image_dir is not None:
        print(f"Saving observation images to {observation_image_dir}")
    actual_streamer = ActualStateStreamer(
        io=io,
        astribot=astribot,
        arm_names=arm_names,
        eef_frame=args.eef_frame,
        logger=logger,
        hz=args.actual_stream_hz,
        enabled=args.actual_log_mode in ("stream", "both"),
    )
    if actual_streamer.enabled:
        print(f"Logging actual feedback stream at {args.actual_stream_hz:g} Hz")

    print("Client running. Press Space to emergency stop.")
    try:
        with actual_streamer, EmergencyStop() as estop:
            if args.rtc_enable:
                latency_tracker = LatencyTracker(
                    window_size=args.rtc_latency_window,
                    percentile=args.rtc_delay_percentile,
                )
                current_actions: dict[str, np.ndarray] | None = None
                current_steps: list[ActionStep] = []
                current_info: dict[str, Any] = {}
                queue_index = 0
                pending_future: Future | None = None
                pending_meta: dict[str, Any] = {}
                rtc_disabled_reason: str | None = None

                def background_get_action(
                    observation: dict[str, Any], request_options: dict[str, Any]
                ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
                    client = PolicyClient(
                        host=args.host,
                        port=args.port,
                        timeout_ms=args.timeout_ms,
                        strict=False,
                    )
                    try:
                        return client.get_action(observation, options=request_options)
                    finally:
                        try:
                            client.socket.close(linger=0)
                            client.context.term()
                        except Exception:
                            pass

                def capture_observation_or_skip() -> dict[str, Any] | None:
                    nonlocal previous_image_counts, skipped_policy_requests
                    fresh_images, image_counts = wait_for_fresh_images(
                        io,
                        previous_image_counts,
                        timeout_sec=args.fresh_image_timeout_sec,
                    )
                    if logger.enabled:
                        logger.log(
                            "observation",
                            {
                                "loop_index": loop_count,
                                "skipped_policy_requests": skipped_policy_requests,
                                "fresh_images": fresh_images,
                                "image_update_counts": image_counts,
                                "video_topics": video_topics,
                                "background_spin": io.background_spin_status(),
                                "rtc_enabled": True,
                                "rtc_disabled_reason": rtc_disabled_reason,
                            },
                        )
                    if not fresh_images:
                        print(
                            "[WARN] Timed out waiting for fresh camera frames; "
                            f"skipping policy request at loop {loop_count}: {image_counts}; "
                            f"background_spin={io.background_spin_status()}",
                            flush=True,
                        )
                        skipped_policy_requests += 1
                        return None
                    observation = build_observation(
                        io=io,
                        astribot=astribot,
                        arm_names=arm_names,
                        eef_frame=args.eef_frame,
                        task=args.task,
                    )
                    previous_image_counts = image_update_counts(io)
                    if (
                        observation_image_dir is not None
                        and loop_count % args.save_observation_image_every == 0
                    ):
                        save_observation_images(
                            observation,
                            observation_image_dir,
                            loop_index=loop_count,
                        )
                    return observation

                with ThreadPoolExecutor(max_workers=1) as executor:
                    while not estop.poll():
                        loop_started = time.monotonic()

                        if pending_future is not None and pending_future.done():
                            latency_sec = time.monotonic() - float(pending_meta["started_at"])
                            try:
                                actions, info = pending_future.result()
                            except Exception as exc:
                                logger.log(
                                    "rtc",
                                    {
                                        **pending_meta,
                                        "latency_sec": latency_sec,
                                        "fallback_reason": repr(exc),
                                        "rtc_enabled": True,
                                    },
                                )
                                pending_future = None
                                pending_meta = {}
                                if args.rtc_server_unsupported == "disable":
                                    rtc_disabled_reason = repr(exc)
                                else:
                                    raise
                            else:
                                latency_tracker.add(latency_sec)
                                real_delay_steps = latency_seconds_to_delay_steps(
                                    latency_sec,
                                    control_hz=args.control_hz,
                                )
                                try:
                                    current_actions = discard_action_prefix(
                                        actions,
                                        delay_steps=real_delay_steps,
                                    )
                                except ValueError as exc:
                                    logger.log(
                                        "rtc",
                                        {
                                            **pending_meta,
                                            "latency_sec": latency_sec,
                                            "real_delay_steps": real_delay_steps,
                                            "fallback_reason": repr(exc),
                                            "rtc_enabled": True,
                                        },
                                    )
                                else:
                                    current_steps = list(
                                        iter_action_steps(
                                            current_actions,
                                            _action_horizon(current_actions),
                                        )
                                    )
                                    current_info = info if isinstance(info, dict) else {}
                                    queue_index = 0
                                    logger.log(
                                        "rtc",
                                        {
                                            **pending_meta,
                                            "latency_sec": latency_sec,
                                            "real_delay_steps": real_delay_steps,
                                            "queue_len": len(current_steps),
                                            "rtc_enabled": True,
                                        },
                                    )
                                pending_future = None
                                pending_meta = {}

                        if current_actions is None or queue_index >= len(current_steps):
                            if pending_future is not None:
                                if pending_future.done():
                                    continue
                                if args.rtc_fallback == "stop" or previous_action_step is None:
                                    logger.log(
                                        "rtc",
                                        {
                                            **pending_meta,
                                            "queue_len": 0,
                                            "fallback_reason": "queue_exhausted",
                                            "rtc_enabled": True,
                                        },
                                    )
                                    break
                                previous_left_pose, previous_right_pose = send_direct_action_step(
                                    astribot=astribot,
                                    io=io,
                                    arm_names=arm_names,
                                    step=previous_action_step,
                                    previous_left_pose=previous_left_pose,
                                    previous_right_pose=previous_right_pose,
                                    workspace=workspace,
                                    args=args,
                                    logger=logger,
                                    loop_index=loop_count,
                                    policy_step=current_info.get("current_step"),
                                    chunk_step_index=queue_index,
                                    rtc_log={
                                        "rtc_enabled": True,
                                        "queue_len": 0,
                                        "fallback_reason": "hold_last",
                                        "rtc_disabled_reason": rtc_disabled_reason,
                                    },
                                )
                                elapsed = time.monotonic() - loop_started
                                sleep_time = period - elapsed
                                if sleep_time > 0:
                                    time.sleep(sleep_time)
                                loop_count += 1
                                continue
                            else:
                                observation = capture_observation_or_skip()
                                if observation is None:
                                    time.sleep(min(period, args.fresh_image_timeout_sec))
                                    continue
                                request_started_at = time.monotonic()
                                actions, info = policy.get_action(observation)
                                latency_tracker.add(time.monotonic() - request_started_at)
                                current_actions = actions
                                current_steps = list(
                                    iter_action_steps(actions, _action_horizon(actions))
                                )
                                current_info = info if isinstance(info, dict) else {}
                                queue_index = 0

                        if (
                            rtc_disabled_reason is None
                            and pending_future is None
                            and current_actions is not None
                        ):
                            full_horizon = _action_horizon(current_actions)
                            resolved_overlap_steps = (
                                max(0, full_horizon - args.execute_horizon)
                                if args.rtc_overlap_steps is None
                                else args.rtc_overlap_steps
                            )
                            estimated_delay_steps = latency_tracker.delay_steps(args.control_hz)
                            trigger_backlog = max(
                                args.rtc_min_leftover,
                                resolved_overlap_steps + estimated_delay_steps,
                            )
                            trigger_index = max(0, full_horizon - trigger_backlog)
                            if queue_index >= trigger_index:
                                leftover = slice_leftover_actions(current_actions, queue_index)
                                leftover_len = _action_horizon(leftover)
                                if leftover_len >= max(args.rtc_min_leftover, resolved_overlap_steps):
                                    observation = capture_observation_or_skip()
                                    if observation is not None:
                                        request_options = build_rtc_options(
                                            prev_chunk_left_over=leftover,
                                            action_horizon=leftover_len,
                                            overlap_steps=resolved_overlap_steps,
                                            frozen_steps=args.rtc_frozen_steps,
                                            estimated_delay_steps=estimated_delay_steps,
                                            execute_horizon=args.execute_horizon,
                                            ramp_rate=args.rtc_ramp_rate,
                                        )
                                        request_started_at = time.monotonic()
                                        pending_future = executor.submit(
                                            background_get_action,
                                            observation,
                                            request_options,
                                        )
                                        pending_meta = {
                                            "request_started_at_step": queue_index,
                                            "estimated_delay_steps": estimated_delay_steps,
                                            "leftover_len": leftover_len,
                                            "overlap_steps": request_options["rtc"][
                                                "rtc_overlap_steps"
                                            ],
                                            "frozen_steps": request_options["rtc"][
                                                "rtc_frozen_steps"
                                            ],
                                            "started_at": request_started_at,
                                            "rtc_disabled_reason": rtc_disabled_reason,
                                        }
                                else:
                                    logger.log(
                                        "rtc",
                                        {
                                            "request_started_at_step": queue_index,
                                            "estimated_delay_steps": estimated_delay_steps,
                                            "leftover_len": leftover_len,
                                            "overlap_steps": resolved_overlap_steps,
                                            "frozen_steps": args.rtc_frozen_steps,
                                            "queue_len": len(current_steps),
                                            "fallback_reason": "short_leftover",
                                            "rtc_enabled": True,
                                            "rtc_disabled_reason": rtc_disabled_reason,
                                        },
                                    )

                        if queue_index >= len(current_steps):
                            continue

                        policy_step = current_info.get("current_step")
                        step = current_steps[queue_index]
                        previous_left_pose, previous_right_pose = send_direct_action_step(
                            astribot=astribot,
                            io=io,
                            arm_names=arm_names,
                            step=step,
                            previous_left_pose=previous_left_pose,
                            previous_right_pose=previous_right_pose,
                            workspace=workspace,
                            args=args,
                            logger=logger,
                            loop_index=loop_count,
                            policy_step=policy_step,
                            chunk_step_index=queue_index,
                            rtc_log={
                                "rtc_enabled": True,
                                "estimated_delay_steps": latency_tracker.delay_steps(
                                    args.control_hz
                                ),
                                "queue_len": len(current_steps) - queue_index,
                                "rtc_disabled_reason": rtc_disabled_reason,
                            },
                        )
                        previous_action_step = step
                        queue_index += 1

                        elapsed = time.monotonic() - loop_started
                        sleep_time = period - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                        loop_count += 1
                        if args.max_loops and loop_count >= args.max_loops:
                            print(f"Reached --max-loops={args.max_loops}; stopping.")
                            break
                        if policy_step is not None:
                            print(f"policy_step={policy_step}", flush=True)
                return

            while not estop.poll():
                loop_started = time.monotonic()
                fresh_images, image_counts = wait_for_fresh_images(
                    io,
                    previous_image_counts,
                    timeout_sec=args.fresh_image_timeout_sec,
                )
                if logger.enabled:
                    logger.log(
                        "observation",
                        {
                            "loop_index": loop_count,
                            "skipped_policy_requests": skipped_policy_requests,
                            "fresh_images": fresh_images,
                            "image_update_counts": image_counts,
                            "video_topics": video_topics,
                            "background_spin": io.background_spin_status(),
                        },
                    )
                if not fresh_images:
                    print(
                        "[WARN] Timed out waiting for fresh camera frames; "
                        f"skipping policy request at loop {loop_count}: {image_counts}; "
                        f"background_spin={io.background_spin_status()}",
                        flush=True,
                    )
                    skipped_policy_requests += 1
                    if period > 0.0:
                        time.sleep(min(period, args.fresh_image_timeout_sec))
                    continue
                observation = build_observation(
                    io=io,
                    astribot=astribot,
                    arm_names=arm_names,
                    eef_frame=args.eef_frame,
                    task=args.task,
                )
                previous_image_counts = image_update_counts(io)
                if (
                    observation_image_dir is not None
                    and loop_count % args.save_observation_image_every == 0
                ):
                    save_observation_images(
                        observation,
                        observation_image_dir,
                        loop_index=loop_count,
                    )
                actions, info = policy.get_action(observation)
                steps = list(iter_action_steps(actions, args.execute_horizon))
                policy_step = info.get("current_step") if isinstance(info, dict) else None
                if args.move_to_first and not moved_to_first and steps:
                    print("Moving Astribot to first policy action...")
                    first_left_pose, first_right_pose = move_to_first_action(
                        astribot,
                        arm_names,
                        steps[0],
                        duration=args.move_to_first_duration,
                        use_wbc=args.use_wbc,
                    )
                    previous_left_pose = first_left_pose
                    previous_right_pose = first_right_pose
                    workspace = workspace_limits_from_current(
                        previous_left_pose,
                        previous_right_pose,
                        radius=np.asarray(args.workspace_radius, dtype=np.float32),
                        max_xyz_step=args.max_eef_step,
                        max_rotation_step_rad=math.radians(args.max_eef_rotation_step_deg),
                    )
                    moved_to_first = True
                    if steps[0].left_hand_joints is not None:
                        io.publish_hand(io.left_hand_pub, clip_hand_joints(steps[0].left_hand_joints))
                        io.publish_hand(
                            io.right_hand_pub,
                            clip_hand_joints(steps[0].right_hand_joints),
                        )
                    logger.log(
                        "move_to_first",
                        {
                            "loop_index": loop_count,
                            "policy_step": policy_step,
                            "duration": args.move_to_first_duration,
                            **sent_command_payload(
                                arm_names=arm_names,
                                left_pose=first_left_pose,
                                right_pose=first_right_pose,
                                left_hand=clip_hand_joints(steps[0].left_hand_joints),
                                right_hand=clip_hand_joints(steps[0].right_hand_joints),
                                control_way="move_cartesian_pose",
                                use_wbc=args.use_wbc,
                            ),
                        },
                    )
                for chunk_step_index, step in enumerate(steps):
                    logger.log(
                        "raw",
                        {
                            "loop_index": loop_count,
                            "policy_step": policy_step,
                            "chunk_step_index": chunk_step_index,
                            "action": action_step_payload(step),
                        },
                    )
                if interpolation_enabled:
                    events = build_interpolated_action_events(
                        steps,
                        previous_step=previous_action_step,
                        source_hz=args.control_hz,
                        arm_interp_hz=args.arm_interp_hz,
                        hand_interp_hz=args.hand_interp_hz,
                    )
                    for event_index, event in enumerate(events):
                        if estop.poll():
                            break
                        logger.log(
                            "interpolated",
                            {
                                "loop_index": loop_count,
                                "policy_step": policy_step,
                                "event_index": event_index,
                                "sleep_seconds": event.sleep_seconds,
                                "arm_action": action_step_payload(event.arm_step),
                                "hand_action": action_step_payload(event.hand_step),
                            },
                        )

                        left_pose = None
                        right_pose = None
                        left_hand = None
                        right_hand = None
                        if event.arm_step is not None:
                            left_pose = clip_eef_pose(
                                eef9_to_astribot_pose(event.arm_step.left_eef),
                                previous_left_pose,
                                workspace["left"],
                            )
                            right_pose = clip_eef_pose(
                                eef9_to_astribot_pose(event.arm_step.right_eef),
                                previous_right_pose,
                                workspace["right"],
                            )
                            command_cartesian_pose(
                                astribot,
                                arm_names,
                                left_pose,
                                right_pose,
                                control_way=args.eef_control_way,
                                use_wbc=args.use_wbc,
                            )
                            previous_left_pose = left_pose
                            previous_right_pose = right_pose

                        if event.hand_step is not None:
                            left_hand = clip_hand_joints(event.hand_step.left_hand_joints)
                            right_hand = clip_hand_joints(event.hand_step.right_hand_joints)
                            io.publish_hand(io.left_hand_pub, left_hand)
                            io.publish_hand(io.right_hand_pub, right_hand)

                        logger.log(
                            "sent",
                            {
                                "loop_index": loop_count,
                                "policy_step": policy_step,
                                "event_index": event_index,
                                **sent_command_payload(
                                    arm_names=arm_names,
                                    left_pose=left_pose,
                                    right_pose=right_pose,
                                    left_hand=left_hand,
                                    right_hand=right_hand,
                                    control_way=args.eef_control_way,
                                    use_wbc=args.use_wbc,
                                ),
                            },
                        )
                        if logger.enabled and should_log_inline_actual(args):
                            logger.log(
                                "actual",
                                {
                                    "loop_index": loop_count,
                                    "policy_step": policy_step,
                                    "event_index": event_index,
                                    **read_actual_payload(
                                        io=io,
                                        astribot=astribot,
                                        arm_names=arm_names,
                                        eef_frame=args.eef_frame,
                                        delay_sec=args.actual_readback_delay_sec,
                                    ),
                                },
                            )

                        if event.sleep_seconds > 0.0:
                            time.sleep(event.sleep_seconds)
                    if steps:
                        previous_action_step = steps[-1]
                else:
                    for chunk_step_index, step in enumerate(steps):
                        if estop.poll():
                            break

                        left_pose = clip_eef_pose(
                            eef9_to_astribot_pose(step.left_eef),
                            previous_left_pose,
                            workspace["left"],
                        )
                        right_pose = clip_eef_pose(
                            eef9_to_astribot_pose(step.right_eef),
                            previous_right_pose,
                            workspace["right"],
                        )
                        left_hand = clip_hand_joints(step.left_hand_joints)
                        right_hand = clip_hand_joints(step.right_hand_joints)

                        command_cartesian_pose(
                            astribot,
                            arm_names,
                            left_pose,
                            right_pose,
                            control_way=args.eef_control_way,
                            use_wbc=args.use_wbc,
                        )
                        io.publish_hand(io.left_hand_pub, left_hand)
                        io.publish_hand(io.right_hand_pub, right_hand)
                        previous_left_pose = left_pose
                        previous_right_pose = right_pose
                        previous_action_step = step
                        logger.log(
                            "sent",
                            {
                                "loop_index": loop_count,
                                "policy_step": policy_step,
                                "chunk_step_index": chunk_step_index,
                                **sent_command_payload(
                                    arm_names=arm_names,
                                    left_pose=left_pose,
                                    right_pose=right_pose,
                                    left_hand=left_hand,
                                    right_hand=right_hand,
                                    control_way=args.eef_control_way,
                                    use_wbc=args.use_wbc,
                                ),
                            },
                        )
                        if logger.enabled and should_log_inline_actual(args):
                            logger.log(
                                "actual",
                                {
                                    "loop_index": loop_count,
                                    "policy_step": policy_step,
                                    "chunk_step_index": chunk_step_index,
                                    **read_actual_payload(
                                        io=io,
                                        astribot=astribot,
                                        arm_names=arm_names,
                                        eef_frame=args.eef_frame,
                                        delay_sec=args.actual_readback_delay_sec,
                                    ),
                                },
                            )

                        elapsed = time.monotonic() - loop_started
                        sleep_time = period - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        loop_started = time.monotonic()

                loop_count += 1
                if args.max_loops and loop_count >= args.max_loops:
                    print(f"Reached --max-loops={args.max_loops}; stopping.")
                    break
                if isinstance(info, dict) and info.get("current_step") is not None:
                    print(f"policy_step={info['current_step']}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted; stopping.")
    finally:
        io.stop_background_spin()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
