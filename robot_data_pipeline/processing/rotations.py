from __future__ import annotations

import numpy as np


class RotationError(ValueError):
    pass


def normalize_quaternions(
    quaternions_xyzw: np.ndarray, *, min_norm: float = 0.5, max_norm: float = 1.5
) -> np.ndarray:
    quaternions = np.asarray(quaternions_xyzw, dtype=np.float64)
    if quaternions.shape[-1:] != (4,):
        raise RotationError("quaternion array must have a final dimension of 4")
    if not np.all(np.isfinite(quaternions)):
        raise RotationError("quaternion contains NaN or Inf")
    norms = np.linalg.norm(quaternions, axis=-1)
    if np.any((norms < min_norm) | (norms > max_norm)):
        raise RotationError(f"quaternion norm outside [{min_norm}, {max_norm}]")
    return quaternions / norms[..., None]


def make_quaternion_signs_continuous(quaternions_xyzw: np.ndarray) -> np.ndarray:
    quaternions = normalize_quaternions(quaternions_xyzw).copy()
    if quaternions.ndim != 2:
        raise RotationError("sign continuity expects an N x 4 quaternion array")
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1
    return quaternions


def quaternion_to_matrix(quaternions_xyzw: np.ndarray) -> np.ndarray:
    q = normalize_quaternions(quaternions_xyzw)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def quaternion_to_rot6d(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Return the first two matrix rows flattened, matching the existing Wuji convention."""
    matrices = quaternion_to_matrix(quaternions_xyzw)
    return matrices[..., :2, :].reshape(matrices.shape[:-2] + (6,))


def slerp(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise RotationError("SLERP fraction must be in [0, 1]")
    q0 = normalize_quaternions(np.asarray(q0_xyzw, dtype=np.float64))
    q1 = normalize_quaternions(np.asarray(q1_xyzw, dtype=np.float64))
    if q0.shape != (4,) or q1.shape != (4,):
        raise RotationError("SLERP inputs must each be a single quaternion")
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternions(q0 + fraction * (q1 - q0))
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    result = (
        np.sin((1.0 - fraction) * angle) / sin_angle * q0
        + np.sin(fraction * angle) / sin_angle * q1
    )
    return normalize_quaternions(result)
