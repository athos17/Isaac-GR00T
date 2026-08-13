from __future__ import annotations

from typing import Any

import numpy as np

from robot_data_pipeline.models import AlignedEpisodeData


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def audit_aligned_episode(aligned: AlignedEpisodeData, *, output_fps: float) -> dict[str, Any]:
    frame_count = len(aligned.timestamps)
    reasons = []
    if any(len(images) != frame_count for images in aligned.images.values()):
        reasons.append("frame_count_mismatch")
    expected_timestamps = np.arange(frame_count, dtype=np.float64) / output_fps
    if not np.array_equal(aligned.timestamps, expected_timestamps):
        reasons.append("output_timestamp_mismatch")
    if not np.all(np.isfinite(aligned.state)) or not np.all(np.isfinite(aligned.action)):
        reasons.append("non_finite_payload")
    stream_metrics = {}
    future_action_violations = 0
    for key, diagnostic in aligned.diagnostics.items():
        metrics: dict[str, Any] = {}
        if "signed_skew_ns" in diagnostic:
            metrics["absolute_skew_sec"] = _distribution(
                np.abs(diagnostic["signed_skew_ns"]) * 1e-9
            )
        if "bracket_gap_ns" in diagnostic:
            metrics["bracket_gap_sec"] = _distribution(diagnostic["bracket_gap_ns"] * 1e-9)
        if "action_age_ns" in diagnostic:
            ages = diagnostic["action_age_ns"]
            future_action_violations += int(np.count_nonzero(ages < 0))
            metrics["action_age_sec"] = _distribution(ages * 1e-9)
        if "frame_reused" in diagnostic:
            reused = int(np.count_nonzero(diagnostic["frame_reused"]))
            metrics["reused_frame_count"] = reused
            metrics["reused_frame_ratio"] = reused / frame_count if frame_count else 0.0
        stream_metrics[key] = metrics
    if future_action_violations:
        reasons.append("future_action_violation")
    return {
        "status": "REJECT" if reasons else "PASS",
        "reject_reasons": sorted(set(reasons)),
        "frame_count": frame_count,
        "state_dimension": aligned.state.shape[1],
        "action_dimension": aligned.action.shape[1],
        "future_action_violation_count": future_action_violations,
        "streams": stream_metrics,
    }
