#!/usr/bin/env python
"""Quality report generation for converted datasets.

Generates quality_report.json with per-episode metrics to help identify:
- Episodes with poor time alignment (high skew)
- Episodes with low motion content
- Episodes with camera issues (dropped frames, duplicates)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np


@dataclass
class EpisodeQualityMetrics:
    """Quality metrics for a single episode."""

    episode_index: int
    source: str

    # Length metrics
    original_length: int
    final_length: int
    idle_prefix_frames: int
    idle_suffix_frames: int
    idle_prefix_seconds: float
    idle_suffix_seconds: float

    # Time alignment metrics
    max_skew_sec: float
    mean_skew_sec: float | None
    p90_skew_sec: float | None
    skew_warning_threshold: float

    # Motion metrics
    mean_eef_velocity: float
    max_eef_velocity: float
    mean_action_state_diff: float

    # Camera metrics
    camera_fps: float
    camera_frame_count: int
    duplicate_frame_ratio: float | None

    # Filter status
    passed_filter: bool
    filter_reasons: list[str]


@dataclass
class DatasetQualitySummary:
    """Summary statistics across the entire dataset."""

    total_episodes: int
    passed_episodes: int
    failed_episodes: int
    total_frames: int
    total_frames_after_trimming: int
    trimmed_frames: int

    # Aggregate metrics
    mean_episode_length: float
    median_episode_length: float
    mean_idle_prefix_sec: float
    mean_skew_sec: float
    max_skew_sec: float
    episodes_exceeding_skew_threshold: int

    # Motion statistics
    mean_eef_velocity_across_dataset: float
    max_eef_velocity_across_dataset: float


def compute_skew_statistics(skews: list[float] | None) -> tuple[float | None, float | None]:
    """Compute mean and p90 skew from a list of per-frame skews.

    Args:
        skews: List of skew values for each frame in the episode, or None if not tracked

    Returns:
        Tuple of (mean_skew, p90_skew), or (None, None) if skews is None
    """
    if skews is None or len(skews) == 0:
        return None, None
    skews_arr = np.array(skews)
    return float(np.mean(skews_arr)), float(np.percentile(skews_arr, 90))


def estimate_duplicate_frame_ratio(video_frames: list[np.ndarray]) -> float:
    """Estimate the ratio of duplicate frames in a video by comparing consecutive frames.

    Args:
        video_frames: List of frames (H, W, 3) uint8 arrays

    Returns:
        Ratio of duplicate frames (0.0 to 1.0)
    """
    if len(video_frames) < 2:
        return 0.0

    # Sample up to 100 random consecutive pairs to avoid processing entire video
    sample_size = min(100, len(video_frames) - 1)
    indices = np.linspace(0, len(video_frames) - 2, sample_size, dtype=int)

    duplicates = 0
    for i in indices:
        frame1 = video_frames[i]
        frame2 = video_frames[i + 1]
        if np.array_equal(frame1, frame2):
            duplicates += 1

    return duplicates / sample_size


def create_episode_quality_metrics(
    episode_index: int,
    source: str,
    original_length: int,
    final_length: int,
    idle_prefix_frames: int,
    idle_suffix_frames: int,
    fps: float,
    max_skew_sec: float,
    skew_warning_threshold: float,
    mean_eef_velocity: float,
    max_eef_velocity: float,
    mean_action_state_diff: float,
    camera_frame_count: int,
    head_rgb_frames: list[np.ndarray] | None = None,
    per_frame_skews: list[float] | None = None,
) -> EpisodeQualityMetrics:
    """Create quality metrics for an episode.

    Args:
        episode_index: Episode index
        source: Source bag path
        original_length: Length before any trimming
        final_length: Length after motion detection trimming
        idle_prefix_frames: Frames trimmed from start
        idle_suffix_frames: Frames trimmed from end
        fps: Frames per second
        max_skew_sec: Maximum time skew observed
        skew_warning_threshold: Threshold for skew warnings
        mean_eef_velocity: Mean EEF velocity during motion
        max_eef_velocity: Max EEF velocity during motion
        mean_action_state_diff: Mean action-state difference
        camera_frame_count: Number of camera frames captured
        head_rgb_frames: Optional list of head RGB frames for duplicate detection
        per_frame_skews: Optional list of per-frame skew values

    Returns:
        EpisodeQualityMetrics instance
    """
    mean_skew, p90_skew = compute_skew_statistics(per_frame_skews)

    duplicate_ratio = None
    if head_rgb_frames is not None and len(head_rgb_frames) > 0:
        duplicate_ratio = estimate_duplicate_frame_ratio(head_rgb_frames)

    # Determine filter status
    filter_reasons = []
    if max_skew_sec > skew_warning_threshold:
        filter_reasons.append(f"max_skew ({max_skew_sec:.4f}s) > threshold ({skew_warning_threshold}s)")
    if final_length < 30:
        filter_reasons.append(f"episode too short ({final_length} frames)")
    if mean_eef_velocity < 0.001:
        filter_reasons.append(f"very low motion (mean_vel={mean_eef_velocity:.6f} m/s)")

    return EpisodeQualityMetrics(
        episode_index=episode_index,
        source=source,
        original_length=original_length,
        final_length=final_length,
        idle_prefix_frames=idle_prefix_frames,
        idle_suffix_frames=idle_suffix_frames,
        idle_prefix_seconds=idle_prefix_frames / fps,
        idle_suffix_seconds=idle_suffix_frames / fps,
        max_skew_sec=max_skew_sec,
        mean_skew_sec=mean_skew,
        p90_skew_sec=p90_skew,
        skew_warning_threshold=skew_warning_threshold,
        mean_eef_velocity=mean_eef_velocity,
        max_eef_velocity=max_eef_velocity,
        mean_action_state_diff=mean_action_state_diff,
        camera_fps=fps,
        camera_frame_count=camera_frame_count,
        duplicate_frame_ratio=duplicate_ratio,
        passed_filter=len(filter_reasons) == 0,
        filter_reasons=filter_reasons,
    )


def create_dataset_summary(metrics: list[EpisodeQualityMetrics]) -> DatasetQualitySummary:
    """Create summary statistics for the entire dataset.

    Args:
        metrics: List of per-episode quality metrics

    Returns:
        DatasetQualitySummary instance
    """
    if len(metrics) == 0:
        return DatasetQualitySummary(
            total_episodes=0,
            passed_episodes=0,
            failed_episodes=0,
            total_frames=0,
            total_frames_after_trimming=0,
            trimmed_frames=0,
            mean_episode_length=0.0,
            median_episode_length=0.0,
            mean_idle_prefix_sec=0.0,
            mean_skew_sec=0.0,
            max_skew_sec=0.0,
            episodes_exceeding_skew_threshold=0,
            mean_eef_velocity_across_dataset=0.0,
            max_eef_velocity_across_dataset=0.0,
        )

    passed = [m for m in metrics if m.passed_filter]
    total_frames = sum(m.original_length for m in metrics)
    total_frames_after = sum(m.final_length for m in metrics)
    lengths = [m.final_length for m in metrics]
    idle_prefixes = [m.idle_prefix_seconds for m in metrics]
    max_skews = [m.max_skew_sec for m in metrics]
    eef_vels = [m.mean_eef_velocity for m in metrics]
    max_eef_vels = [m.max_eef_velocity for m in metrics]

    exceeding_threshold = sum(
        1 for m in metrics if m.max_skew_sec > m.skew_warning_threshold
    )

    return DatasetQualitySummary(
        total_episodes=len(metrics),
        passed_episodes=len(passed),
        failed_episodes=len(metrics) - len(passed),
        total_frames=total_frames,
        total_frames_after_trimming=total_frames_after,
        trimmed_frames=total_frames - total_frames_after,
        mean_episode_length=float(np.mean(lengths)),
        median_episode_length=float(np.median(lengths)),
        mean_idle_prefix_sec=float(np.mean(idle_prefixes)),
        mean_skew_sec=float(np.mean(max_skews)),
        max_skew_sec=float(np.max(max_skews)),
        episodes_exceeding_skew_threshold=exceeding_threshold,
        mean_eef_velocity_across_dataset=float(np.mean(eef_vels)),
        max_eef_velocity_across_dataset=float(np.max(max_eef_vels)),
    )


def write_quality_report(
    output_dir: Path,
    metrics: list[EpisodeQualityMetrics],
    summary: DatasetQualitySummary,
) -> None:
    """Write quality report to JSON files.

    Args:
        output_dir: Dataset output directory
        metrics: List of per-episode metrics
        summary: Dataset summary statistics
    """
    report_dir = output_dir / "meta"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict and handle numpy types
    def to_json_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_json_serializable(item) for item in obj]
        return obj

    # Write detailed per-episode report
    quality_report_path = report_dir / "quality_report.json"
    with quality_report_path.open("w", encoding="utf-8") as f:
        report_data = {
            "episodes": [to_json_serializable(asdict(m)) for m in metrics],
            "summary": to_json_serializable(asdict(summary)),
        }
        json.dump(report_data, f, indent=2)

    print(f"  ✓ Wrote {quality_report_path}")

    # Write human-readable summary
    summary_path = report_dir / "quality_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Dataset Quality Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Episodes: {summary.total_episodes}\n")
        f.write(f"Passed Filter: {summary.passed_episodes}\n")
        f.write(f"Failed Filter: {summary.failed_episodes}\n")
        f.write(f"Pass Rate: {100 * summary.passed_episodes / summary.total_episodes:.1f}%\n\n")

        f.write(f"Total Frames: {summary.total_frames:,}\n")
        f.write(f"After Trimming: {summary.total_frames_after_trimming:,}\n")
        f.write(f"Trimmed Frames: {summary.trimmed_frames:,} ({100 * summary.trimmed_frames / summary.total_frames:.1f}%)\n\n")

        f.write(f"Mean Episode Length: {summary.mean_episode_length:.1f} frames\n")
        f.write(f"Median Episode Length: {summary.median_episode_length:.1f} frames\n")
        f.write(f"Mean Idle Prefix: {summary.mean_idle_prefix_sec:.2f} seconds\n\n")

        f.write(f"Mean Time Skew: {summary.mean_skew_sec:.4f} seconds\n")
        f.write(f"Max Time Skew: {summary.max_skew_sec:.4f} seconds\n")
        f.write(f"Episodes Exceeding Skew Threshold: {summary.episodes_exceeding_skew_threshold}\n\n")

        f.write(f"Mean EEF Velocity: {summary.mean_eef_velocity_across_dataset:.4f} m/s\n")
        f.write(f"Max EEF Velocity: {summary.max_eef_velocity_across_dataset:.4f} m/s\n\n")

        if summary.failed_episodes > 0:
            f.write("Failed Episodes:\n")
            f.write("-" * 60 + "\n")
            for m in metrics:
                if not m.passed_filter:
                    f.write(f"Episode {m.episode_index}: {', '.join(m.filter_reasons)}\n")

    print(f"  ✓ Wrote {summary_path}")
