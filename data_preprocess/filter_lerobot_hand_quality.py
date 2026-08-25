#!/usr/bin/env python3
"""Audit and quarantine LeRobot v2 episodes with inconsistent hand commands.

This postprocessor does not modify samples numerically. It preserves every frame of an
accepted episode and compacts only dataset-level episode/global frame indexes.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd


META_DIR = "meta"
REQUIRED_META_FILES = ("info.json", "modality.json", "episodes.jsonl", "tasks.jsonl")
STATS_FILES = ("stats.json", "relative_stats.json")


@dataclass(frozen=True)
class HandQualityThresholds:
    active_range: float = 0.1
    min_best_lag_correlation: float = 0.6
    correlation_error: float = 0.2
    max_median_offset: float = 0.12
    max_lag_frames: int = 15
    min_correlation_samples: int = 10


@dataclass(frozen=True)
class MappingShiftThresholds:
    session_gap_sec: float = 180.0
    min_session_episodes: int = 3
    min_reliable_episode_fraction: float = 0.6
    min_reliable_correlation: float = 0.8
    absolute_shift: float = 0.25
    mad_scale: float = 6.0


@dataclass(frozen=True)
class HandFeature:
    name: str
    state_column: str
    state_index: int
    action_column: str
    action_index: int


@dataclass
class EpisodeAudit:
    episode_index: int
    tasks: list[str]
    length: int
    passed: bool
    reasons: list[str]
    warnings: list[str]
    joints: list[dict[str, Any]]
    session_id: str | None = None


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, allow_nan=False) + "\n")


def format_episode_path(pattern: str, episode_index: int, chunk_size: int) -> Path:
    return Path(
        pattern.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
    )


def validate_dataset(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    missing = [name for name in REQUIRED_META_FILES if not (root / META_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{root} is missing required metadata files: {missing}")

    info = read_json(root / META_DIR / "info.json")
    modality = read_json(root / META_DIR / "modality.json")
    episodes = read_jsonl(root / META_DIR / "episodes.jsonl")
    if info.get("codebase_version") != "v2.0":
        raise ValueError(f"{root} is not a LeRobot v2.0 dataset")
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError("info.json total_episodes does not match episodes.jsonl")
    return info, modality, episodes


def _modality_column(entry: dict[str, Any], default: str) -> str:
    original_key = entry.get("original_key", default)
    if not isinstance(original_key, str):
        raise ValueError(f"Invalid original_key in modality entry: {entry}")
    return original_key


def discover_hand_features(info: dict[str, Any], modality: dict[str, Any]) -> list[HandFeature]:
    state_modalities = modality.get("state", {})
    action_modalities = modality.get("action", {})
    features = info.get("features", {})
    result: list[HandFeature] = []

    hand_keys = sorted(
        key for key in state_modalities if "hand" in key.lower() and key in action_modalities
    )
    if not hand_keys:
        raise ValueError("modality.json has no matching state/action hand modalities")

    for key in hand_keys:
        state_entry = state_modalities[key]
        action_entry = action_modalities[key]
        state_column = _modality_column(state_entry, "observation.state")
        action_column = _modality_column(action_entry, "action")
        state_start, state_end = int(state_entry["start"]), int(state_entry["end"])
        action_start, action_end = int(action_entry["start"]), int(action_entry["end"])
        if state_end - state_start != action_end - action_start:
            raise ValueError(f"State/action dimensions differ for hand modality {key}")

        state_names = features.get(state_column, {}).get("names")
        action_names = features.get(action_column, {}).get("names")
        for offset in range(state_end - state_start):
            state_index = state_start + offset
            action_index = action_start + offset
            state_name = (
                state_names[state_index]
                if isinstance(state_names, list) and state_index < len(state_names)
                else f"{key}[{offset}]"
            )
            if isinstance(action_names, list) and action_index < len(action_names):
                action_name = action_names[action_index]
                if action_name != state_name:
                    raise ValueError(
                        f"State/action feature mismatch for {key}[{offset}]: "
                        f"{state_name!r} != {action_name!r}"
                    )
            result.append(
                HandFeature(
                    name=state_name,
                    state_column=state_column,
                    state_index=state_index,
                    action_column=action_column,
                    action_index=action_index,
                )
            )
    return result


def _correlation(first: np.ndarray, second: np.ndarray, min_samples: int) -> float | None:
    finite = np.isfinite(first) & np.isfinite(second)
    first = first[finite]
    second = second[finite]
    if len(first) < min_samples or np.std(first) < 1e-8 or np.std(second) < 1e-8:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _best_lag_correlation(
    state: np.ndarray,
    action: np.ndarray,
    max_lag: int,
    min_samples: int,
) -> tuple[float | None, int | None]:
    best_correlation: float | None = None
    best_lag: int | None = None
    for lag in range(max_lag + 1):
        if lag == 0:
            correlation = _correlation(state, action, min_samples)
        else:
            correlation = _correlation(state[lag:], action[:-lag], min_samples)
        if correlation is not None and (best_correlation is None or correlation > best_correlation):
            best_correlation = correlation
            best_lag = lag
    return best_correlation, best_lag


def audit_joint(
    state: np.ndarray,
    action: np.ndarray,
    feature_name: str,
    thresholds: HandQualityThresholds,
) -> tuple[dict[str, Any], list[str], list[str]]:
    finite = np.isfinite(state) & np.isfinite(action)
    finite_count = int(np.count_nonzero(finite))
    if finite_count != len(state):
        metric = {
            "name": feature_name,
            "sample_count": len(state),
            "finite_sample_count": finite_count,
            "state_range_p05_p95": None,
            "action_range_p05_p95": None,
            "same_time_correlation": None,
            "best_lag_correlation": None,
            "best_lag_frames": None,
            "same_time_median_action_state_offset": None,
            "lag_aligned_median_action_state_offset": None,
            "same_time_p95_absolute_error": None,
            "lag_aligned_p95_absolute_error": None,
            "median_action_state_offset": None,
            "p95_absolute_error": None,
            "failures": ["non_finite"],
            "warnings": [],
        }
        return metric, [f"{feature_name}:non_finite"], []

    state_range = float(np.percentile(state, 95) - np.percentile(state, 5))
    action_range = float(np.percentile(action, 95) - np.percentile(action, 5))
    same_time_correlation = _correlation(state, action, thresholds.min_correlation_samples)
    best_correlation, best_lag = _best_lag_correlation(
        state,
        action,
        thresholds.max_lag_frames,
        thresholds.min_correlation_samples,
    )
    same_time_difference = action - state
    same_time_median_offset = float(np.median(same_time_difference))
    same_time_p95_absolute_error = float(np.percentile(np.abs(same_time_difference), 95))

    aligned_lag = best_lag or 0
    if aligned_lag:
        aligned_difference = action[:-aligned_lag] - state[aligned_lag:]
    else:
        aligned_difference = same_time_difference
    aligned_median_offset = float(np.median(aligned_difference))
    aligned_p95_absolute_error = float(np.percentile(np.abs(aligned_difference), 95))

    failures: list[str] = []
    warnings: list[str] = []
    is_active = state_range >= thresholds.active_range and action_range >= thresholds.active_range
    if (
        is_active
        and (best_correlation is None or best_correlation < thresholds.min_best_lag_correlation)
        and aligned_p95_absolute_error > thresholds.correlation_error
    ):
        warnings.append("low_correlation")
    if abs(aligned_median_offset) > thresholds.max_median_offset:
        warnings.append("tracking_offset")

    metric = {
        "name": feature_name,
        "sample_count": len(state),
        "finite_sample_count": finite_count,
        "state_range_p05_p95": state_range,
        "action_range_p05_p95": action_range,
        "same_time_correlation": same_time_correlation,
        "best_lag_correlation": best_correlation,
        "best_lag_frames": best_lag,
        "same_time_median_action_state_offset": same_time_median_offset,
        "lag_aligned_median_action_state_offset": aligned_median_offset,
        "same_time_p95_absolute_error": same_time_p95_absolute_error,
        "lag_aligned_p95_absolute_error": aligned_p95_absolute_error,
        # Backward-compatible aliases now contain lag-aligned metrics.
        "median_action_state_offset": aligned_median_offset,
        "p95_absolute_error": aligned_p95_absolute_error,
        "failures": failures,
        "warnings": warnings,
    }
    reasons = [f"{feature_name}:{failure}" for failure in failures]
    warning_reasons = [f"{feature_name}:{warning}" for warning in warnings]
    return metric, reasons, warning_reasons


def audit_episode(
    parquet_path: Path,
    episode: dict[str, Any],
    hand_features: list[HandFeature],
    thresholds: HandQualityThresholds,
) -> EpisodeAudit:
    columns = sorted(
        {feature.state_column for feature in hand_features}
        | {feature.action_column for feature in hand_features}
    )
    frame = pd.read_parquet(parquet_path, columns=columns)
    expected_length = int(episode.get("length", len(frame)))
    if len(frame) != expected_length:
        raise ValueError(f"{parquet_path} contains {len(frame)} frames, expected {expected_length}")

    arrays = {
        column: np.stack(frame[column].to_numpy()).astype(np.float64, copy=False)
        for column in columns
    }
    metrics: list[dict[str, Any]] = []
    reasons: list[str] = []
    warnings: list[str] = []
    for feature in hand_features:
        metric, joint_reasons, joint_warnings = audit_joint(
            arrays[feature.state_column][:, feature.state_index],
            arrays[feature.action_column][:, feature.action_index],
            feature.name,
            thresholds,
        )
        metrics.append(metric)
        reasons.extend(joint_reasons)
        warnings.extend(joint_warnings)

    return EpisodeAudit(
        episode_index=int(episode["episode_index"]),
        tasks=list(episode.get("tasks", [])),
        length=len(frame),
        passed=not reasons,
        reasons=reasons,
        warnings=warnings,
        joints=metrics,
    )


_SOURCE_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")


def _source_timestamp(source: str | None) -> datetime | None:
    if not source:
        return None
    match = _SOURCE_TIMESTAMP_RE.search(source)
    if match is None:
        return None
    try:
        return datetime.strptime("_".join(match.groups()), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _sessionize_audits(
    audits: list[EpisodeAudit],
    episodes: list[dict[str, Any]],
    thresholds: MappingShiftThresholds,
) -> list[dict[str, Any]]:
    metadata = {int(episode["episode_index"]): episode for episode in episodes}
    sessions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_timestamp: datetime | None = None
    previous_task: tuple[str, ...] | None = None

    for audit in audits:
        episode = metadata[audit.episode_index]
        task_key = tuple(audit.tasks)
        timestamp = _source_timestamp(str(episode.get("source_file", "")))
        gap = (
            None
            if timestamp is None or previous_timestamp is None
            else (timestamp - previous_timestamp).total_seconds()
        )
        starts_new = (
            current is None
            or task_key != previous_task
            or timestamp is None
            or previous_timestamp is None
            or gap is None
            or gap < 0
            or gap > thresholds.session_gap_sec
        )
        if starts_new:
            session_number = len(sessions)
            current = {
                "session_id": f"session_{session_number:04d}",
                "tasks": list(audit.tasks),
                "episode_indices": [],
                "source_start": str(episode.get("source_file", "")),
                "source_end": str(episode.get("source_file", "")),
                "audits": [],
            }
            sessions.append(current)
        current["episode_indices"].append(audit.episode_index)
        current["source_end"] = str(episode.get("source_file", ""))
        current["audits"].append(audit)
        audit.session_id = current["session_id"]
        previous_timestamp = timestamp
        previous_task = task_key
    return sessions


def apply_mapping_shift_filter(
    audits: list[EpisodeAudit],
    episodes: list[dict[str, Any]],
    thresholds: MappingShiftThresholds,
) -> list[dict[str, Any]]:
    """Quarantine sustained session-level mapping changes, not per-episode tracking error."""
    sessions = _sessionize_audits(audits, episodes, thresholds)

    task_joint_values: dict[tuple[tuple[str, ...], str], list[float]] = {}
    task_joint_sessions: dict[tuple[tuple[str, ...], str], set[str]] = {}
    for session in sessions:
        task_key = tuple(session["tasks"])
        for audit in session["audits"]:
            for joint in audit.joints:
                correlation = joint["best_lag_correlation"]
                if correlation is not None and correlation < thresholds.min_reliable_correlation:
                    continue
                offset = joint["lag_aligned_median_action_state_offset"]
                if offset is None:
                    continue
                key = (task_key, joint["name"])
                task_joint_values.setdefault(key, []).append(float(offset))
                task_joint_sessions.setdefault(key, set()).add(session["session_id"])

    baselines: dict[tuple[tuple[str, ...], str], tuple[float, float]] = {}
    for key, values in task_joint_values.items():
        if len(task_joint_sessions[key]) < 2:
            continue
        values_array = np.asarray(values, dtype=np.float64)
        baseline = float(np.median(values_array))
        mad = float(np.median(np.abs(values_array - baseline)))
        baselines[key] = (baseline, mad)

    mapping_shifts: list[dict[str, Any]] = []
    for session in sessions:
        if len(session["audits"]) < thresholds.min_session_episodes:
            continue
        task_key = tuple(session["tasks"])
        session_values: dict[str, list[float]] = {}
        for audit in session["audits"]:
            for joint in audit.joints:
                correlation = joint["best_lag_correlation"]
                if correlation is not None and correlation < thresholds.min_reliable_correlation:
                    continue
                offset = joint["lag_aligned_median_action_state_offset"]
                if offset is not None:
                    session_values.setdefault(joint["name"], []).append(float(offset))

        shifted_joints = []
        for joint_name, values in session_values.items():
            min_reliable_episodes = max(
                thresholds.min_session_episodes,
                math.ceil(len(session["audits"]) * thresholds.min_reliable_episode_fraction),
            )
            if len(values) < min_reliable_episodes:
                continue
            key = (task_key, joint_name)
            if key not in baselines:
                continue
            session_offset = float(np.median(values))
            baseline, mad = baselines[key]
            limit = max(thresholds.absolute_shift, thresholds.mad_scale * 1.4826 * mad)
            delta = session_offset - baseline
            if abs(delta) > limit:
                shifted_joints.append(
                    {
                        "name": joint_name,
                        "session_offset": session_offset,
                        "task_baseline_offset": baseline,
                        "task_baseline_mad": mad,
                        "delta": delta,
                        "threshold": limit,
                    }
                )

        if shifted_joints:
            mapping_shifts.append(
                {
                    "session_id": session["session_id"],
                    "tasks": session["tasks"],
                    "episode_indices": session["episode_indices"],
                    "source_start": session["source_start"],
                    "source_end": session["source_end"],
                    "shifted_joints": shifted_joints,
                }
            )
            reason = f"session_mapping_shift:{session['session_id']}"
            warning = f"session_mapping_shift:{session['session_id']}"
            for audit in session["audits"]:
                audit.reasons.append(reason)
                audit.warnings.append(warning)
                audit.passed = False
    return mapping_shifts


def audit_dataset(
    root: Path,
    thresholds: HandQualityThresholds,
    mapping_thresholds: MappingShiftThresholds | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[EpisodeAudit],
    list[dict[str, Any]],
]:
    info, modality, episodes = validate_dataset(root)
    hand_features = discover_hand_features(info, modality)
    chunk_size = int(info["chunks_size"])
    data_pattern = str(info["data_path"])
    audits = []
    for number, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        path = root / format_episode_path(data_pattern, episode_index, chunk_size)
        audits.append(audit_episode(path, episode, hand_features, thresholds))
        if number % 100 == 0 or number == len(episodes):
            print(f"Audited {number}/{len(episodes)} episodes", flush=True)
    mapping_shifts = apply_mapping_shift_filter(
        audits,
        episodes,
        mapping_thresholds or MappingShiftThresholds(),
    )
    return info, modality, episodes, audits, mapping_shifts


def build_summary(
    audits: list[EpisodeAudit],
    thresholds: HandQualityThresholds,
    mapping_shifts: list[dict[str, Any]] | None = None,
    mapping_thresholds: MappingShiftThresholds | None = None,
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    task_counts: dict[str, dict[str, int]] = {}
    passed_frames = 0
    rejected_frames = 0
    for audit in audits:
        reason_types = {
            "session_mapping_shift"
            if reason.startswith("session_mapping_shift:")
            else reason.rsplit(":", 1)[-1]
            for reason in audit.reasons
        }
        for reason in reason_types:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        warning_types = {
            "session_mapping_shift"
            if warning.startswith("session_mapping_shift:")
            else warning.rsplit(":", 1)[-1]
            for warning in audit.warnings
        }
        for warning in warning_types:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        if audit.passed:
            passed_frames += audit.length
        else:
            rejected_frames += audit.length
        for task in audit.tasks or ["<unknown>"]:
            counts = task_counts.setdefault(task, {"total": 0, "passed": 0, "rejected": 0})
            counts["total"] += 1
            counts["passed" if audit.passed else "rejected"] += 1

    passed_count = sum(audit.passed for audit in audits)
    return {
        "thresholds": asdict(thresholds),
        "mapping_shift_thresholds": asdict(mapping_thresholds or MappingShiftThresholds()),
        "total_episodes": len(audits),
        "passed_episodes": passed_count,
        "rejected_episodes": len(audits) - passed_count,
        "passed_frames": passed_frames,
        "rejected_frames": rejected_frames,
        "rejected_episode_counts_by_reason": dict(sorted(reason_counts.items())),
        "warning_episode_counts_by_type": dict(sorted(warning_counts.items())),
        "mapping_shift_sessions": mapping_shifts or [],
        "episode_counts_by_task": task_counts,
        "stats_files_invalidated": list(STATS_FILES),
    }


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_sidecars(
    source_root: Path,
    destination_root: Path,
    old_episode_index: int,
    new_episode_index: int,
    chunk_size: int,
) -> int:
    copied_videos = 0
    old_chunk = old_episode_index // chunk_size
    new_chunk = new_episode_index // chunk_size
    old_stem = f"episode_{old_episode_index:06d}"
    new_stem = f"episode_{new_episode_index:06d}"
    for directory in ("videos", "masks"):
        source_chunk = source_root / directory / f"chunk-{old_chunk:03d}"
        if not source_chunk.exists():
            continue
        for source_path in source_chunk.rglob(f"{old_stem}.*"):
            relative = source_path.relative_to(source_chunk)
            renamed = relative.with_name(source_path.name.replace(old_stem, new_stem, 1))
            destination = destination_root / directory / f"chunk-{new_chunk:03d}" / renamed
            _link_or_copy(source_path, destination)
            if directory == "videos":
                copied_videos += 1
    return copied_videos


def _rewrite_episode(
    source: Path,
    destination: Path,
    new_episode_index: int,
    global_index: int,
) -> int:
    frame = pd.read_parquet(source)
    if "episode_index" in frame:
        frame["episode_index"] = new_episode_index
    if "index" in frame:
        frame["index"] = range(global_index, global_index + len(frame))
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return len(frame)


def _filtered_info(
    source_info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    total_videos: int,
) -> dict[str, Any]:
    result = copy.deepcopy(source_info)
    chunk_size = int(result["chunks_size"])
    result["total_episodes"] = total_episodes
    result["total_frames"] = total_frames
    result["total_videos"] = total_videos
    result["total_chunks"] = max(1, math.ceil(total_episodes / chunk_size))
    result["splits"] = {"train": f"0:{total_episodes}"}
    return result


def _export_partition(
    source_root: Path,
    destination_root: Path,
    info: dict[str, Any],
    modality: dict[str, Any],
    episodes: list[dict[str, Any]],
    selected_audits: list[EpisodeAudit],
) -> None:
    audit_by_index = {audit.episode_index: audit for audit in selected_audits}
    selected_episodes = [
        episode for episode in episodes if int(episode["episode_index"]) in audit_by_index
    ]
    chunk_size = int(info["chunks_size"])
    data_pattern = str(info["data_path"])
    rewritten_metadata = []
    total_frames = 0
    total_videos = 0

    for new_index, episode in enumerate(selected_episodes):
        old_index = int(episode["episode_index"])
        source_path = source_root / format_episode_path(data_pattern, old_index, chunk_size)
        destination_path = destination_root / format_episode_path(
            data_pattern, new_index, chunk_size
        )
        length = _rewrite_episode(source_path, destination_path, new_index, total_frames)
        rewritten = copy.deepcopy(episode)
        rewritten["episode_index"] = new_index
        rewritten["length"] = length
        rewritten["hand_quality_filter"] = {
            "source_episode_index": old_index,
            "passed": audit_by_index[old_index].passed,
        }
        rewritten_metadata.append(rewritten)
        total_videos += _copy_sidecars(
            source_root,
            destination_root,
            old_index,
            new_index,
            chunk_size,
        )
        total_frames += length

    write_json(destination_root / META_DIR / "modality.json", modality)
    shutil.copy2(
        source_root / META_DIR / "tasks.jsonl", destination_root / META_DIR / "tasks.jsonl"
    )
    write_jsonl(destination_root / META_DIR / "episodes.jsonl", rewritten_metadata)
    write_json(
        destination_root / META_DIR / "info.json",
        _filtered_info(info, len(rewritten_metadata), total_frames, total_videos),
    )


def export_filtered_dataset(
    source_root: Path,
    output_root: Path,
    info: dict[str, Any],
    modality: dict[str, Any],
    episodes: list[dict[str, Any]],
    audits: list[EpisodeAudit],
    thresholds: HandQualityThresholds,
    overwrite: bool,
    mapping_shifts: list[dict[str, Any]] | None = None,
    mapping_thresholds: MappingShiftThresholds | None = None,
) -> dict[str, Any]:
    resolved_source = source_root.resolve()
    resolved_output = output_root.resolve()
    if (
        resolved_source == resolved_output
        or resolved_output.is_relative_to(resolved_source)
        or resolved_source.is_relative_to(resolved_output)
    ):
        raise ValueError("Input and output directories must not be equal or nested")
    _prepare_output_dir(output_root, overwrite)
    accepted = [audit for audit in audits if audit.passed]
    rejected = [audit for audit in audits if not audit.passed]

    _export_partition(source_root, output_root, info, modality, episodes, accepted)
    quarantine_root = output_root / "quarantine"
    _export_partition(source_root, quarantine_root, info, modality, episodes, rejected)

    summary = build_summary(audits, thresholds, mapping_shifts, mapping_thresholds)
    write_jsonl(output_root / META_DIR / "hand_quality.jsonl", [asdict(a) for a in audits])
    write_json(output_root / META_DIR / "hand_quality_summary.json", summary)
    write_json(
        output_root / "quarantine" / "filter_report.json",
        {
            "summary": summary,
            "rejected_episodes": [asdict(audit) for audit in rejected],
        },
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--active-range", type=float, default=0.1)
    parser.add_argument("--min-best-lag-correlation", type=float, default=0.6)
    parser.add_argument("--correlation-error", type=float, default=0.2)
    parser.add_argument("--max-median-offset", type=float, default=0.12)
    parser.add_argument("--max-lag-frames", type=int, default=15)
    parser.add_argument("--min-correlation-samples", type=int, default=10)
    parser.add_argument("--session-gap-sec", type=float, default=180.0)
    parser.add_argument("--min-session-episodes", type=int, default=3)
    parser.add_argument("--min-reliable-episode-fraction", type=float, default=0.6)
    parser.add_argument("--min-reliable-correlation", type=float, default=0.8)
    parser.add_argument("--mapping-shift-threshold", type=float, default=0.25)
    parser.add_argument("--mapping-shift-mad-scale", type=float, default=6.0)
    args = parser.parse_args(argv)
    if not args.audit_only and args.output_dir is None:
        parser.error("--output-dir is required unless --audit-only is used")
    if args.audit_only and args.output_dir is not None:
        parser.error("--output-dir cannot be used with --audit-only; use --report-dir")
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    thresholds = HandQualityThresholds(
        active_range=args.active_range,
        min_best_lag_correlation=args.min_best_lag_correlation,
        correlation_error=args.correlation_error,
        max_median_offset=args.max_median_offset,
        max_lag_frames=args.max_lag_frames,
        min_correlation_samples=args.min_correlation_samples,
    )
    mapping_thresholds = MappingShiftThresholds(
        session_gap_sec=args.session_gap_sec,
        min_session_episodes=args.min_session_episodes,
        min_reliable_episode_fraction=args.min_reliable_episode_fraction,
        min_reliable_correlation=args.min_reliable_correlation,
        absolute_shift=args.mapping_shift_threshold,
        mad_scale=args.mapping_shift_mad_scale,
    )
    if thresholds.active_range < 0:
        raise ValueError("--active-range must be non-negative")
    if not -1 <= thresholds.min_best_lag_correlation <= 1:
        raise ValueError("--min-best-lag-correlation must be between -1 and 1")
    if thresholds.correlation_error < 0 or thresholds.max_median_offset < 0:
        raise ValueError("Error and offset thresholds must be non-negative")
    if thresholds.max_lag_frames < 0 or thresholds.min_correlation_samples < 2:
        raise ValueError("Lag must be non-negative and minimum samples must be at least 2")
    if mapping_thresholds.session_gap_sec <= 0 or mapping_thresholds.min_session_episodes < 2:
        raise ValueError("Session gap must be positive and minimum session episodes at least 2")
    if not 0 <= mapping_thresholds.min_reliable_correlation <= 1:
        raise ValueError("--min-reliable-correlation must be between 0 and 1")
    if not 0 < mapping_thresholds.min_reliable_episode_fraction <= 1:
        raise ValueError("--min-reliable-episode-fraction must be in (0, 1]")
    if mapping_thresholds.absolute_shift < 0 or mapping_thresholds.mad_scale < 0:
        raise ValueError("Mapping shift thresholds must be non-negative")
    source_root = args.input_dir.expanduser().resolve()
    info, modality, episodes, audits, mapping_shifts = audit_dataset(
        source_root,
        thresholds,
        mapping_thresholds,
    )
    summary = build_summary(audits, thresholds, mapping_shifts, mapping_thresholds)

    if args.audit_only:
        if args.report_dir is not None:
            report_dir = args.report_dir.expanduser().resolve()
            report_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(report_dir / "hand_quality.jsonl", [asdict(a) for a in audits])
            write_json(report_dir / "hand_quality_summary.json", summary)
    else:
        summary = export_filtered_dataset(
            source_root,
            args.output_dir.expanduser().resolve(),
            info,
            modality,
            episodes,
            audits,
            thresholds,
            args.overwrite,
            mapping_shifts,
            mapping_thresholds,
        )

    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
