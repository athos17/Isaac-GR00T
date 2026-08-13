from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from robot_data_pipeline.catalog import build_roster, roster_to_dict
from robot_data_pipeline.config import ConfigError
from robot_data_pipeline.export.lerobot_v2 import (
    write_episode,
    write_metadata,
    write_rejection_metadata,
)
from robot_data_pipeline.export.reports import write_alignment_diagnostics
from robot_data_pipeline.io.rosbag2 import RosbagsReader
from robot_data_pipeline.models import EpisodeAudit, JobConfig
from robot_data_pipeline.processing.activity import detect_activity
from robot_data_pipeline.processing.canonicalize import CanonicalizationError, canonicalize_messages
from robot_data_pipeline.processing.filters import filter_state_streams
from robot_data_pipeline.processing.synchronize import SynchronizationError, synchronize_episode
from robot_data_pipeline.quality.aligned import audit_aligned_episode
from robot_data_pipeline.quality.lag import audit_action_state_lag
from robot_data_pipeline.quality.raw import audit_episode
from robot_data_pipeline.quality.signal import audit_active_interval_gaps


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    result = {}
    for package in ("numpy", "scipy", "opencv-python-headless", "pyarrow", "rosbags", "PyYAML"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _prepare_target(path: Path, overwrite: bool) -> Path:
    if path.exists():
        if not overwrite:
            raise ConfigError(f"output path already exists: {path}")
        marker = path / "meta" / "pipeline_manifest.json"
        if not marker.is_file():
            raise ConfigError(f"refusing to overwrite output not created by this pipeline: {path}")
    temporary = path.parent / f".{path.name}.robot-data-pipeline-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    return temporary


def _publish_outputs(outputs, temporary_outputs: dict[str, Path]) -> None:
    backups: dict[str, Path | None] = {}
    published: set[str] = set()
    try:
        for output in outputs:
            backup = None
            if output.path.exists():
                backup = output.path.parent / f".{output.path.name}.backup-{uuid.uuid4().hex}"
                os.replace(output.path, backup)
            backups[output.action_space] = backup
        for output in outputs:
            os.replace(temporary_outputs[output.action_space], output.path)
            published.add(output.action_space)
    except Exception:
        for output in reversed(outputs):
            if output.action_space in published and output.path.exists():
                shutil.rmtree(output.path)
        for output in reversed(outputs):
            backup = backups.get(output.action_space)
            if backup is not None and backup.exists():
                os.replace(backup, output.path)
        raise
    for backup in backups.values():
        if backup is not None:
            shutil.rmtree(backup)


def _write_quality(
    output: Path,
    raw_reports: list[EpisodeAudit],
    aligned_reports: list[dict],
    rejected_reports: list[dict],
) -> None:
    quality = output / "quality"
    quality.mkdir(parents=True, exist_ok=True)

    def write_jsonl(name: str, values: list[dict]) -> None:
        with (quality / name).open("w", encoding="utf-8") as file:
            for value in values:
                file.write(json.dumps(value, sort_keys=True) + "\n")

    write_jsonl("raw_episode_reports.jsonl", [asdict(report) for report in raw_reports])
    write_jsonl("episode_reports.jsonl", aligned_reports)
    write_jsonl("rejected_episodes.jsonl", rejected_reports)
    reason_counts: dict[str, int] = {}
    for report in rejected_reports:
        for reason in report["reject_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "input_episode_count": len(raw_reports),
        "pass_count": len(aligned_reports),
        "reject_count": len(rejected_reports),
        "reject_reasons": dict(sorted(reason_counts.items())),
    }
    (quality / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rejection(
    source_episode,
    reason: str | list[str] | tuple[str, ...],
    stage: str,
    error: str | None = None,
    details: dict | None = None,
) -> dict:
    result = {
        "roster_index": source_episode.roster_index,
        "task_id": source_episode.task_id,
        "source_file": str(source_episode.bag_path),
        "reject_reasons": [reason] if isinstance(reason, str) else list(reason),
        "stage": stage,
    }
    if error is not None:
        result["error"] = error
    if details:
        result["details"] = details
    return result


def _lag_audits(canonical, activity) -> dict:
    result = {}
    pairs = (
        ("arm.left", "action.left_arm_joint", "state.left_arm_joint", "joint"),
        ("arm.right", "action.right_arm_joint", "state.right_arm_joint", "joint"),
        ("hand.left", "action.left_hand_joint", "state.left_hand_joint", "joint"),
        ("hand.right", "action.right_hand_joint", "state.right_hand_joint", "joint"),
        ("eef.left", "action.left_eef", "state.left_eef", "pose"),
        ("eef.right", "action.right_eef", "state.right_eef", "pose"),
    )
    for name, action_key, state_key, kind in pairs:
        action = canonical.streams[action_key]
        state = canonical.streams[state_key]
        action_selected = (action.timestamps_ns >= activity.active_start_ns) & (
            action.timestamps_ns <= activity.active_end_ns
        )
        state_selected = (state.timestamps_ns >= activity.active_start_ns) & (
            state.timestamps_ns <= activity.active_end_ns
        )
        action_values = action.values if kind == "joint" else action.translations
        state_values = state.values if kind == "joint" else state.translations
        result[name] = asdict(
            audit_action_state_lag(
                action.timestamps_ns[action_selected],
                action_values[action_selected],
                state.timestamps_ns[state_selected],
                state_values[state_selected],
            )
        )
    return result


def _remove_episode_artifacts(output: Path, episode_index: int, chunks_size: int = 1000) -> None:
    chunk = episode_index // chunks_size
    parquet = output / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    parquet.unlink(missing_ok=True)
    video_root = output / "videos" / f"chunk-{chunk:03d}"
    if video_root.is_dir():
        for video in video_root.glob(f"*/episode_{episode_index:06d}.mp4"):
            video.unlink()
    diagnostics = (
        output
        / "quality"
        / "alignment"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.json"
    )
    diagnostics.unlink(missing_ok=True)


def _prepare_episode(job: JobConfig, source_episode) -> dict:
    reader = RosbagsReader()
    raw_report = audit_episode(source_episode, job.profile, reader)
    result = {"source": source_episode, "raw": raw_report}
    if raw_report.status == "REJECT":
        result["rejection"] = _rejection(source_episode, raw_report.reject_reasons, "raw")
        return result
    try:
        canonical = canonicalize_messages(
            reader.messages(source_episode, job.profile.streams), job.profile
        )
        activity = detect_activity(
            canonical,
            job.profile,
            padding_before_sec=job.manifest.processing.activity_padding_before_sec,
            padding_after_sec=job.manifest.processing.activity_padding_after_sec,
        )
        if activity is None:
            raise SynchronizationError(
                "no_valid_motion",
                "no measured-state activity found",
                details={
                    "streams": list(job.profile.activity_detection.groups),
                    "eef_velocity_threshold": (
                        job.profile.activity_detection.eef_velocity_threshold
                    ),
                    "joint_velocity_threshold": (
                        job.profile.activity_detection.joint_velocity_threshold
                    ),
                    "window_sec": job.profile.activity_detection.window_sec,
                },
            )
        active_gap_violations = audit_active_interval_gaps(canonical, job.profile, activity)
        if active_gap_violations:
            first = active_gap_violations[0]
            raise SynchronizationError(
                "raw_gap_exceeded",
                f"active interval gap exceeded for {first['stream']}: "
                f"{first['interval_sec']:.6f}s > {first['threshold_sec']:.6f}s",
                details=first,
            )
        lag_audits = _lag_audits(canonical, activity)
        filtered, filter_manifest = filter_state_streams(
            canonical,
            job.profile,
            padded_start_ns=activity.padded_start_ns,
            padded_end_ns=activity.padded_end_ns,
        )
        result.update(
            {
                "filtered": filtered,
                "activity": activity,
                "lag_audits": lag_audits,
                "filter_manifest": filter_manifest,
            }
        )
    except (ValueError, RuntimeError) as exc:
        structured = isinstance(exc, (CanonicalizationError, SynchronizationError))
        reason = exc.reason if structured else "processing_failure"
        details = exc.details if structured else None
        result["rejection"] = _rejection(source_episode, reason, "processing", str(exc), details)
    return result


def _prepared_episodes(job: JobConfig, episodes) -> object:
    workers = job.manifest.processing.num_workers
    if workers == 1:
        for episode in episodes:
            yield _prepare_episode(job, episode)
        return
    episode_iterator = iter(episodes)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rosbag-pipeline") as executor:
        pending: list[Future] = []
        for _ in range(workers):
            try:
                pending.append(executor.submit(_prepare_episode, job, next(episode_iterator)))
            except StopIteration:
                break
        while pending:
            future = pending.pop(0)
            yield future.result()
            try:
                pending.append(executor.submit(_prepare_episode, job, next(episode_iterator)))
            except StopIteration:
                pass


def audit_processing_roster(job: JobConfig, roster, *, episode_indices=None) -> list[dict]:
    selected = set(episode_indices) if episode_indices is not None else None
    episodes = [
        episode
        for episode in roster.episodes
        if selected is None or episode.roster_index in selected
    ]
    reports = []
    for prepared in _prepared_episodes(job, episodes):
        source = prepared["source"]
        report = {
            "roster_index": source.roster_index,
            "task_id": source.task_id,
            "source_file": str(source.bag_path),
            "raw": asdict(prepared["raw"]),
            "outputs": {},
        }
        if "rejection" in prepared:
            report.update(
                {
                    "status": "REJECT",
                    "reject_reasons": prepared["rejection"]["reject_reasons"],
                    "stage": prepared["rejection"]["stage"],
                }
            )
            if "error" in prepared["rejection"]:
                report["error"] = prepared["rejection"]["error"]
            if "details" in prepared["rejection"]:
                report["details"] = prepared["rejection"]["details"]
            reports.append(report)
            continue
        report["activity"] = asdict(prepared["activity"])
        report["filtering"] = prepared["filter_manifest"]
        report["lag_audit"] = prepared["lag_audits"]
        output_reasons = []
        for output in job.manifest.outputs:
            try:
                aligned = synchronize_episode(
                    prepared["filtered"],
                    job.profile,
                    prepared["activity"],
                    action_space=output.action_space,
                    output_fps=job.manifest.processing.output_fps,
                    minimum_output_frames=job.manifest.processing.minimum_output_frames,
                )
                aligned_qa = audit_aligned_episode(
                    aligned, output_fps=job.manifest.processing.output_fps
                )
                report["outputs"][output.action_space] = aligned_qa
                output_reasons.extend(aligned_qa["reject_reasons"])
            except (ValueError, RuntimeError) as exc:
                reason = (
                    exc.reason if isinstance(exc, SynchronizationError) else "processing_failure"
                )
                report["outputs"][output.action_space] = {
                    "status": "REJECT",
                    "reject_reasons": [reason],
                    "error": str(exc),
                }
                if isinstance(exc, SynchronizationError) and exc.details:
                    report["outputs"][output.action_space]["details"] = exc.details
                output_reasons.append(reason)
        report["reject_reasons"] = sorted(set(output_reasons))
        report["status"] = "REJECT" if output_reasons else "PASS"
        report["stage"] = "aligned"
        reports.append(report)
    return reports


def convert_job(job: JobConfig, *, overwrite: bool = False) -> dict:
    roster = build_roster(job)
    temporary_outputs = {}
    try:
        for output in job.manifest.outputs:
            temporary_outputs[output.action_space] = _prepare_target(output.path, overwrite)
    except Exception:
        for temporary in temporary_outputs.values():
            shutil.rmtree(temporary)
        raise
    states = {
        output.action_space: {
            "episodes": [],
            "aligned_reports": [],
            "rejected": [],
            "raw": [],
            "total_frames": 0,
            "video_shapes": None,
            "filter_manifest": {},
        }
        for output in job.manifest.outputs
    }
    try:
        for prepared in _prepared_episodes(job, roster.episodes):
            source_episode = prepared["source"]
            raw_report = prepared["raw"]
            for output_state in states.values():
                output_state["raw"].append(raw_report)
            if "rejection" in prepared:
                for output_state in states.values():
                    output_state["rejected"].append(prepared["rejection"])
                continue
            filtered = prepared["filtered"]
            activity = prepared["activity"]
            lag_audits = prepared["lag_audits"]
            filter_manifest = prepared["filter_manifest"]

            for output in job.manifest.outputs:
                output_state = states[output.action_space]
                try:
                    aligned = synchronize_episode(
                        filtered,
                        job.profile,
                        activity,
                        action_space=output.action_space,
                        output_fps=job.manifest.processing.output_fps,
                        minimum_output_frames=job.manifest.processing.minimum_output_frames,
                    )
                    aligned_qa = audit_aligned_episode(
                        aligned, output_fps=job.manifest.processing.output_fps
                    )
                    if aligned_qa["status"] != "PASS":
                        raise SynchronizationError(
                            aligned_qa["reject_reasons"][0], "aligned QA rejected episode"
                        )
                    episode_index = len(output_state["episodes"])
                    shapes = write_episode(
                        temporary_outputs[output.action_space],
                        aligned,
                        episode_index=episode_index,
                        task_index=source_episode.task_index,
                        global_start_index=output_state["total_frames"],
                        fps=job.manifest.processing.output_fps,
                        video_workers=job.manifest.processing.video_workers,
                        video_encoder_preset=job.manifest.processing.video_encoder_preset,
                        video_encoder_threads=job.manifest.processing.video_encoder_threads,
                    )
                    write_alignment_diagnostics(
                        temporary_outputs[output.action_space],
                        aligned,
                        episode_index=episode_index,
                    )
                    if output_state["video_shapes"] is None:
                        output_state["video_shapes"] = shapes
                    elif output_state["video_shapes"] != shapes:
                        raise RuntimeError("video shapes changed between episodes")
                    output_state["episodes"].append(
                        {
                            "episode_index": episode_index,
                            "tasks": [source_episode.instruction],
                            "length": len(aligned.timestamps),
                            "source_file": str(source_episode.bag_path),
                            "roster_index": source_episode.roster_index,
                            "action_space": output.action_space,
                        }
                    )
                    output_state["total_frames"] += len(aligned.timestamps)
                    output_state["aligned_reports"].append(
                        {
                            **aligned_qa,
                            "episode_index": episode_index,
                            "roster_index": source_episode.roster_index,
                            "task_id": source_episode.task_id,
                            "source_file": str(source_episode.bag_path),
                            "lag_audit": lag_audits,
                        }
                    )
                    output_state["filter_manifest"] = filter_manifest
                except (ValueError, RuntimeError, OSError) as exc:
                    _remove_episode_artifacts(
                        temporary_outputs[output.action_space], len(output_state["episodes"])
                    )
                    reason = (
                        exc.reason if isinstance(exc, SynchronizationError) else "export_failure"
                    )
                    output_state["rejected"].append(
                        _rejection(
                            source_episode,
                            reason,
                            "aligned_or_export",
                            str(exc),
                            exc.details if isinstance(exc, SynchronizationError) else None,
                        )
                    )

        for output in job.manifest.outputs:
            output_state = states[output.action_space]
            pipeline_manifest = {
                "schema_version": "robot_data_pipeline/v1",
                "code_version": _git_revision(),
                "command": sys.argv,
                "profile": str(job.profile.path),
                "profile_hash": job.profile.config_hash,
                "dataset_manifest": str(job.manifest.path),
                "dataset_manifest_hash": job.manifest.config_hash,
                "action_space": output.action_space,
                "clock": asdict(job.profile.clock),
                "video_encoding": {
                    "codec": "libx264",
                    "preset": job.manifest.processing.video_encoder_preset,
                    "threads_per_stream": job.manifest.processing.video_encoder_threads,
                    "parallel_streams": job.manifest.processing.video_workers,
                    "input_mode": "mjpeg_image2pipe",
                },
                "dependencies": _dependency_versions(),
                "filtering": output_state["filter_manifest"],
                "roster": roster_to_dict(roster),
            }
            if output_state["episodes"]:
                write_metadata(
                    temporary_outputs[output.action_space],
                    manifest=job.manifest,
                    profile=job.profile,
                    action_space=output.action_space,
                    episodes=output_state["episodes"],
                    total_frames=output_state["total_frames"],
                    video_shapes=output_state["video_shapes"],
                    pipeline_manifest=pipeline_manifest,
                )
            else:
                write_rejection_metadata(
                    temporary_outputs[output.action_space],
                    manifest=job.manifest,
                    pipeline_manifest=pipeline_manifest,
                )
            _write_quality(
                temporary_outputs[output.action_space],
                output_state["raw"],
                output_state["aligned_reports"],
                output_state["rejected"],
            )
        _publish_outputs(job.manifest.outputs, temporary_outputs)
    except Exception:
        for temporary in temporary_outputs.values():
            if temporary.exists():
                shutil.rmtree(temporary)
        raise
    return {
        output.action_space: {
            "path": str(output.path),
            "episodes": len(states[output.action_space]["episodes"]),
            "frames": states[output.action_space]["total_frames"],
            "rejected": len(states[output.action_space]["rejected"]),
        }
        for output in job.manifest.outputs
    }
