from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robot_data_pipeline.export.video import probe_video, write_video
from robot_data_pipeline.models import AlignedEpisodeData, DatasetManifest, RobotProfile


VIDEO_NAMES = {
    "video.head": "head_view",
    "video.left_wrist": "left_wrist_view",
    "video.right_wrist": "right_wrist_view",
}
ROT6D_NAMES = (
    "rot6d_r0c0",
    "rot6d_r0c1",
    "rot6d_r0c2",
    "rot6d_r1c0",
    "rot6d_r1c1",
    "rot6d_r1c2",
)


def _group_names(profile: RobotProfile, key: str) -> list[str]:
    stream = profile.streams[key]
    group = key.split(".", 1)[1]
    if stream.semantic.startswith("eef_pose"):
        return [f"{group}.{name}" for name in ("x", "y", "z", *ROT6D_NAMES)]
    return [f"{group}.{name}" for name in stream.names]


def feature_layout(
    profile: RobotProfile, action_space: str
) -> tuple[list[str], list[str], dict[str, Any]]:
    space = profile.output_spaces[action_space]
    state_names = [name for key in space.state_groups for name in _group_names(profile, key)]
    action_names = [name for key in space.action_groups for name in _group_names(profile, key)]

    def ranges(keys: tuple[str, ...]) -> dict[str, dict[str, int]]:
        result = {}
        start = 0
        for key in keys:
            end = start + len(_group_names(profile, key))
            result[key.split(".", 1)[1]] = {"start": start, "end": end}
            start = end
        return result

    modality = {
        "state": ranges(space.state_groups),
        "action": ranges(space.action_groups),
        "video": {
            name: {"original_key": f"observation.images.{name}"} for name in VIDEO_NAMES.values()
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }
    return state_names, action_names, modality


def write_episode(
    output_dir: Path,
    aligned: AlignedEpisodeData,
    *,
    episode_index: int,
    task_index: int,
    global_start_index: int,
    fps: float,
    video_workers: int = 3,
    video_encoder_preset: str = "veryfast",
    video_encoder_threads: int = 0,
    chunks_size: int = 1000,
) -> dict[str, tuple[int, int, int]]:
    frame_count = len(aligned.timestamps)
    chunk = episode_index // chunks_size
    data_path = output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = np.arange(frame_count, dtype=np.int64)
    table = pa.table(
        {
            "observation.state": pa.array(
                aligned.state.astype(np.float32).tolist(), type=pa.list_(pa.float32())
            ),
            "action": pa.array(
                aligned.action.astype(np.float32).tolist(), type=pa.list_(pa.float32())
            ),
            "timestamp": pa.array(frame_indices.astype(np.float32) / np.float32(fps)),
            "frame_index": pa.array(frame_indices),
            "episode_index": pa.array(np.full(frame_count, episode_index, dtype=np.int64)),
            "index": pa.array(frame_indices + global_start_index),
            "task_index": pa.array(np.full(frame_count, task_index, dtype=np.int64)),
            "annotation.human.action.task_description": pa.array(
                np.full(frame_count, task_index, dtype=np.int64)
            ),
        }
    )
    pq.write_table(table, data_path)

    def write_one_video(item: tuple[str, str]) -> tuple[str, tuple[int, int, int]]:
        key, modality_name = item
        video_path = (
            output_dir
            / "videos"
            / f"chunk-{chunk:03d}"
            / f"observation.images.{modality_name}"
            / f"episode_{episode_index:06d}.mp4"
        )
        shape = write_video(
            video_path,
            aligned.images[key],
            fps=fps,
            preset=video_encoder_preset,
            encoder_threads=video_encoder_threads,
        )
        written_frames, written_fps = probe_video(video_path)
        if written_frames != frame_count:
            raise RuntimeError(
                f"video frame count mismatch for {key}: {written_frames} != {frame_count}"
            )
        if abs(written_fps - fps) > 1e-6:
            raise RuntimeError(f"video fps mismatch for {key}: {written_fps} != {fps}")
        return key, shape

    workers = min(video_workers, len(VIDEO_NAMES))
    if workers == 1:
        videos = map(write_one_video, VIDEO_NAMES.items())
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-export")
        try:
            videos = list(executor.map(write_one_video, VIDEO_NAMES.items()))
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    shapes = dict(videos)
    return shapes


def write_metadata(
    output_dir: Path,
    *,
    manifest: DatasetManifest,
    profile: RobotProfile,
    action_space: str,
    episodes: list[dict[str, Any]],
    total_frames: int,
    video_shapes: dict[str, tuple[int, int, int]],
    pipeline_manifest: dict[str, Any],
    chunks_size: int = 1000,
) -> None:
    meta = output_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    state_names, action_names, modality = feature_layout(profile, action_space)
    tasks = [
        {"task_index": index, "task": dataset.instruction}
        for index, dataset in enumerate(manifest.datasets)
    ]

    def write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for value in values:
                file.write(json.dumps(value, sort_keys=True) + "\n")

    write_jsonl(meta / "tasks.jsonl", tasks)
    write_jsonl(meta / "episodes.jsonl", episodes)
    write_json(meta / "modality.json", modality)
    write_json(meta / "pipeline_manifest.json", pipeline_manifest)
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(state_names)],
            "names": state_names,
        },
        "action": {"dtype": "float32", "shape": [len(action_names)], "names": action_names},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key, modality_name in VIDEO_NAMES.items():
        height, width, channels = video_shapes[key]
        features[f"observation.images.{modality_name}"] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.fps": manifest.processing.output_fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    total_episodes = len(episodes)
    info = {
        "codebase_version": "v2.0",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "fps": manifest.processing.output_fps,
        "chunks_size": chunks_size,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": total_episodes * len(VIDEO_NAMES),
        "total_chunks": (total_episodes + chunks_size - 1) // chunks_size,
        "splits": {"train": f"0:{total_episodes}"},
        "features": features,
        "robot_type": f"WUJI_ASTRIBOT_{action_space.upper()}_ROT6D",
    }
    write_json(meta / "info.json", info)


def write_rejection_metadata(
    output_dir: Path, *, manifest: DatasetManifest, pipeline_manifest: dict[str, Any]
) -> None:
    """Write job identity for an output with no PASS episodes.

    No ``info.json`` is emitted because this directory is a QA result, not a loadable
    LeRobot dataset.
    """
    meta = output_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    tasks = [
        {"task_index": index, "task": dataset.instruction}
        for index, dataset in enumerate(manifest.datasets)
    ]
    (meta / "pipeline_manifest.json").write_text(
        json.dumps(pipeline_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (meta / "tasks.jsonl").open("w", encoding="utf-8") as file:
        for task in tasks:
            file.write(json.dumps(task, sort_keys=True) + "\n")
    (meta / "episodes.jsonl").touch()
