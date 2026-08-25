#!/usr/bin/env python3
"""Merge multiple LeRobot v2 datasets without carrying over statistics files."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Any

import pandas as pd


META_DIR = "meta"
INFO_FILE = "info.json"
MODALITY_FILE = "modality.json"
EPISODES_FILE = "episodes.jsonl"
TASKS_FILE = "tasks.jsonl"
STATS_FILES = {"stats.json", "relative_stats.json"}
TASK_INDEX_COLUMNS = {"task_index"}


@dataclass(frozen=True)
class DatasetMeta:
    root: Path
    info: dict[str, Any]
    modality: dict[str, Any]
    episodes: list[dict[str, Any]]
    tasks: list[dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge several local LeRobot v2 datasets into one dataset. "
            "The output rewrites episode/frame indexes and task indexes, copies data/videos, "
            "and intentionally does not create stats.json or relative_stats.json."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        help="Input LeRobot v2 dataset directories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the merged dataset will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output directory first if it already exists.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=4)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_dataset_meta(root: Path) -> DatasetMeta:
    root = root.resolve()
    meta_dir = root / META_DIR
    required_files = [INFO_FILE, MODALITY_FILE, EPISODES_FILE, TASKS_FILE]
    missing = [name for name in required_files if not (meta_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{root} is missing required metadata files: {missing}")

    info = read_json(meta_dir / INFO_FILE)
    modality = read_json(meta_dir / MODALITY_FILE)
    episodes = read_jsonl(meta_dir / EPISODES_FILE)
    tasks = read_jsonl(meta_dir / TASKS_FILE)
    return DatasetMeta(root=root, info=info, modality=modality, episodes=episodes, tasks=tasks)


def feature_signature(features: dict[str, Any]) -> dict[str, Any]:
    signature = copy.deepcopy(features)
    for feature in signature.values():
        if isinstance(feature, dict):
            info = feature.get("info")
            if isinstance(info, dict):
                info.pop("video.fps", None)
    return signature


def validate_compatible_datasets(datasets: list[DatasetMeta]) -> None:
    if not datasets:
        raise ValueError("At least one input dataset is required")

    first = datasets[0]
    first_data_path = first.info.get("data_path")
    first_video_path = first.info.get("video_path")
    first_mask_path = first.info.get("mask_path")
    first_chunk_size = first.info.get("chunks_size")
    first_features = feature_signature(first.info.get("features", {}))

    for dataset in datasets:
        if dataset.info.get("codebase_version") != "v2.0":
            raise ValueError(f"{dataset.root} is not a LeRobot v2.0 dataset")
        if dataset.info.get("data_path") != first_data_path:
            raise ValueError(f"{dataset.root} has a different data_path pattern")
        if dataset.info.get("video_path") != first_video_path:
            raise ValueError(f"{dataset.root} has a different video_path pattern")
        if dataset.info.get("mask_path") != first_mask_path:
            raise ValueError(f"{dataset.root} has a different mask_path pattern")
        if dataset.info.get("chunks_size") != first_chunk_size:
            raise ValueError(f"{dataset.root} has a different chunks_size")
        if dataset.modality != first.modality:
            raise ValueError(f"{dataset.root} has a different modality.json")
        if feature_signature(dataset.info.get("features", {})) != first_features:
            raise ValueError(f"{dataset.root} has incompatible features in info.json")

        episode_count = len(dataset.episodes)
        if dataset.info.get("total_episodes") != episode_count:
            raise ValueError(
                f"{dataset.root} info.json total_episodes does not match episodes.jsonl"
            )


def normalize_task_text(value: Any) -> str:
    """Canonicalize formatting differences that do not change an instruction's meaning."""
    return " ".join(str(value).replace("\\n", " ").replace("\\r", " ").split())


def build_task_mapping(datasets: list[DatasetMeta]) -> tuple[list[dict[str, Any]], list[dict[int, int]]]:
    merged_tasks: list[dict[str, Any]] = []
    task_to_new_index: dict[str, int] = {}
    per_dataset_mappings: list[dict[int, int]] = []

    for dataset in datasets:
        mapping: dict[int, int] = {}
        for task in dataset.tasks:
            old_index = int(task["task_index"])
            task_text = normalize_task_text(task["task"])
            if task_text not in task_to_new_index:
                task_to_new_index[task_text] = len(merged_tasks)
                merged_tasks.append({"task_index": task_to_new_index[task_text], "task": task_text})
            mapping[old_index] = task_to_new_index[task_text]
        per_dataset_mappings.append(mapping)

    return merged_tasks, per_dataset_mappings


def task_annotation_columns(modality: dict[str, Any]) -> set[str]:
    columns = set(TASK_INDEX_COLUMNS)
    for annotation in modality.get("annotation", {}).values():
        if isinstance(annotation, dict):
            original_key = annotation.get("original_key")
            if isinstance(original_key, str):
                columns.add(original_key)
    return columns


def format_episode_path(pattern: str, episode_index: int, chunks_size: int) -> Path:
    return Path(
        pattern.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
    )


def remap_task_column(series: pd.Series, task_mapping: dict[int, int], column: str) -> pd.Series:
    unknown_values = sorted({int(value) for value in series.dropna().unique()} - set(task_mapping))
    if unknown_values:
        raise ValueError(f"Column {column} contains task indexes not present in tasks.jsonl: {unknown_values}")
    return series.map(lambda value: task_mapping[int(value)])


def rewrite_parquet_episode(
    source_path: Path,
    output_path: Path,
    new_episode_index: int,
    global_frame_index: int,
    task_mapping: dict[int, int],
    task_columns: set[str],
) -> int:
    df = pd.read_parquet(source_path)
    episode_length = len(df)

    if "episode_index" in df.columns:
        df["episode_index"] = new_episode_index
    if "index" in df.columns:
        df["index"] = range(global_frame_index, global_frame_index + episode_length)

    for column in sorted(task_columns):
        if column in df.columns:
            df[column] = remap_task_column(df[column], task_mapping, column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return episode_length


def copy_episode_sidecar_files(
    source_root: Path,
    output_root: Path,
    old_episode_index: int,
    new_episode_index: int,
    chunks_size: int,
) -> int:
    copied_videos = 0
    old_chunk = old_episode_index // chunks_size
    new_chunk = new_episode_index // chunks_size
    old_name = f"episode_{old_episode_index:06d}"
    new_name = f"episode_{new_episode_index:06d}"

    for root_name in ["videos", "masks"]:
        source_chunk_dir = source_root / root_name / f"chunk-{old_chunk:03d}"
        if not source_chunk_dir.exists():
            continue
        for source_path in source_chunk_dir.rglob(f"{old_name}.*"):
            relative = source_path.relative_to(source_chunk_dir)
            output_relative = Path(f"chunk-{new_chunk:03d}") / relative.with_name(
                source_path.name.replace(old_name, new_name, 1)
            )
            output_path = output_root / root_name / output_relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, output_path)
            if root_name == "videos":
                copied_videos += 1

    return copied_videos


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            existing = next(output_dir.iterdir(), None)
            if existing is not None:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {output_dir}. "
                    "Pass --overwrite to replace it."
                )
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def merged_info_template(
    first_info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_sidecar_videos: int,
) -> dict[str, Any]:
    info = copy.deepcopy(first_info)
    chunks_size = int(info["chunks_size"])
    average_fps = float(first_info.get("fps", 30.0))

    info["fps"] = average_fps
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = total_tasks
    info["total_videos"] = total_sidecar_videos
    info["total_chunks"] = max(1, math.ceil(total_episodes / chunks_size))
    info["splits"] = {"train": f"0:{total_episodes}"}

    for feature in info.get("features", {}).values():
        if isinstance(feature, dict):
            video_info = feature.get("info")
            if isinstance(video_info, dict) and "video.fps" in video_info:
                video_info["video.fps"] = average_fps

    return info


def merge_datasets(input_dirs: list[Path], output_dir: Path, overwrite: bool = False) -> None:
    datasets = [load_dataset_meta(path) for path in input_dirs]
    validate_compatible_datasets(datasets)
    resolved_output = output_dir.resolve()
    for dataset in datasets:
        if resolved_output == dataset.root:
            raise ValueError(f"Output directory must not be the same as an input dataset: {output_dir}")
    prepare_output_dir(output_dir, overwrite)

    merged_tasks, task_mappings = build_task_mapping(datasets)
    chunks_size = int(datasets[0].info["chunks_size"])
    data_path_pattern = str(datasets[0].info["data_path"])
    task_columns = task_annotation_columns(datasets[0].modality)

    merged_episodes: list[dict[str, Any]] = []
    total_frames = 0
    total_sidecar_videos = 0
    new_episode_index = 0

    for dataset, task_mapping in zip(datasets, task_mappings):
        for episode in dataset.episodes:
            old_episode_index = int(episode["episode_index"])
            source_data_path = dataset.root / format_episode_path(
                data_path_pattern, old_episode_index, chunks_size
            )
            output_data_path = output_dir / format_episode_path(
                data_path_pattern, new_episode_index, chunks_size
            )
            episode_length = rewrite_parquet_episode(
                source_data_path,
                output_data_path,
                new_episode_index,
                total_frames,
                task_mapping,
                task_columns,
            )

            rewritten_episode = copy.deepcopy(episode)
            rewritten_episode["episode_index"] = new_episode_index
            if "tasks" in rewritten_episode:
                rewritten_episode["tasks"] = [
                    normalize_task_text(task) for task in rewritten_episode["tasks"]
                ]
            if int(rewritten_episode.get("length", episode_length)) != episode_length:
                raise ValueError(
                    f"{source_data_path} length {episode_length} does not match episodes.jsonl"
                )
            rewritten_episode["length"] = episode_length
            merged_episodes.append(rewritten_episode)

            total_sidecar_videos += copy_episode_sidecar_files(
                dataset.root,
                output_dir,
                old_episode_index,
                new_episode_index,
                chunks_size,
            )
            total_frames += episode_length
            new_episode_index += 1

    write_json(output_dir / META_DIR / MODALITY_FILE, datasets[0].modality)
    write_jsonl(output_dir / META_DIR / TASKS_FILE, merged_tasks)
    write_jsonl(output_dir / META_DIR / EPISODES_FILE, merged_episodes)
    write_json(
        output_dir / META_DIR / INFO_FILE,
        merged_info_template(
            datasets[0].info,
            total_episodes=len(merged_episodes),
            total_frames=total_frames,
            total_tasks=len(merged_tasks),
            total_sidecar_videos=total_sidecar_videos,
        ),
    )

    for stats_file in STATS_FILES:
        stats_path = output_dir / META_DIR / stats_file
        if stats_path.exists():
            stats_path.unlink()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    merge_datasets(args.datasets, args.output_dir, overwrite=args.overwrite)
    print(f"Merged {len(args.datasets)} datasets into {args.output_dir}")


if __name__ == "__main__":
    main()
