from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from robot_data_pipeline.config import ConfigError
from robot_data_pipeline.models import BagMetadata, EpisodeSpec, JobConfig, ProcessingRoster


def _metadata_path(bag_path: Path) -> Path:
    return bag_path / "metadata.yaml"


def _parse_metadata(path: Path) -> tuple[BagMetadata, str]:
    raw = path.read_bytes()
    metadata_hash = hashlib.sha256(raw).hexdigest()
    try:
        document: dict[str, Any] = yaml.safe_load(raw)
        info = document["rosbag2_bagfile_information"]
        topics = {
            item["topic_metadata"]["name"]: (
                item["topic_metadata"]["type"],
                int(item.get("message_count", 0)),
            )
            for item in info.get("topics_with_message_count", [])
        }
        result = BagMetadata(
            storage_identifier=str(info["storage_identifier"]),
            relative_files=tuple(str(value) for value in info["relative_file_paths"]),
            starting_time_ns=int(info["starting_time"]["nanoseconds_since_epoch"]),
            duration_ns=int(info["duration"]["nanoseconds"]),
            message_count=int(info["message_count"]),
            topics=topics,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid rosbag metadata {path}: {exc}") from exc
    if result.storage_identifier not in {"sqlite3", "mcap"}:
        raise ConfigError(f"unsupported rosbag storage {result.storage_identifier!r}: {path}")
    for relative in result.relative_files:
        if not (path.parent / relative).is_file():
            raise ConfigError(f"rosbag data file is missing: {path.parent / relative}")
    return result, metadata_hash


def _discover_root(root: Path) -> list[Path]:
    if _metadata_path(root).is_file():
        return [root]
    return sorted(
        (path.parent for path in root.rglob("metadata.yaml")), key=lambda path: path.as_posix()
    )


def build_roster(job: JobConfig) -> ProcessingRoster:
    episodes = []
    seen_bags: set[Path] = set()
    roster_index = 0
    for task_index, dataset in enumerate(job.manifest.datasets):
        for root in dataset.roots:
            discovered = _discover_root(root)
            if not discovered:
                raise ConfigError(f"no rosbag episodes found under input root: {root}")
            for bag_path in discovered:
                resolved = bag_path.resolve()
                if resolved in seen_bags:
                    raise ConfigError(f"duplicate rosbag episode: {resolved}")
                seen_bags.add(resolved)
                metadata, metadata_hash = _parse_metadata(_metadata_path(resolved))
                episodes.append(
                    EpisodeSpec(
                        roster_index=roster_index,
                        task_index=task_index,
                        task_id=dataset.task_id,
                        instruction=dataset.instruction,
                        root=root,
                        bag_path=resolved,
                        metadata_hash=metadata_hash,
                        metadata=metadata,
                    )
                )
                roster_index += 1
    return ProcessingRoster(
        manifest_hash=job.manifest.config_hash,
        profile_hash=job.profile.config_hash,
        episodes=tuple(episodes),
    )


def roster_to_dict(roster: ProcessingRoster) -> dict[str, Any]:
    return {
        "manifest_hash": roster.manifest_hash,
        "profile_hash": roster.profile_hash,
        "episode_count": len(roster.episodes),
        "episodes": [
            {
                "roster_index": episode.roster_index,
                "task_index": episode.task_index,
                "task_id": episode.task_id,
                "bag_path": str(episode.bag_path),
                "metadata_hash": episode.metadata_hash,
                "storage_identifier": episode.metadata.storage_identifier,
                "message_count": episode.metadata.message_count,
            }
            for episode in roster.episodes
        ],
    }
