#!/usr/bin/env python3
"""Create symlinked R1 Lite train/open-loop-test dataset splits."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def read_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def make_subset(source: Path, output: Path, episode_ids: list[int], split_name: str) -> None:
    if output.exists():
        raise FileExistsError(f"{output} already exists. Remove it or choose a different output.")

    meta_src = source / "meta"
    meta_dst = output / "meta"
    meta_dst.mkdir(parents=True)

    info = json.loads((meta_src / "info.json").read_text())
    episodes = [json.loads(line) for line in (meta_src / "episodes.jsonl").read_text().splitlines()]
    episodes_by_id = {int(ep["episode_index"]): ep for ep in episodes}
    selected_episodes = [episodes_by_id[i] for i in episode_ids]

    info["total_episodes"] = len(selected_episodes)
    info["total_frames"] = sum(int(ep["length"]) for ep in selected_episodes)
    info["splits"] = {split_name: ",".join(str(i) for i in episode_ids)}
    write_json(meta_dst / "info.json", info)

    for name in ["tasks.jsonl", "modality.json", "stats.json", "relative_stats.json"]:
        src = meta_src / name
        if src.exists():
            shutil.copy2(src, meta_dst / name)

    with (meta_dst / "episodes.jsonl").open("w") as f:
        for ep in selected_episodes:
            f.write(json.dumps(ep) + "\n")

    stats_src = meta_src / "episodes_stats.jsonl"
    if stats_src.exists():
        stats_rows = [json.loads(line) for line in stats_src.read_text().splitlines()]
        stats_by_id = {int(row["episode_index"]): row for row in stats_rows}
        with (meta_dst / "episodes_stats.jsonl").open("w") as f:
            for episode_id in episode_ids:
                if episode_id in stats_by_id:
                    f.write(json.dumps(stats_by_id[episode_id]) + "\n")

    for episode_id in episode_ids:
        chunk = episode_id // int(info["chunks_size"])
        parquet_rel = Path(info["data_path"].format(episode_chunk=chunk, episode_index=episode_id))
        copy_or_link(source / parquet_rel, output / parquet_rel)

    video_root = source / "videos"
    if video_root.exists():
        for video_key_dir in sorted((video_root / "chunk-000").iterdir()):
            if not video_key_dir.is_dir():
                continue
            for episode_id in episode_ids:
                video_rel = Path(
                    info["video_path"].format(
                        episode_chunk=episode_id // int(info["chunks_size"]),
                        episode_index=episode_id,
                        video_key=video_key_dir.name,
                    )
                )
                copy_or_link(source / video_rel, output / video_rel)

    for name in ["README.md", ".gitattributes"]:
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("examples/R1_Lite/R1_Lite_move_the_position_of_the_rubiks_cube"),
    )
    parser.add_argument("--train-output", type=Path, default=None)
    parser.add_argument("--test-output", type=Path, default=None)
    args = parser.parse_args()

    source = args.source
    train_ids = read_ids(source / "meta" / "train_traj_ids.txt")
    test_ids = read_ids(source / "meta" / "open_loop_test_traj_ids.txt")

    train_output = args.train_output or source.with_name(source.name + "_train")
    test_output = args.test_output or source.with_name(source.name + "_open_loop_test")

    make_subset(source, train_output, train_ids, "train")
    make_subset(source, test_output, test_ids, "open_loop_test")

    print(f"Train split: {train_output} ({len(train_ids)} episodes)")
    print(f"Open-loop test split: {test_output} ({len(test_ids)} episodes)")


if __name__ == "__main__":
    main()
