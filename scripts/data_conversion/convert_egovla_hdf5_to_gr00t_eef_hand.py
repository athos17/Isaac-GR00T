#!/usr/bin/env python
"""Convert EgoVLA_SIM HDF5 episodes to GR00T LeRobot eef+hand format."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd


TASK_DESCRIPTIONS = {
    "Pour-Balls": "pour balls in cup into bowl",
    "Pick-Place-Safe-Drawer": "Pick up object, place it in drawer and close drawer",
    "Stack-Cubes": "stack one cube on another cube",
    "Dump-Apple": "dump aplle into bowl",
    "Push-Box": "push box to the marker",
    "Sort-Cans": "Put sprite cans to the left box, and orange cans to the right box",
    "Sort-Cans-Old": "Put sprite cans to the left box, and orange cans to the right box",
    "Orient-Cube": "Orient the cube to the rotation as show in the observation",
    "Insert-Cans": "Insert cans into the boxes",
    "Close-Drawer": "Close the opened drawer",
    "Open-Drawer": "Open the closed drawer",
    "Stack-Cubes-From-Drawer": "Open the closed drawer, and stack on cube on another cube on the desk",
    "Insert-And-Unload-Cans": "Insert the left can into the slot and insert the right can into the slot, unload the left cans andd then unload the right cans",
    "Insert-And-Unload-Cans-Old": "Insert the right can into the slot and insert the left can into the slot, unload the right cans andd then unload the left cans",
    "Orient-Pour-Balls": "Reorient the mug and pour balls in cup into bowl",
    "Press-Gamepad-Blue": "Press the blue button on the gamepad",
    "Press-Gamepad-Red": "Press the red button on the gamepad",
    "Press-Gamepad-Blue-Red": "Press the blue button on the gamepad then press the red button",
    "Flip-Mug": "Flip the mug",
    "Unload-Cans": "unload the right cans and then unload the left cans",
    "Press-Gamepad-Red-Blue": "Press the red button on the gamepad then press the blue button",
    "Stack-Single-Cube": "stack right cube on the middle cube",
    "Stack-Single-Cube-From-Drawer": "Open the closed drawer, and stack the cube on the cube in front of the drawer",
    "Stack-Can": "put can on the saucer",
    "Stack-Can-From-Drawer": "Put can on the saucer, and Close the drawer",
    "Stack-Can-Into-Drawer": "Open the drawer, and Put can on the saucer",
    "Open-Laptop": "open the laptop",
}

DEFAULT_TASKS = [
    "Close-Drawer",
    "Flip-Mug",
    "Open-Drawer",
    "Open-Laptop",
    "Pour-Balls",
    "Push-Box",
    "Stack-Can",
]

HAND_JOINT_IDS_BY_SIDE = {
    "left": [26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48],
    "right": [31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49],
}

HAND_JOINT_NAMES_BY_SIDE = {
    "left": [
        "L_index_proximal_joint",
        "L_index_intermediate_joint",
        "L_middle_proximal_joint",
        "L_middle_intermediate_joint",
        "L_pinky_proximal_joint",
        "L_pinky_intermediate_joint",
        "L_ring_proximal_joint",
        "L_ring_intermediate_joint",
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_intermediate_joint",
        "L_thumb_distal_joint",
    ],
    "right": [
        "R_index_proximal_joint",
        "R_index_intermediate_joint",
        "R_middle_proximal_joint",
        "R_middle_intermediate_joint",
        "R_pinky_proximal_joint",
        "R_pinky_intermediate_joint",
        "R_ring_proximal_joint",
        "R_ring_intermediate_joint",
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_intermediate_joint",
        "R_thumb_distal_joint",
    ],
}

STATE_KEYS = ["left_eef", "right_eef", "left_hand_joints", "right_hand_joints"]
STATE_DIMS = [6, 6, 12, 12]
STATE_DIM = sum(STATE_DIMS)
ACTION_KEYS = STATE_KEYS
ACTION_DIMS = STATE_DIMS
ACTION_DIM = STATE_DIM


def _episode_index(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.hdf5$", path.name)
    if match is None:
        raise ValueError(f"Could not parse episode index from {path}")
    return int(match.group(1))


def _resolve_source_files(input_root: Path, source_files: list[str]) -> dict[str, list[Path]]:
    files_by_task: dict[str, list[Path]] = {}
    for source_file in source_files:
        hdf5_path = Path(source_file).expanduser()
        if not hdf5_path.is_absolute():
            hdf5_path = input_root / hdf5_path
        hdf5_path = hdf5_path.resolve()

        if not hdf5_path.is_file():
            raise FileNotFoundError(hdf5_path)
        task_name = hdf5_path.parent.name
        if task_name not in TASK_DESCRIPTIONS:
            raise ValueError(
                f"Could not infer a known task from {hdf5_path}. "
                f"Parent directory must be one of {sorted(TASK_DESCRIPTIONS)}."
            )
        files_by_task.setdefault(task_name, []).append(hdf5_path)
    return files_by_task


def _pose7_wxyz_to_xyz_rotvec(pose: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, qw, qx, qy, qz] to [x, y, z, rx, ry, rz]."""
    pose = np.asarray(pose, dtype=np.float32)
    out = np.empty((*pose.shape[:-1], 6), dtype=np.float32)
    out[..., :3] = pose[..., :3]

    quat = pose[..., 3:7].astype(np.float64, copy=True)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-12)

    # Canonicalize to the shortest rotation, matching scipy Rotation.as_rotvec behavior.
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    w = np.clip(quat[..., 0], -1.0, 1.0)
    xyz = quat[..., 1:4]
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)

    scale = np.zeros_like(angle)
    mask = sin_half > 1e-12
    scale[mask] = angle[mask] / sin_half[mask]
    out[..., 3:6] = (xyz * scale[..., None]).astype(np.float32)
    return out


def _concat_state(obs_group: h5py.Group) -> np.ndarray:
    qpos = np.asarray(obs_group["qpos"][:], dtype=np.float32)
    return np.concatenate(
        [
            _pose7_wxyz_to_xyz_rotvec(obs_group["left_ee_pose"][:]),
            _pose7_wxyz_to_xyz_rotvec(obs_group["right_ee_pose"][:]),
            qpos[:, HAND_JOINT_IDS_BY_SIDE["left"]],
            qpos[:, HAND_JOINT_IDS_BY_SIDE["right"]],
        ],
        axis=-1,
    ).astype(np.float32)


def _concat_action(h5_file: h5py.File) -> np.ndarray:
    action = np.asarray(h5_file["action"][:], dtype=np.float32)
    obs_group = h5_file["observations"]
    return np.concatenate(
        [
            _pose7_wxyz_to_xyz_rotvec(obs_group["left_target_ee_pose"][:]),
            _pose7_wxyz_to_xyz_rotvec(obs_group["right_target_ee_pose"][:]),
            action[:, HAND_JOINT_IDS_BY_SIDE["left"]],
            action[:, HAND_JOINT_IDS_BY_SIDE["right"]],
        ],
        axis=-1,
    ).astype(np.float32)


def _write_video(video_path: Path, frames_rgb: np.ndarray, fps: int) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _rows_to_dataframe(
    state: np.ndarray,
    action: np.ndarray,
    episode_index: int,
    task_index: int,
    global_start_index: int,
    fps: int,
) -> pd.DataFrame:
    length = len(state)
    return pd.DataFrame(
        {
            "observation.state": list(state),
            "action": list(action),
            "timestamp": np.arange(length, dtype=np.float32) / float(fps),
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(global_start_index, global_start_index + length, dtype=np.int64),
            "task_index": np.full(length, task_index, dtype=np.int64),
        }
    )


def _feature_names(prefixes: list[str], dims: list[int]) -> list[str]:
    names = []
    for prefix, dim in zip(prefixes, dims):
        if dim == 6 and prefix.endswith("eef"):
            names.extend(
                [
                    f"{prefix}.x",
                    f"{prefix}.y",
                    f"{prefix}.z",
                    f"{prefix}.rotvec_x",
                    f"{prefix}.rotvec_y",
                    f"{prefix}.rotvec_z",
                ]
            )
        elif prefix == "left_hand_joints":
            names.extend([f"{prefix}.{name}" for name in HAND_JOINT_NAMES_BY_SIDE["left"]])
        elif prefix == "right_hand_joints":
            names.extend([f"{prefix}.{name}" for name in HAND_JOINT_NAMES_BY_SIDE["right"]])
        else:
            names.extend([f"{prefix}_{idx}" for idx in range(dim)])
    return names


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _modality_ranges(keys: list[str], dims: list[int]) -> dict[str, dict[str, int]]:
    ranges = {}
    start = 0
    for key, dim in zip(keys, dims):
        ranges[key] = {"start": start, "end": start + dim}
        start += dim
    return ranges


def _write_metadata(
    output_dir: Path,
    episodes: list[dict],
    tasks: list[dict],
    total_frames: int,
    fps: int,
    chunks_size: int,
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _write_jsonl(meta_dir / "tasks.jsonl", tasks)

    modality = {
        "state": _modality_ranges(STATE_KEYS, STATE_DIMS),
        "action": _modality_ranges(ACTION_KEYS, ACTION_DIMS),
        "video": {"ego_view": {"original_key": "observation.images.camera_0"}},
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }
    with (meta_dir / "modality.json").open("w", encoding="utf-8") as f:
        json.dump(modality, f, indent=2)

    info = {
        "codebase_version": "v2.0",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "fps": fps,
        "chunks_size": chunks_size,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes),
        "total_chunks": max(1, (len(episodes) + chunks_size - 1) // chunks_size),
        "splits": {"train": f"0:{len(episodes)}"},
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": _feature_names(STATE_KEYS, STATE_DIMS),
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": _feature_names(ACTION_KEYS, ACTION_DIMS),
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.images.camera_0": {
                "dtype": "video",
                "shape": [384, 384, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
        },
        "robot_type": "EgoVLA_SIM_EEF_HAND",
    }
    with (meta_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def convert(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_by_task = None
    if args.source_files:
        files_by_task = _resolve_source_files(input_root, args.source_files)
        task_names = list(files_by_task)
    else:
        task_names = args.tasks or DEFAULT_TASKS

    tasks = [{"task_index": idx, "task": TASK_DESCRIPTIONS[name]} for idx, name in enumerate(task_names)]
    episodes_meta = []
    global_frame_index = 0
    output_episode_index = 0

    for task_index, task_name in enumerate(task_names):
        if files_by_task is None:
            task_dir = input_root / task_name
            if not task_dir.is_dir():
                raise FileNotFoundError(task_dir)
            files = sorted(task_dir.glob("episode_*.hdf5"), key=_episode_index)
        else:
            files = files_by_task[task_name]
        if args.max_episodes_per_task is not None:
            files = files[: args.max_episodes_per_task]

        for hdf5_path in files:
            episode_chunk = output_episode_index // args.chunks_size
            data_dir = output_dir / "data" / f"chunk-{episode_chunk:03d}"
            video_dir = (
                output_dir
                / "videos"
                / f"chunk-{episode_chunk:03d}"
                / "observation.images.camera_0"
            )
            data_dir.mkdir(parents=True, exist_ok=True)

            with h5py.File(hdf5_path, "r") as h5_file:
                obs = h5_file["observations"]
                state = _concat_state(obs)
                action = _concat_action(h5_file)
                if state.shape != action.shape or state.shape[1] != STATE_DIM:
                    raise ValueError(
                        f"Unexpected state/action shape for {hdf5_path}: "
                        f"state={state.shape}, action={action.shape}"
                    )

                df = _rows_to_dataframe(
                    state=state,
                    action=action,
                    episode_index=output_episode_index,
                    task_index=task_index,
                    global_start_index=global_frame_index,
                    fps=args.fps,
                )
                parquet_path = data_dir / f"episode_{output_episode_index:06d}.parquet"
                df.to_parquet(parquet_path, index=False)

                frames = np.asarray(obs[args.image_key][:], dtype=np.uint8)
                video_path = video_dir / f"episode_{output_episode_index:06d}.mp4"
                _write_video(video_path, frames, args.fps)

            length = len(df)
            episodes_meta.append(
                {
                    "episode_index": output_episode_index,
                    "tasks": [TASK_DESCRIPTIONS[task_name]],
                    "length": length,
                    "source_file": str(hdf5_path),
                }
            )
            global_frame_index += length
            output_episode_index += 1

    _write_metadata(
        output_dir=output_dir,
        episodes=episodes_meta,
        tasks=tasks,
        total_frames=global_frame_index,
        fps=args.fps,
        chunks_size=args.chunks_size,
    )
    print(f"Wrote {len(episodes_meta)} episodes / {global_frame_index} frames to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--input-root",
        default="/data_all/share/datasets/EgoVLA_SIM",
        help="Root containing EgoVLA_SIM task directories with episode_*.hdf5 files.",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/ego_vla_eef_hand/data",
        help="Output GR00T LeRobot dataset directory.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_DESCRIPTIONS),
        default=None,
        help="Task directories to convert. Defaults to the same 7 tasks as ego_vla_short.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--image-key", default="images/main")
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument(
        "--source-files",
        nargs="+",
        default=None,
        help=(
            "Specific EgoVLA_SIM HDF5 episode files to convert. Paths may be absolute or "
            "relative to --input-root, e.g. Open-Drawer/episode_37.hdf5."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
