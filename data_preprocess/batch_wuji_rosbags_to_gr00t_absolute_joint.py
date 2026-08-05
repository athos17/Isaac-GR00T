#!/usr/bin/env python3
"""Batch-convert Wuji ROS bags to absolute arm-joint + hand-joint datasets.

Joint positions are exported without delta conversion. EEF state/action streams are read only for
motion detection and quality filtering. Motion defaults match the validated EEF conversion command:
0.05 m/s for EEF motion, 0.5 rad/s for hand motion, and 1e9 for action-state mismatch. Rejected
episodes are deleted after an audit record is written, and accepted per-task datasets are merged
before GR00T statistics are generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocess.batch_wuji_rosbags_to_gr00t import (  # noqa: E402
    DEFAULT_CONFIG,
    Task,
    format_command,
    load_tasks,
    validate_output_layout,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "examples/wuji_joint/tactile_vla_teleop_test_joint_absolute"
DEFAULT_MERGED_OUTPUT_DIR = (
    REPO_ROOT / "examples/wuji_joint/tactile_vla_teleop_test_joint_absolute_merged"
)
DEFAULT_MODALITY_CONFIG = REPO_ROOT / "data_preprocess/wuji_joint_hand_absolute_h32_config.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON file containing version and tasks entries.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for per-task absolute-joint datasets.",
    )
    parser.add_argument(
        "--merged-output-dir",
        type=Path,
        default=DEFAULT_MERGED_OUTPUT_DIR,
        help="Output directory for the merged dataset and its statistics.",
    )
    parser.add_argument(
        "--bag-backend",
        choices=["rosbags", "ros2", "auto"],
        default="rosbags",
    )
    parser.add_argument(
        "--timestamp-source",
        choices=["header", "rosbag"],
        default="rosbag",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/wuji_joint_bag_cache"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--converter-python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable with the ROS bag conversion dependencies.",
    )
    parser.add_argument(
        "--gr00t-python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable with GR00T and statistics dependencies.",
    )
    parser.add_argument(
        "--modality-config-path",
        type=Path,
        default=DEFAULT_MODALITY_CONFIG,
        help="Absolute-joint modality config passed to gr00t/data/stats.py.",
    )
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument(
        "--max-time-skew",
        type=float,
        default=0.06,
        help=(
            "Maximum allowed nearest-neighbor alignment skew in seconds. An entire episode is "
            "deleted when its maximum observed skew exceeds this value. Default: 0.06."
        ),
    )
    parser.add_argument("--motion-velocity-threshold", type=float, default=0.05)
    parser.add_argument("--motion-hand-velocity-threshold", type=float, default=0.5)
    parser.add_argument("--motion-action-state-diff-threshold", type=float, default=1e9)
    parser.add_argument("--motion-window-sec", type=float, default=0.5)
    parser.add_argument("--motion-min-frames", type=int, default=30)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-task and merged outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without writing outputs.",
    )
    return parser.parse_args(argv)


def task_output_dir(output_root: Path, task: Task) -> Path:
    return output_root / f"{task.name}_joint_absolute_rosbag_ts"


def conversion_command(args: argparse.Namespace, task: Task, output_dir: Path) -> list[str]:
    return [
        str(args.converter_python),
        str(REPO_ROOT / "data_preprocess/wuji_rosbag_to_gr00t.py"),
        "--input-root",
        str(task.input_dir),
        "--output-dir",
        str(output_dir),
        "--action-space",
        "joint",
        "--eef-rotation-format",
        "rot6d",
        "--task-description",
        task.instruction,
        "--bag-backend",
        args.bag_backend,
        "--work-dir",
        str(args.work_dir / task.name),
        "--timestamp-source",
        args.timestamp_source,
        "--num-workers",
        str(args.num_workers),
        "--max-time-skew",
        str(args.max_time_skew),
        "--quality-max-skew",
        str(args.max_time_skew),
        "--enable-motion-detection",
        "--filter-by-quality",
        "--filtered-episode-policy",
        "delete",
        "--motion-velocity-threshold",
        str(args.motion_velocity_threshold),
        "--motion-hand-velocity-threshold",
        str(args.motion_hand_velocity_threshold),
        "--motion-action-state-diff-threshold",
        str(args.motion_action_state_diff_threshold),
        "--motion-window-sec",
        str(args.motion_window_sec),
        "--motion-min-frames",
        str(args.motion_min_frames),
    ] + (["--overwrite"] if args.overwrite else [])


def merge_command(args: argparse.Namespace, input_dirs: list[Path]) -> list[str]:
    command = [
        str(args.gr00t_python),
        str(REPO_ROOT / "scripts/lerobot_conversion/merge_lerobot_v2_datasets.py"),
        *(str(path) for path in input_dirs),
        "--output-dir",
        str(args.merged_output_dir),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def stats_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.gr00t_python),
        str(REPO_ROOT / "gr00t/data/stats.py"),
        "--dataset-path",
        str(args.merged_output_dir),
        "--embodiment-tag",
        args.embodiment_tag,
        "--modality-config-path",
        str(args.modality_config_path),
        "--num-workers",
        str(args.num_workers),
    ]


def validate_inputs(tasks: list[Task], args: argparse.Namespace) -> None:
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.max_time_skew < 0:
        raise ValueError("--max-time-skew must be >= 0")
    if args.motion_velocity_threshold <= 0:
        raise ValueError("--motion-velocity-threshold must be > 0")
    if args.motion_hand_velocity_threshold <= 0:
        raise ValueError("--motion-hand-velocity-threshold must be > 0")
    if args.motion_action_state_diff_threshold <= 0:
        raise ValueError("--motion-action-state-diff-threshold must be > 0")
    if args.motion_window_sec <= 0:
        raise ValueError("--motion-window-sec must be > 0")
    if args.motion_min_frames < 1:
        raise ValueError("--motion-min-frames must be >= 1")

    for path, description in [
        (args.converter_python, "converter Python executable"),
        (args.gr00t_python, "GR00T Python executable"),
        (args.modality_config_path, "absolute-joint modality config"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")
    for task in tasks:
        if not task.input_dir.is_dir():
            raise FileNotFoundError(
                f"Input directory for task {task.name!r} does not exist: {task.input_dir}"
            )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = load_tasks(args.task_config)
    validate_inputs(tasks, args)
    output_dirs = [task_output_dir(args.output_root, task) for task in tasks]
    validate_output_layout(tasks, output_dirs, args.merged_output_dir, args.overwrite)

    commands = [
        *(
            conversion_command(args, task, output_dir)
            for task, output_dir in zip(tasks, output_dirs)
        ),
        merge_command(args, output_dirs),
        stats_command(args),
    ]
    if args.dry_run:
        print("Validated absolute-joint task configuration. Commands that would run:")
        for command in commands:
            print(format_command(command))
        return

    for task, output_dir in zip(tasks, output_dirs):
        print(f"Converting absolute joints for {task.name} -> {output_dir}")
        subprocess.run(conversion_command(args, task, output_dir), check=True, cwd=REPO_ROOT)

    print(f"Merging {len(output_dirs)} datasets -> {args.merged_output_dir}")
    subprocess.run(merge_command(args, output_dirs), check=True, cwd=REPO_ROOT)
    print(f"Calculating absolute-joint stats for {args.merged_output_dir}")
    subprocess.run(stats_command(args), check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
