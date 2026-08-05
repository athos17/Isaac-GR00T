#!/usr/bin/env python3
"""Convert Wuji EEF ROS bags task-by-task, filter them, merge them, and calculate stats.

The converter and the stats command may require different Python environments. Use
``--converter-python`` for the environment that has ``rosbags`` installed and
``--gr00t-python`` for the environment that has the GR00T package installed.

EEF and hand velocities are used to trim idle prefixes/suffixes. The default thresholds match the
validated standalone Wuji conversion command: 0.05 m/s for EEF motion, 0.5 rad/s for hand motion,
and 1e9 for action-state mismatch so absolute command offsets do not keep idle frames active.
Episodes exceeding the configured maximum timestamp skew, episodes shorter than the converter
quality threshold, and very-low-motion episodes are deleted before indexes are compacted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_CONFIG_VERSION = "wuji_task_config_v1"
DEFAULT_CONFIG = REPO_ROOT / "examples/wuji_rot6d/tactile_vla_teleop_test_tasks.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "examples/wuji_rot6d/tactile_vla_teleop_test_rot6d"
DEFAULT_MERGED_OUTPUT_DIR = REPO_ROOT / "examples/wuji_rot6d/tactile_vla_teleop_test_rot6d_merged"
DEFAULT_MODALITY_CONFIG = REPO_ROOT / "examples/wuji_rot6d/wuji_eef_hand_rot6d_h32_config.py"


@dataclass(frozen=True)
class Task:
    name: str
    input_dir: Path
    instruction: str


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
        help="Parent directory for per-task LeRobot datasets.",
    )
    parser.add_argument(
        "--merged-output-dir",
        type=Path,
        default=DEFAULT_MERGED_OUTPUT_DIR,
        help="Output directory for the merged LeRobot dataset and its stats.",
    )
    parser.add_argument(
        "--eef-rotation-format",
        choices=["rotvec", "rot6d"],
        default="rot6d",
        help="EEF pose representation passed to the ROS bag converter.",
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
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/wuji_bag_cache"))
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
        help="Python executable with GR00T and its stats dependencies.",
    )
    parser.add_argument(
        "--modality-config-path",
        type=Path,
        default=DEFAULT_MODALITY_CONFIG,
        help="Custom modality config passed to gr00t/data/stats.py.",
    )
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument(
        "--max-time-skew",
        type=float,
        default=0.06,
        help=(
            "Maximum allowed nearest-neighbor alignment skew in seconds. Episodes exceeding "
            "this value are deleted. Default: 0.06."
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
        help="Validate inputs and print commands without converting, merging, or calculating stats.",
    )
    return parser.parse_args(argv)


def load_tasks(config_path: Path) -> list[Task]:
    try:
        with config_path.open() as file:
            config = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in task config {config_path}: {error}") from error

    if not isinstance(config, dict) or not isinstance(config.get("tasks"), list):
        raise ValueError(f"{config_path} must contain a top-level 'tasks' list")
    if config.get("version") != SUPPORTED_CONFIG_VERSION:
        raise ValueError(
            f"Unsupported task config version in {config_path}: {config.get('version')!r}; "
            f"expected {SUPPORTED_CONFIG_VERSION!r}"
        )

    tasks: list[Task] = []
    names: set[str] = set()
    for index, raw_task in enumerate(config["tasks"]):
        if not isinstance(raw_task, dict):
            raise ValueError(f"Task {index} in {config_path} must be an object")
        try:
            name = raw_task["task_name"]
            input_dir = raw_task["task_dir"]
            instruction = raw_task["global_task_instruction"]
        except KeyError as error:
            raise ValueError(
                f"Task {index} in {config_path} is missing {error.args[0]!r}"
            ) from error
        if not all(
            isinstance(value, str) and value.strip() for value in (name, input_dir, instruction)
        ):
            raise ValueError(
                f"Task {index} in {config_path} has an empty or non-string required field"
            )
        if name in names:
            raise ValueError(f"Duplicate task_name in {config_path}: {name}")
        names.add(name)
        tasks.append(Task(name=name, input_dir=Path(input_dir), instruction=instruction))

    if not tasks:
        raise ValueError(f"Task config {config_path} does not contain any tasks")
    return tasks


def task_output_dir(output_root: Path, task: Task, rotation_format: str) -> Path:
    return output_root / f"{task.name}_{rotation_format}_rosbag_ts"


def conversion_command(args: argparse.Namespace, task: Task, output_dir: Path) -> list[str]:
    command = [
        str(args.converter_python),
        str(REPO_ROOT / "data_preprocess/wuji_rosbag_to_gr00t.py"),
        "--input-root",
        str(task.input_dir),
        "--output-dir",
        str(output_dir),
        "--eef-rotation-format",
        args.eef_rotation_format,
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
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


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
        (args.modality_config_path, "modality config"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")
    for task in tasks:
        if not task.input_dir.is_dir():
            raise FileNotFoundError(
                f"Input directory for task {task.name!r} does not exist: {task.input_dir}"
            )


def paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def validate_output_layout(
    tasks: list[Task], output_dirs: list[Path], merged_output_dir: Path, overwrite: bool
) -> None:
    for task, output_dir in zip(tasks, output_dirs):
        if paths_overlap(task.input_dir, output_dir):
            raise ValueError(
                f"Input and output directories for task {task.name!r} must not overlap: "
                f"{task.input_dir} and {output_dir}"
            )
        if paths_overlap(output_dir, merged_output_dir):
            raise ValueError(
                f"Per-task output and merged output directories must not overlap: "
                f"{output_dir} and {merged_output_dir}"
            )
        if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Per-task output directory is not empty: {output_dir}. "
                "Pass --overwrite to replace it."
            )


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = load_tasks(args.task_config)
    validate_inputs(tasks, args)
    output_dirs = [
        task_output_dir(args.output_root, task, args.eef_rotation_format) for task in tasks
    ]
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
        print("Validated task configuration. Commands that would run:")
        for command in commands:
            print(format_command(command))
        return

    for task, output_dir in zip(tasks, output_dirs):
        print(f"Converting {task.name} -> {output_dir}")
        subprocess.run(conversion_command(args, task, output_dir), check=True, cwd=REPO_ROOT)

    print(f"Merging {len(output_dirs)} datasets -> {args.merged_output_dir}")
    subprocess.run(merge_command(args, output_dirs), check=True, cwd=REPO_ROOT)
    print(f"Calculating stats for {args.merged_output_dir}")
    subprocess.run(stats_command(args), check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
