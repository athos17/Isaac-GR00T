import importlib.util
import json
from pathlib import Path
import sys

from gr00t.data.types import ActionRepresentation, ActionType


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "data_preprocess" / "batch_wuji_rosbags_to_gr00t_absolute_joint.py"
MODALITY_CONFIG_PATH = REPO_ROOT / "data_preprocess" / "wuji_joint_hand_absolute_h32_config.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_conversion_command_exports_absolute_joints_and_deletes_failed_episodes(tmp_path):
    module = _load_module(SCRIPT_PATH, "batch_wuji_rosbags_to_gr00t_absolute_joint")
    task_dir = tmp_path / "bag"
    task_dir.mkdir()
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "wuji_task_config_v1",
                "tasks": [
                    {
                        "task_name": "pick",
                        "task_dir": str(task_dir),
                        "global_task_instruction": "pick up the object",
                    }
                ],
            }
        )
    )
    args = module.parse_args(
        [
            "--task-config",
            str(config_path),
            "--converter-python",
            sys.executable,
            "--gr00t-python",
            sys.executable,
            "--max-time-skew",
            "0.04",
        ]
    )
    task = module.load_tasks(config_path)[0]
    output_dir = module.task_output_dir(args.output_root, task)

    command = module.conversion_command(args, task, output_dir)

    assert command[command.index("--action-space") + 1] == "joint"
    assert "--enable-motion-detection" in command
    assert "--filter-by-quality" in command
    assert command[command.index("--filtered-episode-policy") + 1] == "delete"
    assert command[command.index("--max-time-skew") + 1] == "0.04"
    assert command[command.index("--quality-max-skew") + 1] == "0.04"
    assert command[command.index("--motion-velocity-threshold") + 1] == "0.05"
    assert command[command.index("--motion-hand-velocity-threshold") + 1] == "0.5"
    assert command[command.index("--motion-action-state-diff-threshold") + 1] == "1000000000.0"
    assert command[command.index("--motion-window-sec") + 1] == "0.5"
    assert command[command.index("--motion-min-frames") + 1] == "30"
    assert "--disable-motion-action-state-diff" not in command
    assert args.modality_config_path == MODALITY_CONFIG_PATH
    stats = module.stats_command(args)
    assert stats[stats.index("--num-workers") + 1] == "8"


def test_joint_modality_config_marks_every_action_absolute():
    module = _load_module(MODALITY_CONFIG_PATH, "wuji_joint_hand_absolute_h32_config")
    config = module.wuji_joint_hand_absolute_h32_config

    assert config["state"].modality_keys == [
        "left_joint_space",
        "right_joint_space",
        "left_hand_joints",
        "right_hand_joints",
    ]
    assert config["action"].delta_indices == list(range(32))
    assert all(
        action_config.rep == ActionRepresentation.ABSOLUTE
        for action_config in config["action"].action_configs
    )
    assert all(
        action_config.type == ActionType.NON_EEF
        for action_config in config["action"].action_configs
    )
