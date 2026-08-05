import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "data_preprocess" / "batch_wuji_rosbags_to_gr00t.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("batch_wuji_rosbags_to_gr00t", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_tasks_and_build_commands(tmp_path: Path):
    module = _load_script_module()
    task_dir = tmp_path / "bag"
    task_dir.mkdir()
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "wuji_task_config_v1",
                "tasks": [
                    {
                        "task_name": "spray_water",
                        "task_dir": str(task_dir),
                        "global_task_instruction": "spray the flowers",
                    }
                ],
            }
        )
    )
    args = module.parse_args(
        [
            "--task-config",
            str(config_path),
            "--output-root",
            str(tmp_path / "outputs"),
            "--merged-output-dir",
            str(tmp_path / "merged"),
            "--converter-python",
            sys.executable,
            "--gr00t-python",
            sys.executable,
        ]
    )

    task = module.load_tasks(config_path)[0]
    output_dir = module.task_output_dir(args.output_root, task, args.eef_rotation_format)
    convert = module.conversion_command(args, task, output_dir)
    merge = module.merge_command(args, [output_dir])
    stats = module.stats_command(args)

    assert output_dir == tmp_path / "outputs" / "spray_water_rot6d_rosbag_ts"
    assert convert[convert.index("--task-description") + 1] == "spray the flowers"
    assert convert[convert.index("--work-dir") + 1].endswith("wuji_bag_cache/spray_water")
    assert "--enable-motion-detection" in convert
    assert "--filter-by-quality" in convert
    assert convert[convert.index("--filtered-episode-policy") + 1] == "delete"
    assert convert[convert.index("--max-time-skew") + 1] == "0.06"
    assert convert[convert.index("--quality-max-skew") + 1] == "0.06"
    assert convert[convert.index("--motion-velocity-threshold") + 1] == "0.05"
    assert convert[convert.index("--motion-hand-velocity-threshold") + 1] == "0.5"
    assert convert[convert.index("--motion-action-state-diff-threshold") + 1] == "1000000000.0"
    assert convert[convert.index("--motion-window-sec") + 1] == "0.5"
    assert convert[convert.index("--motion-min-frames") + 1] == "30"
    assert "--disable-motion-action-state-diff" not in convert
    assert merge[-2:] == ["--output-dir", str(tmp_path / "merged")]
    assert stats[stats.index("--dataset-path") + 1] == str(tmp_path / "merged")
    assert stats[stats.index("--num-workers") + 1] == "8"


def test_load_tasks_rejects_duplicate_names(tmp_path: Path):
    module = _load_script_module()
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "wuji_task_config_v1",
                "tasks": [
                    {"task_name": "same", "task_dir": "/tmp/a", "global_task_instruction": "a"},
                    {"task_name": "same", "task_dir": "/tmp/b", "global_task_instruction": "b"},
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="Duplicate task_name"):
        module.load_tasks(config_path)


def test_validate_output_layout_rejects_merged_parent(tmp_path: Path):
    module = _load_script_module()
    task_dir = tmp_path / "input"
    task_dir.mkdir()
    task = module.Task(name="task", input_dir=task_dir, instruction="do it")
    output_dir = tmp_path / "outputs" / "task_rot6d_rosbag_ts"

    with pytest.raises(ValueError, match="must not overlap"):
        module.validate_output_layout([task], [output_dir], tmp_path / "outputs", overwrite=True)
