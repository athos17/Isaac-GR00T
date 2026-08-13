from pathlib import Path

from robot_data_pipeline.catalog import build_roster
from robot_data_pipeline.config import load_job_config
import yaml


REPO_ROOT = Path(__file__).parents[2]
PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml"


def _write_bag(root: Path, name: str) -> Path:
    bag = root / name
    bag.mkdir(parents=True)
    (bag / "data.db3").touch()
    metadata = {
        "rosbag2_bagfile_information": {
            "storage_identifier": "sqlite3",
            "relative_file_paths": ["data.db3"],
            "starting_time": {"nanoseconds_since_epoch": 10},
            "duration": {"nanoseconds": 20},
            "message_count": 0,
            "topics_with_message_count": [],
        }
    }
    (bag / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    return bag


def test_catalog_is_sorted_independent_of_creation_order(tmp_path: Path) -> None:
    root = tmp_path / "input"
    _write_bag(root, "z_episode")
    _write_bag(root, "a_episode")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {PROFILE}
processing:
  output_fps: 30
  num_workers: 1
  activity_padding_before_sec: 0.5
  activity_padding_after_sec: 0.5
  minimum_output_frames: 30
outputs:
  - action_space: joint_absolute
    path: {tmp_path / "output"}
datasets:
  - task_id: task
    roots: [{root}]
    instruction: Do the task
"""
    )

    roster = build_roster(load_job_config(manifest))

    assert [episode.bag_path.name for episode in roster.episodes] == ["a_episode", "z_episode"]
    assert [episode.roster_index for episode in roster.episodes] == [0, 1]
