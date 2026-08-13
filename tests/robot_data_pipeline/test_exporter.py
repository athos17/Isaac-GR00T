import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from robot_data_pipeline.config import load_robot_profile
from robot_data_pipeline.export.lerobot_v2 import feature_layout, write_metadata
from robot_data_pipeline.models import DatasetManifest, DatasetSource, ProcessingConfig


REPO_ROOT = Path(__file__).parents[2]
PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml"


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        path=Path("/manifest.yaml"),
        schema_version="dataset_manifest/v1",
        profile_path=PROFILE,
        processing=ProcessingConfig(30, 1, 0.5, 0.5, 30),
        outputs=(),
        datasets=(DatasetSource("task", (Path("/input"),), "Do the task"),),
        config_hash="hash",
    )


def test_feature_layout_matches_joint_and_eef_dimensions() -> None:
    profile = load_robot_profile(PROFILE)

    eef_state, eef_action, eef_modality = feature_layout(profile, "eef_absolute_hand_absolute")
    joint_state, joint_action, joint_modality = feature_layout(profile, "joint_absolute")

    assert len(eef_state) == len(eef_action) == 58
    assert eef_modality["state"]["left_eef"] == {"start": 0, "end": 9}
    assert len(joint_state) == len(joint_action) == 54
    assert joint_modality["action"]["right_hand_joint"] == {"start": 34, "end": 54}


def test_metadata_is_lerobot_v2_compatible_and_does_not_write_stats(tmp_path: Path) -> None:
    profile = load_robot_profile(PROFILE)
    shapes = {
        "video.head": (720, 1280, 3),
        "video.left_wrist": (360, 640, 3),
        "video.right_wrist": (360, 640, 3),
    }

    write_metadata(
        tmp_path,
        manifest=_manifest(),
        profile=profile,
        action_space="joint_absolute",
        episodes=[{"episode_index": 0, "tasks": ["task"], "length": 10}],
        total_frames=10,
        video_shapes=shapes,
        pipeline_manifest={"schema_version": "robot_data_pipeline/v1"},
    )

    info = json.loads((tmp_path / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.0"
    assert info["fps"] == 30
    assert info["features"]["observation.state"]["shape"] == [54]
    assert info["features"]["observation.images.head_view"]["shape"] == [720, 1280, 3]
    assert not (tmp_path / "meta/stats.json").exists()
    assert not (tmp_path / "meta/relative_stats.json").exists()


def test_exported_smoke_parquet_has_expected_global_index_when_available() -> None:
    path = Path("/tmp/robot_data_pipeline_legacy_smoke_joint/data/chunk-000/episode_000000.parquet")
    if not path.is_file():
        return
    table = pq.read_table(path)

    assert table.num_rows == 575
    assert np.array_equal(table["frame_index"].to_numpy(), np.arange(575))
    assert np.array_equal(table["index"].to_numpy(), np.arange(575))
