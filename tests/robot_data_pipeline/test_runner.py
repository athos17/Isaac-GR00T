import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from robot_data_pipeline.config import load_robot_profile
from robot_data_pipeline.models import (
    BagMetadata,
    DatasetManifest,
    DatasetSource,
    EpisodeAudit,
    EpisodeSpec,
    JobConfig,
    OutputRequest,
    ProcessingConfig,
    ProcessingRoster,
)
from robot_data_pipeline.runner import _prepared_episodes, _publish_outputs, convert_job


REPO_ROOT = Path(__file__).parents[2]
PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml"


def test_parallel_preparation_yields_roster_order(monkeypatch) -> None:
    job = SimpleNamespace(manifest=SimpleNamespace(processing=SimpleNamespace(num_workers=3)))

    def prepare(unused_job, episode):
        return {"source": episode}

    monkeypatch.setattr("robot_data_pipeline.runner._prepare_episode", prepare)

    results = list(_prepared_episodes(job, [3, 1, 4, 2]))

    assert [result["source"] for result in results] == [3, 1, 4, 2]


def test_all_rejected_job_publishes_quality_without_lerobot_info(tmp_path, monkeypatch) -> None:
    profile = load_robot_profile(PROFILE)
    output_path = tmp_path / "output"
    manifest = DatasetManifest(
        path=tmp_path / "manifest.yaml",
        schema_version="dataset_manifest/v1",
        profile_path=PROFILE,
        processing=ProcessingConfig(30, 1, 0.5, 0.5, 30),
        outputs=(OutputRequest("joint_absolute", output_path),),
        datasets=(DatasetSource("task", (tmp_path,), "Do the task"),),
        config_hash="manifest-hash",
    )
    episode = EpisodeSpec(
        roster_index=0,
        task_index=0,
        task_id="task",
        instruction="Do the task",
        root=tmp_path,
        bag_path=tmp_path / "bag",
        metadata_hash="metadata-hash",
        metadata=BagMetadata("sqlite3", ("data.db3",), 0, 1, 0, {}),
    )
    roster = ProcessingRoster("manifest-hash", "profile-hash", (episode,))
    raw = EpisodeAudit(0, "task", str(episode.bag_path), "REJECT", ("missing_required_topic",), {})
    monkeypatch.setattr("robot_data_pipeline.runner.build_roster", lambda unused_job: roster)
    monkeypatch.setattr(
        "robot_data_pipeline.runner._prepared_episodes",
        lambda unused_job, unused_episodes: iter(
            [
                {
                    "source": episode,
                    "raw": raw,
                    "rejection": {
                        "roster_index": 0,
                        "task_id": "task",
                        "source_file": str(episode.bag_path),
                        "reject_reasons": ["missing_required_topic"],
                        "stage": "raw",
                    },
                }
            ]
        ),
    )

    result = convert_job(JobConfig(manifest, profile))

    assert result["joint_absolute"]["episodes"] == 0
    assert (output_path / "quality/rejected_episodes.jsonl").is_file()
    assert (output_path / "meta/pipeline_manifest.json").is_file()
    assert not (output_path / "meta/info.json").exists()
    summary = json.loads((output_path / "quality/dataset_summary.json").read_text())
    assert summary["reject_reasons"] == {"missing_required_topic": 1}


def test_multi_output_publication_restores_all_old_outputs_on_failure(
    tmp_path, monkeypatch
) -> None:
    output_a = OutputRequest("joint_absolute", tmp_path / "output_a")
    output_b = OutputRequest("eef_absolute_hand_absolute", tmp_path / "output_b")
    temporary_a = tmp_path / "temporary_a"
    temporary_b = tmp_path / "temporary_b"
    for path, content in (
        (output_a.path, "old-a"),
        (output_b.path, "old-b"),
        (temporary_a, "new-a"),
        (temporary_b, "new-b"),
    ):
        path.mkdir()
        (path / "value.txt").write_text(content)

    real_replace = os.replace

    def fail_second_publish(source, destination):
        if Path(source) == temporary_b:
            raise OSError("simulated second-output publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr("robot_data_pipeline.runner.os.replace", fail_second_publish)

    with pytest.raises(OSError, match="second-output"):
        _publish_outputs(
            (output_a, output_b),
            {
                output_a.action_space: temporary_a,
                output_b.action_space: temporary_b,
            },
        )

    assert (output_a.path / "value.txt").read_text() == "old-a"
    assert (output_b.path / "value.txt").read_text() == "old-b"
    assert not list(tmp_path.glob(".*.backup-*"))
