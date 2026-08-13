import json
from pathlib import Path

from robot_data_pipeline.cli import main
from robot_data_pipeline.config import load_robot_profile

from tests.robot_data_pipeline.synthetic_bag import write_synthetic_bag


REPO_ROOT = Path(__file__).parents[2]
PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml"


def _manifest(tmp_path: Path, bag: Path) -> tuple[Path, Path]:
    output = tmp_path / "output"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {PROFILE}
processing:
  output_fps: 30
  num_workers: 1
  activity_padding_before_sec: 0.2
  activity_padding_after_sec: 0.2
  minimum_output_frames: 30
outputs:
  - action_space: joint_absolute
    path: {output}
datasets:
  - task_id: synthetic_task
    roots: [{bag}]
    instruction: Move both hands and arms
"""
    )
    return manifest, output


def test_cli_dry_runs_and_end_to_end_commands(tmp_path: Path, capsys) -> None:
    profile = load_robot_profile(PROFILE)
    bag = write_synthetic_bag(tmp_path / "bag", profile)
    manifest, output = _manifest(tmp_path, bag)
    report_dir = tmp_path / "audit"

    for command in ("validate", "audit", "convert"):
        assert main([command, "--manifest", str(manifest), "--dry-run"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["episode_count"] == 1
        assert not output.exists()
        assert not report_dir.exists()

    assert main(["summarize", "--quality-dir", str(report_dir), "--dry-run"]) == 0
    dry_summary = json.loads(capsys.readouterr().out)
    assert dry_summary["exists"] is False
    assert not report_dir.exists()

    assert main(["audit", "--manifest", str(manifest), "--report-dir", str(report_dir)]) == 0
    audit_summary = json.loads(capsys.readouterr().out)
    assert audit_summary["pass_count"] == 1
    assert (report_dir / "episode_reports.jsonl").is_file()

    assert main(["summarize", "--quality-dir", str(report_dir)]) == 0
    summarized_audit = json.loads(capsys.readouterr().out)
    assert summarized_audit["episode_count"] == 1
    assert summarized_audit["reject_count"] == 0

    assert main(["convert", "--manifest", str(manifest)]) == 0
    converted = json.loads(capsys.readouterr().out)
    assert converted["joint_absolute"]["episodes"] == 1
    assert output.is_dir()

    assert main(["summarize", "--quality-dir", str(output / "quality")]) == 0
    summarized_output = json.loads(capsys.readouterr().out)
    assert summarized_output["episode_count"] == 1
    assert summarized_output["pass_count"] == 1
