from dataclasses import replace
import hashlib
import json
from pathlib import Path

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.stats import generate_stats
from gr00t.data.types import ModalityConfig
import numpy as np
import pyarrow.parquet as pq
import pytest
from robot_data_pipeline.catalog import build_roster
from robot_data_pipeline.config import load_job_config, load_robot_profile
from robot_data_pipeline.export.reports import processing_audit_summary
from robot_data_pipeline.quality.decisions import (
    CAMERA_DECODE_FAILURE,
    MISSING_REQUIRED_TOPIC,
    NON_MONOTONIC_HEADER_TIMESTAMP,
    RAW_GAP_EXCEEDED,
    ZERO_HEADER_TIMESTAMP,
)
from robot_data_pipeline.quality.raw import audit_roster
from robot_data_pipeline.runner import audit_processing_roster, convert_job

from tests.robot_data_pipeline.synthetic_bag import write_synthetic_bag


REPO_ROOT = Path(__file__).parents[2]
PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml"
MANUS_PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_manus.yaml"
GOLDEN = REPO_ROOT / "tests/robot_data_pipeline/golden/synthetic_joint_v1.json"


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _manifest(
    tmp_path: Path,
    bag: Path,
    *,
    action_space: str = "joint_absolute",
    profile_path: Path = PROFILE,
) -> Path:
    manifest = tmp_path / "manifest.yaml"
    rotation = "\n    eef_rotation_format: rot6d" if "eef" in action_space else ""
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {profile_path}
processing:
  output_fps: 30
  num_workers: 2
  activity_padding_before_sec: 0.2
  activity_padding_after_sec: 0.2
  minimum_output_frames: 30
outputs:
  - action_space: {action_space}
    path: {tmp_path / "output"}
    {rotation}
datasets:
  - task_id: synthetic_task
    roots: [{bag}]
    instruction: Move both hands and arms
"""
    )
    return manifest


def _load_low_dim_with_gr00t(output: Path) -> None:
    modality = json.loads((output / "meta/modality.json").read_text())
    generate_stats(output)
    loader = LeRobotEpisodeLoader(
        output,
        {
            "state": ModalityConfig(delta_indices=[0], modality_keys=list(modality["state"])),
            "action": ModalityConfig(delta_indices=[0], modality_keys=list(modality["action"])),
        },
    )
    episode = loader[0]
    assert len(loader) == 1
    assert len(episode) > 0


def test_real_rosbag_reader_audit_and_convert(tmp_path: Path) -> None:
    profile = load_robot_profile(PROFILE)
    bag = write_synthetic_bag(tmp_path / "synthetic_bag", profile)
    input_digest_before = _tree_digest(bag)
    job = load_job_config(_manifest(tmp_path, bag))
    roster = build_roster(job)

    reports = audit_roster(roster, profile, num_workers=2)

    assert reports[0].status == "PASS"
    assert abs(reports[0].streams["video.head"].frequency_hz - 30) < 0.01
    assert abs(reports[0].streams["state.left_hand_joint"].frequency_hz - 200) < 0.01
    assert abs(reports[0].streams["state.left_arm_joint"].frequency_hz - 250) < 0.01
    assert abs(reports[0].streams["video.head"].bag_header_offset_sec["p50"] - 0.001) < 1e-9

    result = convert_job(job)
    assert _tree_digest(bag) == input_digest_before

    output = tmp_path / "output"
    info = json.loads((output / "meta/info.json").read_text())
    parquet = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert result["joint_absolute"]["episodes"] == 1
    assert info["total_frames"] == parquet.num_rows
    assert info["fps"] == 30
    assert len(parquet["observation.state"][0].as_py()) == 54
    assert len(list(output.rglob("*.mp4"))) == 3
    assert not (output / "meta/stats.json").exists()
    diagnostics = json.loads(
        (output / "quality/alignment/chunk-000/episode_000000.json").read_text()
    )
    assert diagnostics["frame_count"] == parquet.num_rows
    assert len(diagnostics["head_timestamp_ns"]) == parquet.num_rows
    assert len(diagnostics["streams"]["video.left_wrist"]["signed_skew_ns"]) == parquet.num_rows
    episode_report = json.loads((output / "quality/episode_reports.jsonl").read_text())
    assert episode_report["lag_audit"] == {}
    pipeline_manifest = json.loads((output / "meta/pipeline_manifest.json").read_text())
    assert (
        pipeline_manifest["filtering"]["state.left_hand_joint"]["implementation"]
        == "scipy_sosfiltfilt_regular_grid/v1"
    )

    processing = replace(job.manifest.processing, run_lag_audit=True)
    processing_job = replace(job, manifest=replace(job.manifest, processing=processing))
    processing_reports = audit_processing_roster(processing_job, roster)
    processing_summary = processing_audit_summary(processing_reports)
    assert processing_summary["pass_count"] == 1
    assert processing_summary["lag_sec"]["hand.left"]["p50"] is not None
    assert (
        processing_summary["filtering"]["state.left_hand_joint"]["high_band_retention"]["p50"]
        is not None
    )
    assert (
        processing_summary["outputs"]["joint_absolute"]["streams"]["video.left_wrist"][
            "absolute_skew_sec"
        ]["p95"]
        == 0.0
    )
    first_state = parquet["observation.state"][0].as_py()[0]
    first_action = parquet["action"][0].as_py()[0]
    actual_golden = {
        "raw": {
            "status": reports[0].status,
            "reject_reasons": list(reports[0].reject_reasons),
            "streams": {
                key: {
                    "frequency_hz": round(reports[0].streams[key].frequency_hz, 3),
                    "message_count": reports[0].streams[key].message_count,
                }
                for key in (
                    "state.left_arm_joint",
                    "state.left_hand_joint",
                    "video.head",
                )
            },
        },
        "output": {
            "first_action": round(first_action, 6),
            "first_state": round(first_state, 6),
            "last_frame_index": parquet["frame_index"][-1].as_py(),
            "last_timestamp": round(parquet["timestamp"][-1].as_py(), 6),
            "robot_type": info["robot_type"],
            "state_dimension": len(parquet["observation.state"][0].as_py()),
            "total_frames": info["total_frames"],
            "total_videos": info["total_videos"],
        },
    }
    assert actual_golden == json.loads(GOLDEN.read_text())
    _load_low_dim_with_gr00t(output)


def test_real_rosbag_eef_output_loads_with_gr00t(tmp_path: Path) -> None:
    profile = load_robot_profile(PROFILE)
    bag = write_synthetic_bag(tmp_path / "synthetic_eef_bag", profile)
    job = load_job_config(_manifest(tmp_path, bag, action_space="eef_absolute_hand_absolute"))

    result = convert_job(job)["eef_absolute_hand_absolute"]

    output = tmp_path / "output"
    parquet = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert result["episodes"] == 1
    assert len(parquet["observation.state"][0].as_py()) == 58
    assert len(parquet["action"][0].as_py()) == 58
    _load_low_dim_with_gr00t(output)


def test_synthetic_manus_rosbag_reports_120_hz_hand_commands(tmp_path: Path) -> None:
    profile = load_robot_profile(MANUS_PROFILE)
    bag = write_synthetic_bag(tmp_path / "synthetic_manus_bag", profile)
    job = load_job_config(_manifest(tmp_path, bag, profile_path=MANUS_PROFILE))

    report = audit_roster(build_roster(job), profile)[0]

    assert report.status == "PASS"
    assert abs(report.streams["video.head"].frequency_hz - 30) < 0.01
    assert abs(report.streams["action.left_hand_joint"].frequency_hz - 120) < 0.01
    assert abs(report.streams["state.left_hand_joint"].frequency_hz - 200) < 0.01
    assert abs(report.streams["state.left_arm_joint"].frequency_hz - 250) < 0.01


def _determinism_manifest(
    tmp_path: Path, bags: tuple[Path, ...], *, workers: int, output_name: str
) -> Path:
    manifest = tmp_path / f"manifest_{workers}.yaml"
    roots = "\n".join(f"      - {bag}" for bag in bags)
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {PROFILE}
processing:
  output_fps: 30
  num_workers: {workers}
  activity_padding_before_sec: 0.2
  activity_padding_after_sec: 0.2
  minimum_output_frames: 30
outputs:
  - action_space: joint_absolute
    path: {tmp_path / output_name}
datasets:
  - task_id: synthetic_task
    roots:
{roots}
    instruction: Move both hands and arms
"""
    )
    return manifest


def test_single_and_multi_worker_outputs_have_identical_values_and_indices(tmp_path: Path) -> None:
    profile = load_robot_profile(PROFILE)
    bag_b = write_synthetic_bag(tmp_path / "bag_b", profile)
    bag_a = write_synthetic_bag(tmp_path / "bag_a", profile, epoch_ns=1_700_000_100_000_000_000)
    bags = (bag_b, bag_a)
    single = load_job_config(_determinism_manifest(tmp_path, bags, workers=1, output_name="single"))
    parallel = load_job_config(
        _determinism_manifest(tmp_path, bags, workers=2, output_name="parallel")
    )

    single_result = convert_job(single)["joint_absolute"]
    parallel_result = convert_job(parallel)["joint_absolute"]

    assert single_result["episodes"] == parallel_result["episodes"] == 2
    assert single_result["frames"] == parallel_result["frames"]
    for episode_index in range(2):
        relative = Path(f"data/chunk-000/episode_{episode_index:06d}.parquet")
        left = pq.read_table(tmp_path / "single" / relative)
        right = pq.read_table(tmp_path / "parallel" / relative)
        assert left.schema == right.schema
        for column in left.column_names:
            if column in {"observation.state", "action"}:
                assert np.array_equal(
                    np.asarray(left[column].to_pylist()), np.asarray(right[column].to_pylist())
                )
            else:
                assert left[column].to_pylist() == right[column].to_pylist()
    assert (tmp_path / "single/meta/tasks.jsonl").read_text() == (
        tmp_path / "parallel/meta/tasks.jsonl"
    ).read_text()
    assert (tmp_path / "single/meta/episodes.jsonl").read_text() == (
        tmp_path / "parallel/meta/episodes.jsonl"
    ).read_text()


def test_rejected_episode_does_not_stop_later_valid_episode(tmp_path: Path) -> None:
    profile = load_robot_profile(PROFILE)
    rejected_bag = write_synthetic_bag(tmp_path / "bag_00_rejected", profile, fault="missing_topic")
    valid_bag = write_synthetic_bag(
        tmp_path / "bag_01_valid", profile, epoch_ns=1_700_000_100_000_000_000
    )
    job = load_job_config(
        _determinism_manifest(
            tmp_path,
            (rejected_bag, valid_bag),
            workers=2,
            output_name="failure_isolation",
        )
    )

    result = convert_job(job)["joint_absolute"]

    output = tmp_path / "failure_isolation"
    rejected = [
        json.loads(line)
        for line in (output / "quality/rejected_episodes.jsonl").read_text().splitlines()
    ]
    episodes = [
        json.loads(line) for line in (output / "meta/episodes.jsonl").read_text().splitlines()
    ]
    parquet = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert result["episodes"] == 1
    assert result["rejected"] == 1
    assert rejected[0]["roster_index"] == 0
    assert rejected[0]["reject_reasons"] == [MISSING_REQUIRED_TOPIC]
    assert episodes[0]["episode_index"] == 0
    assert episodes[0]["roster_index"] == 1
    assert np.all(parquet["episode_index"].to_numpy() == 0)
    assert np.array_equal(parquet["index"].to_numpy(), np.arange(parquet.num_rows))


def _multitask_manifest(tmp_path: Path, bag_a: Path, bag_b: Path) -> Path:
    manifest = tmp_path / "multitask_manifest.yaml"
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {PROFILE}
processing:
  output_fps: 30
  num_workers: 2
  activity_padding_before_sec: 0.2
  activity_padding_after_sec: 0.2
  minimum_output_frames: 30
outputs:
  - action_space: joint_absolute
    path: {tmp_path / "multitask_output"}
datasets:
  - task_id: task_a
    roots: [{bag_a}]
    instruction: Perform task A
  - task_id: task_b
    roots: [{bag_b}]
    instruction: Perform task B
"""
    )
    return manifest


def test_multiple_tasks_are_exported_directly_with_stable_mapping(tmp_path: Path) -> None:
    profile = load_robot_profile(PROFILE)
    bag_a = write_synthetic_bag(tmp_path / "bag_a", profile)
    bag_b = write_synthetic_bag(tmp_path / "bag_b", profile, epoch_ns=1_700_000_100_000_000_000)
    job = load_job_config(_multitask_manifest(tmp_path, bag_a, bag_b))

    result = convert_job(job)["joint_absolute"]

    output = tmp_path / "multitask_output"
    info = json.loads((output / "meta/info.json").read_text())
    tasks = [json.loads(line) for line in (output / "meta/tasks.jsonl").read_text().splitlines()]
    episodes = [
        json.loads(line) for line in (output / "meta/episodes.jsonl").read_text().splitlines()
    ]
    first = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    second = pq.read_table(output / "data/chunk-000/episode_000001.parquet")
    assert result["episodes"] == 2
    assert result["rejected"] == 0
    assert tasks == [
        {"task": "Perform task A", "task_index": 0},
        {"task": "Perform task B", "task_index": 1},
    ]
    assert [episode["tasks"] for episode in episodes] == [
        ["Perform task A"],
        ["Perform task B"],
    ]
    assert np.all(first["task_index"].to_numpy() == 0)
    assert np.all(second["task_index"].to_numpy() == 1)
    assert np.array_equal(first["index"].to_numpy(), np.arange(first.num_rows))
    assert np.array_equal(
        second["index"].to_numpy(), np.arange(first.num_rows, first.num_rows + second.num_rows)
    )
    assert info["total_tasks"] == 2
    assert info["total_episodes"] == 2
    assert info["total_frames"] == first.num_rows + second.num_rows
    assert info["total_videos"] == 6
    assert len(list(output.rglob("*.mp4"))) == 6


@pytest.mark.parametrize(
    ("fault", "reason", "stream"),
    [
        ("missing_topic", MISSING_REQUIRED_TOPIC, "video.right_wrist"),
        ("zero_header", ZERO_HEADER_TIMESTAMP, "state.left_hand_joint"),
        (
            "duplicate_header",
            NON_MONOTONIC_HEADER_TIMESTAMP,
            "state.left_hand_joint",
        ),
        ("corrupt_jpeg", CAMERA_DECODE_FAILURE, "video.head"),
    ],
)
def test_real_rosbag_raw_faults_are_rejected(
    tmp_path: Path, fault: str, reason: str, stream: str
) -> None:
    profile = load_robot_profile(PROFILE)
    bag = write_synthetic_bag(tmp_path / fault, profile, fault=fault)
    job = load_job_config(_manifest(tmp_path, bag))
    report = audit_roster(build_roster(job), profile)[0]

    assert report.status == "REJECT"
    assert reason in report.reject_reasons
    assert reason in report.streams[stream].reject_reasons
    assert any(detail["reason"] == reason for detail in report.streams[stream].details)


def test_real_rosbag_active_interval_gap_is_hard_rejected_during_processing(
    tmp_path: Path,
) -> None:
    profile = load_robot_profile(PROFILE)
    bag = write_synthetic_bag(tmp_path / "active_gap", profile, fault="active_gap")
    job = load_job_config(_manifest(tmp_path, bag))
    roster = build_roster(job)

    raw_report = audit_roster(roster, profile)[0]
    processing_report = audit_processing_roster(job, roster)[0]

    assert raw_report.status == "PASS"
    assert RAW_GAP_EXCEEDED in raw_report.streams["state.left_arm_joint"].warning_reasons
    assert processing_report["status"] == "REJECT"
    assert processing_report["reject_reasons"] == [RAW_GAP_EXCEEDED]
    assert processing_report["details"]["stream"] == "state.left_arm_joint"


def test_real_rosbag_stationary_episode_is_rejected_after_raw_qa(tmp_path: Path) -> None:
    profile = load_robot_profile(PROFILE)
    bag = write_synthetic_bag(tmp_path / "stationary", profile, fault="stationary")
    job = load_job_config(_manifest(tmp_path, bag))
    roster = build_roster(job)

    raw_report = audit_roster(roster, profile)[0]
    processing_report = audit_processing_roster(job, roster)[0]

    assert raw_report.status == "PASS"
    assert processing_report["status"] == "REJECT"
    assert processing_report["reject_reasons"] == ["no_valid_motion"]
    assert processing_report["stage"] == "processing"
    assert processing_report["details"] == {
        "streams": list(profile.activity_detection.groups),
        "eef_velocity_threshold": profile.activity_detection.eef_velocity_threshold,
        "joint_velocity_threshold": profile.activity_detection.joint_velocity_threshold,
        "window_sec": profile.activity_detection.window_sec,
    }
