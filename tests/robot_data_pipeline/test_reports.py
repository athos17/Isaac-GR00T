import json

from robot_data_pipeline.export.reports import processing_audit_summary, summarize_quality_dir


def _write_jsonl(path, values) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def test_summarize_processing_audit_reports(tmp_path) -> None:
    _write_jsonl(
        tmp_path / "processing_episode_reports.jsonl",
        [
            {
                "status": "PASS",
                "reject_reasons": [],
                "activity": {"active_start_ns": 0, "active_end_ns": 1_000_000_000},
                "filtering": {},
                "lag_audit": {},
                "outputs": {},
            },
            {
                "status": "REJECT",
                "reject_reasons": ["wrist_camera_skew_exceeded"],
                "activity": {"active_start_ns": 0, "active_end_ns": 2_000_000_000},
                "filtering": {},
                "lag_audit": {},
                "outputs": {},
            },
        ],
    )

    summary = summarize_quality_dir(tmp_path)

    assert summary["episode_count"] == 2
    assert summary["pass_count"] == 1
    assert summary["reject_count"] == 1
    assert summary["reject_reasons"] == {"wrist_camera_skew_exceeded": 1}
    assert summary["activity_duration_sec"]["max"] == 2.0


def test_processing_summary_includes_wrist_warnings_and_boundary_trims() -> None:
    reports = [
        {
            "status": "PASS_WITH_WARNING",
            "reject_reasons": [],
            "warning_reasons": ["wrist_camera_skew_warning"],
            "outputs": {
                "joint_absolute": {
                    "streams": {
                        "video.head": {
                            "boundary_trimmed_before": 1,
                            "boundary_trimmed_after": 0,
                        },
                        "video.left_wrist": {
                            "soft_skew_violation_count": 1,
                            "soft_skew_violation_ratio": 0.002,
                            "maximum_consecutive_soft_skew_violations": 1,
                        },
                    }
                }
            },
        },
        {
            "status": "PASS",
            "reject_reasons": [],
            "warning_reasons": [],
            "outputs": {
                "joint_absolute": {
                    "streams": {
                        "video.head": {
                            "boundary_trimmed_before": 0,
                            "boundary_trimmed_after": 2,
                        },
                        "video.left_wrist": {
                            "soft_skew_violation_count": 0,
                            "soft_skew_violation_ratio": 0.0,
                            "maximum_consecutive_soft_skew_violations": 0,
                        },
                    }
                }
            },
        },
    ]

    summary = processing_audit_summary(reports)

    assert summary["pass_count"] == 2
    assert summary["pass_with_warning_count"] == 1
    assert summary["warning_reasons"] == {"wrist_camera_skew_warning": 1}
    streams = summary["outputs"]["joint_absolute"]["streams"]
    assert streams["video.head"]["boundary_trimmed_before"]["max"] == 1.0
    assert streams["video.head"]["boundary_trimmed_after"]["max"] == 2.0
    assert streams["video.left_wrist"]["soft_skew_violation_count"]["max"] == 1.0
    assert streams["video.left_wrist"]["soft_skew_violation_ratio"]["max"] == 0.002


def test_summarize_conversion_combines_passed_and_rejected_reports(tmp_path) -> None:
    _write_jsonl(tmp_path / "raw_episode_reports.jsonl", [{"status": "PASS"}])
    _write_jsonl(
        tmp_path / "episode_reports.jsonl",
        [{"status": "PASS", "reject_reasons": []}],
    )
    _write_jsonl(
        tmp_path / "rejected_episodes.jsonl",
        [
            {
                "reject_reasons": ["missing_required_topic"],
            }
        ],
    )

    summary = summarize_quality_dir(tmp_path)

    assert summary == {
        "episode_count": 2,
        "pass_count": 1,
        "pass_with_warning_count": 0,
        "reject_count": 1,
        "reject_reasons": {"missing_required_topic": 1},
        "warning_reasons": {},
    }
