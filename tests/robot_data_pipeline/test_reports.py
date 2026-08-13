import json

from robot_data_pipeline.export.reports import summarize_quality_dir


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
        "reject_count": 1,
        "reject_reasons": {"missing_required_topic": 1},
    }
