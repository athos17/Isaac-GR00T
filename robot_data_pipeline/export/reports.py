from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

from robot_data_pipeline.catalog import roster_to_dict
from robot_data_pipeline.models import EpisodeAudit, ProcessingRoster


def write_alignment_diagnostics(output_dir: Path, aligned, *, episode_index: int) -> None:
    chunk = episode_index // 1000
    path = (
        output_dir
        / "quality"
        / "alignment"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": "robot_data_pipeline/alignment_diagnostics/v1",
        "episode_index": episode_index,
        "action_space": aligned.action_space,
        "frame_count": len(aligned.timestamps),
        "head_timestamp_ns": aligned.head_timestamps_ns.tolist(),
        "streams": {
            key: {name: values.tolist() for name, values in sorted(diagnostic.items())}
            for key, diagnostic in sorted(aligned.diagnostics.items())
        },
    }
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def episode_audit_to_dict(report: EpisodeAudit) -> dict[str, Any]:
    return asdict(report)


def audit_summary(reports: Iterable[EpisodeAudit]) -> dict[str, Any]:
    reports = list(reports)
    reasons: dict[str, int] = {}
    for report in reports:
        for reason in report.reject_reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
    passed = sum(report.status == "PASS" for report in reports)
    stream_names = sorted({name for report in reports for name in report.streams})
    streams = {}
    for name in stream_names:
        stream_reports = [report.streams[name] for report in reports if name in report.streams]

        def distribution(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"p01": None, "p50": None, "p99": None}
            values = sorted(values)

            def percentile(fraction: float) -> float:
                index = round(fraction * (len(values) - 1))
                return float(values[index])

            return {"p01": percentile(0.01), "p50": percentile(0.5), "p99": percentile(0.99)}

        streams[name] = {
            "episode_count": len(stream_reports),
            "frequency_hz": distribution(
                [
                    report.frequency_hz
                    for report in stream_reports
                    if report.frequency_hz is not None
                ]
            ),
            "max_interval_sec": distribution(
                [
                    float(report.interval_sec["max"])
                    for report in stream_reports
                    if report.interval_sec.get("max") is not None
                ]
            ),
            "offset_p50_sec": distribution(
                [
                    float(report.bag_header_offset_sec["p50"])
                    for report in stream_reports
                    if report.bag_header_offset_sec.get("p50") is not None
                ]
            ),
            "offset_drift_sec": distribution(
                [
                    report.offset_drift_sec
                    for report in stream_reports
                    if report.offset_drift_sec is not None
                ]
            ),
        }
    return {
        "episode_count": len(reports),
        "pass_count": passed,
        "reject_count": len(reports) - passed,
        "reject_reasons": dict(sorted(reasons.items())),
        "streams": streams,
    }


def write_audit_reports(
    output_dir: Path, roster: ProcessingRoster, reports: list[EpisodeAudit]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "processing_roster.json").write_text(
        json.dumps(roster_to_dict(roster), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(audit_summary(reports), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "episode_reports.jsonl").open("w", encoding="utf-8") as file:
        for report in reports:
            file.write(json.dumps(episode_audit_to_dict(report), sort_keys=True) + "\n")
    with (output_dir / "rejected_episodes.jsonl").open("w", encoding="utf-8") as file:
        for report in reports:
            if report.status == "REJECT":
                file.write(json.dumps(episode_audit_to_dict(report), sort_keys=True) + "\n")


def summarize_quality_dir(quality_dir: Path) -> dict[str, Any]:
    processing_reports_path = quality_dir / "processing_episode_reports.jsonl"
    if processing_reports_path.is_file():
        reports = [
            json.loads(line)
            for line in processing_reports_path.read_text().splitlines()
            if line.strip()
        ]
        return processing_audit_summary(reports)

    reports_path = quality_dir / "episode_reports.jsonl"
    if not reports_path.is_file():
        raise FileNotFoundError(
            f"missing episode report: expected {reports_path} or {processing_reports_path}"
        )
    reports = [json.loads(line) for line in reports_path.read_text().splitlines() if line.strip()]
    # Conversion outputs keep passed aligned reports and rejected reports in separate files.
    if (quality_dir / "raw_episode_reports.jsonl").is_file():
        rejected_path = quality_dir / "rejected_episodes.jsonl"
        if rejected_path.is_file():
            reports.extend(
                json.loads(line) for line in rejected_path.read_text().splitlines() if line.strip()
            )
    reasons: dict[str, int] = {}
    warnings: dict[str, int] = {}
    for report in reports:
        for reason in report["reject_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason in report.get("warning_reasons", []):
            warnings[reason] = warnings.get(reason, 0) + 1
    passed = sum(report.get("status") in {"PASS", "PASS_WITH_WARNING"} for report in reports)
    return {
        "episode_count": len(reports),
        "pass_count": passed,
        "pass_with_warning_count": sum(
            report.get("status") == "PASS_WITH_WARNING" for report in reports
        ),
        "reject_count": len(reports) - passed,
        "reject_reasons": dict(sorted(reasons.items())),
        "warning_reasons": dict(sorted(warnings.items())),
    }


def _numeric_distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None}
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def processing_audit_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    warnings: dict[str, int] = {}
    for report in reports:
        for reason in report["reject_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason in report.get("warning_reasons", []):
            warnings[reason] = warnings.get(reason, 0) + 1
    passed = sum(report["status"] in {"PASS", "PASS_WITH_WARNING"} for report in reports)
    passed_with_warning = sum(report["status"] == "PASS_WITH_WARNING" for report in reports)
    activity_durations = [
        (report["activity"]["active_end_ns"] - report["activity"]["active_start_ns"]) * 1e-9
        for report in reports
        if "activity" in report
    ]
    filter_streams = sorted({key for report in reports for key in report.get("filtering", {})})
    filtering = {}
    for key in filter_streams:
        entries = [
            report["filtering"][key] for report in reports if key in report.get("filtering", {})
        ]
        filtering[key] = {
            "estimated_input_hz": _numeric_distribution(
                [float(entry["estimated_input_hz"]) for entry in entries]
            ),
            "low_band_retention": _numeric_distribution(
                [
                    float(entry["spectral"]["low_band_retention"])
                    for entry in entries
                    if entry["spectral"]["low_band_retention"] is not None
                ]
            ),
            "high_band_retention": _numeric_distribution(
                [
                    float(entry["spectral"]["high_band_retention"])
                    for entry in entries
                    if entry["spectral"]["high_band_retention"] is not None
                ]
            ),
        }
    lag_groups = sorted({key for report in reports for key in report.get("lag_audit", {})})
    lag = {
        key: _numeric_distribution(
            [
                float(report["lag_audit"][key]["consensus_lag_sec"])
                for report in reports
                if key in report.get("lag_audit", {})
                and report["lag_audit"][key]["consensus_lag_sec"] is not None
            ]
        )
        for key in lag_groups
    }
    outputs = {}
    output_names = sorted({key for report in reports for key in report.get("outputs", {})})
    for output_name in output_names:
        output_reports = [
            report["outputs"][output_name]
            for report in reports
            if output_name in report.get("outputs", {})
            and "streams" in report["outputs"][output_name]
        ]
        stream_names = sorted(
            {key for output_report in output_reports for key in output_report["streams"]}
        )
        stream_summary = {}
        for key in stream_names:
            metrics = [output_report["streams"][key] for output_report in output_reports]
            stream_summary[key] = {}
            for metric_name in ("absolute_skew_sec", "bracket_gap_sec", "action_age_sec"):
                stream_summary[key][metric_name] = _numeric_distribution(
                    [
                        float(metric[metric_name]["p95"])
                        for metric in metrics
                        if metric_name in metric
                    ]
                )
            stream_summary[key]["reused_frame_ratio"] = _numeric_distribution(
                [
                    float(metric["reused_frame_ratio"])
                    for metric in metrics
                    if "reused_frame_ratio" in metric
                ]
            )
            stream_summary[key]["soft_skew_violation_count"] = _numeric_distribution(
                [
                    float(metric["soft_skew_violation_count"])
                    for metric in metrics
                    if "soft_skew_violation_count" in metric
                ]
            )
            stream_summary[key]["soft_skew_violation_ratio"] = _numeric_distribution(
                [
                    float(metric["soft_skew_violation_ratio"])
                    for metric in metrics
                    if "soft_skew_violation_ratio" in metric
                ]
            )
            stream_summary[key]["maximum_consecutive_soft_skew_violations"] = _numeric_distribution(
                [
                    float(metric["maximum_consecutive_soft_skew_violations"])
                    for metric in metrics
                    if "maximum_consecutive_soft_skew_violations" in metric
                ]
            )
            for metric_name in ("boundary_trimmed_before", "boundary_trimmed_after"):
                stream_summary[key][metric_name] = _numeric_distribution(
                    [float(metric[metric_name]) for metric in metrics if metric_name in metric]
                )
        outputs[output_name] = {"streams": stream_summary}
    return {
        "episode_count": len(reports),
        "pass_count": passed,
        "pass_with_warning_count": passed_with_warning,
        "reject_count": len(reports) - passed,
        "reject_reasons": dict(sorted(reasons.items())),
        "warning_reasons": dict(sorted(warnings.items())),
        "activity_duration_sec": _numeric_distribution(activity_durations),
        "filtering": filtering,
        "lag_sec": lag,
        "outputs": outputs,
    }


def write_processing_audit_reports(output_dir: Path, reports: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "processing_episode_reports.jsonl").open("w", encoding="utf-8") as file:
        for report in reports:
            file.write(json.dumps(report, sort_keys=True) + "\n")
    (output_dir / "processing_summary.json").write_text(
        json.dumps(processing_audit_summary(reports), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
