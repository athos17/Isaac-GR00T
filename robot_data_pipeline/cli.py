from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from robot_data_pipeline.catalog import build_roster, roster_to_dict
from robot_data_pipeline.config import ConfigError, load_job_config
from robot_data_pipeline.export.reports import (
    audit_summary,
    processing_audit_summary,
    summarize_quality_dir,
    write_audit_reports,
    write_processing_audit_reports,
)
from robot_data_pipeline.quality.raw import audit_roster
from robot_data_pipeline.runner import audit_processing_roster, convert_job


def _add_manifest_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m robot_data_pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate configuration and input roster")
    _add_manifest_argument(validate)

    audit = subparsers.add_parser("audit", help="run raw rosbag quality checks")
    _add_manifest_argument(audit)
    audit.add_argument("--report-dir", type=Path)
    audit.add_argument("--episode-index", type=int, action="append")
    audit.add_argument(
        "--processing",
        action="store_true",
        help="also run canonicalization, activity, filtering, lag and alignment QA",
    )
    audit.add_argument(
        "--max-episodes", type=int, help="audit only the first N selected episodes (pilot helper)"
    )

    convert = subparsers.add_parser("convert", help="run the complete conversion pipeline")
    _add_manifest_argument(convert)
    convert.add_argument("--overwrite", action="store_true")

    summarize = subparsers.add_parser("summarize", help="summarize an existing quality report")
    summarize.add_argument("--quality-dir", type=Path, required=True)
    summarize.add_argument("--dry-run", action="store_true")
    return parser


def _validate(manifest: Path) -> tuple[object, object]:
    job = load_job_config(manifest)
    roster = build_roster(job)
    return job, roster


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "summarize":
            if args.dry_run:
                quality_dir = args.quality_dir.resolve()
                candidates = (
                    quality_dir / "episode_reports.jsonl",
                    quality_dir / "processing_episode_reports.jsonl",
                )
                report_path = next((path for path in candidates if path.is_file()), candidates[0])
                print(
                    json.dumps({"report_path": str(report_path), "exists": report_path.is_file()})
                )
            else:
                print(json.dumps(summarize_quality_dir(args.quality_dir.resolve()), indent=2))
            return 0

        job, roster = _validate(args.manifest)
        if args.command == "validate" or args.dry_run:
            print(json.dumps(roster_to_dict(roster), indent=2))
            return 0
        if args.command == "convert":
            print(json.dumps(convert_job(job, overwrite=args.overwrite), indent=2, sort_keys=True))
            return 0

        indices = args.episode_index
        if indices is not None:
            available = {episode.roster_index for episode in roster.episodes}
            missing = sorted(set(indices) - available)
            if missing:
                raise ConfigError(f"unknown episode indices: {missing}")
        selected = [
            episode.roster_index
            for episode in roster.episodes
            if indices is None or episode.roster_index in set(indices)
        ]
        if args.max_episodes is not None:
            if args.max_episodes <= 0:
                raise ConfigError("--max-episodes must be positive")
            selected = selected[: args.max_episodes]
        if args.processing:
            reports = audit_processing_roster(job, roster, episode_indices=selected)
            summary = processing_audit_summary(reports)
            if args.report_dir is not None:
                write_processing_audit_reports(args.report_dir.resolve(), reports)
                summary["report_dir"] = str(args.report_dir.resolve())
        else:
            reports = audit_roster(
                roster,
                job.profile,
                episode_indices=selected,
                num_workers=job.manifest.processing.num_workers,
            )
            summary = audit_summary(reports)
            if args.report_dir is not None:
                write_audit_reports(args.report_dir.resolve(), roster, reports)
                summary["report_dir"] = str(args.report_dir.resolve())
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["reject_count"] == 0 else 2
    except (ConfigError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
