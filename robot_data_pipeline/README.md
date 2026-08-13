# ROS2 Bag Data Pipeline

This package implements the configuration-driven pipeline described in
`docs/rosbag_data_pipeline_implementation_plan.md`. It is independent of
`data_preprocess` and `wuji_pipeline` at runtime.

Install the optional dependencies:

```bash
uv sync --extra data-pipeline --extra dev
```

Video export also requires `ffmpeg` and `ffprobe` with H.264 support on `PATH`.

Validate a manifest and its deterministic input roster:

```bash
python -m robot_data_pipeline validate \
  --manifest robot_data_pipeline/configs/datasets/legacy_smoke.yaml
```

Run raw QA without creating a LeRobot dataset:

```bash
python -m robot_data_pipeline audit \
  --manifest robot_data_pipeline/configs/datasets/legacy_pilot.yaml \
  --report-dir /tmp/robot_data_pipeline_legacy_pilot_audit
```

Convert every PASS episode into the requested independent LeRobot v2 datasets:

```bash
python -m robot_data_pipeline convert \
  --manifest robot_data_pipeline/configs/datasets/legacy_smoke.yaml
```

Configure video throughput under `processing`:

```yaml
processing:
  num_workers: 16
  video_workers: 3
  video_encoder_preset: veryfast
  video_encoder_threads: 8
```

`video_workers: 3` encodes the head and two wrist streams concurrently. `veryfast` is the
recommended throughput/size setting; use `medium` to reproduce the previous encoder preset or
`ultrafast` when conversion latency matters more than compression efficiency. A thread count of
`0` lets libx264 choose automatically.

`convert` never edits input bags and refuses existing output paths. `--overwrite`
only accepts a directory containing this pipeline's `meta/pipeline_manifest.json`.
The pipeline deliberately does not create `stats.json` or `relative_stats.json`.
Generate ordinary training statistics explicitly after conversion:

```bash
python -c "from gr00t.data.stats import generate_stats; generate_stats('/path/to/output')"
```

Each PASS episode also has a deterministic per-frame alignment sidecar under
`quality/alignment/`, containing real head timestamps and source timestamps, camera skew,
interpolation gaps, action age, and camera frame reuse. Summarize raw audit, processing audit, or
conversion quality directories with:

```bash
python -m robot_data_pipeline summarize --quality-dir /path/to/quality-or-audit-directory
```

Legacy pilot results and the outstanding confirmed Manus + Orin data requirement are recorded in
`robot_data_pipeline/PILOT_REPORT.md`.
