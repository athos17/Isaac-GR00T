import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "data_preprocess" / "filter_lerobot_hand_quality.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("filter_lerobot_hand_quality", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def _make_dataset(
    root: Path,
    episodes: list[tuple[np.ndarray, np.ndarray]],
    source_files: list[str] | None = None,
) -> None:
    lengths = [len(state) for state, _ in episodes]
    info = {
        "codebase_version": "v2.0",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "fps": 30.0,
        "chunks_size": 1000,
        "total_episodes": len(episodes),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "total_videos": len(episodes),
        "total_chunks": 1,
        "splits": {"train": f"0:{len(episodes)}"},
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["left_hand.joint0", "left_hand.joint1"],
            },
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["left_hand.joint0", "left_hand.joint1"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    modality = {
        "state": {"left_hand_joints": {"start": 0, "end": 2}},
        "action": {"left_hand_joints": {"start": 0, "end": 2}},
        "video": {"head_view": {"original_key": "observation.images.head_view"}},
    }
    source_files = source_files or [
        f"/tmp/test/arm_hand_vr_20260101_120{index:02d}00" for index in range(len(lengths))
    ]
    metadata = [
        {
            "episode_index": index,
            "tasks": ["test task"],
            "length": length,
            "source_file": source_files[index],
        }
        for index, length in enumerate(lengths)
    ]
    _write_json(root / "meta" / "info.json", info)
    _write_json(root / "meta" / "modality.json", modality)
    _write_json(root / "meta" / "stats.json", {"stale": True})
    _write_json(root / "meta" / "relative_stats.json", {"stale": True})
    _write_jsonl(root / "meta" / "episodes.jsonl", metadata)
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "test task"}])

    global_index = 0
    for episode_index, (state, action) in enumerate(episodes):
        length = len(state)
        frame = pd.DataFrame(
            {
                "observation.state": list(state.astype(np.float32)),
                "action": list(action.astype(np.float32)),
                "timestamp": np.arange(length, dtype=np.float32) / 30.0,
                "frame_index": np.arange(length),
                "episode_index": episode_index,
                "index": np.arange(global_index, global_index + length),
                "task_index": 0,
            }
        )
        parquet = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet, index=False)
        video = (
            root
            / "videos"
            / "chunk-000"
            / "observation.images.head_view"
            / f"episode_{episode_index:06d}.mp4"
        )
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{episode_index}".encode())
        global_index += length


def _signals() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_count = 80
    base = np.sin(np.linspace(0, 6 * np.pi, sample_count))
    good_state = np.column_stack([base, np.full(sample_count, 0.25)])
    good_action = good_state.copy()
    shuffled = np.random.default_rng(1234).permutation(base)
    bad_correlation = np.column_stack([shuffled, np.full(sample_count, 0.25)])
    bad_offset = good_state + np.array([0.0, 0.2])
    return good_state, good_action, bad_correlation, bad_offset


def test_tracking_offset_uses_future_state_at_best_lag() -> None:
    module = _load_module()
    action = np.cumsum(np.random.default_rng(7).normal(size=100))
    lag = 4
    state = np.empty_like(action)
    state[:lag] = action[0]
    state[lag:] = action[:-lag]

    metric, failures, _ = module.audit_joint(
        state,
        action,
        "left_hand.joint0",
        module.HandQualityThresholds(),
    )

    assert not failures
    assert metric["best_lag_frames"] == lag
    assert metric["lag_aligned_median_action_state_offset"] == 0.0
    assert metric["lag_aligned_p95_absolute_error"] == 0.0
    assert metric["same_time_p95_absolute_error"] > 0.0


def test_audit_warns_on_tracking_difference_but_allows_static_joint(tmp_path: Path) -> None:
    module = _load_module()
    good_state, good_action, bad_correlation, bad_offset = _signals()
    root = tmp_path / "source"
    _make_dataset(
        root,
        [
            (good_state, good_action),
            (good_state, bad_correlation),
            (good_state, bad_offset),
        ],
    )

    _, _, _, audits, _ = module.audit_dataset(root, module.HandQualityThresholds())

    assert audits[0].passed
    assert audits[1].passed
    assert "left_hand.joint0:low_correlation" in audits[1].warnings
    assert audits[2].passed
    assert "left_hand.joint1:tracking_offset" in audits[2].warnings
    assert not audits[0].joints[1]["failures"]
    assert audits[0].joints[1]["same_time_correlation"] is None


def test_export_compacts_indexes_preserves_frames_and_quarantines_rejections(
    tmp_path: Path,
) -> None:
    module = _load_module()
    good_state, good_action, bad_correlation, bad_offset = _signals()
    bad_non_finite = good_state.copy()
    bad_non_finite[10, 0] = np.nan
    root = tmp_path / "source"
    output = tmp_path / "filtered"
    _make_dataset(
        root,
        [
            (good_state, good_action),
            (good_state, bad_correlation),
            (good_state, bad_offset),
            (bad_non_finite, good_action),
        ],
    )
    source_info_before = (root / "meta" / "info.json").read_bytes()

    summary = module.main(["--input-dir", str(root), "--output-dir", str(output)])

    assert summary["passed_episodes"] == 3
    assert summary["rejected_episodes"] == 1
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 240
    assert not (output / "meta" / "stats.json").exists()
    assert not (output / "meta" / "relative_stats.json").exists()

    second = pd.read_parquet(output / "data" / "chunk-000" / "episode_000002.parquet")
    assert second["episode_index"].tolist() == [2] * 80
    assert second["index"].tolist() == list(range(160, 240))
    assert second["frame_index"].tolist() == list(range(80))
    np.testing.assert_allclose(second["timestamp"], np.arange(80) / 30.0, rtol=0, atol=1e-6)
    assert (
        output / "videos" / "chunk-000" / "observation.images.head_view" / "episode_000001.mp4"
    ).read_bytes() == b"video-1"

    quarantine_info = json.loads((output / "quarantine" / "meta" / "info.json").read_text())
    assert quarantine_info["total_episodes"] == 1
    report = json.loads((output / "quarantine" / "filter_report.json").read_text())
    assert report["rejected_episodes"][0]["episode_index"] == 3
    assert "left_hand.joint0:non_finite" in report["rejected_episodes"][0]["reasons"]
    clean_audit = [
        json.loads(line)
        for line in (output / "meta" / "hand_quality.jsonl").read_text().splitlines()
    ]
    assert "left_hand.joint0:low_correlation" in clean_audit[1]["warnings"]
    assert (root / "meta" / "info.json").read_bytes() == source_info_before
    assert (root / "meta" / "stats.json").exists()

    _, _, _, clean_audits, _ = module.audit_dataset(output, module.HandQualityThresholds())
    assert all(audit.passed for audit in clean_audits)


def test_session_mapping_shift_is_quarantined_as_a_batch(tmp_path: Path) -> None:
    module = _load_module()
    good_state, good_action, _, _ = _signals()
    shifted_action = good_action + 0.5
    source_files = [f"/tmp/test/arm_hand_vr_20260101_120{i:02d}00" for i in range(5)] + [
        "/tmp/test/arm_hand_vr_20260101_121000",
        "/tmp/test/arm_hand_vr_20260101_121030",
        "/tmp/test/arm_hand_vr_20260101_121100",
    ]
    root = tmp_path / "source"
    output = tmp_path / "filtered"
    _make_dataset(
        root,
        [(good_state, good_action)] * 5 + [(good_state, shifted_action)] * 3,
        source_files,
    )

    summary = module.main(["--input-dir", str(root), "--output-dir", str(output)])

    assert summary["passed_episodes"] == 5
    assert summary["rejected_episodes"] == 3
    assert len(summary["mapping_shift_sessions"]) == 1
    assert summary["mapping_shift_sessions"][0]["episode_indices"] == [5, 6, 7]


def test_session_mapping_shift_requires_reliable_episode_support(tmp_path: Path) -> None:
    module = _load_module()
    good_state, good_action, bad_correlation, _ = _signals()
    source_files = [f"/tmp/test/arm_hand_vr_20260101_120{i:02d}00" for i in range(5)] + [
        "/tmp/test/arm_hand_vr_20260101_121000",
        "/tmp/test/arm_hand_vr_20260101_121030",
        "/tmp/test/arm_hand_vr_20260101_121100",
    ]
    root = tmp_path / "source"
    shifted_good_action = good_action.copy()
    shifted_good_action[:, 0] += 0.5
    shifted_bad_action = bad_correlation.copy()
    shifted_bad_action[:, 0] += 0.5
    _make_dataset(
        root,
        [(good_state, good_action)] * 5
        + [
            (good_state, shifted_good_action),
            (good_state, shifted_bad_action),
            (good_state, shifted_bad_action),
        ],
        source_files,
    )

    _, _, _, audits, mapping_shifts = module.audit_dataset(
        root,
        module.HandQualityThresholds(),
    )

    assert not mapping_shifts
    assert all(audit.passed for audit in audits)
