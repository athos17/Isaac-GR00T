import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "lerobot_conversion" / "merge_lerobot_v2_datasets.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("merge_lerobot_v2_datasets", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _make_dataset(root: Path, task: str, lengths: list[int], fps: float = 30.0) -> None:
    info = {
        "codebase_version": "v2.0",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "fps": fps,
        "chunks_size": 1000,
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "total_videos": len(lengths),
        "total_chunks": 1,
        "splits": {"train": f"0:{len(lengths)}"},
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
            "action": {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "observation.images.head_view": {
                "dtype": "video",
                "shape": [4, 4, 3],
                "names": ["height", "width", "rgb"],
                "info": {"video.fps": fps},
            },
        },
    }
    modality = {
        "state": {"state": {"start": 0, "end": 2, "original_key": "observation.state"}},
        "action": {"action": {"start": 0, "end": 2, "original_key": "action"}},
        "video": {"head_view": {"original_key": "observation.images.head_view"}},
        "annotation": {
            "human.action.task_description": {
                "original_key": "annotation.human.action.task_description"
            }
        },
    }
    _write_json(root / "meta" / "info.json", info)
    _write_json(root / "meta" / "modality.json", modality)
    _write_json(root / "meta" / "stats.json", {"unused": True})
    _write_json(root / "meta" / "relative_stats.json", {"unused": True})
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": idx,
                "tasks": [task],
                "length": length,
                "source_file": f"source-{idx}",
            }
            for idx, length in enumerate(lengths)
        ],
    )

    frame_offset = 0
    for episode_index, length in enumerate(lengths):
        df = pd.DataFrame(
            {
                "observation.state": [[episode_index, frame] for frame in range(length)],
                "action": [[episode_index, frame + 1] for frame in range(length)],
                "timestamp": [frame / fps for frame in range(length)],
                "frame_index": list(range(length)),
                "episode_index": [episode_index] * length,
                "index": list(range(frame_offset, frame_offset + length)),
                "task_index": [0] * length,
                "annotation.human.action.task_description": [0] * length,
            }
        )
        data_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path)

        video_path = (
            root
            / "videos"
            / "chunk-000"
            / "observation.images.head_view"
            / f"episode_{episode_index:06d}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"video-{episode_index}".encode())
        frame_offset += length


def test_merges_lerobot_v2_datasets_and_skips_stats(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "merged"
    _make_dataset(first, "grasp the mango", [2, 3], fps=30.0)
    _make_dataset(second, "grasp the rugby ball", [4], fps=31.0)

    module = _load_script_module()
    module.main([str(first), str(second), "--output-dir", str(output)])

    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 9
    assert info["total_tasks"] == 2
    assert info["total_videos"] == 3
    assert info["splits"] == {"train": "0:3"}

    tasks = [json.loads(line) for line in (output / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert tasks == [
        {"task_index": 0, "task": "grasp the mango"},
        {"task_index": 1, "task": "grasp the rugby ball"},
    ]

    episodes = [
        json.loads(line) for line in (output / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert [episode["episode_index"] for episode in episodes] == [0, 1, 2]
    assert [episode["tasks"] for episode in episodes] == [
        ["grasp the mango"],
        ["grasp the mango"],
        ["grasp the rugby ball"],
    ]

    merged_second = pd.read_parquet(output / "data" / "chunk-000" / "episode_000002.parquet")
    assert merged_second["episode_index"].tolist() == [2, 2, 2, 2]
    assert merged_second["index"].tolist() == [5, 6, 7, 8]
    assert merged_second["task_index"].tolist() == [1, 1, 1, 1]
    assert merged_second["annotation.human.action.task_description"].tolist() == [1, 1, 1, 1]

    assert (
        output
        / "videos"
        / "chunk-000"
        / "observation.images.head_view"
        / "episode_000002.mp4"
    ).read_bytes() == b"video-0"
    assert not (output / "meta" / "stats.json").exists()
    assert not (output / "meta" / "relative_stats.json").exists()


def test_normalizes_equivalent_task_text(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "merged"
    canonical = "pump the spray bottle and spray the flowers"
    _make_dataset(first, canonical, [2])
    _make_dataset(second, "pump the spray bottle\\n    and spray the flowers", [3])

    module = _load_script_module()
    module.main([str(first), str(second), "--output-dir", str(output)])

    info = json.loads((output / "meta" / "info.json").read_text())
    tasks = [json.loads(line) for line in (output / "meta" / "tasks.jsonl").read_text().splitlines()]
    episodes = [
        json.loads(line) for line in (output / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert info["total_tasks"] == 1
    assert tasks == [{"task_index": 0, "task": canonical}]
    assert [episode["tasks"] for episode in episodes] == [[canonical], [canonical]]

    merged_second = pd.read_parquet(output / "data" / "chunk-000" / "episode_000001.parquet")
    assert merged_second["task_index"].tolist() == [0, 0, 0]
    assert merged_second["annotation.human.action.task_description"].tolist() == [0, 0, 0]
