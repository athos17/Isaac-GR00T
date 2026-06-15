#!/usr/bin/env python
"""Plot ROS2 bag topic timestamps as horizontal point timelines.

This script reads timestamp metadata directly from a ROS2 sqlite3 bag (`*.db3`).
It does not deserialize messages and does not require a ROS Python environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TOPIC_MAP = {
    "left_eef_state": "/astribot_arm_left/endpoint_current_states",
    "right_eef_state": "/astribot_arm_right/endpoint_current_states",
    "left_eef_action": "/astribot_arm_left/endpoint_desired_states",
    "right_eef_action": "/astribot_arm_right/endpoint_desired_states",
    "left_hand_state": "/left_hand/joint_states",
    "right_hand_state": "/right_hand/joint_states",
    "left_hand_action": "/left_hand/joint_commands",
    "right_hand_action": "/right_hand/joint_commands",
    "head_rgb": "/astribot_camera/head_rgbd/color_compress/compressed",
    "left_wrist_rgb": "/astribot_camera/left_wrist_rgbd/color_compress/compressed",
    "right_wrist_rgb": "/astribot_camera/right_wrist_rgbd/color_compress/compressed",
}


@dataclass(frozen=True)
class TopicTimestamps:
    label: str
    topic: str
    seconds: np.ndarray
    absolute_seconds: np.ndarray


def find_db3_file(bag_path: str | Path) -> Path:
    """Return the sqlite3 database for a ROS2 bag path."""
    path = Path(bag_path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".db3":
            raise ValueError(f"Expected a .db3 file, got: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Bag path does not exist: {path}")

    db_files = sorted(path.glob("*.db3"))
    if not db_files:
        compressed = sorted(path.glob("*.db3.zstd"))
        if compressed:
            raise FileNotFoundError(
                f"No uncompressed .db3 file found in {path}. Decompress first: {compressed[0]}"
            )
        raise FileNotFoundError(f"No .db3 file found in bag directory: {path}")
    if len(db_files) > 1:
        raise ValueError(f"Expected one .db3 file in {path}, found {len(db_files)}: {db_files}")
    return db_files[0]


def _topic_id_map(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute("SELECT id, name FROM topics").fetchall()
    return {str(name): int(topic_id) for topic_id, name in rows}


def _message_timestamps_ns(connection: sqlite3.Connection, topic_id: int) -> np.ndarray:
    rows = connection.execute(
        "SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic_id,),
    ).fetchall()
    if not rows:
        return np.empty((0,), dtype=np.int64)
    return np.asarray([row[0] for row in rows], dtype=np.int64)


def load_topic_timestamps(
    bag_path: str | Path,
    topic_map: dict[str, str] | None = None,
) -> dict[str, TopicTimestamps]:
    """Load requested topic timestamps and normalize them to the global first timestamp."""
    db_path = find_db3_file(bag_path)
    requested = topic_map or DEFAULT_TOPIC_MAP

    with sqlite3.connect(str(db_path)) as connection:
        topic_ids = _topic_id_map(connection)
        raw: dict[str, tuple[str, np.ndarray]] = {}
        for label, topic in requested.items():
            topic_id = topic_ids.get(topic)
            timestamps_ns = (
                _message_timestamps_ns(connection, topic_id)
                if topic_id is not None
                else np.empty((0,), dtype=np.int64)
            )
            raw[label] = (topic, timestamps_ns)

    non_empty = [timestamps for _, timestamps in raw.values() if timestamps.size > 0]
    origin_ns = min(int(timestamps[0]) for timestamps in non_empty) if non_empty else 0

    series = {}
    for label, (topic, timestamps_ns) in raw.items():
        absolute_seconds = timestamps_ns.astype(np.float64) * 1e-9
        seconds = (timestamps_ns.astype(np.float64) - float(origin_ns)) * 1e-9
        series[label] = TopicTimestamps(
            label=label,
            topic=topic,
            seconds=seconds,
            absolute_seconds=absolute_seconds,
        )
    return series


def _parse_topic_overrides(values: list[str] | None) -> dict[str, str]:
    topic_map = dict(DEFAULT_TOPIC_MAP)
    if not values:
        return topic_map
    for value in values:
        if "=" in value:
            label, topic = value.split("=", 1)
        else:
            topic = value
            label = topic.rstrip("/").split("/")[-1] or topic
        if not label or not topic:
            raise ValueError(f"Invalid --topic value: {value!r}")
        topic_map[label] = topic
    return topic_map


def plot_topic_timestamps(
    series: dict[str, TopicTimestamps],
    output_path: str | Path,
    *,
    title: str | None = None,
    marker_style: str = "ticks",
    point_size: float = 8.0,
    dpi: int = 180,
) -> Path:
    """Render a horizontal timestamp scatter plot."""
    if marker_style not in {"ticks", "dots", "both"}:
        raise ValueError("marker_style must be one of: ticks, dots, both")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = list(series.keys())
    fig_height = max(3.0, 0.45 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    y_positions = np.arange(len(labels), dtype=np.float64)
    for y, label in zip(y_positions, labels):
        topic_series = series[label]
        if topic_series.seconds.size:
            if marker_style in {"ticks", "both"}:
                ax.vlines(
                    topic_series.seconds,
                    y - 0.32,
                    y + 0.32,
                    alpha=0.85,
                    linewidth=0.45,
                )
            if marker_style in {"dots", "both"}:
                ax.scatter(
                    topic_series.seconds,
                    np.full_like(topic_series.seconds, y),
                    s=point_size,
                    alpha=0.65,
                    linewidths=0,
                    zorder=3,
                )
        else:
            ax.text(
                0.0,
                y,
                "missing/no messages",
                va="center",
                ha="left",
                fontsize=8,
                color="crimson",
            )

    tick_labels = [
        f"{label} ({series[label].seconds.size})\n{series[label].topic}" for label in labels
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Time from first requested topic message (s)")
    ax.set_title(title or "ROS bag topic timestamps")
    ax.grid(axis="x", color="0.85", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", color="0.92", linestyle="-", linewidth=0.6)

    non_empty = [item.seconds for item in series.values() if item.seconds.size > 0]
    if non_empty:
        max_time = max(float(item[-1]) for item in non_empty)
        ax.set_xlim(left=-0.02 * max(max_time, 1.0), right=max_time + 0.02 * max(max_time, 1.0))

    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_path", help="ROS2 bag directory or uncompressed .db3 file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path. Defaults to <bag_path>/topic_timestamps.png",
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=None,
        help=(
            "Topic to plot. Use LABEL=/topic/name or just /topic/name. "
            "Can be repeated. Defaults to Wuji/Astribot alignment topics."
        ),
    )
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument(
        "--marker-style",
        choices=["ticks", "dots", "both"],
        default="ticks",
        help="How to draw each timestamp. Ticks are easiest to see for dense topics.",
    )
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bag_path = Path(args.bag_path).expanduser().resolve()
    if args.output is None:
        output = (bag_path if bag_path.is_dir() else bag_path.parent) / "topic_timestamps.png"
    else:
        output = Path(args.output)

    topic_map = _parse_topic_overrides(args.topic)
    series = load_topic_timestamps(bag_path, topic_map)
    saved = plot_topic_timestamps(
        series,
        output,
        title=args.title or f"Topic timestamps: {bag_path.name}",
        marker_style=args.marker_style,
        point_size=args.point_size,
        dpi=args.dpi,
    )
    print(f"Wrote {saved}")


if __name__ == "__main__":
    main()
