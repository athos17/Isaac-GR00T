import sqlite3

import numpy as np

from data_preprocess.plot_rosbag_topic_timestamps import (
    DEFAULT_TOPIC_MAP,
    find_db3_file,
    load_topic_timestamps,
    plot_topic_timestamps,
)


def _write_minimal_rosbag_db(path):
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        con.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, timestamp INTEGER NOT NULL, data BLOB)"
        )
        con.executemany(
            "INSERT INTO topics (id, name) VALUES (?, ?)",
            [
                (1, "/astribot_camera/head_rgbd/color_compress/compressed"),
                (2, "/astribot_arm_left/endpoint_current_states"),
            ],
        )
        con.executemany(
            "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
            [
                (1, 1_000_000_000, b""),
                (1, 1_033_000_000, b""),
                (2, 1_020_000_000, b""),
            ],
        )
        con.commit()
    finally:
        con.close()


def test_find_db3_file_finds_single_database(tmp_path):
    db_path = tmp_path / "sample_0.db3"
    _write_minimal_rosbag_db(db_path)

    assert find_db3_file(tmp_path) == db_path


def test_load_topic_timestamps_returns_relative_seconds(tmp_path):
    _write_minimal_rosbag_db(tmp_path / "sample_0.db3")

    series = load_topic_timestamps(
        tmp_path,
        {
            "head_rgb": "/astribot_camera/head_rgbd/color_compress/compressed",
            "left_eef_state": "/astribot_arm_left/endpoint_current_states",
            "missing": "/missing",
        },
    )

    np.testing.assert_allclose(series["head_rgb"].seconds, np.array([0.0, 0.033]))
    np.testing.assert_allclose(series["left_eef_state"].seconds, np.array([0.02]))
    assert series["missing"].seconds.size == 0


def test_default_topic_map_contains_wuji_alignment_topics():
    assert DEFAULT_TOPIC_MAP["right_eef_action"] == "/astribot_arm_right/endpoint_desired_states"
    assert DEFAULT_TOPIC_MAP["left_wrist_rgb"].endswith("/left_wrist_rgbd/color_compress/compressed")


def test_plot_topic_timestamps_accepts_tick_marker_style(tmp_path):
    _write_minimal_rosbag_db(tmp_path / "sample_0.db3")
    series = load_topic_timestamps(
        tmp_path,
        {
            "head_rgb": "/astribot_camera/head_rgbd/color_compress/compressed",
            "left_eef_state": "/astribot_arm_left/endpoint_current_states",
        },
    )

    output = plot_topic_timestamps(
        series,
        tmp_path / "topic_timestamps.png",
        marker_style="ticks",
    )

    assert output.is_file()
    assert output.stat().st_size > 0
