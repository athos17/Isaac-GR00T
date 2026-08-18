from pathlib import Path

import pytest
from robot_data_pipeline.config import ConfigError, load_job_config, load_robot_profile


REPO_ROOT = Path(__file__).parents[2]
PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_legacy.yaml"
MANUS_PROFILE = REPO_ROOT / "robot_data_pipeline/configs/robots/wuji_astribot_manus.yaml"


def test_legacy_profile_is_complete() -> None:
    profile = load_robot_profile(PROFILE)

    assert profile.schema_version == "robot_profile/v1"
    assert set(profile.output_spaces) == {
        "joint_absolute",
        "eef_absolute_hand_absolute",
    }
    assert len(profile.streams) == 15
    assert profile.streams["state.left_arm_joint"].allow_unnamed is True
    assert profile.streams["state.left_hand_joint"].expected_hz == 200
    wrist = profile.streams["video.left_wrist"]
    assert wrist.max_skew_sec == 0.02
    assert wrist.hard_max_skew_sec == 0.04
    assert wrist.max_consecutive_skew_violations == 1
    assert wrist.max_skew_violation_ratio == 0.005


def test_manus_profile_resolves_with_120_hz_hand_commands() -> None:
    profile = load_robot_profile(MANUS_PROFILE)

    assert profile.name == "wuji_astribot_manus"
    assert len(profile.streams) == 15
    assert profile.streams["state.left_hand_joint"].expected_hz == 200
    assert profile.streams["action.left_hand_joint"].expected_hz == 120


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    profile = tmp_path / "duplicate.yaml"
    profile.write_text("schema_version: robot_profile/v1\nschema_version: robot_profile/v1\n")

    with pytest.raises(ConfigError, match="duplicate YAML key"):
        load_robot_profile(profile)


def test_unknown_profile_field_is_rejected(tmp_path: Path) -> None:
    text = PROFILE.read_text().replace(
        "name: wuji_astribot_legacy", "name: wuji_astribot_legacy\nunknown: true"
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text(text)

    with pytest.raises(ConfigError, match="unknown fields"):
        load_robot_profile(profile)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("semantics: publish_time", "semantics: receive_time", "must be 'publish_time'"),
        ("require_nonzero: true", "require_nonzero: false", "must be true"),
    ],
)
def test_v1_clock_contract_is_enforced(tmp_path: Path, old: str, new: str, match: str) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(PROFILE.read_text().replace(old, new, 1))

    with pytest.raises(ConfigError, match=match):
        load_robot_profile(profile_path)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("quaternion_order: xyzw", "quaternion_order: wxyz", "must be 'xyzw'"),
        ("zero_phase: true", "zero_phase: false", "only supports zero-phase"),
    ],
)
def test_unsupported_profile_semantics_are_rejected(tmp_path, old, new, match) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(PROFILE.read_text().replace(old, new, 1))

    with pytest.raises(ConfigError, match=match):
        load_robot_profile(profile_path)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("hard_max_skew_sec: 0.04", "hard_max_skew_sec: 0.02", "must exceed"),
        (
            "max_consecutive_skew_violations: 1",
            "max_consecutive_skew_violations: -1",
            "non-negative integer",
        ),
        ("max_skew_violation_ratio: 0.005", "max_skew_violation_ratio: 1.1", "at most 1"),
    ],
)
def test_invalid_wrist_skew_policy_is_rejected(tmp_path, old, new, match) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(PROFILE.read_text().replace(old, new, 1))

    with pytest.raises(ConfigError, match=match):
        load_robot_profile(profile_path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("expected_hz: 30", "expected_hz: .nan"),
        ("range: [-6.4, 6.4]", "range: [-6.4, .inf]"),
    ],
)
def test_non_finite_profile_numbers_are_rejected(tmp_path: Path, old: str, new: str) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(PROFILE.read_text().replace(old, new, 1))

    with pytest.raises(ConfigError, match="finite"):
        load_robot_profile(profile_path)


def test_activity_detection_rejects_action_stream(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        PROFILE.read_text().replace(
            "activity_detection:\n  groups:\n    - state.left_eef",
            "activity_detection:\n  groups:\n    - action.left_eef",
            1,
        )
    )

    with pytest.raises(ConfigError, match="not measured state"):
        load_robot_profile(profile_path)


def test_input_output_overlap_is_rejected(tmp_path: Path) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {PROFILE}
processing:
  output_fps: 30
  num_workers: 1
  activity_padding_before_sec: 0.5
  activity_padding_after_sec: 0.5
  minimum_output_frames: 30
outputs:
  - action_space: joint_absolute
    path: {bag / "output"}
datasets:
  - task_id: task
    roots: [{bag}]
    instruction: Do the task
"""
    )

    with pytest.raises(ConfigError, match="input/output paths overlap"):
        load_job_config(manifest)


def test_video_encoding_defaults_and_overrides(tmp_path: Path) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {PROFILE}
processing:
  output_fps: 30
  num_workers: 1
  activity_padding_before_sec: 0.5
  activity_padding_after_sec: 0.5
  minimum_output_frames: 30
  video_workers: 2
  video_encoder_preset: ultrafast
  video_encoder_threads: 4
outputs:
  - action_space: joint_absolute
    path: {tmp_path / "output"}
datasets:
  - task_id: task
    roots: [{bag}]
    instruction: Do the task
"""
    )

    processing = load_job_config(manifest).manifest.processing

    assert processing.video_workers == 2
    assert processing.video_encoder_preset == "ultrafast"
    assert processing.video_encoder_threads == 4


def test_lag_audit_defaults_off_and_can_be_enabled(tmp_path: Path) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    base = f"""
schema_version: dataset_manifest/v1
profile: {PROFILE}
processing:
  output_fps: 30
  num_workers: 1
  activity_padding_before_sec: 0.5
  activity_padding_after_sec: 0.5
  minimum_output_frames: 30
outputs:
  - action_space: joint_absolute
    path: {{output}}
datasets:
  - task_id: task
    roots: [{bag}]
    instruction: Do the task
"""
    default_manifest = tmp_path / "default.yaml"
    default_manifest.write_text(base.format(output=tmp_path / "default-output"))
    enabled_manifest = tmp_path / "enabled.yaml"
    enabled_manifest.write_text(
        base.replace(
            "minimum_output_frames: 30", "minimum_output_frames: 30\n  run_lag_audit: true"
        ).format(output=tmp_path / "enabled-output")
    )

    assert load_job_config(default_manifest).manifest.processing.run_lag_audit is False
    assert load_job_config(enabled_manifest).manifest.processing.run_lag_audit is True


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("video_workers", "0", "video_workers"),
        ("video_encoder_preset", "turbo", "video_encoder_preset"),
        ("video_encoder_threads", "-1", "video_encoder_threads"),
    ],
)
def test_invalid_video_encoding_config_is_rejected(tmp_path, field, value, match) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {PROFILE}
processing:
  output_fps: 30
  num_workers: 1
  activity_padding_before_sec: 0.5
  activity_padding_after_sec: 0.5
  minimum_output_frames: 30
  {field}: {value}
outputs:
  - action_space: joint_absolute
    path: {tmp_path / "output"}
datasets:
  - task_id: task
    roots: [{bag}]
    instruction: Do the task
"""
    )

    with pytest.raises(ConfigError, match=match):
        load_job_config(manifest)


def test_nested_input_roots_are_rejected_during_configuration(tmp_path: Path) -> None:
    root = tmp_path / "input"
    nested = root / "nested"
    nested.mkdir(parents=True)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {PROFILE}
processing:
  output_fps: 30
  num_workers: 1
  activity_padding_before_sec: 0.5
  activity_padding_after_sec: 0.5
  minimum_output_frames: 30
outputs:
  - action_space: joint_absolute
    path: {tmp_path / "output"}
datasets:
  - task_id: task
    roots: [{root}, {nested}]
    instruction: Do the task
"""
    )

    with pytest.raises(ConfigError, match="input roots overlap"):
        load_job_config(manifest)


@pytest.mark.parametrize("include_rotation", [False, True])
def test_eef_rotation_requirement_uses_stream_semantics(
    tmp_path: Path, include_rotation: bool
) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        PROFILE.read_text().replace("eef_absolute_hand_absolute:", "cartesian_absolute:", 1)
    )
    input_root = tmp_path / "input"
    input_root.mkdir()
    rotation = "\n    eef_rotation_format: rot6d" if include_rotation else ""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {profile_path}
processing:
  output_fps: 30
  num_workers: 1
  activity_padding_before_sec: 0.5
  activity_padding_after_sec: 0.5
  minimum_output_frames: 30
outputs:
  - action_space: cartesian_absolute
    path: {tmp_path / "output"}{rotation}
datasets:
  - task_id: task
    roots: [{input_root}]
    instruction: Do the task
"""
    )

    if include_rotation:
        load_job_config(manifest)
    else:
        with pytest.raises(ConfigError, match="requires eef_rotation_format"):
            load_job_config(manifest)


@pytest.mark.parametrize(
    ("second_task_id", "match"),
    [("task", "duplicate task_id"), ("other", "duplicate input root")],
)
def test_duplicate_task_and_input_root_are_rejected(tmp_path, second_task_id, match) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {PROFILE}
processing:
  output_fps: 30
  num_workers: 1
  activity_padding_before_sec: 0.5
  activity_padding_after_sec: 0.5
  minimum_output_frames: 30
outputs:
  - action_space: joint_absolute
    path: {tmp_path / "output"}
datasets:
  - task_id: task
    roots: [{bag}]
    instruction: First
  - task_id: {second_task_id}
    roots: [{bag}]
    instruction: Second
"""
    )

    with pytest.raises(ConfigError, match=match):
        load_job_config(manifest)


def test_duplicate_output_action_space_is_rejected(tmp_path) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: dataset_manifest/v1
profile: {PROFILE}
processing:
  output_fps: 30
  num_workers: 1
  activity_padding_before_sec: 0.5
  activity_padding_after_sec: 0.5
  minimum_output_frames: 30
outputs:
  - action_space: joint_absolute
    path: {tmp_path / "output_a"}
  - action_space: joint_absolute
    path: {tmp_path / "output_b"}
datasets:
  - task_id: task
    roots: [{bag}]
    instruction: Do the task
"""
    )

    with pytest.raises(ConfigError, match="duplicate output action_space"):
        load_job_config(manifest)
