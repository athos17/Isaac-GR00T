from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import yaml

from robot_data_pipeline.adapters.registry import adapter_names
from robot_data_pipeline.models import (
    ActivityConfig,
    ClockConfig,
    DatasetManifest,
    DatasetSource,
    JobConfig,
    OutputRequest,
    OutputSpaceConfig,
    ProcessingConfig,
    RobotProfile,
    SmoothingConfig,
    StreamConfig,
)


class ConfigError(ValueError):
    """Raised when a profile or manifest violates the v1 schema."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _load_yaml(path: Path) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    raw = path.read_bytes()
    try:
        value = yaml.load(raw, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"top-level YAML value must be a mapping: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _sequence(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{where} must be a list")
    return value


def _strict_keys(
    value: dict[str, Any], *, required: Iterable[str], optional: Iterable[str] = (), where: str
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - value.keys())
    unknown = sorted(value.keys() - allowed)
    if missing:
        raise ConfigError(f"{where} is missing fields: {missing}")
    if unknown:
        raise ConfigError(f"{where} has unknown fields: {unknown}")


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return value


def _positive_float(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigError(f"{where} must be a finite positive number")
    return float(value)


def _nonnegative_float(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ConfigError(f"{where} must be a finite non-negative number")
    return float(value)


def _string_tuple(value: Any, where: str, *, nonempty: bool = True) -> tuple[str, ...]:
    items = _sequence(value, where)
    result = tuple(_nonempty_string(item, f"{where}[]") for item in items)
    if nonempty and not result:
        raise ConfigError(f"{where} must not be empty")
    if len(set(result)) != len(result):
        raise ConfigError(f"{where} contains duplicates")
    return result


def _load_profile_document(path: Path, stack: tuple[Path, ...] = ()) -> tuple[dict[str, Any], str]:
    data, own_hash = _load_yaml(path)
    if "extends" not in data:
        return data, own_hash
    _strict_keys(
        data,
        required=("schema_version", "name", "extends", "stream_overrides"),
        where="robot profile overlay",
    )
    if data["schema_version"] != "robot_profile/v1":
        raise ConfigError("robot profile schema_version must be 'robot_profile/v1'")
    base_path = (path.parent / _nonempty_string(data["extends"], "extends")).resolve()
    if base_path in stack or base_path == path:
        raise ConfigError(f"robot profile inheritance cycle: {base_path}")
    base, base_hash = _load_profile_document(base_path, (*stack, path))
    merged = deepcopy(base)
    merged["name"] = _nonempty_string(data["name"], "name")
    overrides = _mapping(data["stream_overrides"], "stream_overrides")
    for stream_key, values in overrides.items():
        if stream_key not in merged["streams"]:
            raise ConfigError(f"stream_overrides references unknown stream: {stream_key}")
        values = _mapping(values, f"stream_overrides.{stream_key}")
        unknown = sorted(values.keys() - merged["streams"][stream_key].keys())
        if unknown:
            raise ConfigError(f"stream_overrides.{stream_key} has unknown fields: {unknown}")
        merged["streams"][stream_key].update(values)
    resolved_hash = hashlib.sha256(f"{base_hash}:{own_hash}".encode()).hexdigest()
    return merged, resolved_hash


def load_robot_profile(path: Path | str) -> RobotProfile:
    profile_path = Path(path).expanduser().resolve()
    data, config_hash = _load_profile_document(profile_path)
    _strict_keys(
        data,
        required=(
            "schema_version",
            "name",
            "clock",
            "output_spaces",
            "streams",
            "activity_detection",
        ),
        where="robot profile",
    )
    if data["schema_version"] != "robot_profile/v1":
        raise ConfigError("robot profile schema_version must be 'robot_profile/v1'")

    clock_data = _mapping(data["clock"], "clock")
    _strict_keys(
        clock_data,
        required=("source", "semantics", "require_nonzero"),
        where="clock",
    )
    if clock_data["source"] != "header":
        raise ConfigError("v1 clock.source must be 'header'")
    if clock_data["semantics"] != "publish_time":
        raise ConfigError("v1 clock.semantics must be 'publish_time'")
    if not isinstance(clock_data["require_nonzero"], bool):
        raise ConfigError("clock.require_nonzero must be boolean")
    if not clock_data["require_nonzero"]:
        raise ConfigError("v1 clock.require_nonzero must be true")
    clock = ClockConfig(
        source="header",
        semantics=_nonempty_string(clock_data["semantics"], "clock.semantics"),
        require_nonzero=clock_data["require_nonzero"],
    )

    streams_data = _mapping(data["streams"], "streams")
    if not streams_data:
        raise ConfigError("streams must not be empty")
    streams: dict[str, StreamConfig] = {}
    seen_topics: set[str] = set()
    for key, raw_stream in streams_data.items():
        stream_key = _nonempty_string(key, "stream key")
        item = _mapping(raw_stream, f"streams.{stream_key}")
        _strict_keys(
            item,
            required=(
                "topic",
                "adapter",
                "message_type",
                "semantic",
                "required",
                "expected_hz",
                "alignment",
            ),
            optional=(
                "max_skew_sec",
                "hard_max_skew_sec",
                "max_consecutive_skew_violations",
                "max_skew_violation_ratio",
                "max_gap_sec",
                "max_action_age_sec",
                "names",
                "allow_unnamed",
                "unit",
                "range",
                "continuous_joints",
                "quaternion_order",
                "base_frame",
                "tool_frame",
                "smoothing",
            ),
            where=f"streams.{stream_key}",
        )
        topic = _nonempty_string(item["topic"], f"streams.{stream_key}.topic")
        if topic in seen_topics:
            raise ConfigError(f"duplicate topic in streams: {topic}")
        seen_topics.add(topic)
        if not isinstance(item["required"], bool):
            raise ConfigError(f"streams.{stream_key}.required must be boolean")
        semantic = _nonempty_string(item["semantic"], f"streams.{stream_key}.semantic")
        allowed_semantics = {
            "rgb_image",
            "joint_position_measured",
            "joint_position_command_absolute",
            "eef_pose_measured",
            "eef_pose_command_absolute",
        }
        if semantic not in allowed_semantics:
            raise ConfigError(f"streams.{stream_key}.semantic is invalid: {semantic!r}")
        alignment = item["alignment"]
        if alignment not in {"anchor", "nearest", "linear", "slerp", "pose", "previous"}:
            raise ConfigError(f"streams.{stream_key}.alignment is invalid: {alignment!r}")
        smoothing_data = _mapping(item.get("smoothing", {"type": "none"}), "smoothing")
        _strict_keys(
            smoothing_data,
            required=("type",),
            optional=("cutoff_hz", "order", "zero_phase"),
            where=f"streams.{stream_key}.smoothing",
        )
        smoothing_type = smoothing_data["type"]
        if smoothing_type not in {"none", "butterworth"}:
            raise ConfigError(f"unsupported smoothing type: {smoothing_type!r}")
        cutoff = smoothing_data.get("cutoff_hz")
        order = smoothing_data.get("order")
        zero_phase = smoothing_data.get("zero_phase")
        if smoothing_type == "butterworth":
            if cutoff is None or order is None or zero_phase is None:
                raise ConfigError(f"streams.{stream_key}.smoothing is incomplete")
            cutoff = _positive_float(cutoff, f"streams.{stream_key}.smoothing.cutoff_hz")
            if not isinstance(order, int) or isinstance(order, bool) or order <= 0:
                raise ConfigError(
                    f"streams.{stream_key}.smoothing.order must be a positive integer"
                )
            if not isinstance(zero_phase, bool):
                raise ConfigError(f"streams.{stream_key}.smoothing.zero_phase must be boolean")
            if not zero_phase:
                raise ConfigError(f"streams.{stream_key} only supports zero-phase filtering in v1")
        names = _string_tuple(item.get("names", []), f"streams.{stream_key}.names", nonempty=False)
        allow_unnamed = item.get("allow_unnamed", False)
        if not isinstance(allow_unnamed, bool):
            raise ConfigError(f"streams.{stream_key}.allow_unnamed must be boolean")
        if allow_unnamed and not names:
            raise ConfigError(f"streams.{stream_key}.allow_unnamed requires configured names")
        continuous = _string_tuple(
            item.get("continuous_joints", []),
            f"streams.{stream_key}.continuous_joints",
            nonempty=False,
        )
        if not set(continuous).issubset(names):
            raise ConfigError(f"streams.{stream_key}.continuous_joints must be included in names")
        value_range = item.get("range")
        parsed_range = None
        if value_range is not None:
            values = _sequence(value_range, f"streams.{stream_key}.range")
            if len(values) != 2 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in values
            ):
                raise ConfigError(f"streams.{stream_key}.range must contain two finite numbers")
            parsed_range = (float(values[0]), float(values[1]))
            if parsed_range[0] >= parsed_range[1]:
                raise ConfigError(f"streams.{stream_key}.range lower bound must be smaller")
        if semantic == "rgb_image":
            expected_alignment = "anchor" if stream_key == "video.head" else "nearest"
            if alignment != expected_alignment:
                raise ConfigError(f"streams.{stream_key}.alignment must be {expected_alignment!r}")
            if smoothing_type != "none":
                raise ConfigError(f"streams.{stream_key} camera smoothing is not supported")
            if alignment == "nearest" and "max_skew_sec" not in item:
                raise ConfigError(f"streams.{stream_key}.max_skew_sec is required")
            if alignment == "nearest":
                required_policy = {
                    "hard_max_skew_sec",
                    "max_consecutive_skew_violations",
                    "max_skew_violation_ratio",
                }
                missing_policy = sorted(required_policy - item.keys())
                if missing_policy:
                    raise ConfigError(
                        f"streams.{stream_key} is missing wrist skew policy fields: "
                        f"{missing_policy}"
                    )
        else:
            if not names and semantic.startswith("joint_position"):
                raise ConfigError(f"streams.{stream_key}.names must not be empty")
            if parsed_range is None:
                raise ConfigError(f"streams.{stream_key}.range is required")
            _nonempty_string(item.get("unit"), f"streams.{stream_key}.unit")
            if semantic.endswith("measured"):
                expected_alignment = "pose" if semantic.startswith("eef_pose") else "linear"
                if not stream_key.startswith("state.") or alignment != expected_alignment:
                    raise ConfigError(
                        f"streams.{stream_key} measured semantic requires a state key and "
                        f"{expected_alignment!r} alignment"
                    )
                if "max_gap_sec" not in item:
                    raise ConfigError(f"streams.{stream_key}.max_gap_sec is required")
            else:
                if not stream_key.startswith("action.") or alignment != "previous":
                    raise ConfigError(
                        f"streams.{stream_key} command semantic requires an action key and "
                        "'previous' alignment"
                    )
                if "max_action_age_sec" not in item:
                    raise ConfigError(f"streams.{stream_key}.max_action_age_sec is required")
                if smoothing_type != "none":
                    raise ConfigError(f"streams.{stream_key} action smoothing is not supported")
        if semantic.startswith("eef_pose"):
            if item.get("quaternion_order") != "xyzw":
                raise ConfigError(f"streams.{stream_key}.quaternion_order must be 'xyzw' in v1")
            _nonempty_string(item.get("base_frame"), f"streams.{stream_key}.base_frame")
            _nonempty_string(item.get("tool_frame"), f"streams.{stream_key}.tool_frame")
        if item["required"] and "max_gap_sec" not in item:
            raise ConfigError(f"streams.{stream_key}.max_gap_sec is required")
        max_skew_sec = (
            _positive_float(item["max_skew_sec"], f"streams.{stream_key}.max_skew_sec")
            if "max_skew_sec" in item
            else None
        )
        hard_max_skew_sec = (
            _positive_float(item["hard_max_skew_sec"], f"streams.{stream_key}.hard_max_skew_sec")
            if "hard_max_skew_sec" in item
            else None
        )
        max_consecutive_skew_violations = item.get("max_consecutive_skew_violations")
        if max_consecutive_skew_violations is not None and (
            not isinstance(max_consecutive_skew_violations, int)
            or isinstance(max_consecutive_skew_violations, bool)
            or max_consecutive_skew_violations < 0
        ):
            raise ConfigError(
                f"streams.{stream_key}.max_consecutive_skew_violations must be a "
                "non-negative integer"
            )
        max_skew_violation_ratio = item.get("max_skew_violation_ratio")
        if max_skew_violation_ratio is not None:
            max_skew_violation_ratio = _nonnegative_float(
                max_skew_violation_ratio,
                f"streams.{stream_key}.max_skew_violation_ratio",
            )
            if max_skew_violation_ratio > 1:
                raise ConfigError(
                    f"streams.{stream_key}.max_skew_violation_ratio must be at most 1"
                )
        if (
            max_skew_sec is not None
            and hard_max_skew_sec is not None
            and hard_max_skew_sec <= max_skew_sec
        ):
            raise ConfigError(f"streams.{stream_key}.hard_max_skew_sec must exceed max_skew_sec")
        streams[stream_key] = StreamConfig(
            key=stream_key,
            topic=topic,
            adapter=_nonempty_string(item["adapter"], f"streams.{stream_key}.adapter"),
            message_type=_nonempty_string(
                item["message_type"], f"streams.{stream_key}.message_type"
            ),
            semantic=semantic,
            required=item["required"],
            expected_hz=_positive_float(item["expected_hz"], f"streams.{stream_key}.expected_hz"),
            alignment=alignment,
            max_skew_sec=max_skew_sec,
            hard_max_skew_sec=hard_max_skew_sec,
            max_consecutive_skew_violations=max_consecutive_skew_violations,
            max_skew_violation_ratio=max_skew_violation_ratio,
            max_gap_sec=(
                _positive_float(item["max_gap_sec"], f"streams.{stream_key}.max_gap_sec")
                if "max_gap_sec" in item
                else None
            ),
            max_action_age_sec=(
                _positive_float(
                    item["max_action_age_sec"], f"streams.{stream_key}.max_action_age_sec"
                )
                if "max_action_age_sec" in item
                else None
            ),
            names=names,
            allow_unnamed=allow_unnamed,
            unit=item.get("unit"),
            value_range=parsed_range,
            continuous_joints=continuous,
            quaternion_order=item.get("quaternion_order"),
            base_frame=item.get("base_frame"),
            tool_frame=item.get("tool_frame"),
            smoothing=SmoothingConfig(smoothing_type, cutoff, order, zero_phase),
        )
        if streams[stream_key].adapter not in adapter_names():
            raise ConfigError(
                f"streams.{stream_key}.adapter is not registered: {streams[stream_key].adapter!r}"
            )

    spaces_data = _mapping(data["output_spaces"], "output_spaces")
    if not spaces_data:
        raise ConfigError("output_spaces must not be empty")
    output_spaces = {}
    for name, raw_space in spaces_data.items():
        space_name = _nonempty_string(name, "output space name")
        item = _mapping(raw_space, f"output_spaces.{space_name}")
        _strict_keys(
            item,
            required=("state_groups", "action_groups"),
            where=f"output_spaces.{space_name}",
        )
        state_groups = _string_tuple(item["state_groups"], f"{space_name}.state_groups")
        action_groups = _string_tuple(item["action_groups"], f"{space_name}.action_groups")
        missing = sorted((set(state_groups) | set(action_groups)) - streams.keys())
        if missing:
            raise ConfigError(f"output space {space_name!r} references missing streams: {missing}")
        invalid_state = [group for group in state_groups if not group.startswith("state.")]
        invalid_action = [group for group in action_groups if not group.startswith("action.")]
        if invalid_state or invalid_action:
            raise ConfigError(
                f"output space {space_name!r} has invalid state/action groups: "
                f"state={invalid_state}, action={invalid_action}"
            )
        output_spaces[space_name] = OutputSpaceConfig(state_groups, action_groups)

    activity_data = _mapping(data["activity_detection"], "activity_detection")
    _strict_keys(
        activity_data,
        required=(
            "groups",
            "eef_velocity_threshold",
            "joint_velocity_threshold",
            "window_sec",
        ),
        where="activity_detection",
    )
    groups = _string_tuple(activity_data["groups"], "activity_detection.groups")
    missing_activity = sorted(set(groups) - streams.keys())
    if missing_activity:
        raise ConfigError(f"activity_detection references missing streams: {missing_activity}")
    for group in groups:
        if not group.startswith("state.") or not streams[group].semantic.endswith("measured"):
            raise ConfigError(f"activity_detection group is not measured state: {group}")

    return RobotProfile(
        path=profile_path,
        schema_version=data["schema_version"],
        name=_nonempty_string(data["name"], "name"),
        clock=clock,
        output_spaces=output_spaces,
        streams=streams,
        activity_detection=ActivityConfig(
            groups=groups,
            eef_velocity_threshold=_nonnegative_float(
                activity_data["eef_velocity_threshold"], "eef_velocity_threshold"
            ),
            joint_velocity_threshold=_nonnegative_float(
                activity_data["joint_velocity_threshold"], "joint_velocity_threshold"
            ),
            window_sec=_positive_float(activity_data["window_sec"], "window_sec"),
        ),
        config_hash=config_hash,
    )


def _resolve_config_path(value: Any, manifest_path: Path) -> Path:
    raw = Path(_nonempty_string(value, "profile"))
    candidates = [raw] if raw.is_absolute() else [manifest_path.parent / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def load_dataset_manifest(path: Path | str) -> DatasetManifest:
    manifest_path = Path(path).expanduser().resolve()
    data, config_hash = _load_yaml(manifest_path)
    _strict_keys(
        data,
        required=("schema_version", "profile", "processing", "outputs", "datasets"),
        where="dataset manifest",
    )
    if data["schema_version"] != "dataset_manifest/v1":
        raise ConfigError("dataset manifest schema_version must be 'dataset_manifest/v1'")

    processing_data = _mapping(data["processing"], "processing")
    _strict_keys(
        processing_data,
        required=(
            "output_fps",
            "num_workers",
            "activity_padding_before_sec",
            "activity_padding_after_sec",
            "minimum_output_frames",
        ),
        optional=(
            "video_workers",
            "video_encoder_preset",
            "video_encoder_threads",
            "run_lag_audit",
        ),
        where="processing",
    )
    workers = processing_data["num_workers"]
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ConfigError("processing.num_workers must be a positive integer")
    minimum_frames = processing_data["minimum_output_frames"]
    if (
        not isinstance(minimum_frames, int)
        or isinstance(minimum_frames, bool)
        or minimum_frames <= 0
    ):
        raise ConfigError("processing.minimum_output_frames must be a positive integer")
    video_workers = processing_data.get("video_workers", 3)
    if not isinstance(video_workers, int) or isinstance(video_workers, bool) or video_workers <= 0:
        raise ConfigError("processing.video_workers must be a positive integer")
    video_encoder_preset = _nonempty_string(
        processing_data.get("video_encoder_preset", "veryfast"),
        "processing.video_encoder_preset",
    )
    if video_encoder_preset not in {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }:
        raise ConfigError("processing.video_encoder_preset is not a valid libx264 preset")
    video_encoder_threads = processing_data.get("video_encoder_threads", 0)
    if (
        not isinstance(video_encoder_threads, int)
        or isinstance(video_encoder_threads, bool)
        or video_encoder_threads < 0
    ):
        raise ConfigError("processing.video_encoder_threads must be a non-negative integer")
    run_lag_audit = processing_data.get("run_lag_audit", False)
    if not isinstance(run_lag_audit, bool):
        raise ConfigError("processing.run_lag_audit must be boolean")
    processing = ProcessingConfig(
        output_fps=_positive_float(processing_data["output_fps"], "processing.output_fps"),
        num_workers=workers,
        activity_padding_before_sec=_nonnegative_float(
            processing_data["activity_padding_before_sec"], "activity_padding_before_sec"
        ),
        activity_padding_after_sec=_nonnegative_float(
            processing_data["activity_padding_after_sec"], "activity_padding_after_sec"
        ),
        minimum_output_frames=minimum_frames,
        video_workers=video_workers,
        video_encoder_preset=video_encoder_preset,
        video_encoder_threads=video_encoder_threads,
        run_lag_audit=run_lag_audit,
    )

    outputs = []
    output_paths: set[Path] = set()
    output_spaces: set[str] = set()
    for index, raw_output in enumerate(_sequence(data["outputs"], "outputs")):
        item = _mapping(raw_output, f"outputs[{index}]")
        _strict_keys(
            item,
            required=("action_space", "path"),
            optional=("eef_rotation_format",),
            where=f"outputs[{index}]",
        )
        output_path = (
            Path(_nonempty_string(item["path"], f"outputs[{index}].path")).expanduser().resolve()
        )
        if output_path in output_paths:
            raise ConfigError(f"duplicate output path: {output_path}")
        output_paths.add(output_path)
        action_space = _nonempty_string(item["action_space"], "action_space")
        if action_space in output_spaces:
            raise ConfigError(f"duplicate output action_space: {action_space}")
        output_spaces.add(action_space)
        outputs.append(
            OutputRequest(
                action_space=action_space,
                path=output_path,
                eef_rotation_format=item.get("eef_rotation_format"),
            )
        )
    if not outputs:
        raise ConfigError("outputs must not be empty")
    for index, left in enumerate(outputs):
        for right in outputs[index + 1 :]:
            if left.path.is_relative_to(right.path) or right.path.is_relative_to(left.path):
                raise ConfigError(f"output paths overlap: {left.path} and {right.path}")

    datasets = []
    task_ids: set[str] = set()
    roots_seen: set[Path] = set()
    for index, raw_dataset in enumerate(_sequence(data["datasets"], "datasets")):
        item = _mapping(raw_dataset, f"datasets[{index}]")
        _strict_keys(
            item,
            required=("task_id", "roots", "instruction"),
            where=f"datasets[{index}]",
        )
        task_id = _nonempty_string(item["task_id"], f"datasets[{index}].task_id")
        if task_id in task_ids:
            raise ConfigError(f"duplicate task_id: {task_id}")
        task_ids.add(task_id)
        roots = tuple(
            Path(_nonempty_string(root, f"datasets[{index}].roots[]")).expanduser().resolve()
            for root in _sequence(item["roots"], f"datasets[{index}].roots")
        )
        if not roots:
            raise ConfigError(f"datasets[{index}].roots must not be empty")
        for root in roots:
            if not root.is_dir():
                raise ConfigError(f"input root does not exist: {root}")
            if root in roots_seen:
                raise ConfigError(f"duplicate input root: {root}")
            roots_seen.add(root)
        datasets.append(
            DatasetSource(
                task_id=task_id,
                roots=roots,
                instruction=_nonempty_string(item["instruction"], "instruction"),
            )
        )
    if not datasets:
        raise ConfigError("datasets must not be empty")

    all_roots = [root for dataset in datasets for root in dataset.roots]
    for index, left in enumerate(all_roots):
        for right in all_roots[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ConfigError(f"input roots overlap: {left} and {right}")
    for output in outputs:
        for root in all_roots:
            if (
                output.path == root
                or output.path.is_relative_to(root)
                or root.is_relative_to(output.path)
            ):
                raise ConfigError(f"input/output paths overlap: {root} and {output.path}")

    return DatasetManifest(
        path=manifest_path,
        schema_version=data["schema_version"],
        profile_path=_resolve_config_path(data["profile"], manifest_path),
        processing=processing,
        outputs=tuple(outputs),
        datasets=tuple(datasets),
        config_hash=config_hash,
    )


def load_job_config(path: Path | str) -> JobConfig:
    manifest = load_dataset_manifest(path)
    profile = load_robot_profile(manifest.profile_path)
    for output in manifest.outputs:
        if output.action_space not in profile.output_spaces:
            raise ConfigError(f"unknown output action_space: {output.action_space}")
        space = profile.output_spaces[output.action_space]
        uses_eef = any(
            profile.streams[key].semantic.startswith("eef_pose")
            for key in (*space.state_groups, *space.action_groups)
        )
        if uses_eef and output.eef_rotation_format != "rot6d":
            raise ConfigError(
                f"EEF output {output.action_space!r} requires eef_rotation_format: rot6d"
            )
        if not uses_eef and output.eef_rotation_format is not None:
            raise ConfigError("eef_rotation_format is only valid for EEF output spaces")
    if manifest.processing.output_fps != 30.0:
        raise ConfigError("v1 processing.output_fps must be 30")
    for stream in profile.streams.values():
        if stream.smoothing.type == "butterworth":
            assert stream.smoothing.cutoff_hz is not None
            if stream.smoothing.cutoff_hz >= manifest.processing.output_fps / 2:
                raise ConfigError(
                    f"{stream.key} smoothing cutoff must be below output Nyquist frequency"
                )
    return JobConfig(manifest=manifest, profile=profile)
