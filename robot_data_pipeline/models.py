from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class ClockConfig:
    source: str
    semantics: str
    require_nonzero: bool


@dataclass(frozen=True)
class SmoothingConfig:
    type: str = "none"
    cutoff_hz: float | None = None
    order: int | None = None
    zero_phase: bool | None = None


@dataclass(frozen=True)
class StreamConfig:
    key: str
    topic: str
    adapter: str
    message_type: str
    semantic: str
    required: bool
    expected_hz: float
    alignment: str
    max_skew_sec: float | None = None
    max_gap_sec: float | None = None
    max_action_age_sec: float | None = None
    names: tuple[str, ...] = ()
    allow_unnamed: bool = False
    unit: str | None = None
    value_range: tuple[float, float] | None = None
    continuous_joints: tuple[str, ...] = ()
    quaternion_order: str | None = None
    base_frame: str | None = None
    tool_frame: str | None = None
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)


@dataclass(frozen=True)
class OutputSpaceConfig:
    state_groups: tuple[str, ...]
    action_groups: tuple[str, ...]


@dataclass(frozen=True)
class ActivityConfig:
    groups: tuple[str, ...]
    eef_velocity_threshold: float
    joint_velocity_threshold: float
    window_sec: float


@dataclass(frozen=True)
class RobotProfile:
    path: Path
    schema_version: str
    name: str
    clock: ClockConfig
    output_spaces: dict[str, OutputSpaceConfig]
    streams: dict[str, StreamConfig]
    activity_detection: ActivityConfig
    config_hash: str


@dataclass(frozen=True)
class ProcessingConfig:
    output_fps: float
    num_workers: int
    activity_padding_before_sec: float
    activity_padding_after_sec: float
    minimum_output_frames: int
    video_workers: int = 3
    video_encoder_preset: str = "veryfast"
    video_encoder_threads: int = 0


@dataclass(frozen=True)
class OutputRequest:
    action_space: str
    path: Path
    eef_rotation_format: str | None = None


@dataclass(frozen=True)
class DatasetSource:
    task_id: str
    roots: tuple[Path, ...]
    instruction: str


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    schema_version: str
    profile_path: Path
    processing: ProcessingConfig
    outputs: tuple[OutputRequest, ...]
    datasets: tuple[DatasetSource, ...]
    config_hash: str


@dataclass(frozen=True)
class JobConfig:
    manifest: DatasetManifest
    profile: RobotProfile


@dataclass(frozen=True)
class BagMetadata:
    storage_identifier: str
    relative_files: tuple[str, ...]
    starting_time_ns: int
    duration_ns: int
    message_count: int
    topics: dict[str, tuple[str, int]]


@dataclass(frozen=True)
class EpisodeSpec:
    roster_index: int
    task_index: int
    task_id: str
    instruction: str
    root: Path
    bag_path: Path
    metadata_hash: str
    metadata: BagMetadata


@dataclass(frozen=True)
class ProcessingRoster:
    manifest_hash: str
    profile_hash: str
    episodes: tuple[EpisodeSpec, ...]


@dataclass(frozen=True)
class RawMessage:
    stream_key: str
    topic: str
    message_type: str
    header_time_ns: int
    bag_time_ns: int
    sequence: int
    message: Any


@dataclass(frozen=True)
class AdaptedPayload:
    values: tuple[float, ...] = ()
    names: tuple[str, ...] = ()
    encoded_image: bytes | None = None
    image_format: str | None = None


@dataclass(frozen=True)
class JointPositionSeries:
    timestamps_ns: np.ndarray
    values: np.ndarray
    names: tuple[str, ...]


@dataclass(frozen=True)
class PositionCommandSeries:
    timestamps_ns: np.ndarray
    values: np.ndarray
    names: tuple[str, ...]


@dataclass(frozen=True)
class PoseSeries:
    timestamps_ns: np.ndarray
    translations: np.ndarray
    quaternions_xyzw: np.ndarray


@dataclass(frozen=True)
class ImageSeries:
    timestamps_ns: np.ndarray
    bag_timestamps_ns: np.ndarray
    encoded_images: tuple[bytes, ...]
    formats: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalEpisode:
    streams: dict[str, JointPositionSeries | PositionCommandSeries | PoseSeries | ImageSeries]


@dataclass(frozen=True)
class ActivityInterval:
    active_start_ns: int
    active_end_ns: int
    padded_start_ns: int
    padded_end_ns: int


@dataclass(frozen=True)
class AxisLagAudit:
    status: str
    best_lag_sec: float | None
    peak_correlation: float | None
    secondary_peak_margin: float | None
    direction_agreement: float | None
    tracking_error: dict[str, float | None]
    valid_duration_sec: float
    windowed_lag_sec: dict[str, float | None]


@dataclass(frozen=True)
class LagAudit:
    axes: tuple[AxisLagAudit, ...]
    consensus_lag_sec: float | None
    active_axis_count: int


@dataclass(frozen=True)
class AlignedEpisodeData:
    action_space: str
    state: np.ndarray
    action: np.ndarray
    timestamps: np.ndarray
    head_timestamps_ns: np.ndarray
    images: dict[str, tuple[bytes, ...]]
    diagnostics: dict[str, dict[str, np.ndarray]]


@dataclass
class StreamAudit:
    stream_key: str
    topic: str
    message_type: str
    expected_hz: float
    message_count: int = 0
    header_start_ns: int | None = None
    header_end_ns: int | None = None
    zero_header_count: int = 0
    duplicate_timestamp_count: int = 0
    backward_timestamp_count: int = 0
    non_finite_payload_count: int = 0
    schema_mismatch_count: int = 0
    camera_decode_failure_count: int = 0
    duplicate_image_payload_count: int = 0
    max_consecutive_frozen_frames: int = 0
    image_shape: tuple[int, int, int] | None = None
    image_format: str | None = None
    interval_sec: dict[str, float | int | None] = field(default_factory=dict)
    frequency_hz: float | None = None
    estimated_dropped_messages: int = 0
    large_gap_count: int = 0
    largest_gap_start_ns: int | None = None
    largest_gap_end_ns: int | None = None
    bag_header_offset_sec: dict[str, float | None] = field(default_factory=dict)
    offset_drift_sec: float | None = None
    reject_reasons: list[str] = field(default_factory=list)
    warning_reasons: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class EpisodeAudit:
    roster_index: int
    task_id: str
    bag_path: str
    status: str
    reject_reasons: tuple[str, ...]
    streams: dict[str, StreamAudit]


class BagReader:
    def messages(
        self, episode: EpisodeSpec, streams: dict[str, StreamConfig]
    ) -> Iterator[RawMessage]:
        raise NotImplementedError
