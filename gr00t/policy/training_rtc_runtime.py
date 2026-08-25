"""Runtime guards and request context for TrainingRTC.

The model sampler is intentionally separate from this scheduler.  This module
owns action-grid latency conversion, request metadata, and handoff decisions;
it does not choose or train the execution step ``s``.
"""

from dataclasses import dataclass, replace
from math import ceil
from copy import deepcopy
from typing import Any

import numpy as np


class RTCGuardError(RuntimeError):
    """A request cannot be served by the configured TrainingRTC checkpoint."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class TrainingRTCRequestContext:
    reference_timestamp: float
    raw_state_snapshot: dict[str, np.ndarray]
    c_obs: int
    d_cond: int
    chunk_version: int
    stats_version: str
    committed_prefix: np.ndarray | None = None
    semantic_dim: int | None = None
    model_dim: int | None = None
    layout_version: str | None = None
    t_launch: float | None = None
    t_ready: float | None = None
    c_ready: int | None = None
    t_handoff: float | None = None
    c_handoff: int | None = None


@dataclass
class AbsoluteTargetChunkCache:
    """Absolute action targets indexed on one monotonically increasing action grid."""

    targets: dict[str, np.ndarray]
    start_cursor: int
    chunk_version: int
    stats_version: str

    def coverage_from(self, cursor: int) -> int:
        offset = int(cursor) - int(self.start_cursor)
        if offset < 0:
            return 0
        lengths = [np.asarray(value).shape[0] for value in self.targets.values()]
        return max(0, min(lengths, default=0) - offset)

    def slice(self, cursor: int, length: int) -> dict[str, np.ndarray]:
        if length < 0:
            raise ValueError("length must be non-negative")
        offset = int(cursor) - int(self.start_cursor)
        if offset < 0 or self.coverage_from(cursor) < length:
            raise RTCGuardError(
                "RTC_CHUNK_COVERAGE",
                f"cache cannot provide {length} actions from cursor {cursor}",
            )
        return {
            key: np.asarray(value)[offset : offset + length].copy()
            for key, value in self.targets.items()
        }


@dataclass(frozen=True)
class RTCHandoff:
    accepted: bool
    reason: str
    d_actual: int
    c_handoff: int
    t_handoff: float | None = None
    relaunch_required: bool = False


class TrainingRTCScheduler:
    """Validate and account for TrainingRTC requests on the action grid."""

    def __init__(
        self,
        action_horizon: int,
        training_rtc_max_delay: int,
        action_step_hz: float = 30.0,
    ):
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if training_rtc_max_delay < 0:
            raise ValueError("training_rtc_max_delay must be non-negative")
        if action_step_hz <= 0:
            raise ValueError("action_step_hz must be positive")
        self.action_horizon = int(action_horizon)
        self.training_rtc_max_delay = int(training_rtc_max_delay)
        self.action_step_hz = float(action_step_hz)
        self.dt_action = 1.0 / self.action_step_hz

    def estimate_delay(self, latency_seconds: float, jitter_margin: int = 0) -> int:
        if latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative")
        if jitter_margin < 0:
            raise ValueError("jitter_margin must be non-negative")
        return ceil(latency_seconds / self.dt_action) + int(jitter_margin)

    def compute_handoff_cursor(self, *, t_ready: float, t_obs: float, c_obs: int) -> int:
        """Return the first action-grid cursor available at controller-ready time.

        Grid intervals are half-open: a result ready exactly on a boundary can
        take effect at that boundary; any positive amount after it waits for the
        next boundary.  This cursor is intentionally distinct from ``c_ready``
        (the cursor observed when inference finished).
        """
        elapsed = float(t_ready) - float(t_obs)
        if elapsed < 0:
            raise RTCGuardError("RTC_TIME_INVALID", "t_ready precedes t_obs")
        # Small tolerance makes an exact boundary stable under timestamp roundoff.
        grid_steps = int(ceil(elapsed / self.dt_action - 1e-9))
        return int(c_obs) + max(0, grid_steps)

    def decide_handoff_at_ready(
        self,
        context: TrainingRTCRequestContext,
        *,
        t_ready: float,
        chunk_version: int,
    ) -> RTCHandoff:
        c_handoff = self.compute_handoff_cursor(
            t_ready=t_ready, t_obs=context.reference_timestamp, c_obs=context.c_obs
        )
        result = self.decide_handoff(context, c_handoff=c_handoff, chunk_version=chunk_version)
        return RTCHandoff(
            accepted=result.accepted,
            reason=result.reason,
            d_actual=result.d_actual,
            c_handoff=result.c_handoff,
            t_handoff=float(t_ready),
            relaunch_required=(result.reason == "RTC_DELAY_EXCEEDED"),
        )

    def validate_d_cond(self, d_cond: int, *, s: int | None = None) -> None:
        if d_cond < 0 or d_cond >= self.action_horizon:
            raise RTCGuardError("RTC_DELAY_INVALID", f"d_cond must satisfy 0 <= d < {self.action_horizon}")
        if d_cond > self.training_rtc_max_delay:
            raise RTCGuardError(
                "RTC_DELAY_OOD",
                f"d_cond={d_cond} exceeds checkpoint support {self.training_rtc_max_delay}",
            )
        if s is not None:
            if s <= 0:
                raise RTCGuardError("RTC_SCHEDULER_CONSTRAINT", "s must be positive")
            if d_cond > self.action_horizon - s:
                raise RTCGuardError(
                    "RTC_SCHEDULER_CONSTRAINT",
                    f"d_cond={d_cond} exceeds H-s={self.action_horizon - s}",
                )

    def validate_context(self, context: TrainingRTCRequestContext) -> None:
        if not isinstance(context, TrainingRTCRequestContext):
            raise RTCGuardError("RTC_CONTEXT_INVALID", "invalid request context type")
        if not np.isfinite(context.reference_timestamp):
            raise RTCGuardError("RTC_CONTEXT_INVALID", "reference_timestamp must be finite")
        if not context.raw_state_snapshot:
            raise RTCGuardError("RTC_CONTEXT_INVALID", "raw_state_snapshot is missing")
        if not context.stats_version or context.chunk_version < 0:
            raise RTCGuardError("RTC_CONTEXT_INVALID", "chunk/stats version is invalid")
        self.validate_d_cond(context.d_cond)

    def create_request(
        self,
        *,
        t_obs: float,
        c_obs: int,
        raw_state_snapshot: dict[str, np.ndarray],
        latency_estimate_seconds: float,
        old_chunk_coverage: int,
        chunk_version: int,
        stats_version: str,
        s: int | None = None,
        jitter_margin: int = 0,
        absolute_cache: AbsoluteTargetChunkCache | None = None,
        processor: Any | None = None,
        embodiment: Any | None = None,
        layout_version: str | None = None,
        t_launch: float | None = None,
    ) -> TrainingRTCRequestContext:
        d_cond = self.estimate_delay(latency_estimate_seconds, jitter_margin)
        self.validate_d_cond(d_cond, s=s)
        if absolute_cache is not None:
            if absolute_cache.chunk_version != chunk_version:
                raise RTCGuardError("RTC_STALE_REQUEST", "absolute cache version does not match request")
            if absolute_cache.stats_version != stats_version:
                raise RTCGuardError("RTC_STATS_MISMATCH", "absolute cache statistics version does not match")
            old_chunk_coverage = absolute_cache.coverage_from(c_obs)
        if old_chunk_coverage < d_cond:
            raise RTCGuardError(
                "RTC_CHUNK_COVERAGE",
                f"old chunk coverage {old_chunk_coverage} is shorter than d_cond={d_cond}",
            )
        if not raw_state_snapshot:
            raise RTCGuardError("RTC_CONTEXT_INVALID", "raw_state_snapshot is required")
        if not stats_version:
            raise RTCGuardError("RTC_CONTEXT_INVALID", "stats_version is required")
        committed_prefix = None
        semantic_dim = model_dim = None
        if d_cond and absolute_cache is not None:
            if processor is None or embodiment is None:
                raise RTCGuardError(
                    "RTC_CONTEXT_INVALID",
                    "processor and embodiment are required to build a committed prefix",
                )
            committed_targets = absolute_cache.slice(c_obs, d_cond)
            committed_prefix = processor.encode_absolute_action_targets(
                committed_targets, embodiment, raw_state_snapshot
            )
            layout = processor.get_action_layout(embodiment)
            semantic_dim = int(layout["semantic_dim"])
            model_dim = int(layout["model_dim"])
        return TrainingRTCRequestContext(
            reference_timestamp=float(t_obs),
            raw_state_snapshot=deepcopy(raw_state_snapshot),
            c_obs=int(c_obs),
            d_cond=int(d_cond),
            chunk_version=int(chunk_version),
            stats_version=str(stats_version),
            committed_prefix=committed_prefix,
            semantic_dim=semantic_dim,
            model_dim=model_dim,
            layout_version=layout_version,
            t_launch=None if t_launch is None else float(t_launch),
        )

    @staticmethod
    def decode_absolute_chunk(processor: Any, normalized_action: np.ndarray, embodiment: Any, context: TrainingRTCRequestContext):
        """Decode the complete chunk using only the request's t_obs snapshot."""
        # Keep this validation independent from any newer observation supplied by
        # the caller: the request snapshot is the only legal relative reference.
        if (
            context.raw_state_snapshot is None
            or not context.stats_version
            or not np.isfinite(context.reference_timestamp)
            or context.c_obs < 0
            or context.d_cond < 0
        ):
            raise RTCGuardError("RTC_CONTEXT_INVALID", "request reference context is incomplete")
        if not context.stats_version:
            raise RTCGuardError("RTC_CONTEXT_INVALID", "stats_version is missing")
        return processor.decode_action(
            normalized_action,
            embodiment,
            state=context.raw_state_snapshot,
        )

    @staticmethod
    def mark_ready(
        context: TrainingRTCRequestContext, *, t_ready: float, c_ready: int
    ) -> TrainingRTCRequestContext:
        if t_ready < context.reference_timestamp:
            raise RTCGuardError("RTC_TIME_INVALID", "t_ready precedes t_obs")
        return replace(context, t_ready=float(t_ready), c_ready=int(c_ready))

    @staticmethod
    def mark_handoff(
        context: TrainingRTCRequestContext, *, t_handoff: float, c_handoff: int
    ) -> TrainingRTCRequestContext:
        if t_handoff < context.reference_timestamp:
            raise RTCGuardError("RTC_TIME_INVALID", "t_handoff precedes t_obs")
        if context.t_ready is not None and t_handoff < context.t_ready:
            raise RTCGuardError("RTC_TIME_INVALID", "t_handoff precedes t_ready")
        return replace(context, t_handoff=float(t_handoff), c_handoff=int(c_handoff))

    @classmethod
    def decode_to_cache(
        cls,
        processor: Any,
        normalized_action: np.ndarray,
        embodiment: Any,
        context: TrainingRTCRequestContext,
    ) -> AbsoluteTargetChunkCache:
        """Decode once at request reference and immediately create an absolute cache."""
        decoded = cls.decode_absolute_chunk(processor, normalized_action, embodiment, context)
        targets = {}
        for key, value in decoded.items():
            array = np.asarray(value)
            if array.ndim == 3:
                if array.shape[0] != 1:
                    raise RTCGuardError(
                        "RTC_CONTEXT_INVALID", "decode_to_cache currently requires batch size 1"
                    )
                array = array[0]
            targets[key] = array.copy()
        return AbsoluteTargetChunkCache(
            targets=targets,
            start_cursor=context.c_obs,
            chunk_version=context.chunk_version,
            stats_version=context.stats_version,
        )

    def decide_handoff(
        self,
        context: TrainingRTCRequestContext,
        *,
        c_handoff: int,
        chunk_version: int,
    ) -> RTCHandoff:
        self.validate_context(context)
        d_actual = int(c_handoff) - context.c_obs
        if chunk_version != context.chunk_version:
            return RTCHandoff(False, "RTC_STALE_REQUEST", d_actual, int(c_handoff))
        if d_actual > context.d_cond:
            return RTCHandoff(
                False,
                "RTC_DELAY_EXCEEDED",
                d_actual,
                int(c_handoff),
                relaunch_required=True,
            )
        if d_actual < 0:
            return RTCHandoff(False, "RTC_CURSOR_INVALID", d_actual, int(c_handoff))
        return RTCHandoff(True, "OK", d_actual, int(c_handoff))
