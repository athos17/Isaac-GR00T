import numpy as np
import pytest

from gr00t.policy.training_rtc_runtime import (
    AbsoluteTargetChunkCache,
    RTCGuardError,
    TrainingRTCScheduler,
)


def test_action_grid_delay_uses_action_step_rate():
    scheduler = TrainingRTCScheduler(action_horizon=32, training_rtc_max_delay=6, action_step_hz=30)
    assert scheduler.estimate_delay(0.070) == 3
    assert scheduler.estimate_delay(0.120) == 4


def test_handoff_cursor_before_on_and_after_action_boundary():
    scheduler = TrainingRTCScheduler(action_horizon=32, training_rtc_max_delay=6, action_step_hz=30)
    dt = 1.0 / 30.0
    assert scheduler.compute_handoff_cursor(t_ready=0.099, t_obs=0.0, c_obs=10) == 13
    assert scheduler.compute_handoff_cursor(t_ready=dt, t_obs=0.0, c_obs=10) == 11
    assert scheduler.compute_handoff_cursor(t_ready=dt + 1e-4, t_obs=0.0, c_obs=10) == 12


def test_ready_and_handoff_timeline_is_monotonic():
    scheduler = TrainingRTCScheduler(action_horizon=32, training_rtc_max_delay=6)
    context = scheduler.create_request(
        t_obs=1.0,
        c_obs=0,
        raw_state_snapshot={"eef": np.zeros((1, 1))},
        latency_estimate_seconds=0.0,
        old_chunk_coverage=32,
        chunk_version=1,
        stats_version="v1",
    )
    context = scheduler.mark_ready(context, t_ready=1.1, c_ready=3)
    context = scheduler.mark_handoff(context, t_handoff=1.11, c_handoff=3)
    assert context.t_ready == 1.1
    assert context.t_handoff == 1.11
    with pytest.raises(RTCGuardError, match="RTC_TIME_INVALID"):
        scheduler.mark_handoff(context, t_handoff=1.05, c_handoff=2)


def test_delay_ood_is_not_clamped():
    scheduler = TrainingRTCScheduler(action_horizon=32, training_rtc_max_delay=6)
    with pytest.raises(RTCGuardError, match="RTC_DELAY_OOD"):
        scheduler.create_request(
            t_obs=1.0,
            c_obs=4,
            raw_state_snapshot={"left_eef": np.zeros(9)},
            latency_estimate_seconds=0.25,
            old_chunk_coverage=32,
            chunk_version=1,
            stats_version="v1",
        )


def test_handoff_uses_c_handoff_and_rejects_late_result():
    scheduler = TrainingRTCScheduler(action_horizon=32, training_rtc_max_delay=6)
    context = scheduler.create_request(
        t_obs=1.0,
        c_obs=10,
        raw_state_snapshot={"left_eef": np.zeros(9)},
        latency_estimate_seconds=0.100,
        old_chunk_coverage=32,
        chunk_version=2,
        stats_version="v1",
    )
    assert context.d_cond == 3
    assert scheduler.decide_handoff(context, c_handoff=13, chunk_version=2).accepted
    late = scheduler.decide_handoff(context, c_handoff=14, chunk_version=2)
    assert not late.accepted
    assert late.reason == "RTC_DELAY_EXCEEDED"


def test_scheduler_boundary_and_stale_context():
    scheduler = TrainingRTCScheduler(action_horizon=32, training_rtc_max_delay=6)
    context = scheduler.create_request(
        t_obs=1.0,
        c_obs=0,
        raw_state_snapshot={"left_eef": np.zeros(9)},
        latency_estimate_seconds=0.0,
        old_chunk_coverage=32,
        chunk_version=5,
        stats_version="v1",
    )
    assert scheduler.decide_handoff(context, c_handoff=0, chunk_version=5).d_actual == 0
    assert scheduler.decide_handoff(context, c_handoff=1, chunk_version=99).reason == "RTC_STALE_REQUEST"


def test_absolute_cache_prefix_is_rebased_at_t_obs_and_full_decode_uses_snapshot():
    class FakeProcessor:
        def __init__(self):
            self.encode_state = None
            self.decode_state = None

        def get_action_layout(self, embodiment):
            return {"semantic_dim": 2, "model_dim": 4, "horizon": 4}

        def encode_absolute_action_targets(self, targets, embodiment, state):
            self.encode_state = state
            values = targets["eef"]
            return np.pad(values[None], ((0, 0), (0, 0), (0, 2))).astype(np.float32)

        def decode_action(self, normalized, embodiment, state=None):
            self.decode_state = state
            return {"eef": np.asarray(normalized)[0, :, :2]}

    processor = FakeProcessor()
    scheduler = TrainingRTCScheduler(action_horizon=4, training_rtc_max_delay=3)
    cache = AbsoluteTargetChunkCache(
        targets={"eef": np.arange(8, dtype=np.float32).reshape(4, 2)},
        start_cursor=10,
        chunk_version=7,
        stats_version="stats-a",
    )
    snapshot = {"eef": np.array([[100.0, 101.0]], dtype=np.float32)}
    context = scheduler.create_request(
        t_obs=12.5,
        c_obs=10,
        raw_state_snapshot=snapshot,
        latency_estimate_seconds=2 / 30,
        old_chunk_coverage=0,
        chunk_version=7,
        stats_version="stats-a",
        absolute_cache=cache,
        processor=processor,
        embodiment="wuji",
    )
    assert context.d_cond == 2
    np.testing.assert_array_equal(processor.encode_state["eef"], snapshot["eef"])
    assert context.committed_prefix.shape == (1, 2, 4)

    normalized = np.zeros((1, 4, 4), dtype=np.float32)
    decoded_cache = scheduler.decode_to_cache(processor, normalized, "wuji", context)
    np.testing.assert_array_equal(processor.decode_state["eef"], snapshot["eef"])
    assert decoded_cache.start_cursor == context.c_obs
    np.testing.assert_array_equal(decoded_cache.targets["eef"], np.zeros((4, 2), dtype=np.float32))


def test_absolute_cache_rejects_version_or_stats_mismatch():
    scheduler = TrainingRTCScheduler(action_horizon=8, training_rtc_max_delay=3)
    cache = AbsoluteTargetChunkCache(
        targets={"joint": np.zeros((8, 1), dtype=np.float32)},
        start_cursor=0,
        chunk_version=1,
        stats_version="v1",
    )
    with pytest.raises(RTCGuardError, match="RTC_STALE_REQUEST"):
        scheduler.create_request(
            t_obs=0.0,
            c_obs=0,
            raw_state_snapshot={"joint": np.zeros((1, 1), dtype=np.float32)},
            latency_estimate_seconds=0.0,
            old_chunk_coverage=0,
            chunk_version=2,
            stats_version="v1",
            absolute_cache=cache,
        )
    with pytest.raises(RTCGuardError, match="RTC_STATS_MISMATCH"):
        scheduler.create_request(
            t_obs=0.0,
            c_obs=0,
            raw_state_snapshot={"joint": np.zeros((1, 1), dtype=np.float32)},
            latency_estimate_seconds=0.0,
            old_chunk_coverage=0,
            chunk_version=1,
            stats_version="v2",
            absolute_cache=cache,
        )
