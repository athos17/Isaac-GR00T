# GR00T RTC Design

Date: 2026-06-17

## Context

The Wuji real-robot deployment currently uses a synchronous loop in
`examples/wuji_rot6d/run_gr00t_client.py`: request one action chunk from the GR00T policy server,
execute the first `--execute-horizon` steps, then request another chunk. This can introduce
stop-and-go behavior because policy inference blocks the next control segment.

The RTC paper and LeRobot implementation treat real-time chunking as an inpainting problem for
diffusion or flow policies. The next chunk is generated while the current chunk is being executed,
and generation is constrained by unexecuted actions from the previous chunk. LeRobot implements
this with `prev_chunk_left_over`, delay-aware queue replacement, soft prefix weights, and
per-denoising-step guidance.

GR00T N1.7 already has a simplified RTC-like branch in
`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`: when action input is present during inference, it uses the
previous action prefix to initialize part of the denoising trajectory and applies a `vel_strength`
mask to freeze or ramp the overlap region. This is not the full paper/LeRobot autograd guidance
algorithm, but it is a practical baseline for real-robot validation.

## Decision

Implement RTC in two stages.

Stage 1 is an engineering baseline:

- Add a general GR00T policy/model RTC interface.
- Add Wuji-specific asynchronous client scheduling.
- Reuse the existing simplified GR00T RTC branch.
- Keep parameters adjustable from the command line and request `options`.

Stage 2 is a paper-aligned implementation:

- Add LeRobot-style per-denoising-step RTC guidance to GR00T.
- Preserve the Stage 1 client protocol so the client does not need another redesign.

## Goals

- Reduce visible stop-and-go during Wuji real-robot deployment.
- Keep the Stage 1 implementation small enough to validate on hardware quickly.
- Keep RTC request metadata generic enough for other GR00T clients later.
- Preserve existing synchronous behavior when RTC is disabled.
- Log enough metadata to diagnose latency, queue replacement, and fallback decisions.

## Non-Goals

- Stage 1 will not implement the RTC paper's autograd guidance correction.
- Stage 1 will not introduce a fully generic real-robot client abstraction.
- Stage 1 will not require retraining or checkpoint format changes.
- Stage 1 will not depend on ROS in unit tests.

## Stage 1 Architecture

### Wuji Client

`examples/wuji_rot6d/run_gr00t_client.py` remains the hardware integration entry point. It gains an
RTC execution mode that maintains:

- Current action chunk in physical action space.
- Current execution index within the chunk.
- A background inference future.
- A latency tracker.
- A delay estimate in control steps.
- Fallback state when inference is late or fails.

The main thread continues to send commands at `--control-hz`. In RTC mode, it executes from the
current queue while a background request computes the next chunk. The client sends the previous
chunk's unexecuted suffix as `prev_chunk_left_over`.

The Wuji client owns robot-specific details:

- Observation capture from ROS/Astribot.
- Conversion between EEF rot6d actions and Astribot pose commands.
- Workspace and per-step safety clipping.
- Hand command clipping.
- Command logging.

### Policy Interface

`Gr00tPolicy.get_action(observation, options)` accepts an optional RTC payload:

```python
{
    "rtc": {
        "enabled": True,
        "prev_chunk_left_over": {
            "left_eef": np.ndarray,
            "right_eef": np.ndarray,
            "left_hand_joints": np.ndarray,
            "right_hand_joints": np.ndarray,
        },
        "action_horizon": int,
        "rtc_overlap_steps": int,
        "rtc_frozen_steps": int,
        "rtc_ramp_rate": float,
    }
}
```

The payload uses physical action space, matching normal GR00T decoded actions. `Gr00tPolicy`
injects `prev_chunk_left_over` into `VLAStepData.actions` before processor collation. This lets the
existing state/action processor handle:

- EEF absolute-to-relative conversion.
- Hand absolute action normalization.
- Action concatenation.
- Padding to max action horizon and max action dimension.

The policy then forwards model-specific RTC options to `self.model.get_action(..., options=...)`.

### Model Interface

The GR00T N1.7 model receives normalized action input plus model options. Stage 1 reuses and
hardens the existing simplified branch:

- Validate `rtc_overlap_steps <= action_horizon`.
- Validate `rtc_frozen_steps <= rtc_overlap_steps`.
- Copy the selected previous action prefix into the denoising initialization.
- Set `vel_strength=0` for frozen prefix steps.
- Apply an exponential ramp over the remaining overlap steps.
- Leave non-overlap steps fully denoised.

If RTC is disabled or no `prev_chunk_left_over` is supplied, inference follows the existing normal
path.

## Stage 1 Data Flow

1. The first client request is synchronous and does not include RTC options.
2. The client starts executing the returned chunk.
3. When the trigger point is reached, the client captures a fresh observation.
4. The client slices the unexecuted suffix from the current chunk and sends it as
   `options["rtc"]["prev_chunk_left_over"]`.
5. The server normalizes that suffix through the standard processor path.
6. The model generates a new chunk using the simplified RTC branch.
7. The client measures actual inference latency and converts it to `real_delay_steps`.
8. The client discards the first `real_delay_steps` from the returned chunk.
9. The remaining new chunk replaces the execution queue.

The first discarded steps correspond to controller timesteps that passed while inference was
running.

## Default Parameters

Defaults are tuned for the current Wuji deployment:

- `control_hz`: 30 Hz
- model action horizon: read from processor/modality config; current Wuji config is 32
- `execute_horizon`: 16
- `rtc_overlap_steps`: `action_horizon - execute_horizon`
- `rtc_frozen_steps`: estimated delay in control steps
- delay estimate: recent p95 latency converted to control steps
- fallback: `hold-last`

All RTC parameters remain adjustable through command-line flags and request options.

Planned client flags:

- `--rtc-enable`
- `--rtc-overlap-steps`
- `--rtc-frozen-steps`
- `--rtc-ramp-rate`
- `--rtc-latency-window`
- `--rtc-delay-percentile`
- `--rtc-min-leftover`
- `--rtc-fallback {hold-last,stop}`
- `--rtc-server-unsupported {fail,disable}`

## Scheduling Rules

The trigger point is derived from the estimated delay:

- Start inference early enough that the result should arrive before the current executable segment
  ends.
- Use `estimated_delay_steps` as the default `rtc_frozen_steps`.
- Clamp delay estimates to valid model ranges.
- If the queue has insufficient leftover for RTC, degrade according to the configured behavior.

The client records both estimated and real delay. Real delay is used for queue replacement, while
estimated delay is used for the next request's frozen prefix.

## Error Handling

First request failure:

- Fail fast and do not start robot execution.

Async request timeout or failure:

- Continue executing the current queue if actions remain.
- If the queue is exhausted, use `--rtc-fallback`.
- Log the failure reason and latency metadata.

Invalid RTC parameters:

- Validate at startup where possible.
- Validate again in the server/model for request-specific values.
- Raise clear errors for `frozen > overlap`, `overlap > action_horizon`, or non-positive horizons.

Server does not support RTC:

- Default behavior is fail-fast.
- Optional behavior is to disable RTC and continue in synchronous mode.

Short leftover:

- If leftover is shorter than `--rtc-min-leftover`, do not request RTC guidance for that call.
- Continue with a normal async request or synchronous fallback according to configuration.

## Logging

Add RTC fields to command logs:

- `rtc_enabled`
- `request_started_at_step`
- `estimated_delay_steps`
- `real_delay_steps`
- `latency_sec`
- `leftover_len`
- `overlap_steps`
- `frozen_steps`
- `queue_len`
- `fallback_reason`
- `rtc_disabled_reason`

These fields support post-run analysis without requiring robot hardware during development.

## Testing

Stage 1 tests should avoid ROS and hardware.

Policy/model tests:

- `Gr00tPolicy` passes `options` to `model.get_action`.
- `prev_chunk_left_over` is injected as action data and produces `action` and `action_mask` in
  processed inputs.
- Simplified RTC model options validate shape and horizon constraints.
- RTC-disabled inference keeps the existing call path.

Client logic tests:

- Convert latency seconds to delay steps.
- Slice leftover actions from a multi-key action chunk.
- Replace queue after discarding `real_delay_steps`.
- Handle timeout fallback.
- Preserve existing synchronous behavior when `--rtc-enable` is false.

Integration tests can use mocked `PolicyClient` and fake action chunks to validate scheduling.

## Stage 2 Architecture

Stage 2 replaces the simplified model-side RTC behavior with paper-aligned guidance while keeping
the Stage 1 client protocol.

Add a GR00T RTC module based on LeRobot's concepts:

- `RTCConfig`
- `RTCProcessor`
- `LatencyTracker`
- soft prefix weight schedule
- optional debug tracker

The processor wraps each flow denoising step:

- Compute prefix weights from delay, execution horizon, and action horizon.
- Estimate final denoised action `x1_t`.
- Compute weighted error against `prev_chunk_left_over`.
- Use autograd to calculate the vector-Jacobian product correction.
- Clamp guidance with `max_guidance_weight`.
- Apply guided velocity during Euler integration.

Supported schedules should include at least `exp` and `linear`; `exp` is the default target because
it matches the RTC paper's recommendation.

## Stage 2 Tests

- Prefix weights match expected zeros, ones, linear, and exponential schedules.
- Guidance preserves tensor shapes and device/dtype.
- No-prefix behavior matches normal denoising.
- Debug trace captures time, velocity, correction, weights, and guidance weight.
- Stage 1 client requests work unchanged with the Stage 2 server.

## Rollout Plan

1. Implement Stage 1 server/model interface.
2. Add client-side RTC scheduling behind `--rtc-enable`.
3. Add logic tests and mocked policy-client tests.
4. Run a dry client simulation with mocked actions.
5. Validate on robot with conservative limits and verbose logging.
6. Use collected logs to tune delay percentile, overlap, frozen steps, and fallback.
7. Implement Stage 2 guidance after Stage 1 is stable.

## Open Decisions Locked for Stage 1

- First implementation targets Wuji client scheduling but keeps server/model RTC options generic.
- Defaults follow the current 30 Hz Wuji deployment and 32-step action horizon.
- Parameters are adjustable from the command line.
- Stage 1 uses simplified GR00T RTC, not LeRobot autograd guidance.
- Stage 2 will preserve the Stage 1 client protocol.
