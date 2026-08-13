# Data Pipeline Pilot Report

Date: 2026-08-12

## Legacy raw audit

The first 20 bags selected by `configs/datasets/legacy_pilot.yaml` passed raw QA.

- Head and wrist cameras: approximately 30 Hz.
- Hand measured state: approximately 200 Hz.
- Arm and EEF state/command: approximately 250 Hz.
- Legacy hand command: approximately 209-212 Hz.
- Report directory: `/tmp/robot_data_pipeline_legacy_pilot_audit`.

## Legacy processing audit

The same 20 bags were processed with the v1 profile. The result was 18 PASS and 2 REJECT.
The complete regenerated report is in
`/tmp/robot_data_pipeline_legacy_processing_pilot_v3`. Its aggregate summary is identical to the
v2 report after the configuration and lag-audit hardening described in `IMPLEMENTATION_AUDIT.md`.

- Active duration: 20.92-30.69 s; median 26.49 s.
- Arm response lag median across episodes: 12.5-15 ms.
- EEF response lag median across episodes: 15-17.5 ms.
- Hand response lag median across episodes: 27.5-30 ms.
- Arm/EEF action age p95: approximately 3.8 ms.
- Hand action age p95: approximately 4.9-5.7 ms.
- State bracket gaps: approximately 4 ms for arm/EEF and 5 ms for hand.
- Wrist skew p95 across episodes: approximately 15-16.25 ms.
- Wrist frame reuse ratio was generally below 3.4%.

Filter spectrum QA uses velocity on a regularized grid, linear detrending, and Welch PSD.
The 0.5-10 Hz power retention across the six filtered state streams was 0.937-0.998.
The 15 Hz-to-Nyquist power retention was 3.7e-6 to 6.7e-5. A synthetic 5 Hz plus 30 Hz
test independently verifies low-band preservation and high-band attenuation.

No NumPy correlation warnings occurred after shifted window slices with insufficient variance were
excluded from lag estimation.

The v3 report also records windowed lag trend and maximum adjacent-window step for all 1,200
audited axes. A tail review found that 781 axes had a step of at least 100 ms, but 715 of those had
a global secondary-peak margin below 0.005. The only axis with an absolute trend of at least
20 ms/s had peak correlation 0.421 and secondary-peak margin 0.00324. These ambiguous correlation
tails support keeping lag metrics report-only; they are not evidence for a production hard reject.

## Legacy conversion smoke

The current code converted `arm_hand_vr_20260805_151803` into both requested action spaces:

- EEF + hand output: 1 episode, 575 frames, 0 rejected.
- Joint output: 1 episode, 575 frames, 0 rejected.
- Each output has three 30 fps H.264 videos and a 575-frame alignment sidecar.
- SHA-256 hashes of the source `metadata.yaml` and DB3 file were identical before and after
  conversion.
- After the separate training-preparation stats step, both outputs loaded as one episode with
  `LeRobotEpisodeLoader`.

The regenerated outputs are `/tmp/robot_data_pipeline_legacy_smoke_eef` and
`/tmp/robot_data_pipeline_legacy_smoke_joint`.

## Wrist skew tail inspection

Both rejected bags contain an isolated missing left-wrist frame. Normal camera intervals near the
failure are approximately 33.34 ms; the left-wrist interval at each failure is approximately
66.68 ms.

| Roster | Source | Violations | Maximum skew | Evidence |
| --- | --- | ---: | ---: | --- |
| 5 | `arm_hand_vr_20260804_212748` | 2 / 823 | 33.002 ms | left-wrist 66.673 ms interval |
| 15 | `arm_hand_vr_20260804_213347` | 1 / 705 | 29.933 ms | left-wrist 66.684 ms interval |

The 20 ms wrist skew threshold is retained. It separates these dropped-frame events from the
normal approximately 16 ms nearest-frame tail. Rejection reports now include the stream, anchor
and source timestamps, signed and absolute skew, maximum skew, threshold, and violation count.

The revised policy keeps an isolated nearest-neighbor match between 20 and 40 ms as
`PASS_WITH_WARNING` only when no more than one violating anchor is consecutive and no more than
0.5% of the episode's anchors violate 20 ms. A match above 40 ms, a longer run, or a larger ratio
still rejects the episode. Head anchors outside required stream coverage at an activity boundary
are trimmed instead of rejecting the full episode.

A targeted replay of all 20 episodes previously rejected for wrist skew produced 18 recoveries:
15 `PASS_WITH_WARNING` episodes with isolated missing wrist frames and 3 plain `PASS` episodes
after boundary trimming. Two episodes remain rejected by the 0.5% ratio guard:

| Roster | Soft violations | Ratio | Maximum consecutive | Decision |
| --- | ---: | ---: | ---: | --- |
| 81 | 7 / 566 | 1.2367% | 1 | REJECT |
| 96 | 4 / 526 | 0.7605% | 1 | REJECT |

This preserves the occasional duplicated nearest-neighbor wrist image while continuing to reject
episodes where frame loss is too frequent to be treated as an isolated acquisition artifact.

## Manus and Orin status

Synthetic rosbag coverage verifies 30/120/200/250 Hz streams with the Manus profile. The search
was expanded from `/data_all/share/datasets` to all readable `/data_all` and parsed 2,127 rosbag
`metadata.yaml` files. No path or metadata explicitly identifies a confirmed Manus plus Jetson
Orin capture, and no bag had both configured hand-command topics at a coarse 90-150 Hz rate.

A broader hand command/cmd topic search produced two weak single-side candidates. Auditing their
message header timestamps excluded both: the tactile scrub-pan bag had left/right rates of
206.6/166.5 Hz, and the grasp-mango bag had left/right rates of 150.6/193.5 Hz. A wider
count-balanced 70-180 Hz scan found 19 bags, all in known tactile or teleoperation paths with
coarse rates around 160-180 Hz. Legacy, tactile, or teleoperation data is not treated as Manus
based only on topic shape or frequency.

The required pilot of at least 50 confirmed Manus plus Orin bags, tail review, and final production
threshold freeze remain blocked until that dataset is available.
