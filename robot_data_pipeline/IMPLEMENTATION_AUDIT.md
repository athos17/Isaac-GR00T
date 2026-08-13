# Implementation Audit

Date: 2026-08-12

This audit maps `docs/rosbag_data_pipeline_implementation_plan.md` to implementation evidence.
`VERIFIED` means the behavior is implemented and exercised. `BLOCKED` means completion requires
confirmed external data that is not currently available.

## Goals and non-goals

| Requirement | Status | Evidence |
| --- | --- | --- |
| Configuration-driven profiles and manifests | VERIFIED | Strict loaders in `config.py`; finite numeric values, fixed v1 clock semantics, measured-state activity groups, path overlap, semantic EEF rotation checks, and legacy/Manus profiles |
| Header timestamp is the alignment clock | VERIFIED | Pure Python reader preserves header and bag timestamps; synchronization uses header only |
| Raw topic, timestamp, payload, joint and camera QA | VERIFIED | `quality/raw.py`; unit and real synthetic fault tests |
| Measured-state activity crop | VERIFIED | `processing/activity.py`; rate-independent and stationary tests |
| Action-state lag audit without time shifting | VERIFIED | `quality/lag.py`; known lag, low-excitation, insufficient-overlap, constant-slice, window trend and maximum-step tests |
| State-only low-pass before 30 Hz sampling | VERIFIED | `processing/filters.py`; frequency and zero-phase tests |
| Head-anchor 30 Hz synchronization | VERIFIED | Bounded interpolation, SLERP, causal ZOH and nearest camera tests |
| Joint and EEF + hand output spaces | VERIFIED | Both real synthetic conversions and GR00T loader smoke tests |
| Multi-task LeRobot v2 plus QA | VERIFIED | Direct task mapping, deterministic runner, aggregate and per-frame QA |
| Inputs remain read-only | VERIFIED | Reader-only input path; integration test hashes the complete input bag tree before audit/convert and proves it is unchanged afterward |
| Listed v1 non-goals remain excluded | VERIFIED | No normalization, delta conversion, horizon, FK/IK, repair, split, or stats generation |

## Pipeline stages

| Stage | Status | Evidence and policy |
| --- | --- | --- |
| 0. Configuration and roster | VERIFIED | Unknown/duplicate fields, non-finite values, clock/activity/EEF semantics, duplicate topics/tasks/roots/outputs/bags, nested or input/output paths, missing paths, hashes and stable ordering are checked |
| 1. Raw integrity | VERIFIED | Required topic/type, zero/duplicate/backward header, frequency/gap/drop, bag offset/drift, finite payload, joint schema, JPEG decode/exception/shape/format and frozen payload metrics |
| 2. Canonicalization | VERIFIED | Typed series, named reorder, explicit unnamed legacy mapping, continuous unwrap, xyz/quaternion canonical form and stable quaternion norm diagnostics |
| 3. Activity | VERIFIED | EEF/hand measured-state velocity, time-window thresholds, padding and no-motion rejection |
| 4. Lag audit | VERIFIED | Positive 0-300 ms search at 200 Hz, 3 s windows, per-axis confidence/median/MAD/range/trend/max-step and group consensus; insufficient overlap remains report-only |
| 5. Filtering | VERIFIED | Fourth-order 10 Hz Butterworth, regularized high-rate grid, zero phase, padding, implementation version and Welch velocity spectrum QA |
| 6. Synchronization | VERIFIED | Real head timestamps, linear state, SLERP, previous action, nearest wrist, bounds and structured failures |
| 7. Aligned QA | VERIFIED | Frame counts, finite vectors, configured ranges, exact logical timestamps, skew/gap/age/reuse metrics and stable reject reasons |
| 8. Export | VERIFIED | Parquet fields, fixed 30 fps H.264 video with ffprobe, metadata, no stats, per-frame sidecars and independent outputs |

Raw gaps are warnings until activity is known. Only gaps overlapping the unpadded active interval
are hard rejects. Invalid camera payload tolerance is zero in v1: any required JPEG decode failure
rejects the episode rather than attempting repair.

## CLI, determinism and reliability

| Requirement | Status | Evidence |
| --- | --- | --- |
| `validate`, `audit`, `convert`, `summarize`, dry-run | VERIFIED | Synthetic end-to-end CLI regression plus real legacy dry-run; raw/processing/conversion summary support |
| Guarded overwrite | VERIFIED | Existing targets require `--overwrite` and a pipeline manifest marker |
| Episode-level bounded parallelism | VERIFIED | Threaded preparation with roster-ordered collection |
| Worker-count determinism | VERIFIED | One-worker and two-worker outputs have identical schemas, values, indices and task maps |
| Failure isolation | VERIFIED | Expected per-episode raw, processing, alignment and export failures become rejection records; a rejected first episode does not prevent a later valid episode from being exported |
| Transactional multi-output publication | VERIFIED | All old outputs are backed up first; a later publish failure rolls back every output |
| Run provenance | VERIFIED | Git revision, command, profile/manifest paths and hashes, dependencies, clock and roster |
| Runtime independence | VERIFIED | No imports from `data_preprocess` or `wuji_pipeline` |
| Optional dependency installation | VERIFIED | `data-pipeline` extra in `pyproject.toml`; locked by `uv.lock` |

Cross-directory publication cannot provide a single filesystem rename visible atomically to
concurrent readers. It does provide failure atomicity: after an error, all pre-existing outputs are
restored and no persistent mixed generation remains.

## Test plan

| Acceptance item | Status |
| --- | --- |
| YAML schema, duplicate keys/tasks/paths and overlap | VERIFIED |
| Timestamp monotonicity, zero, duplicate, gap and offset | VERIFIED |
| Joint reorder, missing names and duplicate names | VERIFIED |
| Quaternion normalize, zero norm, sign, SLERP and rot6d | VERIFIED |
| Rate-independent activity and stationary input | VERIFIED |
| Butterworth response, spectrum and zero-phase timing | VERIFIED |
| Linear bounds, causal ZOH, camera tie/skew/reuse | VERIFIED |
| Stable PASS/REJECT details | VERIFIED |
| Metadata dimensions, indices and task mapping | VERIFIED |
| Real synthetic rosbag at 30/120/200/250 Hz | VERIFIED |
| Missing topic, zero/duplicate header, bad JPEG, active gap bags | VERIFIED |
| Joint and EEF conversion plus GR00T stats/loader | VERIFIED |
| Checked-in golden QA/metadata/parquet subset | VERIFIED |
| Input bag content hash unchanged after audit and convert | VERIFIED |
| Mixed rejected/valid episode job continues and preserves indices | VERIFIED |
| Direct two-task export, instructions, task/global indices, videos and counts | VERIFIED |
| Camera shape/format/frozen payload and bag offset/drift metrics | VERIFIED |
| Structured action-age, short-episode and aligned-range rejection details | VERIFIED |
| Non-finite config, invalid clock/activity semantics and nested roots | VERIFIED |
| OpenCV decode exceptions remain episode/export failures, not job crashes | VERIFIED |
| CLI dry-run read-only behavior and actual audit/convert/summarize | VERIFIED |
| Real legacy raw/processing/convert pilot | VERIFIED; see `PILOT_REPORT.md` |
| One confirmed real Manus bag smoke test | BLOCKED: no confirmed dataset found |
| At least 50 confirmed Manus + Orin bags | BLOCKED: no confirmed dataset found |

The locally executable suite contains 75 passing tests. It includes a stationary real synthetic
bag that passes raw QA and is rejected during processing with `no_valid_motion`, in addition to
the rate, synchronization, fault-injection, export and loader coverage above.

## Pilot policy status

The current-code legacy 20-bag processing pilot is complete. Two rejects were manually traced to isolated
left-wrist missing frames, and the 20 ms skew threshold is retained. Filter and lag distributions
are documented in `PILOT_REPORT.md`.

The following production policy remains provisional until the confirmed Manus + Orin pilot:

- final filter cutoff by arm, hand and EEF group;
- final activity thresholds and padding;
- final camera gap/skew, action-age and state-gap thresholds;
- whether lag/direction metrics remain report-only or become hard checks;
- quaternion norm bounds and any EEF angular-velocity threshold;
- joint velocity/acceleration limits beyond current configured position ranges.

The current v1 implementation uses configurable fourth-order 10 Hz zero-phase filters, a 0-300 ms
lag search with 3 s windows, report-only lag decisions, configured position ranges, quaternion norm
0.5-1.5, and one deterministic instruction per task. These are implementation defaults, not a
claim that Manus production thresholds are frozen.

## Completion gate

Implementation and all locally executable acceptance tests are complete. Final plan completion is
blocked solely on locating and auditing at least 50 confirmed Manus + Jetson Orin bags, including
the required motion and anomaly tails, then freezing the production profile thresholds from that
evidence.
