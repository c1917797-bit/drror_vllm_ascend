# Conv layout graph gate and end-to-end screen

Date: 2026-09-08. Joint performance/accuracy target remains unachieved.

## Stage and budget

This is performance-first exploration, not final acceptance. The question is whether
the already implemented load-time convolution-weight packing gives measurable whole-model
benefit. Budget: one original-layout and one packed-layout Dk64 run, same frozen driver,
40 requests, concurrency16, input32768/prefix16384, fixed1024 outputs and two one-token
warmups. Profiling and diagnostic-determinism are disabled. Preserve both outcomes.
A small or inconsistent difference is not a reason for an unbounded repeatability branch.

## Change-specific graph evidence

`tools/conv_layout_graph_probe.py` compares four independent state chains on physical NPU4:
eager original, eager packed, graphed original and graphed packed. Inputs and cache indices
change in place at each step, retaining the captured buffer addresses. Capture/warmup state
is restored before measurement. Each path is compared with eager original, including the
entire convolution-state buffer, not just active slots.

Dk64/128 x batch1/2/4/8/16/24/32 x32 steps: all14 cases/448 steps pass. Output and full
state are finite and bitwise equal; captured pointers remain unchanged. This complements
the prior64-case eager matrix and four-worker48-layer exact loaded-weight checks.
It is an isolated synthetic convolution graph, NOT TP/full-model logit or task-accuracy parity.

Artifact:
`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/conv-layout-e2e-v1/graph-parity-v1/report.json`

- Report SHA256: 5d1dbea6ee54c30d03fa402a8e41ce77c68646e7a89ef662bdb53f89cd394b25
- Script SHA256: 8facf2827ec366f94dcce45715affc47577ac04fa7090d9da4b6d42f243d8241
- Conv module SHA256: 0dd0da18098600fcc921d9c3af36b30b2dea7f7f73e328e12c63bd64c92d713e
- Installed local0.1.7 wheel SHA256: c5f137872059a60c6a13893dcc850af5658999547d5c7bffdaab1b6f42683ba6

CPU regression:54 tests pass; includes four graph-report validator tests rejecting missing,
duplicate, incomplete or numerically failed evidence. NPU execution operation009e exited0.
Initial attempt009c stopped before NPU execution at metadata validation: the reused helper
prepended the repo, exposing stale0.1.6 egg-info. Read-only audit009d established that the
installed purelib distribution remained0.1.7. The probe now resolves installed metadata
explicitly from purelib and checks installed/repository package hashes. No package or
upstream source was changed to fix this test-harness issue.

## Original-layout screen recovered after connection interruption

`conv-layout-screen-v1/dk64-original/performance.run1.json` completed while the local
tool context was unavailable. Its client operation handle later expired; the completed
receipt, report hash, service manifest and40 request records were checked instead.
The original service operation00a6 was still live. After confirming zero running/waiting
requests, only its API PID30831 was sent SIGTERM; launcher exited0. No benchmark was rerun.

- Throughput:133.060989687 tok/s
- TTFT mean:36824.037230 ms
- TPOT mean:69.812184743 ms
- Measured duration:307.828764060 s;40/40 success,40960 total output tokens, all length-finished.
- Report SHA256: dbecf8a2d460b979730e1276b44831c25b612a58e2e3dbd25d08060aecad89f6

This is Dk64 WITHOUT packing. Comparing it with the historical120.6607 tok/s mean does
not establish reproducible10% gain; a contemporaneous unpruned control is still required.
It is also not an accuracy result. The packed-layout counterpart completed successfully.

## Completed layout pair

| Metric | Dk64 original | Dk64 packed | Change |
| --- | ---: | ---: | ---: |
| Throughput, tok/s | 133.060990 | 134.205617 | +0.8602% |
| TTFT mean, ms | 36824.037230 | 36832.961721 | +0.0242% |
| TPOT mean, ms | 69.812185 | 69.030633 | -1.1195% |
| Measured duration, s | 307.828764 | 305.203320 | -2.625444 s |

Both runs completed40/40 requests,1024 output tokens each, with two one-token warmups.
Packed client operation00b0 exited0. Packed report SHA256:
be4655ee28424ee24519acc1990b08a60225e5126d97a7625f5903595474c43d.

Comparison operation00c3 verified identical benchmark configs, server commands, runtime,
installed module/wheel/plan/launcher hashes and client/driver hashes. Only the layout flag
and evidence destination differ. All four packed workers have48 exact-value-verified layer
records, with exactly the intended linear-attention layer indices. No profiler or custom
determinism worker was enabled. Structured result: conv-layout-screen-comparison.v1.json.

Decision: retain as an opt-in candidate, not the main optimization focus. The0.86% single
observation is not a statistical significance or repeatability claim; runs were separated
by a connection interruption. There is no new contemporaneous unpruned control or task
accuracy result. Do not attribute a comparison against the historical baseline to packing.

## Larger-hotspot follow-up, not enabled in this screen

Frozen runtime source inspection identifies an existing NZ-weight control:
`additional_config.weight_nz_mode=2`. Default1 converts quantized weights only; mode2 also
converts BF16/FP16 linear weights. AscendUnquantizedLinearMethod explicitly excludes conv1d
weights. This is a candidate for the measured MatMul hotspot, not yet a tested improvement.
Test it as a separate intervention, with numerical checks appropriate to any changed
accumulation and an unpruned same-optimization control before DRRQR-specific attribution.

Generic determinism investigation remains deferred; no LoRA, thinking-mode evaluation,
upstream/checkpoint edits, commits or pushes.
