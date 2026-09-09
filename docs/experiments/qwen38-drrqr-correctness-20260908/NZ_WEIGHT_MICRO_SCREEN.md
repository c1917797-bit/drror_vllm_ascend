# NZ BF16 linear micro screen

Date: 2026-09-08. Performance-first exploration; joint target unachieved.

## Hypothesis, source and budget

Test whether persistent NZ BF16 linear weights reduce the observed MatMul hotspot.
Installed vLLM-Ascend defines FRACTAL_NZ=29 and weight_nz_mode=2 converts BF16/FP16
weights during process_weights_after_loading; conv1d is explicitly excluded.
The unquantized GEMM implementation calls torch.nn.functional.linear. The installed
model runner sets torch.npu.config.allow_internal_format=True; the probe matches it.

This is a native configuration candidate, not a DRRQR-specific algorithm improvement.
No upstream changes or full-model configuration changes are made for the micro screen.

One 16-case matrix on physical NPU4 (audited container logical0), external timeout900s:
six observed decode weight shapes at M=1 and16, four main prefill shapes at M=32768.
The source trace is profile-first-v1/dk64/profile-summary.v1.json. Shapes are actual;
probe inputs and weights are synthetic and seeded, not checkpoint/activation samples.
Script: tools/nz_weight_probe.py. CPU regression before launch:58 tests pass.

## Frozen numerical and timing criteria

- NZ-to-ND weight roundtrip must be bitwise exact; actual ND/NZ storage formats verified.
- All output elements compared between ND and NZ, finite values required.
- Both paths independently compared with CPU FP32 for first min(M,8) rows, all columns.
- Arithmetic-kernel screening thresholds, declared before execution: relative L2 <=
  2^-7 (one BF16 epsilon); max absolute error / reference max absolute <= 2^-6.
  These global error screens are not a task-accuracy or recurrent-state proof.
- Decode graph outputs must be bitwise equal to same-layout eager outputs.
- Stop on first failure; never loosen thresholds after observing results.
- Profiler disabled. Three warmup calls/layout; eight alternating timing blocks
  ND,NZ,NZ,ND,ND,NZ,NZ,ND. Each block uses50 graph replays/decode or5 eager calls/prefill.
  NPU-event elapsed time, four block means/layout; report median and individual means.

Repeated weights may be cache-resident and isolated graph replay includes dispatch effects.
A micro gain is not an end-to-end gain. Advance only with sound numeric evidence and
material gains in major shapes. Any eventual DRRQR attribution needs the unpruned model
with the same generic runtime optimization, and final matched quality at max_tokens=2048.

## Execution

Operation exec-00000000000000ce. Artifact directory:
`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/nz-weight-micro-v1/`.

Completed with exit0 in114.165 seconds. All16 cases passed the frozen numerical screen.
All16 ND/NZ outputs are bitwise equal; every weight roundtrip is exact. All12 decode
cases pass same-layout graph/eager bitwise comparisons. Maximum sampled CPU-FP32
relative L2 error:0.0020339072; normalized max absolute error:0.0034548047.

Report SHA256: ab2de6c3d5944a98fae7f106d7b39e36aa4467896da36fb1fe3ce3347096d999

Script SHA256: 21077a9018bc939381996fd7de6013eafb3a1b3f4e919b0257ce551953b30bf5

## Results and decision

Numbers are isolated per-call median latency changes; positive means NZ is slower.
They are not model throughput changes or multi-run statistical estimates.

| Shape | Decode M=1 | Decode M=16 | Prefill M=32768 |
| --- | ---: | ---: | ---: |
| out_proj [5120,1536] | +34.10% | unstable, see below | +0.94% |
| down_proj [5120,4352] | +22.61% | +0.73% | -1.63% |
| gate [24,5120] | -3.06% | -1.11% | not tested |
| in_proj [3584,5120] | +20.69% | +2.04% | -1.18% |
| lm_head [62080,5120] | -5.46% | -5.60% | not tested |
| gate_up_proj [8704,5120] | +6.77% | -2.08% | -0.98% |

For M=16 out_proj, the four NZ block means are115.434,67.480,19.093,19.080 us,
versus ND18.298,17.458,17.457,17.453 us. The recorded NZ median43.287 us (+147.96%)
is not a stable slowdown estimate. Preserve all samples; timing for this case is
inconclusive. No post-hoc sample deletion or rerun was used to manufacture a gain.
Some small differences in other shapes also overlap block variability.

Decision: do not promote global weight_nz_mode=2 to a full-model screen now. Major
shapes show little gain or regression; LM-head-only savings are a lower-priority lead,
not evidence of a 10% end-to-end improvement. This rejects a tuning candidate, not DRRQR.
The service launcher/configuration and installed plugin remain unchanged.

Next candidate is the installed enable_matmul_allreduce path in linear_op.py:
MatmulAllreduceRowParallelOp calls npu_mm_all_reduce_base using the existing TP HCCL
group. It has only been inspected, not enabled or benchmarked. First verify representative
TP4 numerical behavior and timing, then consider one profiler-disabled end-to-end screen.
Do not combine it with NZ in the first intervention or resume generic determinism work.
