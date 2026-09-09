# Findings: Qwen3.8 DRRQR GDN correctness gate

## Outcome

The preregistered numerical gate found a shape-specific defect in the
vLLM-Ascend v0.23 recurrent decode path. This is not a monotonic DRRQR pruning
result.

| phase | Dk128 | Dk104 | Dk88 | Dk64 |
|---|---:|---:|---:|---:|
| prefill | 2/2 pass | 2/2 pass | 2/2 pass | 2/2 pass |
| decode, two seeds x two batches x eight recurrent steps | 32/32 pass | 0/32 pass | 0/32 pass | 32/32 pass |

The Dk128 control passed, so the corrected v2 harness is valid. All prefill
dimensions passed. Decode state and output remained near the CPU float32
reference for Dk128 and Dk64. Dk104 and Dk88 diverged after recurrent state was
written back and reused.

| decode Dk | worst output relative L2 | worst state relative L2 |
|---:|---:|---:|
| 128 | 2.993e-5 | 7.192e-8 |
| 104 | 1.474 | 1.381 |
| 88 | 1.377 | 1.377 |
| 64 | 3.150e-5 | 6.746e-8 |

## Interpretation

Dk104 and Dk88 both require the recurrent kernel to pad the key dimension to
the next multiple of 16; Dk128 and Dk64 do not. Inspection of
`recurrent_gated_delta_rule.h` shows an aligned local state layout and a
`CopyOutState` operation configured with zero source-row stride. The observed
pass/fail boundary is therefore consistent with padded local rows being copied
back as if they were tightly packed. This is the leading root-cause hypothesis;
the native-kernel patch must still compile and pass the same frozen matrix
before it is promoted to a confirmed fix.

The earlier end-to-end results now have a narrower meaning:

| configuration | performance | quality interpretation |
|---|---|---|
| Dk104, actual 18.75% pruning | throughput -2.746%; TTFT +4.015%; TPOT +2.462% | invalid as an algorithm-trend measurement because recurrent decode parity fails |
| Dk88, actual 31.25% pruning | throughput -1.944%; TTFT +2.823%; TPOT +1.793% | invalid as an algorithm-trend measurement because recurrent decode parity fails |
| Dk64, 50% pruning | throughput +8.302%; TTFT -6.527%; TPOT -8.153% | kernel parity passes, but LongBench drops from 45.406824% to 35.958005% (-9.449 percentage points) |

Thus the apparent “50% works while 20%/30% do not” phenomenon has two
different causes: Dk104/Dk88 were corrupted by a runtime shape path, while Dk64
is a real performance/quality trade-off. The performance-quality target remains
unmet.

## Scientific decision

1. Preserve the rejected parent experiment; do not rewrite its historical
   observations.
2. Mark Dk104/Dk88 quality results as operational failures, not DRRQR accuracy
   evidence.
3. Keep Dk64 as the current performance-bearing candidate because both prefill
   and decode kernel parity pass.
4. Before another end-to-end sweep, either repair the native recurrent
   copy-out path or restrict candidate dimensions to kernel-safe multiples of
   16.
5. Profile baseline versus Dk64 separately for prefill, decode and end-to-end.
   Only after confirming the hot-path mechanism should accuracy recovery use
   the paper's missing calibration/RFT/teacher-distillation stages.

## Evidence boundary

This gate used physical NPU4 only, mapped to container NPU0, and reproduced one
TP4 worker's local Qwen3.8 tensor shapes. It proves a kernel-local numerical
property; it is not itself TP4 end-to-end performance or model-quality
evidence. Physical NPU0-3 were not touched and NPU5-7 were not occupied.

The full v2 report is
`/cache/cch/state-reduction-qwen38-drrqr-correctness-20260908/gdn-parity-v2/report.json`
with SHA-256
`9284def78e9cd839a457090d1581bc4c21f4e9dda338662c14c3f881b892a38a`.
