# Prefill MatMul + AllReduce micro screen

Date: 2026-09-08. Performance-first exploration, not joint acceptance.

## Audited scope and hypothesis

The existing Dk64 profile has substantial prefill MatMul and AllReduce cost.
Test compute/communication fusion as a separate intervention, with ND BF16 weights.
This is a generic runtime optimization and cannot be attributed solely to DRRQR.

Source checks in the installed runtime:
- NPUCommunicator inherits DeviceCommunicatorBase.all_reduce; the latter calls
  torch.distributed.all_reduce in place on the HCCL device group.
- Native MatmulAllreduceRowParallelOp calls npu_mm_all_reduce_base with weight.t()
  and the same ProcessGroup HCCL communication name.
- Installed torch_npu API documentation directs use in full/prefill scenarios, not
  incremental/decode. Therefore do NOT globally turn on enable_matmul_allreduce:
  that native selection is not phase-gated. If promising, integrate a prefill-only
  plugin route and leave decode on its original path.
- The API supports BF16, ND weight, TP4 and HCCS all-mesh. No NZ, quantization,
  LoRA, thinking-mode test or upstream changes are included.

## Frozen exploratory protocol

tools/matmul_allreduce_probe.py uses torchrun TP4 mapped to physical NPU4–7.
One six-case matrix: out_proj [5120,1536] and down_proj [5120,4352],
at M=2048/8192/32768. M32768 matches the observed primary prefill shape;
the two smaller M values cover shorter prefill chunks. Inputs/weights are seeded
synthetic rank-specific data, not recorded checkpoint activations.

All4 ranks independently compare every output element between separate and fused
paths. Both are also compared against first8-row CPU FP32 matmul + Gloo FP32
cross-rank sum. Finite values required; frozen thresholds: relative L2 <=2^-7,
normalized max absolute <=2^-6. These accommodate BF16 arithmetic/reduction order
in this exploratory screen, not a model-level quality guarantee. Stop all ranks on
any numerical failure.

Timing:10 warmup calls per path, then8 alternating blocks (four per path),5 calls
per block. CPU barrier and NPU synchronization before the timed region. Report
NPU events AND synchronized host wall time, using the slowest rank per matched block.
No profiler during timing. No decode fusion or graph claim in this prefill-only screen.

Budget:one matrix, external900-second timeout; no automatic rerun or threshold
relaxation. If substantial gains survive the numerical screen, implement the
phase-gated plugin integration and run one end-to-end screen before broader testing.

Final requirements remain a same-configuration reproducible10%+ end-to-end gain and
no LongBench/GSM8K decline. Accuracy max_tokens2048 on both sides; performance remains
fixed1024. Generic gains require an unpruned same-optimization control.

## Execution and result

Completed: operation exec-00000000000000dc, exit0,55.194 seconds. All24 rank-cases
pass the predeclared numerical criteria. Outputs are NOT bitwise equal, consistent
with different arithmetic/reduction order. Maximum observed relative L2:0.0033969435;
normalized maximum absolute error:0.0111731844. This does not establish task accuracy.
CPU regression before launch:62 tests pass.

Report SHA256: a9b52aad062a2ce97ab21655557d5d347083e8c7d2c5f3dc6291b7f7291caf0d

Script SHA256: e94dfa89479caf7206042b6939c65597f20bda8ba8f500382343d69ecce929d0

| M | Shape | Separate device ms | Fused device ms | Device latency change | Wall latency change |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2048 | out_proj | 0.703802 | 0.703184 | -0.09% | -0.47% |
| 2048 | down_proj | 0.987116 | 0.768180 | -22.18% | -22.25% |
| 8192 | out_proj | 2.777706 | 2.406642 | -13.36% | -13.30% |
| 8192 | down_proj | 3.862140 | 2.622778 | -32.09% | -32.05% |
| 32768 | out_proj | 10.802282 | 9.858686 | -8.74% | -8.75% |
| 32768 | down_proj | 14.880386 | 10.128626 | -31.93% | -31.96% |

Each entry is the median of four slowest-rank block means. Do not add these kernel
savings as if they were a causal whole-model critical path. The inherited container
has HCCL_OP_EXPANSION_MODE=AIV and LCCL_DETERMINISTIC=0; neither was changed.
The watchdog180s versus HCCL execution-timeout warning was retained in execution
output; no timeout occurred and all ranks exited successfully.

Decision: promote the prefill-only fusion candidate to plugin integration and one
end-to-end screen, with decode and mixed-batch fallback kept on the original path.
Do not enable the ungated native global flag or treat this as the final10% result.

Generated rank reports and aggregate report are preserved under
`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/matmul-allreduce-micro-v1/`.
