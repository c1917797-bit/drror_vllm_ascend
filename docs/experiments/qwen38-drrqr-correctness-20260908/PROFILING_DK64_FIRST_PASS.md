# Dk64-energy profiling observations, first pass

Status: eight traces validated; paired Dk128 profiling is pending.
This is diagnostic evidence, not an accepted throughput/accuracy result.

## Workload and provenance

- Original Qwen3.8-27B, TP4 on physical NPU4-7.
- Frozen Dk64-energy plan:
  b1fcb727d3e58be66cc24148e30a5a1722d3bdc0914784f01cad02f11f741b8c.
- Built-in vLLM-Ascend v0.23 TorchNPUProfilerWrapper:
  Level1 + PipeUtilization, stack/memory/shape tracing disabled,
  ignore_frontend=true, max_iterations=30, delay_iterations=0.
- Compilation and workload warmup completed before profiling.
- Prefill: one request, 32768 input tokens, 16384-token shared-prefix
  construction, one generated token (no decode model execution).
- Decode: 16 concurrent requests, same prompt construction, 256 generated
  tokens per request. Start recording only after all requests have at least
  eight nonempty streaming chunks. Chunk counts are readiness signals, not
  claimed token counts. The server bounds the recording to 30 worker steps.
- All decode requests returned exactly 256 completion tokens and length
  termination. Final performance experiments remain fixed at 1024 tokens.
- Four-worker DRRQR activation audit passed, including actual reduced
  prefill/decode branch evidence.

Artifact root:
`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/profile-first-v1/dk64/`

Key artifacts:
`service-manifest.json`, `profile-client.json`, `activation-audit.json`,
`profile-summary.v1.json`, and per-rank `traces/*/ASCEND_PROFILER_OUTPUT/`.

The summary binds kernel CSVs and trace JSONs by SHA256.

## Capture recovery and validity

The daemon vLLM workers cannot spawn the profiler's analysis subprocess.
Runtime logs explicitly reported this restriction. Raw device and framework
captures were present, and offline parsing succeeded for all eight traces:

`torch_npu.profiler.profiler.analyse(path, max_process_number=4)`

The installed default schedule also logs a RECORD-state warning on stop.
The source still finalizes the raw trace; validity was checked from outputs
rather than inferred from HTTP 200 or absence of an exception.

Every trace has a populated 46-column kernel_details.csv, op_statistic.csv,
api_statistic.csv and a parseable, nonempty trace_view.json.
Each prefill rank has 3639 kernel rows, 48 causal convolutions, 48
ChunkGatedDeltaRuleFwdH calls, 16 full-attention calls and zero recurrent
decode calls. Each decode rank has 57540 rows, exactly 1440 recurrent GDN
calls (48 layers x 30 steps), 1440 convolutions and 480 full-attention calls.
The native wrapper has no ProfilerStep markers; phase/step coverage is
verified using these independent model-operator counts.

## Observed time distribution

Rank0, diagnostic recording only:

| Phase/operator | Count | Kernel time sum |
| --- | ---: | ---: |
| Prefill MatMulV3 | 256 | 1656.872 ms |
| Prefill AllReduce | 129 | 1113.330 ms |
| Prefill full attention | 16 | 402.603 ms |
| Prefill AddRmsNormBias | 128 | 285.994 ms |
| Prefill GDN FwdH | 48 | 118.455 ms |
| Decode MatMulV2 | 9150 | 636.886 ms |
| Decode full attention | 480 | 360.839 ms |
| Decode AllReduce | 3870 | 59.874 ms |
| Decode recurrent GDN | 1440 | 54.275 ms |
| Decode conv-weight transpose | 1440 | 29.523 ms |

Prefill kernel envelope across ranks: 4345.285-4346.572 ms.
Communication intervals with no concurrent compute: 1110.890-1114.528 ms.
Uncovered intervals inside the kernel envelope: 14.510-14.848 ms.

Decode 30-step kernel envelope across ranks: 1380.101-1380.670 ms
(about 46.0 ms per step in this profiling window).
Communication intervals with no concurrent compute: 63.820-70.972 ms.
Uncovered intervals inside the kernel envelope: 31.125-37.212 ms.

Interval calculations use exact integer nanoseconds and union overlapping
ranges rather than summing across streams. These observations do not by
themselves identify dependency-critical time or prove achievable savings.
The kernel envelope also does not include every frontend/service cost.

## Pipeline evidence and interpretation

Rank0 duration-weighted per-kernel pipeline metrics:

- Decode MatMulV2: MAC ratio ~0.0755, MTE2 ratio ~0.9496.
- Decode full attention: MAC ratio ~0.0722, MTE2 ratio ~0.9078.
- Prefill MatMulV3: MAC ratio ~0.8927, MTE2 ratio ~0.8815.

These are pipeline-activity ratios, not achieved FLOPS percentages or
measured HBM bandwidth saturation. Decode's low MAC/high MTE2 pattern
supports investigating weight/KV movement, with shape/byte-volume and
bandwidth measurements needed for stronger attribution.

Recurrent GDN alone occupies about 3.93% of rank0's decode envelope.
Even hypothetically eliminating it would save only that portion of this
decode window, before accounting for overlap and other bottlenecks.
The GDN pipeline includes additional projection, convolution, gating,
normalization and layout work, so this number is not the total GDN-layer
cost or an end-to-end limit for every possible DRRQR optimization.

## Next experiments

1. Finish matched Dk128 profiling and compare per-shape kernels, cache layout,
   communication and interval unions.
2. Investigate prefill AllReduce plus residual/norm scheduling; the exposed
   communication fraction is a substantial candidate, but no savings have
   been demonstrated.
3. Investigate decode matrix/KV data movement and repeated GDN-side layout
   work. Prioritize candidates with measured end-to-end headroom.
4. Any generic optimization must also be measured on Dk128; report generic
   deployment improvement separately from incremental pruning improvement.
5. Keep numerical parity mandatory. Task-accuracy repair follows the
   performance milestone according to PERFORMANCE_FIRST_PLAN.md.

No runtime algorithm/operator implementation was changed for this capture.
