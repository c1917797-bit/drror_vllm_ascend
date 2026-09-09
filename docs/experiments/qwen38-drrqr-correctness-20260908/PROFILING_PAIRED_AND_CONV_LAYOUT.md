# Dk128/Dk64 paired profiling and convolution-layout candidate

Date: 2026-09-08. Status: diagnostic evidence and isolated candidate parity passed.
The final performance × accuracy objective is NOT achieved.

## Matched baseline comparison

Artifacts:
`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/profile-first-v1/`

Each baseline and dk64 directory contains service-manifest.json, profile-client.json,
profile-summary.v1.json and all eight raw/parsed rank-phase traces.
`paired-profile.v1.json` records matched workload/settings, file hashes and limitations;
reproduce with `tools/compare_service_profiles.py`.

Four-rank averages (rank spread is NOT independent-run statistical uncertainty):

| Metric | Dk128 | Dk64 |
| --- | ---: | ---: |
| Prefill kernel envelope, ms | 4498.045 | 4346.159 |
| Prefill communication without concurrent compute, ms | 1113.316 | 1113.117 |
| Decode 30-step kernel envelope, ms | 1414.881 | 1380.522 |
| Decode MatMulV2 kernel sum, ms | 666.131 | 636.917 |
| Decode recurrent GDN kernel sum, ms | 62.810 | 55.514 |
| Decode convolution kernel sum, ms | 28.754 | 28.785 |
| Decode weight transpose kernel sum, ms | 28.994 | 28.769 |

Diagnostic envelope reduction is 3.3767% prefill and 2.4284% decode. These are NOT formal
throughput improvements. Prefill communication does not shrink materially with Dk; compute
dimension reduction does not automatically solve that cost. Recurrent GDN is only one part
of each GDN layer; its timing is not the timing of the whole layer.

Shared: exact 18 prompt hashes, 32768 input/16384 prefix, decode concurrency16, 30 captured
decode executions, same runtime/plugin source hashes, graph and scheduling settings.
All 16 rank-phase traces passed. Native callback daemon parsing errors were recovered by
offline analyse in the same directories.

Limitations: different allocated state/cache pools under the same memory budget (baseline
455 state slots vs candidate925; full-attention pools5460 vs5550), nondeterministic request
decode positions under async scheduling, single capture per variant, profiler overhead.
Client/service hashes differ due to previously recorded EOF whitespace cleanup; matched
protocol/settings are checked. No causal attribution from sums alone.

## Candidate: prepare fixed conv-weight layout once after loading

The installed Ascend core repeatedly views [C,1,4] as [C,4] and transposes to [4,C].
The operator profile contains one device Transpose per layer per decode step.

New experimental utility:
`drror_vllm_ascend/patches/conv_layout.py`.

It preserves tensor values, shape, dtype and Parameter object/loader metadata. It changes
storage to tap-major so the unchanged core's transposed view is already contiguous.
It is inference-only, shape/dtype guarded and idempotent. It is NOT automatically installed
and must never be called during inference, after graph capture, or while offloading/training.
No original vLLM-Ascend source or checkpoint was modified.

CPU tests: 2 tests passed, including four supported channel counts, exact values, parameter
identity, metadata preservation, idempotence, in-place reload and unsupported-input rejection.

Physical NPU4 operator gate:
`conv-layout-v1/parity.json`
SHA256: 20b8dbc32e5d9875f120357f8faaa7a2bac60723123dd3b42e383ef2f1fd6600.

64 cases passed: Dk128/112/96/64; 48 prefill cases including 32768-token sequences,
initial-state on/off and bias on/off; 16 decode cases totaling512 checked steps.
Both output and full recurrent convolution state are finite and bitwise equal between
original-weight-layout and packed-weight-layout calls. This is equivalence to the unchanged
operator, not a new full-model accuracy proof.

Independent Dk64/concurrency16 operator profiling, 30 calls each, after10 warmup calls:

| Capture | Conv calls | Transpose calls | Conv sum, us | Transpose sum, us |
| --- | ---: | ---: | ---: | ---: |
| Original layout | 30 | 30 | 470.26 | 376.16 |
| Packed layout | 30 | 0 | 329.90 | 0 |

Both captures have47 columns and complete kernel/operator/API/trace artifacts. Kernel CSV SHA256:
- Original: 32df5e62a24a717479ffcc9b2a632ff62fc194735938be76fb8ae776c8b9becd
- Packed: 5329baa5f8c45f5bc65fe8815a6a7a5092ed38630c28967473e324c9e05ca7d6

This verifies removal of the repeated copy in isolation. Absolute microprobe timings include
profiler effects and do not predict model-level gain. Production model graph execution is
not yet checked.

## Next required work

1. Integrate an explicit opt-in load-time hook in this plugin, after final weight loading
   and before graph capture, with audited48-layer targeting and activation evidence.
2. Support a no-pruning + same-optimization control; do not attribute generic gains to DRRQR.
3. Verify graph replay and model-level token/output parity for each fixed weight set.
4. Confirm the1440 transpose calls disappear in model profiling, then run profiler-disabled
   paired performance tests. Continue investigating the larger prefill communication and
   decode matmul/attention costs; this one change is not expected by itself to close10%+.
5. Performance first remains the strategy, followed by accuracy repair and final joint
   acceptance using2048-token accuracy outputs. No LoRA or thinking-mode tests.

Both diagnostic model services exited normally; the probe also completed. No commit or push.
