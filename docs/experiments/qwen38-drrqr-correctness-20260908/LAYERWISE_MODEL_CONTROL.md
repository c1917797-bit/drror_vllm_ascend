# Layerwise all-Dk64 model qualification

Date: 2026-09-09
Status: paired model control FAILED its predeclared logprob gates. No promotion to heterogeneous performance testing.

## Deciding question and bounded protocol

Does the plugin's layerwise runtime preserve the existing energy-kernel Dk64 model when every one of its 48 GDN layers has Dk64 and the selected coordinates are bound to the exact existing uniform plan?

One legacy-uniform64 service and one layerwise-uniform64 service are planned sequentially, each with a 1,800-second service budget and explicit process-group cleanup. They use the same frozen runtime, wheel, model, TP4 on physical NPU4-7, packed convolution layout, mixed prefill MC2, context 40,960, maximum 32 sequences, prefix caching disabled and FULL_DECODE_ONLY graph configuration. The shared HF dimension is 64 for both, not 128.

Deciding evidence, specified before representative requests:

- Four ranks must each qualify all 48 GDN modules, local projection/convolution/state geometry, effective cache spec and actual runner/static-forward-context cache binding.
- Corresponding local parameter byte hashes, names, shapes and dtypes must match exactly between services.
- Cache layout, shapes, strides, offsets and intra-process storage aliasing must match; absolute pointers naturally differ across processes.
- Four synthetic no-thinking requests use one sequential request followed by three concurrent requests and exactly 32 output tokens each, temperature 0, seed 0, ignore_eos true, with five logprobs returned.
- Prompt token hashes and generated token IDs must match exactly. Selected logprob maximum absolute difference must be <=0.02 and mean absolute difference <=0.005. Exact logprob equality is also reported. These endpoint tolerances do not replace the independent bitwise NPU storage/output-state gates.
- Any failure remains a failure with the original artifact retained. No automatic rerun or changed threshold.

This is model-construction/cache-binding and representative output-trajectory qualification. Neither the 32-token requests nor startup timings are performance or benchmark-accuracy evidence. The diagnostic worker must be removed for formal performance tests. Graph configuration/capture evidence is not represented as a new trace proving every request's dispatch path.

## Identity and current artifacts

Result root inside container: `/drrqr-results/layerwise-model-control-v1`.

- Wheel SHA256: `e0692641901ff3936c50d6165067186807cb5a5ce57cb2f686b94fa223640d21`.
- Build manifest SHA256: `846746502c61620a1f51a32318ac033681b96f0fe4e333195d95f99b200d34a1`.
- Uniform source plan SHA256: `b1fcb727d3e58be66cc24148e30a5a1722d3bdc0914784f01cad02f11f741b8c`.
- Layerwise all64 manifest SHA256: `761b976a996d0969bb3abb2387b860b24195adf2c20722fd1415a00cd3041d7f`.
- Probe worker SHA256: `678f714458f81a6b73c5dd19fc50a69e11c2e444837908460aa8e4aab9a46feb`.
- Legacy launch operation: `exec-0000000000000430`; result directory `legacy64`.
- Layerwise launch operation: `exec-000000000000045e`; result directory `layerwise64`.
- Comparison operation: `exec-0000000000000493`; report `comparison.json`, SHA256 `e1048c1eeea1cdafdf0eb168eec28a45e40cb13adcd3e5fdd94639cdbb51f36e`.

## Paired result and next deciding experiment

Both services completed all four representative requests and exited through stop.request with exit code 0 and no remaining process group. All four ranks qualified 48 GDN layers. Each rank's 336 parameter identities and 96 cache tensor layouts/alias patterns matched exactly; both had 925 blocks. All four prompt hashes and all 128 generated token IDs matched exactly.

The selected logprobs did NOT pass: maximum absolute difference 0.07280218601226807 (limit 0.02), overall mean absolute difference 0.007996276115264322 (limit 0.005). The 32-token-input request was issued alone and also exceeded both limits (maximum 0.06878197193145752, mean 0.01052058945282397). Do not attribute the failure solely to concurrent-request ordering. The report preserves the failure; the thresholds have not changed.

No same-implementation A/A control has yet measured the frozen service's repeat-run variation under this protocol. Such a bounded control, together with source/config/dispatch comparison, is the next deciding experiment. Existing-stack nondeterminism is a hypothesis, not a proven explanation. A/A variation would not retroactively turn this failed result into a pass, nor establish the candidate's correctness. No diagnostic-determinism flags are authorized by this protocol.

Additional read-only diagnosis compared all four ranks' model_class, parent_worker, worker_source and runtime_sources, plus all 192 GDN module_class/method_sources records: the recorded identities match. The twelve worker_dispatch_verified events also match after removing run-specific provenance; core/prefill event shapes match. Source inspection found no intended all64 arithmetic change, but this does not rule out unobserved configuration effects or numerical execution differences.

There is a concrete scheduling confound for the three concurrent requests: the first recorded mixed MC2 execution used 40,960 rows in legacy64 and 12,831 rows in layerwise64; the first pure-prefill down_proj event used 3,240 rows in both. This difference is not evidence that a storage-only wrapper changes arithmetic, nor can it explain the first isolated request: its 32-row prefill is below the 2,048-row MC2 threshold. Preserve this distinction when designing A/A and intermediate-value diagnostics.

The real model cache has BF16 convolution state and FP32 recurrent state, unlike the earlier BF16-recurrent ABI variant. Future heterogeneous hybrid experiments must use the actual service dtype and budget, and must validate multi-step state values rather than only metadata. Both service logs contain an aclgraph replay message during requests; this is direct runtime dispatch logging, not a new NPU trace covering all ranks and all requests.

Software regression: 153 tests passed, 14 torch.jit deprecation warnings (`exec-0000000000000495`). This software result does not override the failed NPU model-control gate. Post-run physical NPU4-7 were idle. A separate workload had started on NPU0-3; it was not touched.

Prelaunch checks confirmed physical 4-7 idle, mappings physical4-7 to container0-3, repository/model mounts read-only, unchanged image identity, no service process in the dedicated container, and about 79 GiB host available RAM. These checks are historical once launch begins and must be repeated before the second service.

## Relationship to joint acceptance

The prior NPU arena and GDN-only native-runner allocation experiments passed their own bounded eager/graph bitwise gates. They did not qualify hybrid attention/GDN shared pools. The separate hybrid-pool source audit found concrete overlapping-byte mappings for different Dk layouts. The new plugin allocation guard rejects those unsafe combinations; mixed-Dk serving is not admitted by this control.

After all64 qualification, implement and NPU-test a safe hybrid cache design, then evaluate bounded layerwise precision/speed candidates and the actual Ascend decode optimization route. General optimizations must also be applied to dense. Any numerically valid candidate gets a single fixed-1024-token end-to-end screen before expensive repeated performance and same-stack, no-thinking, normal-EOS, max_tokens=2048 LongBench-v2/GSM8K qualification.

Numerical failure returns to the offending implementation. Throughput failure returns to measured critical-path/layout work. Accuracy failure returns to sensitivity/selection/rank-allocation work without unapproved training. Nonreproducible success remains unaccepted. A rejected configuration does not establish that every heterogeneous or native-operator approach is infeasible. Current joint acceptance remains unmet; commit/push or report completion is not goal completion.
