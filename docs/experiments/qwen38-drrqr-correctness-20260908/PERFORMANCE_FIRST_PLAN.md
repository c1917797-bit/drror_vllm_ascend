# Qwen3.8 DRRQR: performance-first iteration

Status: active; joint target not achieved.
User steering: pursue performance first, then repair accuracy.
This supersedes the execution ordering previously described in conversation.
It does not reopen or alter historical rejected results.

## Objective and invariants

Demonstrate reproducible end-to-end throughput gain >=10% and no measured
LongBench/GSM8K score decline in the same final candidate, without thinking.
All algorithm/operator/runner changes belong to this plugin repository.
Upstream vLLM-Ascend and source checkpoints are read-only references.
No LoRA, commits or pushes are authorized by this iteration.
Only physical NPU4-7 may be used; no operations target physical NPU0-3.

The performance milestone is provisional while task accuracy is below baseline.
Runtime numerical correctness remains mandatory throughout performance work:
golden parity compares implementations of the same mathematical workload;
it does not prove the pruned model preserves the full model's capability.

## Frozen starting point

- Baseline Dk128 mean: 120.660715883 tok/s; TTFT 40029.11975 ms;
  TPOT 77.4988467 ms; historical LongBench 45.406824%.
- Dk64-energy preflight: 126.01112234036557 tok/s; TTFT
  37399.47126375628 ms; TPOT 74.94070224754351 ms.
- Performance: 40/40 requests, fixed 1024 output tokens; single run only.
- Historical Dk64-energy LongBench: 50/127 = 39.37007874015748%;
  approximately -6.04 percentage points from the historical baseline mean.
- Initial performance milestone relative to the historical mean:
  >=132.7267874713 tok/s. Re-establish a contemporaneous paired baseline
  before accepting the milestone; the historical number is a reference.
- Plan SHA256:
  b1fcb727d3e58be66cc24148e30a5a1722d3bdc0914784f01cad02f11f741b8c
- Local wheel SHA256:
  98e7356647939923581fbe40a7cb10261638b8ceb7422cc33cbc6a689f232ccc

Artifacts:
`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/`

The observed +4.43% is a single-run association with the reduced configuration.
It is not a controlled measurement attributing every percentage point to
pruning; runtime variability and shared optimizations require paired controls.

## Phase 1: profiling and the performance milestone

1. Freeze Dk64-energy weights, selection and service settings.
2. Use the built-in v0.23 TorchNPUProfilerWrapper. Its source already enables
   Level1 + PipeUtilization; do not introduce a second profiler.
3. Default sampling is both prefill and decode, with 30 steady decode steps.
   The user asked for our recommendation; use both phases and 30 decode steps
   as the initial diagnostic window, expanding it if variability warrants.
   This is an engineering sampling choice, not an Ascend-mandated step count.
   Keep compilation and
   warmup outside recording. Confirm the installed wrapper's delay/max-step
   semantics and inspect actual recorded steps; do not assume generic torch
   schedule options work in the Ascend wrapper.
4. Use the frozen 32K input / 16K shared-prefix prompt construction and
   representative concurrency (including 16), with prefix caching and service
   settings matched to the existing performance workload. Tag mixed prefill/
   decode intervals explicitly; do not treat a mixed step as pure decode.
5. Compare Dk128 and Dk64 profiles: projection/conv/GDN/state copies,
   normalization, layout changes, graph replay, communication and host gaps.
   Analyze critical-path time and overlap, not summed kernel durations alone.
6. Check optimization headroom before implementation. A 10% throughput gain
   requires ~9.09% lower end-to-end duration at fixed work. If only fraction f
   can improve by factor s, use 1/((1-f)+f/s) as an idealized serial estimate,
   with measured overlap/communication limitations stated.
7. Implement the highest-impact numerically equivalent change in this plugin.
   Prioritize measured state traffic/layout/tiling/fusion hotspots. Run golden
   parity after every change to relevant kernel math or memory layout.
8. Measure with profiler disabled. Use paired baseline/candidate rounds and
   identical requests, fixed output lengths, concurrency and warmed graphs.
   Require three successful paired rounds; report individual values and
   aggregate throughput, TTFT and TPOT.
9. If an optimization also benefits Dk128, include optimized Dk128 as a
   control. Report total deployment gain and incremental pruning gain
   separately. Do not label a generic kernel gain as a DRRQR-specific gain.

Full task-accuracy evaluation is deferred in this phase. Lightweight numerical
and response-validity checks continue. Profiling traces are diagnostic and
must not be reported as production throughput measurements.

## Phase 2: repair accuracy while protecting the performance gain

- Keep Dk64 as the first candidate; optimize coordinate selection offline.
- Available captures contain post-convolution Q/K only. They do not contain
  V, g, beta, recurrent state, output or teacher logits. Collect missing
  quantities before claiming real recurrent replay or teacher fidelity.
- Discrete Q/K coordinate selection is compatible with depthwise convolution
  and elementwise SiLU. A dense projection generally cannot be folded exactly
  through that sequence. Do not promise a zero-cost dense projection.
- Use disjoint calibration-request subsets for selection and held-out proxy
  assessment; fit the energy comparator on training requests only.
- Measure both normalized QK read geometry and KK historical interactions.
  Also report actual Dk^-0.5 output-score scaling, but account for downstream
  per-value-head normalization before interpreting its task impact.
- If Dk64 selection improvements plateau, try already parity-validated
  Dk96/Dk112 with the optimized operators. Dk80 requires a new parity gate
  and explicit runtime support; 16-alignment alone is not evidence of support.
- Predeclare a bounded search budget before each new experiment. If the
  held-out improvement is negligible or reverses, stop that candidate rather
  than repeatedly evaluating the same benchmark for selection.
- Re-measure performance after any change in Dk or runtime path. A Dk64
  speed result and a Dk96 accuracy result cannot establish the joint target.

## Phase 3: final paired acceptance

- Accuracy: explicit enable_thinking=false and max_tokens=2048 for BOTH
  baseline and candidate, honoring the user's instruction. Historical
  1024-cap runs remain historical evidence, not the new final baseline.
- Performance: fixed 1024 completion tokens remains a separate benchmark
  setting; it is not the accuracy generation cap.
- Use identical frozen evaluation ids, templates, truncation rules, scoring,
  seeds/decoding settings and model revision for each pair.
- Report LongBench frozen-127 sample results as such, not as the complete
  LongBench population. Preserve GSM8K's frozen evaluation scope.
- Require candidate observed aggregate scores >= paired baseline on both
  tasks. Also report paired disagreements and uncertainty; small-sample
  equality does not prove population-level equivalence. No post-hoc loss
  tolerance or selective task averaging may turn a decline into a pass.
- Re-run three profiler-disabled performance pairs on the same final
  configuration, require >=10% throughput improvement, report TTFT/TPOT.
- Confirm relevant golden parity, four-worker plugin activation and hashes
  for source, built wheel, installed modules, plan and loaded custom op.
- Final success requires all of the above simultaneously.

## Current verified preparation

On this continuation the old preflight container was stopped, docker ps was
empty, and physical NPU4/5/6/7 each reported zero AI core/vector use and 8%
HBM allocation. No profiling or NPU inference was launched during this
preparation. Recheck device/container state immediately before launch.

Official workflow references (local frozen source governs version details):
- https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html
- https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/graph_mode.html
