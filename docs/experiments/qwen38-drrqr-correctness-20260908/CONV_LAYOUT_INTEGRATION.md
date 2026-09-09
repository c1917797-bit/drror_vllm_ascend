# Convolution layout integration: v0.1.7 validation

Date: 2026-09-08. This updates the historical stage snapshot in
PROFILING_PAIRED_AND_CONV_LAYOUT.md; it does not replace or revise those measurements.
Status: integrated and activated; whole-model output equivalence is NOT yet established.

## Scope and build provenance

All implementation changes remain in this repository. No upstream vLLM-Ascend or checkpoint
edits, commit or push. The dedicated container maps physical NPU4–7 to logical0–3;
the mapping was rechecked before the diagnostic requests.

The opt-in flag is VLLM_ASCEND_DRRQR_CONV_LAYOUT=1. The plugin's load_model wrapper
repackages the 48 convolution weights after loading/post-processing and before graph capture.
It requires exact model/TP/dtype/topology, rejects offloading and reload, and checks every
loaded parameter value for exact equality. The no-pruning control can use the same layout
optimization without enabling DRRQR.

The local offline wheel is drror_vllm_ascend_plugin-0.1.7-py3-none-any.whl.
SHA256: c5f137872059a60c6a13893dcc850af5658999547d5c7bffdaab1b6f42683ba6.
The service launcher requires all 17 installed package sources to match its local build
manifest and the repository, and pins the frozen upstream source and pruning-plan hashes.
The wheel was installed only in the dedicated experimental container.

Artifact root:
`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/conv-layout-e2e-v1/`

Build evidence: build/build-manifest.json.
Packed service evidence: dk64-packed/service-manifest.json and evidence.jsonl.

## Completed checks

- CPU regression suite: 37 existing tests passed in a fresh recheck (operation006a, exit0).
  With three new diagnostic-validator tests, all40 pass (operation0074, exit0);
  the complete result is saved in validation-conv-integration-v1.json.
- Earlier isolated Ascend convolution parity: all64 cases/512 decode steps passed;
  output and full convolution state were bitwise equal. See the historical profile report.
- Loaded-model activation: four distinct TP worker PIDs, each recording all48 target layers;
  every layer has exact_values_verified=true.
- Full decode graph capture completed. Runtime startup recovered from four Triton AOT
  cache-load warnings by recompilation. Recovery is not a numerical equivalence proof.
- Both reduced GDN prefill and decode hot-path evidence were observed in all four workers.

Packed service manifest SHA256:
907f1ded171606c9793d200aa1202aa3f56ebb0615a055b87c67ff5ae0e280bc.
Packed evidence SHA256 at inspection:
68970aa6912bf1e6dcd2a202374f198ecd8dcfc68e7eb9513c5b72b167b02f9b.

## Initial output repeatability failure — preserved

tools/collect_layout_equivalence.py sends two rounds of frozen synthetic prompts:
two sequential32768-token inputs and sixteen concurrently submitted2048-token inputs,
each with128 output tokens, temperature0, seed0, ignore_eos=true and token IDs returned.

The client completed both rounds but exited1 at its explicit repeatability guard:

- Sequential long prompts: 2/2 have identical token IDs across rounds.
- Concurrent prompts: 0/16 have identical token IDs across rounds.
- Seven concurrent prompts first differ at output position0 (zero-based); the others
  first differ at positions13–72.
- All requests completed with the expected token counts. Successful HTTP responses do
  not supersede the failed output check.
- The automatic profiling stage was NOT started, and no new formal throughput result exists.

Preserved report: dk64-packed/output-probe.json.
SHA256: 696572123aa7cae8016b2a33d7e17d94abb10bb665c49a6c4cdc99a146456b4b.

This does not identify the cause. Async batching, floating-point implementation effects,
a pre-existing runtime issue and a layout-integration issue remain hypotheses. Do not
label the new layout correct merely because the isolated operator was bitwise equal.
Do not label it defective merely because this same-configuration repeatability test failed.

## Isolation protocol and next decisions

tools/collect_layout_isolation.py preserves the failed report and uses the exact same16
medium prompt hashes. It records two rounds of serial requests and two rounds submitted
as a single API prompt-list, including top5 logprobs. A single API prompt-list does not
prove a fixed device batch schedule. Results are diagnostics, not acceptance metrics.

Packed isolation completed (operation0069, exit0 means collection completed, NOT parity
passed): only1/16 serial cases and1/16 prompt-list cases are repeat-stable. Thus a
concurrent-submission-only explanation is insufficient. Some first-divergence top-token
probabilities move substantially; ordinary rounding has not been established as the cause.
No profiling was collected. The packed service was stopped gracefully after verifying
zero running/waiting requests; its launcher exited0.

Use an original-layout Dk64 service with the identical0.1.7 wheel, plan, runtime and graph
settings for the same initial and isolation probes. Compare within-mode repeatability first,
then cross-layout token/logprob differences. If API scheduling prevents a meaningful
equivalence gate, use a controlled graph/operator replay; do not silently relax tolerances
or substitute HTTP success for correctness.

Only after a justified numerical/graph gate may the candidate proceed to whole-model
transpose-removal profiling and profiler-disabled paired performance tests. A no-pruning
plus same-optimization control remains required for attribution. The final10%+ throughput
and no-accuracy-decline objective remains unachieved. Accuracy evaluation uses2048 output
tokens for both sides; this128-token runtime probe is not an accuracy evaluation.
