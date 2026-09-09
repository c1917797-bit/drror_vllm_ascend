# Extended heterogeneous-cache and native-operator research

Date: 2026-09-09
Status: authorized research; joint acceptance not yet met.

## Authorization and unchanged constraints

The user authorized research beyond 1–3 days and subsequently allowed push to
https://github.com/c1917797-bit/drror_vllm_ascend.
All implementation stays in this plugin repository. Reference trees and the
original vLLM-Ascend image remain unchanged. Use physical NPU4–7 only.
No LoRA or thinking-mode evaluation is authorized by this extension.
Use the existing research branch; do not force-push.

The joint target remains at least 10% end-to-end output throughput improvement
against a contemporaneous equally optimized dense control, with no LongBench-v2
or GSM8K decline. Performance uses fixed 1024 output tokens; final accuracy uses
thinking off, max_tokens=2048 and normal EOS. Confirmation requires paired rounds.
Longer research is permission to investigate, not a guarantee of success.

## Correction of previous feasibility conclusions

The earlier decode-fusion report inspected the generic vLLM Triton implementation
and incorrectly treated it as the actual Ascend service path.

Read-only verification in the running dedicated container found:
- Ascend source: /vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py
- SHA256: d6ec29919268178f5bf6e70e689c1d273d04b1cb1d84dc94efa7bbbc35490816
- gating at line 313; non-spec decode Q/K normalization at lines 423–424;
  native torch.ops._C_ascend.npu_recurrent_gated_delta_rule at line 427.
- Plugin bootstrap verifies that the worker core is the Ascend core.
- The saved dk64-mixed-on/evidence.jsonl contains worker_dispatch_verified
  records with this exact Ascend source and hash.
- Existing NPU profiles contain separate recurrent, gating and normalization ops.

Thus the claim that the current service already uses the generic packed Triton
decode kernel is withdrawn. This does not establish a speedup for any new fusion.
The exact binary/operator loading provenance must still be captured for prototypes.

Kernel-time sums and an assumed inverse-throughput envelope are not rigorous
end-to-end impossibility bounds. The +0.097074 percentage-point retained-energy
proxy gain is not a task-accuracy bound either. Neither excludes heterogeneous
cache or fusion research. Historical negative measurements remain valid only
for the configurations actually tested.

## Work packages and deciding evidence

1. Native execution and integration contract (first bounded work package).
   Inspect native ABI, state layout, custom-op loader and existing graph path.
   Freeze a replay of actual TP4 worker shapes and state indices. Design a
   plugin-owned opt-in custom namespace/library with source/build/load hashes.
   Budget: one focused source/trace audit and one representative NPU probe after
   implementation, not repeated full-model launches.

2. Decode operator prototype.
   Hypothesis: packed QKV access and normalization/gating integration can reduce
   launches/intermediate traffic on the actual native path. Start with the
   smallest change supported by ABI and numerical evidence; do not immediately
   fuse every stage. Compare Dk64 and Dk128 under the same optimization.
   Decide using NPU multi-step output/state checks, eager/graph compatibility,
   operator timings, and a profiler-disabled paired endpoint screen. A micro
   win is not an endpoint win. Set tolerances before executing arithmetic changes.

3. Layerwise heterogeneous state/cache prototype.
   Hypothesis: per-layer widths can retain sensitive layers while reducing
   less-sensitive state/projection cost. Audit weight slicing, layer shapes,
   cache allocation/grouping, prefix-cache identity, scheduling and graph keys.
   Start with an all-Dk64 map as a behavior-preserving control, then a small
   heterogeneous map. A proxy selects candidates, never proves accuracy.
   Pure storage transformations require exact-value roundtrip checks on NPU.

4. Joint evaluation.
   Promote only numerically qualified candidates with promising endpoint results.
   Compare common optimizations on both arms. Use matched repeated throughput,
   TTFT and TPOT, then the same-stack 2048-token accuracy protocol for both arms.
   Report task counts, truncation, paired outcomes and variability; do not reuse
   historical 1024-token accuracy as final acceptance.

Each prototype has a declared hypothesis and bounded experiment budget before
launch. Keep failed evidence and change direction based on measured results,
without changing the acceptance target or declaring universal impossibility.

## Starting evidence and session state

Best endpoint screen: dense 135.142470 versus Dk64 141.564496 tok/s (+4.752041%),
one pair only. TTFT improvement 3.529799%; TPOT improvement 3.955922%.
Current accuracy preflight: 39.370% (50/127), compared with historical baseline
mean 45.407%, at max_tokens=1024. Final 2048 evaluation and GSM8K remain missing.

On resumption, npu-smi showed no processes on physical NPU4–7 and docker inspect
confirmed host devices 4,5,6,7 map to container devices 0,1,2,3 in
drrqr_qwen38_energy_preflight_20260908. Recheck before new NPU work.
This authorization/correction update did not launch an NPU workload.
