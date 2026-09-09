# DRRQR no-training performance closeout

> Superseded on 2026-09-09: the user authorized extended heterogeneous-cache/operator
> research and push. See [EXTENDED_RUNTIME_RESEARCH.md](EXTENDED_RUNTIME_RESEARCH.md).
> The packed-Triton hot-path inference and broad fusion/layerwise impossibility
> conclusions below are withdrawn; retained here as historical audit evidence.
> Measurements are unchanged. Current status is research authorized, not success.

Date:2026-09-09

## Decision

The current no-training DRRQR optimization branch is closed with a bounded
negative result. It produced a real and numerically gated Ascend improvement,
but it did not meet the predeclared final acceptance threshold.

Best credible same-stack single-screen result:

| Metric | Common-optimized dense | Dk64 | Paired change |
| --- | ---: | ---: | ---: |
| Output throughput | 135.142470 tok/s | 141.564496 tok/s | +4.752041% |
| TTFT | 34693.578 ms | 33468.964 ms | 3.529799% improvement |
| TPOT | 69.436227 ms | 66.689384 ms | 3.955922% improvement |

The locked success threshold was148.656717tok/s, or at least10% above the
common-optimized dense control. The best candidate therefore remains below the
performance gate. Three-round confirmation and final LongBench/GSM8K testing
were intentionally not run; there is no final accuracy-no-decline claim.

## Why the result is scientifically usable

- The success metric, treatment, workload, numerical gates, experiment budget
  and stop rules were declared before each deciding run.
- Candidate and control were matched on runtime, workload and common
  optimizations. Generic optimizations were not credited to DRRQR.
- Native GDN golden checks and model-level sequence checks ran before endpoint
  performance. Dk32 passed68/68 golden cases and all256 sequence token IDs,
  within the frozen logprob bounds.
- Profiler-enabled diagnostics were kept separate from profiler-disabled
  endpoint measurements.
- Failed attempts and negative screens were retained rather than hidden or
  converted into post-hoc threshold changes.
- Source, wheel, installed package and upstream provenance were hash-checked.
  All changes stayed in the plugin repository; upstream vLLM-Ascend and the
  original image were not modified.
- Only physical NPU4-7 were used. They were verified idle after closeout.

## Falsified hypotheses

1. **Dk64 recurrent-kernel rescue.** The current-stack profile showed recurrent
   GDN at only4.07% of the Dk64 decode envelope. Even eliminating it entirely
   could not credibly supply the remaining approximately5.01% candidate
   throughput.
2. **Projection fusion as a DRRQR differential.** The paired graph micro-screen
   projected13.703903ms saving for Dk64 and14.329437ms for dense. It is a
   generic optimization, not a candidate-specific gain.
3. **Concurrency32 scaling.** Dk64 throughput was0.077779% below its matched
   dense control and TPOT was7.635361% worse, despite a higher reported capacity
   estimate.
4. **More aggressive Dk32 pruning.** Dk32 reached137.991359tok/s,2.524% below
   Dk64 and7.729% below the target.
5. **Dk32-specific operator rescue.** The final paired profile found Dk32
   envelopes only1.296969% lower in prefill and1.242699% lower in decode.
   Its largest regressions were all-reduce1.766920ms, attention1.433630ms and
   recurrent GDN1.084250ms. None supplies the required optimistic7.174% time
   reduction, and aggregated kernel times are not additive critical-path proof.
6. **Decode conv/recurrent fusion.** The installed decode path already feeds
   causal-convolution output to one packed recurrent kernel that fuses Q/K/V
   addressing, Q/K normalization, gating, state update and output. Reaching the
   locked target would require an optimistic64.446691ms decode-envelope saving:
   more than the entire55.046810ms recurrent kernel, or77.697044% of recurrent
   plus convolution time. The remaining mathematical and state traffic cannot
   credibly be eliminated by fusion.
7. **Layerwise mixed Dk.** Exact optimization over Dk32/Dk64/Dk128 at the same
   average Dk64 budget improves the retained-energy proxy by only0.097074
   percentage points. It requires a heterogeneous cache/runtime contract while
   preserving total Q/K arithmetic, so it has no credible route to the missing
   candidate throughput.

## Evidence pointers

- Strict Dk64/dense endpoint comparison:
  `prefill-mc2-mixed-v1/dk64-vs-dense-common-optimized.json`,
  SHA256
  `d222aa3a39fa0933c8f2c72d0fba405aae6aa32fcc95e5823d3911092ffdffbd`.
- Current-stack Dk64/dense profile:
  `prefill-mc2-mixed-v1/profile-current-v1/profile-comparison.v1.json`,
  SHA256
  `9cca1cd11a9eecd12c6e45988505d767c92ac6aa41fe3a857445a50d5b64115b`.
- Concurrency32 comparison:
  `prefill-mc2-mixed-v1/concurrency32-v1/dk64-vs-dense-c32.json`,
  SHA256
  `98f19276fea95e26220db961afb0757425ce7c45a1738cb031a4fea824313d9a`.
- Final Dk32/Dk64 paired profile:
  `prefill-mc2-mixed-v1/dk32-vs-dk64-profile-v1/dk32-vs-dk64.profile-comparison.v2.json`,
  SHA256
  `1ca32e165b297a6b3a407d19757ab8c6552f9444a3b21c04b632f49613b06fc4`.
- Detailed projection bound: `OPERATOR_FEASIBILITY_BOUND.md`.
- Detailed decode-fusion bound: `DECODE_FUSION_FEASIBILITY_BOUND.md`.
- Layerwise algorithm bound: `LAYERWISE_ENERGY_BUDGET.md`.
- Full chronological integration record: `MC2_MIXED_INTEGRATION.md`.

## What would constitute a new phase

Continuing now would require a new, explicitly approved research scope rather
than more blind operator tuning. Plausible new scopes are algorithmic
redesign/calibration that changes the accuracy-performance frontier, or
training-based recovery such as LoRA/distillation after a new performance
candidate is established. Neither is part of this closed no-training branch.

The final requirement-by-requirement status is recorded in
`JOINT_ACCEPTANCE_AUDIT.md` and `JOINT_ACCEPTANCE_AUDIT.json`.
