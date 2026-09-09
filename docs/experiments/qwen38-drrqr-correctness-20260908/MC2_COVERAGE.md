# MC2 phase coverage diagnostic

Date:2026-09-08. Performance-first diagnosis after the frozen MC2 ON/OFF screen
showed-0.068501% throughput change. Goal remains joint10%+ throughput/no-quality-loss.

Hypothesis: pure-prefill-only eligibility may cover too little of the real mixed
workload to expose the native MatMul/AllReduce microbenchmark benefit. This is not
yet an observed cause. Measure phase/rows AND actual fused calls to decide.

Use the same installed0.1.8 wheel, Dk64 energy plan, packed convolution, TP4 physical
NPU4–7, BF16, async scheduling, no prefix cache and FULL_DECODE_ONLY. Keep upstream
sources read-only and retain original arithmetic, phase gate and communicator.
A repository-only custom worker wraps model_runner._model_forward outside compiled
model execution. It observes metadata, eligibility and the existing post-dispatch
evidence callback. It temporarily resets/restores only the once-per-target logging
flag so calls can be counted; ordinary evidence still writes only once per target.
No tensor, parameter, state-cache, output, random setting or scheduling knob changes.

Each rank writes a separate JSONL file. All four normalized model-forward sequences
must agree. Record unknown/inconsistent metadata instead of inventing token counts.
These counters can perturb timing/scheduling; do not treat diagnostic duration as
a performance result or assume its exact execution sequence reproduces the prior run.

Match prior warmup: one32K/32-output and two concurrent8K/16-output requests,
then two one-token warmups from the frozen driver's42 generated prompts.
Only then create the measurement marker, BEFORE the frozen driver's _run_measured
submits40 prompts in16/16/8 waves with fixed1024 outputs. Remove the marker after
all requests finish. No NPU calls/synchronization are added by the counter.

Budget: one diagnostic workload, startup<=900s, total service<=1800s,
at most8192 recorded forward calls per rank. No automatic rerun, no LoRA,
no thinking-mode test and no generic determinism-worker investigation.

Tools: mc2_coverage.py, mc2_coverage_worker.py, collect_mc2_coverage.py and
run_conv_layout_service.py --diagnostic-mc2-coverage --prefill-mc2 --plugin-version0.1.8.
Artifacts: /drrqr-results/mc2-coverage-v1/dk64-on.

Deciding evidence: pure/mixed/decode batch and token counts, eligible rows and actual
fused row-projection counts per K. If coverage is low, evaluate a safe large-mixed-batch
route only after numerical/shape checks; if high, inspect critical-path/dispatch costs.
No gate is relaxed merely to turn this diagnostic into a passing result.

## Completed evidence

Operations011b(service) and011c(client) both completed with exit0. All four
normalized sequences agree:3106 forwards/rank, comprising3 pure-prefill,
32 mixed and3071 pure-decode forwards. The40 measured requests all returned
exactly1024 tokens. Prompt count1310720 and recurrent decode count40920 match
the frozen workload; only98304 prompt tokens(7.5%) actually used fusion.
Both projection families have the same7.5% coverage; mixed batches contributed
1212416 prompt tokens(92.5%) and226 decode-token rows but no fused calls.
This establishes low route coverage in this diagnostic, not an additive
critical-path attribution of the earlier throughput result.

Report: /drrqr-results/mc2-coverage-v1/dk64-on/coverage-report.json
SHA256:1bd4d638f1660d174f3182215d2f3c38625b94352065d812a178981a73ece7d6.
See MC2_MIXED_SHAPES.md for the resulting bounded native operator check.
The historical diagnostic worker remains pinned to0.1.8; the launcher explicitly
rejects reusing it with0.1.9, whose phase adapter has a different hash.
