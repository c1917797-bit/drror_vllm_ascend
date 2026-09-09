# DRRQR joint acceptance audit

> Superseded on 2026-09-09: the user authorized extended heterogeneous-cache/operator
> research and push. See [EXTENDED_RUNTIME_RESEARCH.md](EXTENDED_RUNTIME_RESEARCH.md).
> The packed-Triton hot-path inference and broad fusion/layerwise impossibility
> conclusions below are withdrawn; retained here as historical audit evidence.
> Measurements are unchanged. Current status is research authorized, not success.

Date: 2026-09-09

## Outcome

The current no-training branch is blocked on the locked joint target. It has a
real Ascend performance benefit, but neither the performance threshold nor the
accuracy threshold is satisfied.

| Gate | Required | Authoritative evidence | Status |
| --- | --- | --- | --- |
| Numerical | Representative state/output checks pass | 256 token IDs exact; logprob max 0.014418 and mean 0.000620 | Pass within stated scope |
| Throughput | At least +10%, then three matched rounds | +4.752041% in one matched screen | Fail |
| LongBench-v2 | No decline, thinking off, max_tokens=2048, normal EOS | Dk64 39.370% versus baseline mean 45.407%; available run uses max_tokens=1024 | Preflight fail; final protocol missing |
| GSM8K | No decline under the final protocol | No Qwen3.8 candidate result | Missing |
| Paper claim | Distinguish paper-exact selector | Measured Dk64 uses energy-kernel | Not paper-exact |

The current LongBench-v2 preflight delta is -6.036745 percentage points. It
supersedes the older 35.958% Dk64 result for describing the current
energy-kernel candidate, but it is still only one 1024-token run and cannot be
promoted to final accuracy evidence.

## Why no further short NPU experiment is admitted

The same blocking condition now survives three consecutive bounded reviews:

1. the main no-training closeout found the best matched throughput at only
   +4.752041%;
2. source and profiling bounds rejected projection and decode-fusion rescue;
3. exact layerwise energy-budget optimization found only +0.097074 percentage
   points of selector-proxy headroom while preserving total Q/K arithmetic.

Other explored routes also failed: concurrency32, Dk32, Dk32 operator rescue,
packed convolution layout, NZ weights, and candidate-specific MC2 attribution.
Running more trials without a new mechanism would violate the predeclared stop
rule and increase false-positive risk.

## Scope required to resume

At least one material boundary must change:

- allow training recovery (LoRA/RFT plus teacher distillation) after selecting
  a performance point;
- authorize a separately scoped multi-day custom-runtime project whose
  feasibility target is established before NPU work;
- or change the locked +10% / no-decline acceptance target.

Until then, the scientifically correct state is blocked, not success and not an
open-ended tuning loop.

The machine-readable requirement-by-requirement record is
`JOINT_ACCEPTANCE_AUDIT.json`.
