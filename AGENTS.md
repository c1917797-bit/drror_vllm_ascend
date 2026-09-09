# Project working instructions

## Skill review before execution

The user requires reviewing each relevant skill before acting on it. Read the skill fully,
then check its technical evidence, installed-version assumptions, task fit, authorization
boundaries, side effects and whether its validation really proves its claimed outcome.

Correct demonstrated problems directly within the user's authorized scope before following
the affected instructions. Do not treat speculation as a reason for broad rewrites. Record
the issue, evidence and correction; a format-validator pass is not technical validation.
Do not repeatedly ask questions already answered by the user.

For Ascend profiling in this project, use the audited task-local skill at
`skills/model-infer-profiling/SKILL.md` instead of the uncorrected cann bot reference.
If another skill needs adaptation, keep project-specific corrections in this repository
rather than silently editing shared skills or upstream runtime code.

## Current experiment boundaries

All implementation changes belong here. Reference vLLM-Ascend, ops-transformer and cann bot
trees are read-only. On 2026-09-09 the user authorized push to this repository and
research beyond 1–3 days on heterogeneous cache/native operators; all other boundaries
remain unchanged. See docs/experiments/qwen38-drrqr-correctness-20260908/EXTENDED_RUNTIME_RESEARCH.md.
Use only physical NPU4–7 and verify
container device mappings; never affect physical NPU0–3 or unrelated processes.

The current strategy is performance first, with numerical correctness mandatory. No LoRA
or thinking-mode evaluation currently. Final success requires the same configuration to
deliver reproducible 10%+ end-to-end throughput gain without accuracy decline. Accuracy
baseline and candidate use max_tokens=2048; the separate fixed-output performance protocol
uses 1024. Preserve historical results and label invalid or non-comparable evidence.

## Mainline experiment admission and scope control

The user's latest steering is algorithm-first: adapt DRRQR selectors, rank
budgets and recurrence behavior to obtain the joint objective. Operator/cache
changes are authorized when they enable or improve a concrete DRRQR candidate;
generic operator development is not an independent completion target. Continue
to apply any general optimization equally to the contemporaneous dense control.

Before each experiment state its performance-first phase, concrete hypothesis, deciding
evidence, and bounded run/time budget. After completion report the evidence and return to
the mainline. No automatic unbounded retry/search or new acceptance criteria.

Use minimal controls to determine whether an anomaly requires our change. A phenomenon
also present in the unmodified baseline is not alone proof that our change is safe, nor
grounds to invalidate historical evaluations. Test our incremental numerical effect.
Generic framework investigations are deferred unless necessary to trust a comparison;
if that requires expanding scope, stop that candidate and request the user's direction.
Do not launch the prepared diagnostic-determinism worker experiment without renewed
direction. No skill, diagnostic or passing microbenchmark overrides the final joint target.

Exploration uses representative change-specific numerical/state checks and a single
performance screen to prioritize candidates. Expand coverage and repeat paired runs for
promising candidates and final acceptance, not for every speculative change. Bitwise
equality is appropriate for pure storage-layout transformations; arithmetic/fusion changes
use justified predeclared error criteria and multi-step state checks. Never relax criteria
after a numerical failure just to obtain a pass. Final accuracy requirements are unchanged.
