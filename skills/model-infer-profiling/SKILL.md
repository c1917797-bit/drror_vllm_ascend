---
name: model-infer-profiling
description: Collect and validate Ascend NPU inference profiles, preferring the installed framework's profiler. Use for prefill/decode bottleneck investigation; distinguish diagnostic timings from profiler-disabled performance results.
---

# Ascend inference profiling — audited project edition

This is the maintained task-local correction of
`/cache/cch/cannbot-skills/model/model-infer-profiling/SKILL.md`.
Use it for this repository; the reference skill and upstream runtime are not modified.

## Before using this skill

Review the installed framework/version, task scope and evidence behind each relevant
instruction. Correct a demonstrated mismatch before executing it. A skill is guidance,
not proof and not authorization to mutate additional systems. Record material corrections
in the experiment findings. Do not repeat approvals or questions the user has answered.

## Select the collector and workload

- Prefer an existing profiler integrated with the actual execution loop. Inspect its
  start/step/stop implementation and runtime source hashes; a configuration name alone
  does not establish what was captured.
- Do not assume cann-recipes-infer's YAML or schedule exists in vLLM-Ascend. Inject a
  collector only if the native route cannot capture the required scope reliably.
- Use the user's selected phases/steps. If unspecified, recommend separate prefill and
  steady decode captures, with enough decode steps to check stability (30 is a starting
  point, not a universal requirement). Ask only when the choice materially changes cost
  or scope. The present task has already selected both phases and 30 decode steps.
- Freeze prompts, tokenization, concurrency, output limits, dtype, TP, graph mode,
  scheduling, prefix-cache settings, image/plugin hashes and device mapping for comparison.
  Distinguish sampling output lengths from formal benchmark and accuracy output limits.
- Keep first compilation and workload-specific warmup outside capture. Confirm actual
  batch/sequence shapes and request completion, not only HTTP status.
- Operate only on authorized physical devices. Recheck device mappings before new work.
  Do not stop unrelated services, reset devices or edit upstream code.

## Metrics

For this task use `ExperimentalConfig(Level1 + PipeUtilization)`; the frozen v0.23
native wrapper already supplies it. Confirm the parsed fields exist. Other investigations
may need a different supported metric set (for example memory traffic); do not present
PipeUtilization as the only valid profiler mode.

Pipeline ratios are per-kernel observations, not whole-device achieved FLOPS or measured
HBM bandwidth. High MTE activity is a reason to investigate data movement, not proof of
HBM saturation. Field names/counts vary with the installed CANN version.

## Capture-window correctness

Determine boundaries from the collector's actual state machine and where step() occurs
relative to execute_model(). There is no universal total-step formula for every wrapper.

- For a scheduled injected torch_npu collector, derive transitions from its installed
  schedule implementation and ensure recording reaches the trace-ready/finalize path.
  An extra profiler step, when needed, must not silently add an extra model execution.
- For the audited vLLM/vLLM-Ascend 0.23 image in this task, WorkerProfiler.step() runs
  before model execution. max_iterations=30 stops before execution 31, retaining 30
  complete decode executions. The NPU wrapper does not call the underlying scheduled
  profiler step; there are no useful ProfilerStep markers. Validate model-operator
  counts instead. Re-audit this behavior if the source hashes change.
- Prefill may contain several scheduler chunks for a request. Only expect one model
  execution when the actual input/token budget and trace establish that.
- A RECORD-state stop warning warrants investigation; it does not alone prove raw data
  loss. Conversely, a successful stop endpoint does not prove a complete usable trace.

## Parse and recover from evidence

Inspect the exact error and preserve raw captures before choosing a remedy.

- In this frozen runtime, the trace callback executes in daemon workers and explicitly
  reports that parsing is unavailable there. After capture completes, parse the same
  raw directory from a non-daemon process:
  `from torch_npu.profiler.profiler import analyse; analyse(raw_dir, max_process_number=4)`.
  The process limit is a bounded choice for this host, not a correctness requirement.
- For other failures, investigate raw-file completeness, parser/version compatibility,
  permissions, free space, resource limits and the reported error. Do not assume storage
  placement is the root cause.
- Try a different filesystem only when evidence supports that hypothesis. Preserve the
  original and compare parsing of identical raw data; no unconditional moves, deletes,
  symlink replacement or rerunning a costly capture merely because a CSV is missing.
- An observation timeout is not process termination. Poll the original live handle or
  inspect authoritative process state before retrying; do not start duplicate captures.

## Validate every intended rank and phase

Do not select the first glob match as representative of the whole capture.

Check all expected directories and populated kernel/operator/API artifacts; parse the
trace JSON completely. Verify required timing, stream, shape and requested metric fields.
A column-count heuristic may flag missing metrics but cannot establish correctness.

Match actual phase, step, batch, shapes and successful output lengths against the frozen
protocol. Where ProfilerStep markers exist, validate them; otherwise use independent
model-operator counts and reject mixed-phase or wrong-batch windows. Never silently
loosen coverage checks to make a failed capture pass. Save input/artifact/script hashes.

For this audited Qwen model, an isolated prefill execution contains 48 causal convolutions,
48 ChunkGatedDeltaRuleFwdH calls and 16 full-attention calls; 30 decode executions contain
1440 recurrent GDN, 1440 convolution and 480 full-attention calls on each TP rank.
These counts are model-specific and must not be copied to another architecture.

## Interpret and verify the optimization

- Separate prefill and decode. Aggregate by operator and shape, and examine all ranks.
- Preserve timestamp precision (integer nanoseconds or Decimal for epoch microseconds).
  Use interval unions to avoid double-counting overlapping streams.
- Kernel sums, communication without concurrent compute and gaps are observations.
  They are not automatically additive critical-path costs. Trace dependencies and test
  controlled interventions before claiming causal end-to-end attribution.
- Compare baseline and candidate with the same workload. Report unavoidable differences,
  including cache allocations and nondeterministic scheduling positions.
- After a change, pass the relevant numerical/state/graph correctness gates and verify
  which installed plugin actually ran. Then benchmark with profiling disabled.
- A generic optimization also needs the unpruned baseline with that optimization.
  Do not attribute generic runtime improvements solely to DRRQR.
- Never replace the final joint performance/accuracy gate with an operator microbenchmark.

## Existing project tools

Read the relevant tool before changing or invoking it:

- `../../tools/run_profile_service.py`: frozen service/provenance capture in the audited container.
- `../../tools/collect_service_profile.py`: workload/warmup and separate phase capture.
- `../../tools/summarize_service_profile.py`: four-rank artifact/coverage and interval validation.

These are task-specific tools, not universal launchers. Their fixed dimensions and step
checks are intentional for the frozen experiment; a different experiment needs a new,
explicit protocol rather than retroactively editing historical evidence.
