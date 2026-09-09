# Skill preflight audit — 2026-09-08

User instruction: review a skill's reasonableness before using it; correct demonstrated
problems before following affected instructions. This review does not authorize unrelated
changes. The policy is now in the repository AGENTS.md.

## Reviewed and adopted

- host-development: purpose-built remote read/patch/exec actions with explicit risk lanes.
  Appropriate for this work. Retain exact target-peer verification, force-direct route,
  explicit allowed patch paths and stable-handle polling. No functional correction needed.
- skill-creator: suitable for this scoped revision. In particular, do not universalize a
  past incident, preserve user scope, and distinguish format checks from behavioral proof.
  No functional correction needed for the workflow used here.
- model-infer-profiling: useful native-collector preference and artifact validation, but
  several absolute instructions were not valid across runtimes. Use the corrected project
  edition at skills/model-infer-profiling/SKILL.md. The shared reference was left unchanged.

## Evidence-backed corrections

| Original assumption | Correction | Evidence / scope |
| --- | --- | --- |
| Parser failure usually means filesystem placement; first switch disks | Diagnose the actual error; daemon-worker parser errors use offline analysis of the same raw files | Both Dk64 and Dk128 native captures parsed successfully in the same original directories |
| A universal scheduled-step formula applies to collectors | Inspect actual start/step/stop state machine and call position | Native NPU wrapper returns True from _profiler_step without calling profiler.step; max_iterations stops before model execution 31 |
| RECORD-state stop warning implies invalid data | Treat as warning requiring evidence, not automatic loss or automatic success | All 16 rank/phase traces validated after offline parsing, with exact expected operator counts |
| Fixed column count proves correct metrics | Check required fields and populated artifacts; count is only a version-specific heuristic | Service captures have 46 columns; microprobe captures have 47; both include requested metrics |
| Always ask phases and steps again | Honor existing user choices; ask only for unresolved material scope | Both phases and 30 decode steps were already agreed |
| First matching trace can represent the run | Validate every expected rank and phase | Four ranks times two phases per variant were checked |
| Pipeline ratio is achieved device bandwidth/utilization | Keep ratios at their per-kernel scope; measure traffic/bandwidth when needed | PipeUtilization does not supply a measured whole-device HBM throughput claim |
| Operator sums establish critical-path benefit | Use unions, dependency analysis, controlled changes and profiler-off verification | Parallel streams and distributed execution invalidate simple additive attribution |

## Validation beyond wording

The bundled skill-creator quick_validate.py returned "Skill is valid!" for the corrected
skill. This checks frontmatter/scaffolding only.

An executable check extracted the actual installed WorkerProfiler class from the frozen
source and simulated its real before-model step position. With delay=0/max_iterations=30,
only model executions 1 through 30 remained in the recording state; stop reset both
counters. Source SHA256:
31f8a33afd23e67a2ade5a95d0d837a03b0a9164d5b18acf561d948c9ae6c06c.

The NPU wrapper was inspected directly:
0c9ffcaf695a9b1fcb44a51bdd92e69f763a7c5be64a8efbd498d6536b9e09e5.
It supplies Level1/PipeUtilization and does not forward underlying scheduled steps.

Behavioral checks used actual captures, not just matching text:
- Daemon parser failure: same raw captures recovered with non-daemon offline analyse;
  no disk move or unnecessary recapture.
- Native no-ProfilerStep path: exact per-model operator counts validate 30 decode steps.
- Corrected raw-operator schedule: 30 calls captured and parsed without RECORD stop warning.
- Every intended rank/phase validated, including complete trace JSON and kernel/API/operator files.

The new skill does not imply that numerical gates, profiler-disabled speed or final accuracy
already pass. Any new skill or runtime version still needs its own relevant review.
