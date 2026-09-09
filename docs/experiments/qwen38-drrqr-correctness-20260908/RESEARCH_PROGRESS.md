# Current research checkpoint

Updated: 2026-09-09
Main objective: still in progress, not achieved.

Read EXTENDED_RUNTIME_RESEARCH.md and HETEROGENEOUS_RUNNER_ALLOCATION.md.
The earlier handoff saying the arena probe is unrun is stale:
v4 arena ABI and the new real-runner GDN allocation NPU probes both passed.

Latest completed NPU operation: exec-000000000000039c.
There is no active experiment handle to poll from this checkpoint.
No DRRQR heartbeat was successfully created; a prior automatic-run request was
rejected by the safety reviewer and must not be recreated without explicit
user authorization for recurring automation.

Next concrete implementation: per-layer Dk model/weight contract and full
attention+GDN cache grouping/allocation, preserving the all-Dk64 control.
Do not restart the already-passing arena/runner matrices unchanged.
Keep NPU4–7-only, no LoRA, no upstream/image edits. Commit/push on the research
branch is authorized. Do not treat this checkpoint or a push as goal completion.
