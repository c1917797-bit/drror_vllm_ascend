# Prefill MC2 end-to-end screen protocol

Date:2026-09-08. Performance-first candidate screening, not final acceptance.

Hypothesis: the measured prefill GEMM/AllReduce savings survive native service
compilation/scheduling and improve end-to-end throughput. No algorithm/quality change
is accepted solely from kernel timing. The native TP4 micro screen has already passed.

One ordered pair: Dk64 MC2 ON, then Dk64 MC2 OFF. Both use the same local0.1.8 wheel,
same packed convolution layout, frozen energy plan, TP4 physical NPU4–7, BF16,
FULL_DECODE_ONLY graph setting and the unchanged frozen benchmark driver.
No NZ, native global enable_matmul_allreduce, profiler, diagnostic worker, training
or thinking-mode evaluation is added.

Each side first runs identical activation smoke: one32768-token input with32 outputs,
then two concurrent8192-token inputs with16 outputs each. These are raw completions,
seed0, temperature0, ignore_eos. This is a short compiled/decode functionality check,
not task accuracy or a claim of full-model numerical parity.

Before performance, require four worker local0.1.8/flag records, four complete exact
convolution-layout preparations, and (ON only) four128-projection MC2 preparations
plus512 real first32K fused-call records. All adapter hashes must match the manifest.
OFF must have no MC2 preparation/calls. A health response or preparation alone is insufficient.

Performance remains40 requests, concurrency16, input32768/prefix16384,
fixed1024 outputs, two one-token benchmark warmups, no prefix caching, async scheduling.
Both clients validate all40 unique requests and exact output work. The activation smoke
receipt is required and bound to that service manifest before the benchmark starts.

Budget: one pair, each service capped at1800seconds. Startup readiness <=900seconds;
smoke requests individually <=180seconds. Preserve failures and investigate an integration
failure before any retry; do not start a duplicate server after an observation timeout.
Do not extend into unrelated framework nondeterminism.

Expected artifacts: prefill-mc2-screen-v1/dk64-on and dk64-off under the preflight root.
Source helpers: run_conv_layout_service.py (explicit --plugin-version0.1.8/--prefill-mc2),
prefill_mc2_service_smoke.py, collect_layout_performance.py
(--experiment prefill-mc2-screen-v1).

Compare matching workload/runtime/plan/plugin hashes and report single-pair limits.
A promising result still needs unpruned same-optimization control, repeated paired
performance and matched LongBench/GSM8K max_tokens2048. Final joint target is unchanged.

Execution complete: see PREFILL_MC2_SCREEN_RESULT.md and
prefill-mc2-screen-v1-result.json. Both sides pass activation and exact-work checks;
ON/OFF throughput change is -0.068501%. Candidate remains opt-in/default-off.
