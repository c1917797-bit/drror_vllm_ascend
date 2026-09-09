# Prefill MC2 single-pair result

Date:2026-09-08. Status: integration passed; no material end-to-end gain observed.
This is a performance-first screen, not final performance/accuracy acceptance.

## Result

Both sides use Dk64 energy pruning, packed convolution layout, the identical installed
local0.1.8 wheel, BF16/TP4, physical NPU4–7 and FULL_DECODE_ONLY. Only the prefill MC2
flag changes. Profiling and diagnostic determinism are disabled.

| Metric | MC2 OFF | MC2 ON | ON/OFF change |
| --- | ---: | ---: | ---: |
| Throughput (tok/s) | 134.606909753 | 134.514702159 | -0.068501% |
| Mean TTFT (ms) | 36648.522519 | 36500.970183 | -0.402615% |
| Mean TPOT (ms) | 68.874962 | 69.122274 | +0.359074% |
| Measured duration (s) | 304.293443 | 304.502031 | +0.068548% |

Each side completes40/40 requests, exactly1024 output tokens per request
(40960 total), finish_reason=length, plus2 one-token warmups. The frozen workload
is input32768/prefix16384, concurrency16, no prefix caching, raw completions.
Both services report11.02 GiB KV allocation and664701 KV-cache tokens.

Interpretation: the native microbenchmark improvement did not produce meaningful
throughput gain in this end-to-end screen. The tiny negative throughput difference
does NOT establish a statistically significant regression or equivalence: only one
ordered pair was run. Do not promote this candidate as a10% optimization.

## Integration and numerical evidence

The earlier native TP4 operator matrix passes24 rank-cases under its predeclared
BF16 error criteria (MATMUL_ALLREDUCE_MICRO_SCREEN.md). Fusion is not bitwise exact.
CPU regression00eb passes71 tests; the added read-only pair comparator itself is
exercised successfully in010d.

ON smoke00ef passes: four workers,128 prepared projections each,512 actual first32K
pure-prefill fused calls, then successful short compiled/decode requests.
OFF smoke0100 passes: no MC2 preparation/calls, same exact convolution layout checks.
The three corresponding short request output hashes match, with32/16/16 outputs.
These are integration/small-sample observations, NOT LongBench/GSM8K quality parity.
Smoke timings include first-use effects and are not treated as formal performance.

The existing evidence hook records only the first fused call per target. Its
unchanged hash during the benchmark is expected and does not measure the fraction
of benchmark batches or tokens that used fusion. In particular, it cannot prove
that mixed-batch fallback is the cause of the near-zero gain.

## Reproducibility and artifacts

Protocol: PREFILL_MC2_SCREEN.md.
Raw host root:
`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/prefill-mc2-screen-v1`

Subdirectories: dk64-on and dk64-off. Preserve manifests, evidence.jsonl,
activation-smoke.json, performance.run1.json, performance-screen.json and logs.
Complete hashes and comparison fields are in prefill-mc2-screen-v1-result.json.

Read-only reproduction:
`python3 /cache/cch/drror_vllm_ascend/tools/compare_prefill_mc2_screen.py`

Comparator SHA256:185f83dc850aeb4f3b3433cd42d712536b8a098eb1e2e153ab57fc624d61007c.
It checks receipt/report hashes, identical runtime/package/plan/protocol fields,
matching workload configuration, exact measured work and short-smoke output hashes.
The service manifests differ only in treatment flag, evidence path and creation time.

| Side | Service operation | Smoke operation | Performance operation |
| --- | --- | --- | --- |
| ON | 00ee | 00ef | 00f5 |
| OFF | 00ff | 0100 | 0107 |

Both clients and services terminate successfully with exit0. Before stopping each
API, running/waiting request gauges are0. Service lifetimes are747.559s and787.886s,
within each1800s cap. Physical NPU4–7 are all free after shutdown (0111).
CPU affinity binding is skipped on both sides by the existing runtime; it is
preserved, not changed as part of this experiment.

## Decision and next mainline step

Keep MC2 opt-in/default-off. Preserve the integration for a targeted follow-up, but
do not spend three formal repetitions or a full accuracy evaluation on this
near-zero screening result.

Next, measure the phase/row coverage of the actual workload with a bounded diagnostic
counter/profile, separate from timed performance. The adapter deliberately rejects
mixed prefill/decode batches; limited coverage is a hypothesis, not an observed
root cause. If coverage is low, assess whether large mixed batches can safely use
fusion with appropriate shape/numerical/short-sequence checks before changing the
gate. If coverage is already high, inspect critical-path timings and fallback or
dispatch overhead instead. No generic determinism investigation is resumed.

There is still no contemporaneous unpruned same-optimization control here. Any
future generic runtime gain needs that control before attribution to DRRQR.
Final acceptance remains a reproducible10%+ throughput gain and no LongBench/GSM8K
decline in the SAME configuration; quality baseline and candidate use max_tokens2048.
No new task-quality test, LoRA, thinking-mode test, upstream edit, commit or push
occurred in this screen. All implementation changes remain in this repository.
