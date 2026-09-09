# MC2 observed mixed-shape micro protocol

Date:2026-09-08. Performance-first follow-up, not a full-model correctness result.

The completed four-rank coverage diagnostic (011c) finds3 pure-prefill steps,
32 mixed steps and3071 decode steps per rank. Of1310720 prompt tokens, only98304
(7.5%) enter the pure-prefill fusion path;1212416 (92.5%) are in mixed batches.
Actual decode-token rows sum40920, exactly40*(1024-1). All four normalized sequences
match. Client40/40 exact1024 outputs; the diagnostic service011b exits0.

Coverage report:
 /drrqr-results/mc2-coverage-v1/dk64-on/coverage-report.json
SHA256:1bd4d638f1660d174f3182215d2f3c38625b94352065d812a178981a73ece7d6

Observed mixed M values/counts:103(1),8292(1),24598(1),32769(1),40960(28).
M103 stays below the existing fusion thresholds and is not promoted.
The remaining four observed M values cover the large mixed linear matrices.

Hypothesis: native mm+all_reduce remains numerically valid and faster at these actual
large shapes, including non-aligned M. This experiment is a standalone matrix test;
it does NOT establish autoregressive state or task-quality correctness of a mixed-batch
model route. Installed API docs specify no pure-incremental fusion and accept general
nonquantized M; mixed-model use still requires targeted native integration checks.

One8-case TP4 matrix:
 M=8292,24598,32769,40960; N5120; K1536(out_proj) or4352(down_proj).
Reuse the previously checked operation, metrics and timing implementation. Same BF16 ND
weights, same HCCL communicator, physical NPU4–7 only. The source coverage report and
rank0 artifact hashes must match; requested row values must actually appear in mixed
records. No new plugin wheel, no changed numerical thresholds, no pure-decode fusion.

All-element fused versus separate comparison, both versus CPU FP32 product+Gloo SUM
for first8 rows; finite, relative L2<=2^-7, normalized max absolute<=2^-6.
10 warmup calls per path;8 alternating blocks of5 calls, NPU events and synchronized
wall clocks, slowest TP rank per matched block. One matrix, external900s timeout.
On numerical failure, stop all ranks and preserve artifacts.

Tool extension: matmul_allreduce_probe.py now accepts explicit --matrix-rows and
--shape-context observed-mixed-linear --shape-evidence. Default six-case matrix
is retained. CPU0129 caught a new parameter/local-variable name collision in the
summary helper; fixed before any NPU invocation. CPU012d then passes all76 tests.
The already-completed coverage service used unchanged tools and was not affected.

Artifact root: /drrqr-results/matmul-allreduce-mixed-shapes-v1.
If successful, next implement an opt-in large-prefill-containing mixed-batch path,
with original small/decode/graph/capture paths retained, then short native sequence
checks and one profiler-disabled end-to-end pair. Neither coverage nor micro timing
replaces the final10%+ performance/no-accuracy-decline target.

## Completed result

Operation012f completed exit0 in91.296s. All32 rank-cases passed the frozen
numerical checks, and all8 shape timings completed. Slowest-rank block medians:

| M | Projection | Separate ms | Fused ms | Device duration change |
| --- | --- | ---: | ---: | ---: |
|8292|out|2.787092|2.429104|-12.8445%|
|8292|down|3.859792|2.653690|-31.2478%|
|24598|out|8.133336|7.459354|-8.2867%|
|24598|down|11.205946|7.993666|-28.6659%|
|32769|out|10.741818|9.855902|-8.2474%|
|32769|down|14.815962|10.203544|-31.1314%|
|40960|out|13.741476|12.085932|-12.0478%|
|40960|down|19.335400|12.859436|-33.4928%|

Report: /drrqr-results/matmul-allreduce-mixed-shapes-v1/report.json
SHA256:514b82c9f7fee214436b51378db8cb622dc55b422b7019c9cb1f8470ead04104.
Synthetic standalone inputs and weights remain a limitation: this is neither
full-model mixed-state parity nor end-to-end throughput evidence. Proceed to
the opt-in integration tracked in MC2_MIXED_INTEGRATION.md.
