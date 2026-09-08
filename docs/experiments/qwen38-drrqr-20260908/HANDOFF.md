# Workstream Handoff — WS-20260904-qwen38-dense-paper-to-ascend

Updated: 2026-09-07T20:10:00Z

## Active Qwen3.8 DRRQR plugin piercing

The active identity is
`state-reduction-qwen38-drrqr-plugin-tp4-20260907`. It uses the unmodified
BF16 checkpoint `/cache/austinov/Qwen3.8-27B`, TP4, exactly host NPU4-7
mapped to container-visible 0-3, CPU0-95 and host port8227. Host NPU0-3 is a
protected external lane and must not be inspected, stopped or used.

The checkpoint itself declares
`Qwen3_5ForConditionalGeneration`/`qwen3_5_text`; those strings are its
runtime ABI and do not mean that a Qwen3.5 checkpoint was substituted. The
frozen config has 64 layers: 48 linear-attention and 16 full-attention layers,
Dk=128 and Dv=128. Config SHA256 is
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`;
the 18-shard index SHA256 is
`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`.

The first baseline attempts failed before model initialization with kernel
`Conflict open udevid` evidence. An unrelated container was computing on
host NPU0-1 but had all host devices 0-7 mapped. After the operator stopped
that container, the unchanged plugin-free baseline recovered. This establishes
the cause for this incident only; it is not a universal driver-defect claim.
Persisted incident evidence is under
`/cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907/incident-device-namespace`.

The current plugin-free baseline container is
`qwen38_drrqr_baseline_exact_host4_7_tp4_20260907_v6`. It loaded all 18
shards, `/v1/models` returned HTTP 200, the plugin-absence audit passed and
the deterministic smoke returned exactly `BASELINE_OK` with 22 prompt tokens
and four completion tokens. The historical smoke response bytes were not
persisted and its copied digest was invalid, so its response hash is explicitly
not established. Repeat that bounded smoke after performance and save the raw
bytes before hashing them.

Formal baseline performance operation `exec-0000000000000d03` is
terminal-successful with exit code 0. The artifact root is
`/cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907/baseline`.

All three repetitions passed the independent count/configuration/hash audit:

| Run | accepted requests | output tokens | output tok/s | TTFT mean ms | TPOT mean ms | duration s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40/40 | 40960 | 117.343093 | 42317.222010 | 78.957545 | 349.061876 |
| 2 | 40/40 | 40960 | 122.326739 | 38885.543035 | 76.761132 | 334.840938 |
| 3 | 40/40 | 40960 | 122.312316 | 38884.594205 | 76.777863 | 334.880422 |
| mean | — | — | 120.660716 | 40029.119750 | 77.498847 | 339.594412 |

The performance audit is `performance.audit.json`, SHA256
`fca7cbb68778a33ac6c0d790a0a878014a865a7c974967cbec12370e7c83d996`,
`ok=true`, `errors=[]`. A replacement smoke operation
`exec-0000000000000d1e` saved both request and raw response. It returned
HTTP200, exactly `BASELINE_OK`, and the response SHA256 is
`054cc8fa256cac8bb85b4d9924f03c84364ec15d2abc687d7cacd3a15a943a9e`.

Baseline LongBench-v2 completed three audited runs with 127/127 successful
requests and zero failures in every run. Accuracy was 44.881889763779526%,
46.45669291338583% and 44.881889763779526%; mean is
45.40682414698163% and sample SD is 0.9092130223458699 percentage points.
The baseline then exited 0 without OOM; port8227 is free and NPU4-7 each report
`No process in device.` Quality audit SHA256 is
`1db4c87362d6986cbbdd4af61b226fad1c182407d1643ae75223292cf84767a0`;
release audit SHA256 is
`a2ce93a29d290e62da251e355ea3163895f758041f8227371b395e23f961afc1`.
Capture attempt `exec-0000000000000e29` failed closed on its first
calibration request and accepted zero tensors. Its immutable
`capture-qwen38-v2` evidence proves that the scheduler held all 2048 prompt
tokens while the Qwen3.8 multimodal wrapper passed `inputs_embeds` and
`input_ids=None` at the text-model boundary. It exited cleanly and released
NPU4-7/port8227. The incident is recorded in
`incident-qwen38-drrqr-capture-token-binding-20260907.json`.

The fix binds and hashes the token slice at the pinned
`NPUModelRunner._model_forward` boundary, then requires the text-model and
GDN capture hooks to observe the same request context. It remains fail-closed
for missing buffers, non-calibration tokens or runner/model disagreement.
Fresh-clone tests passed 25/25. Retest operation
`exec-0000000000000e4d` then proved the fix in the real runtime: the first
2048-token request returned HTTP 200 and produced all 192 expected tensors
(48 linear-attention layers x four TP ranks). The attempt stopped afterward
because the host controller serialized `row_index` from a calibration row
whose frozen schema names that source field `index`. The immutable
`capture-qwen38-v3` root, all 192 tensors and the successful release audit
are preserved. This is a controller-bookkeeping incident, not a plugin or
algorithm failure.

Source-model full-shard hash operation `exec-0000000000000d21` exited 0.
All 18 shards and 55,563,006,776 bytes are bound in
`source-model-hashes/source-model-sha256.json`, SHA256
`b51f7db23673241578e3b680f3464b134fc0e731083243fceec239a43a55df41`.
Independent structural audit `exec-0000000000000d28` passed with
`ok=true`, `errors=[]`. The controller now validates the calibration
`index` field before writing the normalized ledger `row_index`. Its
syntax/self-test passed, and fresh capture operation
`exec-0000000000000e64` is active in immutable `capture-qwen38-v4`.

The monkeypatch review targets the real v0.23 call chain: the module-global
`vllm_ascend.ops.gdn.chunk_gated_delta_rule` prefill entry, the recurrent
decode branch, Qwen GDN Q/K weight loading, and actual post-convolution shapes.
It must leave V/Z and all full-attention layers unchanged. Installation alone
is not activation proof: each of four workers must emit bound prefill/decode
hot-path evidence with the selected Dk before performance is accepted.

The frozen candidate order is Dk102 (20.3125% reduction), Dk89 (30.46875%) and
Dk64 (50%). They use separate treatment containers and the same employee-style
performance ruler: input32768, shared prefix16384, output1024, 40 requests,
concurrency16, two warmups and three repetitions. CEval is excluded. Repeated
LongBench-v2 and GSM8K are run only for runtime-valid candidates that show
meaningful repeated performance improvement; native profiling is conditional
on performance and quality gates.

### Exact Qwen3.8 recovery

First poll the existing AWMCP exec operation
`exec-0000000000000e64`. Do not launch another capture from missing plan
files alone. Inspect the current artifacts without mutating them:

```bash
cd /cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907
cat controller-status.json
find capture-qwen38-v4 -maxdepth 1 -name '*.pt' | wc -l
wc -l capture-qwen38-v4/requests.jsonl 2>/dev/null || true
ls -l plans/Dk102.json plans/Dk89.json plans/Dk64.json 2>/dev/null || true
```

After terminal success, require exactly 3072 tensors, 16 request records and
all three plans bound to the frozen model, calibration and plugin hashes. On
terminal failure, preserve the partial capture directory and do not restart
into it; diagnose and choose a fresh capture root as required by the runner.

Current records:

- `incident-qwen38-baseline-device-namespace-conflict-20260907.json`
- `incident-qwen38-drrqr-capture-token-binding-20260907.json`
- `incident-qwen38-drrqr-controller-row-index-20260907.json`
- `experiment-state-reduction-qwen38-drrqr-plugin-tp4-20260907.json`
- `evidence-state-reduction-qwen38-drrqr-plugin-tp4-20260907.json`
- `evidence-drrqr-core-audit-and-main-publish-20260907.json`
- `decision-state-reduction-qwen38-drrqr-plugin-tp4-20260907.json`
- `knowledge-proposals-state-reduction-qwen38-drrqr-plugin-20260907.json`

The corrected v0.23 plugin now reproduces the official sequential seed-42
sampling contract, binds Qwen3.8 calibration tokens at the real Ascend runner
boundary and fails closed on Strong-RRQR non-convergence. It is published on
`main` at commit
`9822d4c746c4bceed63e4e2122dd04b91f3801ed`. A new clone using default branch
`main` resolved to that exact commit and passed 25/25 CPU tests in the exact
v0.23 image. The wheel's two changed runtime sources match that clone
byte-for-byte. The hash-pinned wheel is
`/cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907/plugin-artifact-runnerfix/drror_vllm_ascend_plugin-0.1.0-py3-none-any.whl`,
SHA256
`d25bb26492d823facc746763e3c9bb75a1641273285c70331adadf38a304ffaf`.
This proves publication and package checks, not NPU activation or benefit.

## Historical Qwen3.6 status

A new quality-first DRRQR follow-up is active under experiment identity
`state-reduction-qwen36-dk112-quality-first-20260907`. It does not revise or
overwrite the completed Dk64/Dk96 rejection. The only new candidate is Dk112
(128 -> 112, 12.5% aligned Q/K reduction). Its checkpoint build is the
CPU/storage-only AWMCP operation `exec-0000000000000cb7`; it exited 0 after
190904 ms. Independent audit operation `exec-0000000000000cbb` passed:
Dk112, 48 layers, 96 tensors, six rewritten shards and nine unchanged
hardlinked shards. Manifest SHA256 is
`033eac4d29e12e9f219b213f12a15ebd3217e183f353c43c2200bb47f149243a`.
The persisted structural audit SHA256 is
`0d9efd2418d9b34e3ef16b78949c366d79992b587bc53cc6110b7fd058e6a12c`.
No Dk112 NPU service, benchmark, quality run or profile has begun,
and no Dk112 benefit or quality claim is currently allowed.

The new records are:

- `experiment-state-reduction-qwen36-dk112-quality-first-20260907.json`;
- `evidence-state-reduction-qwen36-dk112-quality-first-20260907.json`;
- `decision-state-reduction-qwen36-dk112-quality-first-20260907.json`;
- `knowledge-proposals-state-reduction-qwen36-dk112-20260907.json`.

The allowed future runtime lane is exactly host NPU4-7, container-visible
0-3, TP4, CPU0-95, port8226 and HCCL range 27020-27120. NPU0-3 remains outside
scope even if a read-only outer status reports it idle. At the 2026-09-07
preflight all eight cards were empty; that observation does not grant authority
outside NPU4-7.

### Dk112 exact recovery

Build operation `exec-0000000000000cb7` is terminal-successful and must not
be restarted. To re-audit without mutation, run:

```bash
sha256sum /cache/cch/Qwen3.6-27B-gdn-drrqr-dk112/drrqr_manifest.json
python3 - <<'PY'
import json
from pathlib import Path
p = Path('/cache/cch/Qwen3.6-27B-gdn-drrqr-dk112/drrqr_manifest.json')
x = json.loads(p.read_text())
assert x['target_head_k_dim'] == 112
assert x['changed_layers'] == 48
assert x['changed_tensor_count'] == 96
assert len(x['changed_shards']) == 6
print('Dk112 manifest structural gate: PASS')
PY
```

If and only if the operation is terminal-failed, preserve the partial output
before retrying:

```bash
mv /cache/cch/Qwen3.6-27B-gdn-drrqr-dk112 /cache/cch/Qwen3.6-27B-gdn-drrqr-dk112.failed-exec-0000000000000cb7
docker run --rm --network none -v /cache:/cache sha256:660ce23a83574fdcd28ba126b07615829c06770c06fbdf6533f03cc4d90acdb1 python3 /cache/cch/.worktrees/WS-20260904-qwen38-dense-paper-to-ascend/.agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/build_qwen36_drrqr_checkpoint.py --src /cache/efim/Qwen3.6-27B --dst /cache/cch/Qwen3.6-27B-gdn-drrqr-dk112 --capture-dir /cache/cch/state-reduction-qwen36-formal-npu4-7-tp4-20260906/calibration/captures --calibration-jsonl /cache/cch/state-reduction-qwen36-formal-npu4-7-tp4-20260906/calibration/longbenchv2-16x2048-tokenids.jsonl --official-rrqr /cache/cch/LinearAttentionPruning/src/key_reduction/pruners/rrqr.py --new-head-k-dim 112 --expected-linear-layers 48 --tp-size 4 --captures-per-rank 16
```

The manifest is bound in `state_reduction_dk112_runner.py`; py_compile and
`git diff --check` passed in operation `exec-0000000000000cbc`. NPU execution
was intentionally not started because the current task scope excludes operating
NPU4-7. After explicit renewed authorization, the exact first runtime command is:

```bash
cd /cache/cch/.worktrees/WS-20260904-qwen38-dense-paper-to-ascend
python3 .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/state_reduction_dk112_runner.py --arm drrqr-dk112 --phase smoke
```

Only after that smoke passes, run fresh baseline quality and Dk112 quality
first. Passing means only `no-material-regression-observed`; it is not
`lossless-proven`. Performance and profiling are conditional on both datasets
passing the fresh baseline-variability gate.

The state-reduction paper piercing is **complete**. Official Strong RRQR Dk64
(50%) and Dk96 (25%) both failed the frozen GSM8K quality gate and are rejected
for this workload. Dk96 performance means were slightly favorable, but all
three-run intervals overlap, so no stable serving acceleration is claimed.
The final evidence audit passed and NPU4-7/port8225 are released. Profiling was
not started because the preregistered quality gate failed.
The consolidated success/failure and recovery analysis is in
`MAINTAINER_RETROSPECTIVE.md`; the state-reduction failure ledger contains
16 classified rejected attempts and its knowledge-proposal ledger contains
11 scoped proposals.
The completed `DRRQR论文穿刺续跑` heartbeat was deleted; it will not restart
or re-poll this experiment.

The operator-selected HYPIC historical reproduction is **complete**:

- experiment: Qwen3.6-27B, TP4, four Ascend 910B4 cards;
- resource lane used: host NPU4-7, CPU0-95, port 8224;
- baseline: native prefix cache off, HYPIC off;
- treatment: native prefix cache off, HYPIC on;
- only arm switch: `VLLM_ASCEND_HYPIC_ENABLE`;
- accepted units: **18/18**;
- independent evidence audit: **passed, zero errors**;
- decision: adopt for this frozen workload with quality guardrails;
- integration state: ready for maintainer review.

NPU0-3 was a protected external lane throughout the formal run. Do not use
this handoff as authority to inspect or alter it.

Controller operation `exec-00000000000009ab` exited 0 at
2026-09-06T17:31:34Z. Both exact experiment containers are exited,
`OOMKilled=false`, and port 8224 has no listener.

## Frozen ruler

| Variable | Value |
|---|---|
| model | `/cache/efim/Qwen3.6-27B` |
| image | `sha256:660ce23a83574fdcd28ba126b07615829c06770c06fbdf6533f03cc4d90acdb1` |
| HYPIC | 0.8.5, commit `3ad0967589304cd8fc4e7cff02f48ecc6125c89f` |
| topology | TP4, host NPU4-7, container-visible 0,1,2,3 |
| budget | `max_model_len=40960`, `max_num_batched_tokens=40960` |
| performance | 32768 input, 16384 shared, 1024 fixed output, 40 requests, concurrency 16, two warmups, three runs per arm |
| quality | LongBench-v2 127 and GSM8K 1319; concurrency 32, max output 1024, seed 0, temperature 0, thinking off, three runs per arm |
| excluded | CEval by operator decision |
| unavailable | authoritative NIAH+ files; no substitute |

The employee reference is contextual evidence, not pooled evidence:
Qwen3.5-27B, TP2, 2× Ascend 910C, HYPIC 0.8.5.

## Final results

| Metric | Baseline mean | HYPIC mean | Change | Employee change |
|---|---:|---:|---:|---:|
| TTFT ms | 40357.590939 | 22747.767959 | -43.6345% | -46.6% |
| TPOT ms | 74.507463 | 60.357689 | -18.9911% | -18.4% |
| output tok/s | 123.765347 | 168.402919 | +36.0663% | +38.5% |
| duration s | 331.419626 | 243.252621 | -26.6028% | -27.8% |

The performance direction and approximate magnitude reproduced. Absolute
latencies are not expected to match because model version, hardware generation
and tensor parallelism differ.

| Dataset | Baseline mean | HYPIC mean | Delta | Baseline three-run range |
|---|---:|---:|---:|---:|
| LongBench-v2 | 46.194226% | 45.931759% | -0.262467 pp | 5.511811 pp |
| GSM8K | 96.158706% | 96.007076% | -0.151630 pp | 0.379075 pp |

Both quality deltas are inside measured baseline repeat ranges. This supports
"no material regression observed" for the frozen ruler; it does not prove
universal losslessness.

## Evidence

- output root:
  `/cache/cch/hypic-qwen36-formal-npu4-7-tp4-budget40960-20260906`
- aggregate:
  `historical-reproduction-summary.json`,
  SHA256 `741a5ad225682452062143f776af81949961d11ecf946b4da64ade3a118f935e`
- report:
  `HISTORICAL_REPRODUCTION_REPORT.md`,
  SHA256 `f30ac5decb07145f78ee41a038874aaf3c059423f16ed73df0ea617fcd5fcb79`
- independent audit:
  `historical-reproduction-evidence-audit.json`, `ok=true`,
  `accepted_units=18`, `errors=[]`
- treatment activation:
  `hypic-only/performance.run1.json` has
  `hot_path_verified=true`, `hot_path_missing=[]` and
  `evidence_failures=[]`
- experiment, evidence and decision ledgers:
  `experiment-hypic-formal-npu4-7-tp4-b40960.json`,
  `evidence-hypic-formal-npu4-7-tp4-b40960.json`,
  `decision-hypic-formal-npu4-7-tp4-b40960.json`

Formal no-cache/HYPIC profiling was not repeated because the employee ruler was
a serving benchmark plus accuracy comparison. HYPIC activation is proven by
strict runtime event evidence. Existing native msprof captures belong to the
separate PDC-coexistence experiment and must not be pooled.

## Successes worth reusing

1. Freeze model, image, dataset hashes, prompt/scorer, topology, traffic shape
   and budget; change one algorithm switch only.
2. Treat host-to-container device remapping as an explicit invariant.
3. Budget the private suffix plus seam sink: 16384 + 8 required at least
   16392 physical tokens; changing the budget required a new artifact root.
4. Require hot-path proof in addition to favorable latency.
5. Run three repetitions and compare accuracy deltas with measured baseline
   variability rather than an invented generic tolerance.
6. Persist per-sample JSONL and resume only indexes without an accepted
   `ok=true` row.
7. Stop/restart the identical service between performance and quality phases.
8. Accept completion only after an independent hash/count/index audit.

## Failures and recoveries

- A preflight attempt used host IDs 4-7 inside a remapped container. Measurement
  never began; the failure was archived. Use container-visible 0-3.
- The earlier 16384 token budget could not admit a 16384 suffix plus an
  8-token seam. It was rejected and never pooled with the corrected case.
- Baseline GSM8K run2 lost service after 923 successes. The resumable evaluator
  preserved valid rows and retried only missing indexes.
- An incomplete-report renderer indexed an absent GSM8K aggregate. The renderer
  was changed to render only present datasets and self-tested before resume.
- The first resumed treatment service exited cleanly before measurement. The
  controller restarted the same exact configuration; no accepted unit was
  discarded and no algorithm failure was inferred.

## Exact verification command

```bash
cd /cache/cch/.worktrees/WS-20260904-qwen38-dense-paper-to-ascend
python3 .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/audit_hypic_historical_reproduction.py \
  --root /cache/cch/hypic-qwen36-formal-npu4-7-tp4-budget40960-20260906 \
  --output /cache/cch/hypic-qwen36-formal-npu4-7-tp4-budget40960-20260906/historical-reproduction-evidence-audit.json
```

Expected: exit 0, `ok=true`, `accepted_units=18`, `errors=[]`.

## Exact recovery/resume command

This command is idempotent against validated artifacts: it verifies and skips
accepted units. Run its self-test first and never start it while another
authoritative controller is live.

```bash
cd /cache/cch/.worktrees/WS-20260904-qwen38-dense-paper-to-ascend
export HYPIC_OUTPUT_ROOT=/cache/cch/hypic-qwen36-formal-npu4-7-tp4-budget40960-20260906
export HYPIC_MODEL_NAME=qwen36-formal-npu4-7-tp4-b40960
export HYPIC_PORT=8224
export HYPIC_CPUSET=0-95
export HYPIC_BASELINE_CONTAINER=qwen36_hypic_nocache_baseline_npu4_7_tp4_b40960
export HYPIC_TREATMENT_CONTAINER=qwen36_hypic_only_npu4_7_tp4_b40960
export HYPIC_EXPERIMENT_ID=hypic-qwen36-formal-npu4-7-tp4-budget40960-20260906
export HYPIC_HOST_NPU_FIRST=4
export HYPIC_HOST_NPU_LAST=7
export HYPIC_VISIBLE_DEVICES=0,1,2,3
export HYPIC_CONTAINER_DEVICE_FIRST=0
export HYPIC_TP_SIZE=4
export HYPIC_HCCL_PORT_RANGE=26000-26100
export HYPIC_MAX_NUM_BATCHED_TOKENS=40960
export HYPIC_REFERENCE_SHA256=204b01d76eff8a0fd69a4a048de44da8385c6bfdc250c1efb8af542b63fb438c
bash .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/resume_hypic_historical_reproduction.sh --self-test
bash .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/resume_hypic_historical_reproduction.sh
```

## Delivery

Worktree branch:
`work/WS-20260904-qwen38-dense-paper-to-ascend`.

Intended review destination is
`https://github.com/c1917797-bit/model-tuning-lab`, but this workstream did
not push and did not fabricate or alter Git identity.

## Completed state-reduction paper piercing

This is a separate experiment from HYPIC and must not be pooled with it.

- paper: *The Key to State Reduction in Linear Attention: A Rank-based Perspective*,
  arXiv 2602.04852v2;
- official code: `camail-official/LinearAttentionPruning` at
  `919d8667d951c385e08510bc1267c2e7049a4f56`;
- model: unmodified `/cache/efim/Qwen3.6-27B` baseline versus independent
  `/cache/cch/Qwen3.6-27B-gdn-drrqr-dk64` and
  `/cache/cch/Qwen3.6-27B-gdn-drrqr-dk96` treatments;
- resource lane used: host NPU4-7, container-visible 0-3, TP4, CPU0-95,
  port 8225; now released;
- calibration caveat: delivered LongBench-v2 first 16 source-ordered rows,
  frozen to 2048 tokens, because FineWeb-Edu is absent. This is an Ascend
  adaptation, not paper-exact calibration;
- completed: 3072-file capture audit, official Strong-RRQR Dk64 checkpoint,
  real-weight Dk64 Ascend smoke, baseline performance 3/3, baseline quality
  LongBench-v2 3/3 plus GSM8K 3/3, Dk64 treatment performance 3/3 and
  Dk64 treatment quality 6/6; quality operation
  `exec-0000000000000b5a` exited 0;
- Dk64 quality decision: LongBench-v2 passed, but GSM8K mean was
  96.108163%, -0.202173 pp versus baseline. The loss exceeded the baseline
  three-run range of 0.151630 pp, so Dk64 failed the preregistered quality
  gate and is not adoptable;
- Dk96 fallback checkpoint build `exec-0000000000000be9` exited 0. Manifest
  SHA256 is
  `a8f8f162f15fb6e4717ed509927d93c549a2ef01eca4c88e45e75cdf900b16fc`;
  independent audit confirmed 48 layers, 96 target shapes, six rewritten
  shards, nine unchanged hardlinked shards and the sole config delta
  `linear_key_head_dim: 128 -> 96`;
- Dk96 real-weight smoke `exec-0000000000000bf0` exited 0, returned exactly
  `ASCEND_DRRQR_OK`, and logged the GDN path with `head_k_dim=96`; container
  exited without OOM and NPU4-7 were released;
- Dk96 formal performance operation `exec-0000000000000bf2` exited 0.
  Every repetition accepted 40/40 requests with exact 1024-token output.
  Mean changes versus baseline are TTFT -0.401006%, TPOT -0.442686%,
  output throughput +0.380717%, and duration -0.327093%. All raw three-run
  intervals overlap the baseline intervals, so no repeat-separated serving
  benefit is claimed;
- launcher attempt `exec-0000000000000c15` exited 2 after 20 ms because
  its path contained a nonexistent `scripts/` component. No measurement,
  container or NPU workload started; this is logged as an orchestration failure;
- complete: corrected Dk96 formal repeated-quality operation
  `exec-0000000000000c1b` exited 0 after 7,988,385 ms;
- accepted Dk96 LongBench-v2 run1: 60/127 = 47.244094%, 127/127
  successful, zero failures, exact unique indexes 0-126; summary SHA256
  `005f533016d8c63e68e77f664036f1d50cd76fd643f9d4c328e3c2d42a58b728`,
  details SHA256
  `63f717d4bec3a3fbca2af77b7b5d5d8b51a0007fbdc86541e1d55e6be7c77d66`.
  The recorded and independently computed details hashes agree. The controller
  then advanced to LongBench-v2 run2;
- accepted Dk96 LongBench-v2 run2: 66/127 = 51.968504%, 127/127
  successful, zero failures, exact unique indexes 0-126; summary SHA256
  `826c51296dd7f31bb8f68448a7ae65f79a3acda3eb3a36c6a3749ff68aaeb3a9`,
  details SHA256
  `8f83c0477ed486a115bc9d4a90b8edad4085b2e109bf76d1704b23b28316e71d`.
  The recorded and independently computed details hashes agree. The controller
  then advanced to LongBench-v2 run3;
- accepted Dk96 LongBench-v2 run3: 62/127 = 48.818898%, 127/127
  successful, zero failures, exact unique indexes 0-126; summary SHA256
  `acbc12b9f7aa71a7436b2c3ae882c0110d5934a60d4839df0c2af6507355e256`,
  details SHA256
  `59773f08263dd4366e1ccd5a12b0ac32bf5bed0b19ebabbb01af3da1fcb63ffc`.
  Across the three Dk96 runs, mean accuracy is 49.343832%, sample SD is
  2.405552 pp and the delta versus baseline is +3.674541 pp, so
  LongBench-v2 passes the preregistered gate. The controller then advanced to
  GSM8K run1;
- accepted Dk96 GSM8K run1: 1265/1319 = 95.905989%, 1319/1319
  successful, zero failures, exact unique indexes 0-1318; summary SHA256
  `b11d73a90aa7af15752f99b7fa7e32deee75ed37874353dcdae8553e9633dd12`,
  details SHA256
  `6bdc392670d17a05a196356c4251d2857fc0951c1365ae3edf762a93a5205ac4`.
  The recorded and independently computed details hashes agree. This single
  run is not a gate decision; the controller advanced to GSM8K run2;
- accepted Dk96 GSM8K run2: 1268/1319 = 96.133434%, 1319/1319
  successful, zero failures, exact unique indexes 0-1318; summary SHA256
  `19b6d7ae20f2bdb28315f5ae932bac842877486b0893a23972c666a7027ba751`,
  details SHA256
  `6fffa04b79357e1cc9d179550e44b849732a3c1bcbb0cf03e751e5c10a3e08f9`.
  The recorded and independently computed details hashes agree. Run3 requires
  at least 1272 correct for the final three-run mean to pass the frozen GSM8K
  gate; the controller advanced to run3;
- accepted Dk96 GSM8K run3: 1266/1319 = 95.981804%, 1319/1319
  successful, zero failures, exact unique indexes 0-1318; summary SHA256
  `c836f122866709835377dfbb5e2e60964adac6dda8baa28c3ccd80ff9a37231d`,
  details SHA256
  `5e11e9cbffe39d3262fc6dd88df5fd5a622404f9670bc4a70ec71776cd59df29`;
- final Dk96 GSM8K mean is 96.007076%, -0.303260 pp versus baseline.
  This exceeds the baseline three-run range 0.151630 pp, so Dk96 fails the
  frozen quality gate and is rejected. Profiling was intentionally not
  started under the preregistered rule;
- final summary completed with errors=0. Final evidence audit verified 9
  performance runs, 18 quality summaries/details, Dk96 container exited 0
  without OOM, NPU4-7 empty and port8225 free. NPU0-3 were not inspected.
  Report: `/cache/cch/state-reduction-qwen36-formal-npu4-7-tp4-20260906/STATE_REDUCTION_PIERCING_REPORT.md`.
  Audit: `/cache/cch/state-reduction-qwen36-formal-npu4-7-tp4-20260906/final-evidence-audit.json`;
- accepted treatment quality: LongBench-v2 run1 is 63/127 =
  49.606299%, summary SHA256
  `4396f56d515d2da5054b849c82757221d769ac183ec44d16df51d9527289046d`,
  details SHA256
  `5250567dfb8c4e5467e79e5d5f599bc4161e499432dabfda7f3d32e83f6079c5`;
  run2 is 59/127 = 46.456693%, summary SHA256
  `11aa6e11f133adef83234807b69fa743a202db0606bb9fee0fdcdd982ebfb5c8`,
  details SHA256
  `9cd5e4c2b93003a9f44cc9d9f0ea5dcd9943147f5a1356f990a5b3ec7638ec2e`.
  run3 is 62/127 = 48.818898%, summary SHA256
  `3821194a84c9c3eb4830a810aff349919fcef82b02db565b1161b18d7991b6af`,
  details SHA256
  `0ae2e181bdb0e0910b66fc62736532f543603715c2e16ad9698f3360e935b59a`.
  All three runs are 127/127 successful, zero-failure, exact-index audited.
  Their mean is 48.293963% (sample SD 1.639107 pp), +2.624672 pp
  versus the baseline mean, so LongBench-v2 passes the preregistered gate;
- accepted treatment GSM8K run1: 1269/1319 = 96.209249%, 1319/1319
  successful, zero failures, exact indexes 0-1318; summary SHA256
  `fab6abc4d9d53507831197951135a44854bb1e8eb3e7de9823adffee639e906d`,
  details SHA256
  `87045034f4144e8f25d707783c90bca3268560915f491c04bac896f1edd0a035`;
- accepted treatment GSM8K run2: 1267/1319 = 96.057619%, 1319/1319
  successful, zero failures, exact indexes 0-1318; summary SHA256
  `5929311e641cbbf32361b425b31fd137e4f9e759da1a08111ef9587d84d4c56d`,
  details SHA256
  `e72b72c3d623ae1236095a1f7256fd9bdcc11697d4366059bc8abb2c4237f274`;
- accepted treatment GSM8K run3: 1267/1319 = 96.057619%, 1319/1319
  successful, zero failures, exact indexes 0-1318; summary SHA256
  `bd62db3b496746471c76bf3d16bf55ccb3a5e076edc028e31a1bd66d7e71a508`,
  details SHA256
  `98925497b09f93df524a28058475e7b8a6d03a762ec12c91f1dfbf023b114314`;
- forbidden: any benefit, lossless, adoption or full-reproduction claim before
  treatment performance, repeated quality, native profiling and independent
  final audit.

Baseline performance raw run means:

| Run | TTFT ms | TPOT ms | output tok/s | measured duration s |
|---:|---:|---:|---:|---:|
| 1 | 40556.828397 | 75.804468 | 120.010388 | 341.303788 |
| 2 | 37443.801750 | 72.544506 | 128.687989 | 318.289223 |
| 3 | 37370.040124 | 72.537494 | 128.759025 | 318.113624 |
| mean | 38456.890090 | 73.628823 | 125.819134 | 325.902211 |

Dk64 treatment performance raw run means:

| Run | TTFT ms | TPOT ms | output tok/s | measured duration s |
|---:|---:|---:|---:|---:|
| 1 | 39843.659653 | 74.246411 | 122.245494 | 335.063475 |
| 2 | 36193.819163 | 70.412754 | 132.735702 | 308.583142 |
| 3 | 36198.085898 | 70.439270 | 132.710317 | 308.642169 |
| mean | 37411.854905 | 71.699479 | 129.230504 | 317.429595 |

Mean Dk64 changes versus baseline are TTFT -2.717420%, TPOT
-2.620365%, output throughput +2.711329%, and duration -2.599742%.
All directions are favorable, but the baseline and treatment three-run ranges
overlap. Therefore this is not yet evidence of a repeat-separated stable serving
benefit. Every performance file passed 40/40 success and exact 1024-token output
validation; Dk64 run3 SHA256 is
`17bc2e3f569bb07c524a329d22cb0540771cbb75b57b5301f54213961e1ac1d8`.

Dk96 treatment performance raw run means:

| Run | TTFT ms | TPOT ms | output tok/s | measured duration s | artifact SHA256 |
|---:|---:|---:|---:|---:|---|
| 1 | 40837.282183 | 76.028506 | 119.223907 | 343.555258 | `32a7d0136b98ac66056dd897c3516e14a8cc5af7a3f87ae5c649d0650f40e694` |
| 2 | 37025.522791 | 71.945349 | 129.837447 | 315.471391 | `91bcc4632fb251f95fe8ae422ae00b20ac8a3df76cc51b27803be4ae223b3190` |
| 3 | 37045.221750 | 71.934779 | 129.833092 | 315.481972 | `9e95f1c810a44e00bd33132cddfa2b5046d5e57906708dc3c8eecf2d848fdfc6` |
| mean | 38302.675575 | 73.302878 | 126.298149 | 324.836207 | — |

The Dk96 performance container exited 0 without OOM. Host NPU4-7 were
independently observed with no running process before quality started. NPU0-3
retained the external Qwen3.5 workload and were not altered.

Baseline repeated quality evidence:

| Dataset | runs (correct/total) | mean | sample SD | range |
|---|---|---:|---:|---:|
| LongBench-v2 | 56/127, 61/127, 57/127 | 45.669291% | 2.083269 pp | 3.937008 pp |
| GSM8K | 1269/1319, 1271/1319, 1271/1319 | 96.310336% | 0.087544 pp | 0.151630 pp |

These values define the predeclared quality guardrail. Dk64 and Dk96 both
completed all six quality repetitions. Dk64 GSM8K mean loss was 0.202173 pp
and Dk96 loss was 0.303260 pp; both exceed the baseline range 0.151630 pp.

Exact current-state check:

```bash
cat /cache/cch/state-reduction-qwen36-formal-npu4-7-tp4-20260906/controller-status.json
```

Operation `exec-0000000000000c1b` is terminal and successful. No experiment
resume is required. To regenerate and independently re-audit derived artifacts:

```bash
python3 /cache/cch/.worktrees/WS-20260904-qwen38-dense-paper-to-ascend/.agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/summarize_state_reduction_formal.py --root /cache/cch/state-reduction-qwen36-formal-npu4-7-tp4-20260906
```

Do not run the native-profile runner for this completed identity: the frozen
protocol intentionally skipped profiling after the Dk96 quality failure.
LoRA recovery, a new calibration corpus or another pruning ratio must receive
a new experiment identity and fresh same-denominator baseline.

## Active Qwen3.8 DRRQR monkeypatch piercing

The Qwen3.8 plugin-free TP4 baseline and complete source-checkpoint identity are
finished and audited. The baseline container
`qwen38_drrqr_baseline_exact_host4_7_tp4_20260907_v6` is preserved in exited
state; host NPU4-7 and port8227 are released. NPU0-3 were not inspected.

Calibration is also frozen without using an NPU workload:

- delivered full LongBench-v2 source:
  `/cache/cch/hhs-aisbench/ais_bench/datasets/LongBench-v2/data_full.json`,
  503 unique rows, SHA256
  `f3833b46bf4ccca48fc2f81a0efdcf9cd362ac2ec426c66068b1363e094250a1`;
- frozen quality set:
  `/cache/cch/hhs-aisbench/ais_bench/datasets/LongBench-v2/data.json`,
  127 unique rows, SHA256
  `cdaa9e98deaefbba86f5eadeb287dd2ae90ed8f33313587dc547218b770fa39a`;
- the 127 quality IDs are all present in the 503-row source, leaving 376
  non-evaluation rows;
- the calibration artifact contains the first 16 source-ordered non-evaluation
  rows with at least 2048 Qwen3.8 tokens, each truncated to exactly 2048:
  `/cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907/calibration/longbenchv2-non-eval-16x2048-qwen38-inputids.jsonl`;
- calibration SHA256:
  `81748f94c4eff0ad5e4053eb7c4a30b1a397a82d45676a8b0b42394bd612bd47`;
- independent audit:
  `/cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907/calibration/calibration.audit.json`,
  SHA256
  `67bb45bca7cb1ff0cc216af49746f53748e4f8c66bbe58a2ad667fd0362bc905`,
  16/16 rows, 2048/2048 tokens, 16 unique source IDs, zero overlap with
  the frozen quality set and no audit errors.

Operation `exec-0000000000000d30` is retained as failure evidence: a
CPU-only, no-device container auto-loaded `torch_npu` and failed because
`libascend_hal.so` was intentionally unavailable. The same no-device,
network-disabled preparation succeeded as
`exec-0000000000000d31` after setting
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`; independent audit operation is
`exec-0000000000000d32`.

This fixes a weakness in the older Qwen3.6 adaptation, which calibrated on the
first 16 rows of the same 127-row quality set. The new artifact is still not a
paper-exact reproduction because FineWeb-Edu is absent; it is a documented
Ascend adaptation using the delivered colleague dataset.

Capture and plan generation are complete. Operation
`exec-0000000000000e64` produced 3072/3072 tensors from 16/16 disjoint
LongBench-v2 calibration requests. Independent audit
`exec-0000000000000e82` returned `ok=true`, `errors=[]`, verified all 48
linear-attention layers and preserved all 16 full-attention layers. Frozen plan
hashes are:

- Dk102 / 20.3125%:
  `2ad98436eb6889cfd902c5ce8245a0ee6413e9049672614137ce8cc7914dc0c3`;
- Dk89 / 30.46875%:
  `046cd580e99bcd4376a0229eebfd97515e14e4bd4ed8aeabeeeb4206d80d5e35`;
- Dk64 / 50%:
  `3824e60fb9137fa1a1b134bff4594495014625583fdd62fdf5b9bc92f39b8bea`.

The Qwen3.8 plugin-free baseline is complete:

| Metric | three-run mean |
|---|---:|
| output throughput | 120.660716 tok/s |
| TTFT | 40029.120 ms |
| TPOT | 77.498847 ms |
| LongBench-v2 | 45.406824% |

All three baseline performance runs accepted 40/40 requests with exact
1024-token outputs. All three LongBench-v2 runs accepted 127/127 requests with
zero failures; individual accuracies were 44.881890%, 46.456693% and
44.881890%.

### Treatment startup failures retained

These are compatibility/orchestration failures before any benchmark, not
negative algorithm results:

1. `exec-0000000000000e83`: vLLM-Ascend 0.23 required exact hybrid-cache
   page alignment after Dk128->102. The failed service and release audit are in
   `prune20/performance-failed-start-v1`.
2. `exec-0000000000000ee9`: the first bounded alignment patch passed its
   calculation but used classmethod source introspection incorrectly. Evidence
   is in `prune20/performance-failed-start-v2`.
3. `exec-0000000000000f00`: plugin 0.1.1 proved installation, four-worker
   dispatch, Dk102 model configuration and a live ceil-padded cache layout, but
   asserted whole-checkpoint coverage inside an inner Qwen loader that
   AutoWeightsLoader invokes repeatedly. Evidence is in
   `prune20/performance-failed-start-v3`.
4. `exec-0000000000000f2e`: plugin 0.1.2 completed 96/96 target
   transformations and loaded the model on every TP worker, then the frozen
   v0.23 AscendC recurrent decode operator failed during graph capture with an
   AI-vector-core MTE invalid-parameter error. Its float32 state-row write is
   408 bytes at Dk102, which is not 32-byte aligned. Evidence is preserved in
   `prune20/performance-failed-start-v4`; the container exited 1 and was not
   OOM-killed.

The live cache evidence in attempt 3 is internally consistent with the actual
float32 SSM cache dtype: `block_size=1280`,
`attn_page_size=1310720`, `ssm_page_size=626688`,
`conv_page_size=14112` and `mamba_page_size_padded=1324832`.

### Current exact plugin and operation

Plugin repository: `/cache/cch/drror_vllm_ascend`.

Local main commit:
`84836c15837acded55e38429b01629ad5fecfa6f` (plugin 0.1.2). It moves the
Q/K plus convolution-row transform to the top-level
`Qwen3_5ForConditionalGeneration.load_weights`, which is the checkpoint-
declared runtime ABI of the hash-pinned Qwen3.8 asset. It retains a strict
96-target completeness assertion over the single full checkpoint stream.

Wheel:
`/cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907/plugin-artifact-toploader-v3/drror_vllm_ascend_plugin-0.1.2-py3-none-any.whl`.

Wheel SHA256:
`3aa1f11d7f54be2f0d1b5ca136878bec9d0700822d5a237e088cc178d8ea2fd8`.

The source checkout and the installed wheel each passed 28/28 tests in the
exact v0.23 image; `git diff --check` passed and the wheel build completed.
The GitHub push is operation `exec-0000000000000f22`. Do not claim remote
main contains 0.1.2 until that operation succeeds and remote HEAD is verified;
the previous verified remote main was
`9822d4c746c4bceed63e4e2122dd04b91f3801ed`.

Active Dk64/50% performance operation:
`exec-0000000000000f48`.

Completed container:
`qwen38_drrqr_prune50_performance_host4_7_tp4_20260907_toploader_v4`.

Dk64/50% performance operation `exec-0000000000000f48` completed with exit
0 and released host NPU4-7. All four workers reported the complete 96-target
weight transform, reduced prefill and reduced decode branches; activation
audit has `ok=true`, `errors=[]`. Each of the three benchmark repetitions
completed 40/40 requests and exactly 40960 requested output tokens. Mean
throughput is 130.67781536728035 tok/s (sample SD 3.4598077556703584), versus
baseline 120.66071588314415 tok/s: +8.301873075108745%. Mean TTFT is
37416.28059982322 ms (-6.527346008343782%); mean TPOT is
71.18047284457467 ms (-8.152861783716503%). These are performance facts only;
Dk64 quality and profiling are still pending.

Plugin 0.1.3 commit
`0e0104826ae787e168de5da29ec8507b2dd859ab` is now published and independently
verified at remote main. Source and independently installed-wheel tests both
pass 29/29 in exact image
`sha256:660ce23a83574fdcd28ba126b07615829c06770c06fbdf6533f03cc4d90acdb1`.
The immutable wheel is:
`/cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907/plugin-artifact-aligned-v4/drror_vllm_ascend_plugin-0.1.3-py3-none-any.whl`,
SHA256
`7e98f9745041e539d4c80f33c827583bbe4043a564ae26eedbc476bf527b581a`.

Aligned plan generation `exec-0000000000000f99` completed. Dk104/18.75%
is SHA `5bc3eabe4f98df880b28a287e28fe3c0fc9a8036b120c85f211d3cf41f7f9830`;
Dk88/31.25% is SHA
`c75468decf7752388f3e6beb57b207bb6d426aa69b8aaf588431ceb22f7d6daf`.
Both cover all 48 linear-attention layers and report method DRRQR. Current
operation `exec-0000000000000fb0` is Dk104 performance and exclusively owns
host NPU4-7. NPU0-3 must not be inspected or changed.

### Closed-state verification and recovery

All Qwen3.8 operations are terminal. Do not poll or restart the historical
operation IDs. Verify the closed evidence without starting an NPU workload:

```bash
cd /cache/cch/.worktrees/WS-20260904-qwen38-dense-paper-to-ascend
python3 .agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/audit_qwen38_drrqr_final.py
cat /cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907/final-evidence-audit.json
```

The verified plugin pull recovery is:

```bash
cd /cache/cch/drror_vllm_ascend
git fetch --prune origin
test "$(git rev-parse origin/main)" = "d26dd8cde240fb578442f108620e7bb33ef703c4"
git switch main
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = "d26dd8cde240fb578442f108620e7bb33ef703c4"
git status --short
```

Dk102 is not runnable through the frozen v0.23 decode operator because its
float32 state row violates the operator's 32-byte write alignment. Dk89 has the
same issue. Dk104 and Dk88 are hardware-aligned neighboring experiments with
exact reductions of 18.75% and 31.25%; never label them as exact 20%/30%.
LongBench-v2 is measured next for runtime-valid, meaningful candidates; GSM8K
follows a successful piercing. CEval remains excluded. Native profiling is
restricted to the final candidate that passes both performance and quality
gates.

## Qwen3.8 DRRQR three-arm decision (2026-09-08)

The three runtime-valid, hardware-aligned arms are complete. Every arm used the
same BF16 Qwen3.8-27B checkpoint, TP4 serving contract, host NPU4-7 lane,
plugin wheel and frozen LongBench-v2 denominator. All four workers proved the
real reduced prefill and recurrent decode hot paths before measurement.

| Arm | actual reduction | throughput delta | LongBench runs | mean | delta vs baseline |
|---|---:|---:|---|---:|---:|
| Dk104 | 18.75% | -2.746264% | 20.472441%, 21.259843%, 18.110236% | 19.947507% | -25.459318 pp |
| Dk88 | 31.25% | -1.943667% | 10.236220%, 16.535433%, 16.535433% | 14.435696% | -30.971129 pp |
| Dk64 | 50.00% | +8.301873% | 35.433071%, 33.858268%, 38.582677% | 35.958005% | -9.448819 pp |

Baseline LongBench-v2 mean is 45.406824%. All runs completed 127/127 requests
with zero failures. Dk104 and Dk88 lose both performance and quality. Dk64
improves throughput but has a material, repeated quality loss. Therefore no
candidate passes both gates and the adoption decision is **reject**.

The plans are nested for all 48 linear-attention layers, so nesting alone does
not explain the non-monotonic quality. Cross-run correct-set Jaccard is
0.2250-0.2619 for Dk104 and 0.0968-0.1724 for Dk88, versus
0.7692-0.8333 for Dk64 and 0.8413-0.9016 for baseline. This confirms
shape-dependent instability but does not establish its cause. GSM8K and native
profiling were not started because the preregistered protocol reserves them
for a candidate that passes the LongBench gate.

Exact evidence record:

```bash
cat /cache/cch/.worktrees/WS-20260904-qwen38-dense-paper-to-ascend/.agents/workstreams/WS-20260904-qwen38-dense-paper-to-ascend/evidence-qwen38-drrqr-three-arm-quality-20260908.json
```

All treatment operations are terminal and resource-release audits passed.
There is no command to resume this experiment. To investigate the quality
anomaly, create a new experiment identity, preserve the current artifacts,
freeze a fresh same-denominator baseline, and add direct numerical-parity
checks across Dk128/Dk104/Dk88/Dk64 before any serving benchmark.
