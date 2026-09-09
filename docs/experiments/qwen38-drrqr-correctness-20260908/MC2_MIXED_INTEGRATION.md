# Opt-in mixed-batch MC2 integration

Date:2026-09-08. Performance-first candidate. Final success is reproducible
end-to-end OUTPUT TOKEN THROUGHPUT>=10% above a contemporaneous matched unpruned
baseline, with no task-accuracy decline in the same candidate configuration.
TTFT/TPOT are separately reported latency metrics, not substitutes for throughput.

## Current verified checkpoint

- Source0.1.9 implements VLLM_ASCEND_DRRQR_PREFILL_MC2_MIXED(default0).
  It requires the existing PREFILL_MC2 flag; metadata must contain real prefill,
  no padding, and one row per decode sequence. Existing projection-specific row
  thresholds remain. Pure decode/profile/capture/graph paths remain original.
- Runtime phase detection stays inside the opaque custom operator, so compiling
  dummy/profile calls cannot freeze the subsequent mixed-request route.
  Independent once-per-target mixed-call evidence establishes actual activation.
- Launcher explicitly accepts0.1.9 and --prefill-mc2-mixed, records the flag,
  and preserves source/installed/wheel/upstream provenance checks.
  Old coverage-worker instrumentation is restricted to its audited0.1.8 adapter.
- CPU operation0135:80 tests passed(internal1.991s). New cases cover default-off,
  malformed/mixed/pure-decode metadata, dynamic compiled phase transitions,
  evidence only after fused return, ContextVar reset, incompatible reinstall,
  and launcher version/diagnostic boundaries. No NPU correctness is inferred.
- Offline build0136 and installation0139 both exit0. The dedicated container
  now has0.1.9 installed. All18 Python package files agree across source,
  wheel and site-packages, with exact file coverage. Audited upstream hashes
  are unchanged. Prior0.1.8 build and all historical results remain preserved.

Wheel:
/drrqr-results/prefill-mc2-mixed-v1/build/wheels/drror_vllm_ascend_plugin-0.1.9-py3-none-any.whl
SHA256:049ac88312f262ec7a193b4ad77245583122ffa3a0d77d16ccc38699809e4fae.
Build manifest SHA256:
11d09b682a1b4f17669e7e097d4a79ffa2b550cdacbfca3656ad2ddb4dc3009e.
Installed adapter SHA256:
fbdab3b5857958cf0d531015ca0e394f282cfe0fe27b1713d59c4258fe800291.
Install receipt:
/drrqr-results/prefill-mc2-mixed-v1/install-receipt.json.

## Next experiments (not yet run)

1. Representative native mixed-model activation plus change-specific multi-step
   state/output checks. Standalone32/32 matrix checks are already complete,
   but do not establish state propagation. Freeze numerical criteria and scope
   before execution; preserve any failure rather than loosening criteria.
   Existing0.1.8 smoke hardcodes old version and pure-only first calls, so must
   be explicitly adapted or replaced before use. Do not run it unchanged.
   The frozen paired trajectory requires16 concurrent32K prompts and16 generated
   steps each. Control and candidate prompt hashes and token ids must match exactly;
   selected-token logprob absolute delta must be<=0.02 per step and<=0.005 mean
   over256 steps. These thresholds were fixed before either service was launched.
2. If those checks pass, one profiler-disabled mixed ON/control pair on0.1.9,
   with identical warmup, prompts, config and fixed1024 outputs. Keep diagnostic
   counters out of timings. Record all four workers' actual mixed activation.
3. If promising, repeat with a contemporaneous unpruned control. Also apply
   common optimizations to unpruned control to isolate DRRQR-specific benefit.
4. Recover pruning accuracy using calibration/selection/layer allocation before
   considering training. Final no-thinking baseline/candidate accuracy cap2048,
   normal EOS; LongBench and GSM8K. No LoRA currently.

## Scope and decision policy

The1-3day statement is a time-to-decision estimate, not a promise that a
positive result will be manufactured. The invariant final gate is one candidate
configuration with reproducible end-to-end output-token throughput>=10% over a
contemporaneous matched unpruned baseline and no LongBench/GSM8K decline.
TTFT/TPOT, microbenchmarks and historical noncontemporaneous baselines cannot
substitute for that gate.

Each experiment predeclares one treatment, evidence threshold and bounded run.
Negative artifacts are retained and thresholds are not changed after seeing a
failure. A single run only screens; formal performance uses at least three
matched rounds and reports mean, spread and worst round. Generic optimizations
must also be applied to the unpruned control before attribution to DRRQR.

Stop rules: reject this fusion candidate on failed change-specific numerical
evidence; do not repeat it automatically. Do not expand a sequence mismatch into
the deferred generic determinism project: either use a same-process shadow
comparison at the changed operator or stop the candidate. Do not promote a
single screen below roughly5% clear throughput gain to three-round validation.
If no candidate approaches10% within48hours, issue a bounded negative
performance conclusion with the collected bottleneck evidence instead of
unbounded parameter search. Accuracy repair starts only after a viable
performance candidate; no training/LoRA without a separate decision.

No0.1.9 native model service or benchmark was launched at this checkpoint.
Physical4-7 were free at0133; dedicated container had only its shell at0138.
Device mapping rechecked0137: host4/5/6/7 -> container0/1/2/3, source/checkpoints
read-only. Recheck occupancy before the next NPU job. No physical0-3 operation,
no commit, no push, no upstream edit, no determinism-worker experiment.

## Integration incidents

The first control probe0147 completed all16 requests but its post-run validator
failed because it required exactly four plugin_build_active PIDs. The event is
correctly emitted by the API server, engine core and four workers(six PIDs);
only the four prepared workers are relevant. Evidence014d confirms four
prefill_mc2_prepared records and six matching build records. The validator now
derives worker PIDs from preparation and requires those four to be a subset of
matching build records, consistent with the older audited smoke validator.
No model/NPU result failed. The failed receipt is preserved and will be
revalidated without repeating requests, with a separately hashed receipt.
The first revalidator invocation014f stopped before writing because prompt
regeneration used an invalid placeholder URL even though the frozen driver
tokenizes its seed text through the live API. It was corrected to the explicit
service URL; no request result or experiment threshold changed.

## Native integration evidence

Control service0146 and candidate service0156 use the same0.1.9 wheel and
verified source/upstream hashes; the only treatment is the mixed flag and
per-service evidence path. Control probe0147 completed all16x16 output steps;
corrected revalidation0150 confirms four prepared workers and zero mixed calls.
Candidate probe0157 completed exit0 with four prepared workers and512 mixed
calls: every one of128 target projections on all four TP workers executed the
new path.

Offline comparison015c passed all predeclared criteria across256 generated
steps: prompt hashes exact, token ids exact, selected-token logprob maximum
absolute delta0.014418145642<=0.02 and mean delta0.000619808347<=0.005.
This is representative end-to-end propagation through recurrent decode, not
full task accuracy or direct cache-tensor equality.

The mixed-disabled0.1.9 control single performance screen0151 completed40/40
fixed1024-token requests:134.907539999tok/s, TTFT mean36306.521ms, TPOT
mean68.986460ms, duration303.615350s. This is the paired control value, not a
claim about the treatment. Candidate screen015d was started only after015c
passed and remains the next deciding evidence.

Candidate screen015d completed40/40 fixed1024-token requests:
141.564496153tok/s, TTFT33468.964ms, TPOT66.689384ms,
duration289.338083s. Validated pair015e gives +4.934458% throughput,
7.815558% TTFT improvement and3.329750% TPOT improvement. This is a
promising single Dk64 treatment screen near the5% continuation boundary,
not a repeatability or final>=10% claim.

The contemporaneous unpruned Dk128 service uses the same0.1.9 wheel,
packed-conv and mixed-MC2 flags. Its sequence probe completed16/16 requests
with four prepared workers and512 mixed calls, so the common path is active on
both sides. Its single40/40 fixed1024-token screen completed at
135.142469917tok/s, TTFT34693.578ms, TPOT69.436227ms and duration303.087549s.
The strict cross-variant comparison is the next artifact: it must verify common
manifest fields, remove only the explicit Dk64 head-dimension override, require
the enable/plan treatment contract, bind both sequence receipts, and retain the
existing Dk64 numerical trajectory gate. Since the raw Dk64/dense ratio is only
about1.048, this does not approach the final>=10% throughput gate. Do not run
three-round validation yet; first profile one bounded Dk64/dense pair to locate
the remaining Dk64-specific critical path or issue a negative candidate
decision.

Strict comparison operation completed successfully. Dk64 versus the
common-optimized dense control is +4.752041% output throughput,
3.529799% TTFT improvement,3.955922% TPOT improvement and4.536467%
duration improvement. The report is
`prefill-mc2-mixed-v1/dk64-vs-dense-common-optimized.json`, SHA256
`d222aa3a39fa0933c8f2c72d0fba405aae6aa32fcc95e5823d3911092ffdffbd`.
This rejects promotion to three-round validation now. Reaching the10% dense
gate would require148.656717tok/s; from the current141.564496tok/s candidate,
the next Dk64-specific improvement must add about5.01%.

## Predeclared current-stack profiling pair

Hypothesis: after applying packed convolution and mixed MC2 equally, the
remaining Dk64 advantage and missing approximately5% are governed by a
Dk64-specific decode critical path, not the already-shared prefill fusion.
Capture exactly one diagnostic profile per variant with the current0.1.9
wheel, packed layout and both MC2 flags enabled. The frozen workload is one
32K prefill request plus16-way warm decode; record exactly30 stable decode
worker steps. This is diagnostic and is never mixed with profiler-disabled
throughput.

The validity gate is four ranks for each phase and variant, exact expected GDN,
attention and convolution call counts,512 mixed-call activation records per
service, identical prompts/runtime/wheel/common flags, and only the explicit
Dk64 enable/plan/head-dimension treatment. Budget is one capture per variant
and no blind repeat. A new implementation candidate is allowed only when the
profile exposes a mechanistically Dk64-specific operator/path large enough to
plausibly supply the missing5.01%; kernel sums alone are not accepted as
causality. Otherwise retain the profile and issue a bounded negative decision
for this branch.

Operational note: the first current-stack dense capture completed all requests,
but the initial summary attempt correctly failed because the service was stopped
before torch-NPU's asynchronous trace export created
`ASCEND_PROFILER_OUTPUT`. All eight raw trace directories are preserved.
The recovery is deterministic offline `torch_npu.profiler.analyse` export of
those exact client-recorded directories, followed by the unchanged coverage
gate; no model requests are repeated. Subsequent services are also exported
explicitly rather than depending on implicit background timing.

The explicit exports and summaries completed for both variants. The validated
pair reports3.612514% lower Dk64 prefill envelope and2.678076% lower decode
envelope. Report:
`prefill-mc2-mixed-v1/profile-current-v1/profile-comparison.v1.json`,
SHA256
`9cca1cd11a9eecd12c6e45988505d767c92ac6aa41fe3a857445a50d5b64115b`.
In Dk64 decode, MatMulV2 is637.008ms(47.16% of the envelope), full attention
361.408ms(26.75%), all-reduce122.618ms(9.08%), and recurrent GDN55.047ms
(4.07%). The recurrent kernel alone cannot supply the approximately5.01%
additional candidate throughput even under an impossible zero-cost bound.
Generic MatMul/attention/communication tuning must also benefit the dense
control and therefore cannot be credited to DRRQR without a new paired result.

## Predeclared Dk32 boundary screen

The next and final pruning-rate screen is uniform Dk32(75% key-state
reduction). It is a power-of-two multiple of16, so it avoids the known v0.23
non-aligned state copy-out defect. It uses the same frozen3072 captures,
disjoint16-row calibration artifact and joint normalized Q/K energy selector as
the Dk64 energy plan. This is a performance-bound experiment, not a claim that
quality will pass.

Before any endpoint benchmark, Dk32 must pass the existing native GDN golden
prefill/decode matrix alongside Dk128 as the harness-validity control, and a
Dk32 mixed-OFF/ON16x16 sequence comparison with
exact prompt/token identity and the already frozen logprob thresholds. Only
then run one profiler-disabled40x1024 screen with packed-conv and mixed MC2.
Promotion requires at least148.656717tok/s, i.e.10% above the current
common-optimized dense control. If Dk32 misses that threshold, stop the
pruning-ratio branch rather than sweep Dk48/Dk16. If it passes, start
no-training accuracy repair at the same Dk32 runtime configuration.

## Dk32 boundary result

The Dk32 energy-kernel plan was generated CPU-only with no NPU devices and no
network. It retains32 coordinates per head in all48 GDN layers and is bound to
the same3072 captures and16-row calibration set. Plan SHA256:
`47ca4d1eada4f31f1a9bf1e07b8f1ff050fec6451ae37579add9a9c004ca3e79`.

The first service attempt was rejected before model load because the packed
convolution integration still had the old audited dimension/channel allowlist.
The second reached real TP4 weight loading but was rejected during graph capture
because the unchanged GDN observer still had its old reduced-dimension allowlist.
Both failures are preserved. The plugin changes only added the already aligned
Dk32/1792-channel cases to those two fail-closed allowlists, with new tests and
distinct local versions. No upstream vLLM-Ascend or image file was modified.
The final local0.1.11 wheel SHA256 is
`8cee9fdd11d29f640ea86dcb19303249213b36a95933dac106cfa37a55ef6779`;
its build manifest SHA256 is
`bc0e697153591de05522d824fafc8a0ddf14a7130d8626c9ecab61543080f3fc`.
All81 tests pass and the install receipt verifies all18 source/wheel/installed
modules plus unchanged upstream hashes.

Native GDN golden parity passed68/68 Dk128/Dk32 cases across the frozen
prefill/decode matrix. Report SHA256:
`2526f4a416952a38a58c891757f569264d480d2b75257f94b025dbe5df4053e1`.
The independent mixed-OFF/ON model sequence gate also passed: all prompt hashes
and256 token ids are exact; selected-token logprob maximum delta is
0.0101455902<=0.02 and mean delta is0.0003976166<=0.005. Four workers prepared
both services, with zero mixed calls in control and512 in treatment. Comparison
SHA256:
`92ecf743ee63143a8ca61d40134a50728c90c072544c3857e018aac7d860bdd4`.

Only after those gates, the single profiler-disabled Dk32 screen completed40/40
fixed1024-token requests at137.991358649tok/s, TTFT33548.008ms,
TPOT68.638083ms and duration296.830181s. Raw report SHA256:
`e168541751cf91728d00b1af63d1402046a6e60896a2df865383a200c69bad66`.
This is7.729% below the predeclared148.656717tok/s boundary and2.524% slower
than the existing Dk64 candidate. Relative to the locked135.142470tok/s dense
reference it is only a contextual+2.108%, not a cross-version paired claim.

Decision: reject Dk32 and close the pruning-ratio sweep. Do not run Dk48/Dk16,
Dk32 task accuracy, three-round performance, or a new dense0.1.11 endpoint
screen. Any subsequent work must be a separately predeclared operator hypothesis
supported by the existing current-stack profile; generic gains must still be
paired against an equally optimized dense control.

## Dk64 operator feasibility closeout

The bounded follow-up audited the only concrete large-enough historical lead:
single-QKVZBA projection. A paired physical-NPU4 graph micro screen passed its
numerical gate but projected 14.329437 ms saving for dense and only13.703903 ms
for Dk64 over48 layers x30 decode steps. The Dk64 differential is-0.625534 ms,
versus the predeclared+64.446696 ms needed under the optimistic fixed-dense
profile-envelope bound. It is a generic Qwen optimization and cannot be enabled
only for the candidate in a credible DRRQR comparison.

The full audit and bound are in `OPERATOR_FEASIBILITY_BOUND.md`. This closes
the current operator/pruning branch with a bounded negative decision. The best
same-stack single-screen result remains Dk64 at+4.752041% throughput versus the
common-optimized dense control. No three-round or final accuracy run is
authorized for a below-bound candidate.

## Preregistered concurrency-scaling screen

This is a separate workload-sensitivity experiment, not a reopening of the
closed pruning-rate/operator sweep and not a replacement for the concurrency16
result. The service already fixes `--max-num-seqs 32`, while the frozen
performance screen used concurrency16. The hypothesis is that Dk64's halved
recurrent state reduces state/cache traffic enough for its relative throughput
benefit to increase when all32 admitted sequence slots are active.

Run exactly one contemporaneous dense/Dk64 pair at concurrency32. Both sides
must use the same locally built and hash-verified plugin snapshot, packed
convolution, mixed-prefill MC2, max model length40960, max batched tokens40960,
two warmups,40 measured requests,32768 input tokens,16384 prefix tokens and
fixed1024 output tokens with EOS ignored. Only the Dk64 plan, enable flag and
head-dimension override may differ. Re-run the native mixed sequence gate for
the new service manifests before timing, and retain all failures.

The screen is valid only with40/40 successful requests per side, exactly40960
output tokens per side, finite metrics, matching workload configs and verified
four-worker mixed-path activation. Advance to three matched rounds only if the
single-pair Dk64 throughput gain is at least8%; the final acceptance threshold
remains a reproducible mean gain of at least10% with no LongBench/GSM8K decline.
Below8%, reject concurrency scaling without trying other concurrency values.
If dense fails solely because the workload exceeds capacity, record a capacity
result but do not claim a like-for-like throughput win.

### Concurrency-scaling result

The bounded concurrency32 pair completed and rejects the hypothesis. Before the
NPU run, the repository snapshot passed83 CPU tests and was rebuilt offline as
a hash-pinned local0.1.11 wheel. The installed18 modules exactly matched the
source and wheel, and the audited upstream vLLM-Ascend hashes were unchanged.
Wheel SHA256:
`a358826b5bf1d3eb7e8a8d534b2958018566cccb026666782fcbb43fea296dae`.
Install-receipt SHA256:
`7499aef7bde6fddf0939554ba69d9468e80edbd000161bdb1fde34ef3277a5c2`.

The fresh Dk64 mixed-off/on trajectory gate passed all256 token steps with
exact prompts and token IDs. Selected-token logprob maximum absolute delta was
0.0150799602<=0.02 and mean delta was0.0004894121<=0.005. Four workers were
prepared; the control recorded zero mixed calls and the candidate recorded512.
Comparison SHA256:
`1c5e82cd6c08ba576e0c82d6c897be3833fc039d3b163d5cebfe3fe41db8010b`.

Both profiler-disabled performance sides completed40/40 requests and exactly
40960 fixed output tokens at concurrency32. Dense reached132.441158481tok/s,
TTFT80568.087ms and TPOT85.791833ms. Dk64 reached132.338146436tok/s,
TTFT80098.414ms and TPOT92.342349ms. The strict paired changes are therefore
-0.077779% throughput, +0.582952% TTFT improvement and -7.635361% TPOT
improvement. Report:
`prefill-mc2-mixed-v1/concurrency32-v1/dk64-vs-dense-c32.json`,
SHA256
`98f19276fea95e26220db961afb0757425ce7c45a1738cb031a4fea824313d9a`.

The engine reported maximum40960-token concurrency15.17x for dense and16.23x
for Dk64, so state reduction did increase the model's capacity estimate by
about6.99%. That capacity headroom did not become throughput: versus each
variant's concurrency16 screen, dense throughput fell1.999% while Dk64 fell
6.517%. Decision: reject concurrency scaling; do not run three rounds or sweep
other concurrency values. The best same-workload performance evidence remains
the concurrency16 Dk64 result at+4.752041% versus common-optimized dense.

## Preregistered Dk32-versus-Dk64 diagnostic profile

This is the final bounded no-training performance investigation. It does not
reopen the rejected pruning-rate or concurrency sweeps. The hypothesis is that
Dk32 is slower than Dk64 despite stronger state reduction because at least one
small-Dk-specific kernel or tiling path loses enough efficiency to erase the
theoretical memory saving.

Capture exactly one profile for Dk64 and one for Dk32 using the same installed
local0.1.11 snapshot, packed convolution, mixed prefill MC2, max model length
40960 and max batched tokens40960. The frozen diagnostic workload is one32K
prefill request plus16-way warm decode, with exactly30 profiled worker steps.
Dk64 is bound to plan SHA256
`b1fcb727d3e58be66cc24148e30a5a1722d3bdc0914784f01cad02f11f741b8c`;
Dk32 is bound to plan SHA256
`47ca4d1eada4f31f1a9bf1e07b8f1ff050fec6451ae37579add9a9c004ca3e79`.
Only the plan, plan path and head-dimension override may differ.

Each side must complete all requests, validate four prepared workers and512
mixed calls, export eight traces, pass rank/operator coverage, and match exact
source, wheel, upstream and workload provenance. The existing native numerical
gates remain mandatory and are not weakened by profiling.

The current Dk32 endpoint result is137.991359tok/s; reaching the locked
148.656717tok/s target needs7.729% more throughput, corresponding to an
optimistic7.174% reduction in candidate time if throughput scaled inversely.
An implementation candidate is permitted only if the paired profile identifies
a concrete Dk32-specific differential hotspot with at least that plausible
critical-path headroom. Generic MatMul, attention or communication work is not
candidate-specific and must also be applied to dense. Budget is one profile per
variant and no blind repeat. If no such hotspot exists, close the no-training
performance branch with a bounded negative conclusion; do not run another
endpoint screen, three-round validation or final task accuracy.

### Dk32-versus-Dk64 profile result and branch decision

Both captures passed the predeclared validity contract. They used the same
installed local0.1.11 snapshot, wheel, audited upstream files, packed layout,
mixed-MC2 path and frozen workload. Each variant completed all requests,
recorded four prepared workers and512 mixed calls, exported eight traces, and
passed the four-rank phase/operator coverage gate. The existing68-case native
golden and256-step sequence gates were not relaxed.

The Dk32 prefill envelope was3944.144095ms versus3995.970605ms for Dk64, a
1.296969% reduction. Its decode envelope was1339.413890ms versus1356.268225ms,
a1.242699% reduction. Both are far below the predeclared optimistic7.174% time
reduction required to bridge Dk32 from137.991359tok/s to148.656717tok/s.
Moreover, the endpoint screen already showed Dk32 to be2.524% slower than
Dk64, so the small profile-envelope improvement is not an endpoint benefit.

The largest Dk32 decode improvements were MatMulV2(-16.869980ms) and
Slice(-2.037980ms). The largest regressions were all-reduce(+1.766920ms),
full attention(+1.433630ms), recurrent GDN(+1.084250ms), and
ZerosLike(+0.310965ms), all expressed as candidate minus control aggregated
kernel time. These kernel sums overlap and are not additive critical-path
causality. No concrete Dk32-specific regression is remotely large enough to
supply the required7.174% endpoint time reduction.

Report:
`prefill-mc2-mixed-v1/dk32-vs-dk64-profile-v1/dk32-vs-dk64.profile-comparison.v2.json`,
SHA256
`1ca32e165b297a6b3a407d19757ab8c6552f9444a3b21c04b632f49613b06fc4`.

Decision: reject small-Dk operator rescue and close the current no-training
performance branch. Do not run another pruning ratio, concurrency value,
endpoint screen, three-round performance validation, or final task-accuracy
suite for this below-threshold candidate. The best credible same-stack
single-screen result remains Dk64 at+4.752041% output-token throughput,
3.529799% TTFT improvement and3.955922% TPOT improvement versus the
common-optimized dense control. It did not meet the>=10% throughput gate, and
because the performance gate was not met, no final LongBench/GSM8K
no-decline claim is made.

After the service processes were terminated, the dedicated container had no
matching vLLM/service process. Physical NPU4-7 each reported0% AICore,
0% AIVector and0% HBM-bandwidth utilization, with8% idle HBM occupancy.
Physical NPU0-3 were not queried or operated. No commit or push was made.
