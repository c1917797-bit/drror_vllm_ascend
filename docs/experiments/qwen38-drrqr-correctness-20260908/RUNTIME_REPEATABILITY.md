# Runtime repeatability investigation

Date: 2026-09-08. Status: unresolved; no new performance/quality acceptance claim.
This is a synthetic runtime diagnostic, not the2048-output-token accuracy evaluation.

## Frozen inputs and controls

All services use the same locally built0.1.7 wheel, frozen upstream source, TP4/bfloat16,
async scheduling and FULL_DECODE_ONLY graph mode. Only physical NPU4–7 are used.

Initial probe: two sequential32768-input-token prompts and16 concurrently submitted
2048-input-token prompts, two rounds,128 output tokens each, temperature0/seed0,
ignore_eos=true. Prompts are raw token inputs from the frozen performance driver;
they do not use or enable a thinking-mode chat-template evaluation.

Isolation probe: the same16 medium prompts, two rounds serial and two rounds submitted
as one API prompt-list,128 output tokens, top5 logprobs retained. A prompt-list API
request does not prove a fixed device batch schedule.

Artifact root:
`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/conv-layout-e2e-v1/`

## Completed observations

Counts below mean completely identical token-ID sequences across both rounds.
They are NOT accuracy scores.

| Probe | Dk64 original | Dk64 packed | Dk128 original |
| --- | ---: | ---: | ---: |
| Initial concurrent2048 input | 1/16 | 0/16 | 2/16 |
| Initial sequential32768 input | 2/2 | 2/2 | 1/2 |
| Isolation serial2048 input | 1/16 | 1/16 | 3/16 |
| Isolation prompt-list2048 input | 1/16 | 1/16 | 4/16 |

Dk64 long-input outputs also match across all four original/packed runs.
The provenance-checked comparison is output-comparison.v1.json, verdict
inconclusive_repeatability. Its validator rejects mismatched runtime/flags/commands,
missing or duplicate cases, incomplete tokens and empty coverage. Three unit tests pass.
Do not reinterpret this inconclusive verdict as a layout-parity pass.

Dk128 uses ENABLE=0, CAPTURE_ENABLE=0 and CONV_LAYOUT=0. No hf-overrides are present,
and the disabled plugin emitted no evidence file. Dk128 initial first differences range
from token0 to108 in medium prompts; one long prompt first differs at63.

Thus neither new weight packing nor DRRQR activation is necessary for the observed
repeatability failure. This does not establish that all variants have one identical root
cause, nor does it prove that packing/pruning introduce no additional error.

Some first-divergence top-token distributions move substantially. Neither ordinary
rounding nor concurrent-submission-only variation has been established as the explanation.
The existing short-sequence GDN parity matrix covers lengths63/64/65 and eight decode
steps per seed/batch; it is not a direct2048-input full-model repeatability test.

Dk64 isolation report hashes:
- Original: 013707bc1a0a05a8929e776b6e07f5ef4b56c989a574d819adccb7ffadcc91b6
- Packed: 2ef0e65094816ebd4fe4ed34654299918e15ce9e39bec440ca83586a6f6142ca

Both Dk64 services exited0 after idle-request verification and targeted graceful shutdown.
All failed initial reports are preserved; profiling did not proceed.

## Completed discriminator: one-token outputs

tools/collect_first_token_probe.py freezes five prompts before this probe is executed:
input2048-0, input2048-1, input2048-12, input2048-15 and input32768-0.
One warmup per case, then four serial rounds, max_tokens=1, temperature0/seed0,
ignore_eos=true, token IDs and top5 logprobs retained.

The response validator has three passing CPU tests for expected coverage, token type,
missing probabilities and nonfinite values. Collection completion is not a parity pass.

Baseline collection completed in operation008a. Across four repeats, input2048-12 returned
three distinct first-token IDs (271,1622,1835); input2048-15 returned two (271,1622).
The other three selected cases kept their first-token ID, but all five top5 distributions
varied. Report: baseline-original/first-token-probe.v1.json. This demonstrates variability
before a generated decode chain, not a locked root cause. The baseline service launcher
operation0080 subsequently completed with exit0 after targeted graceful shutdown.

Interpretation:
- Variation with one-token output directly demonstrates prefill-output variability without
  a generated decode chain in that request. Investigate prefill kernels, initial state,
  buffers and TP reduction; do not blame accumulated decode errors alone.
- Stable one-token outputs with unstable longer generation motivates controlled decode,
  graph/scheduling and state-reuse tests. Four stable repeats do not prove all prefill
  states correct or rule out rarer variation.
- Do not silently relax numerical tolerances or accept HTTP200 as correctness.

## Original image / runtime audit

Current image ID exactly matches the preregistered correctness protocol:
sha256:660ce23a83574fdcd28ba126b07615829c06770c06fbdf6533f03cc4d90acdb1.
Local RepoDigest:
quay.io/ascend/vllm-ascend@sha256:471744200c5d0c768c1f6c6a6816dcae0fd551de2487d4d129b6fb750024ce6a.

Installed source repositories are tracked-clean:
- vLLM: 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
- vLLM-Ascend: 5cb98caaadeff42b5b62b996e34bb2aaa29d20fd

Read-only docker diff audit found no non-cache/non-plugin Python source, pth, so/a/o/wheel
changes, no non-cache shared-library changes, and no runtime-repo configuration changes
among the checked suffixes. The locally installed plugin and generated caches are expected
container-layer differences. This is not a blanket forensic proof about every file.

No replacement vLLM runtime source bind mount exists. Driver/firmware are host bind mounts
(read-only from the container); their historical host content is not proven by image ID.
Runtime monkeypatches and compiler caches are also distinct from immutable image identity.

Container defaults at inspection: VLLM_BATCH_INVARIANT and HCCL_DETERMINISTIC unset,
ASCEND_LAUNCH_BLOCKING unset, HCCL_OP_EXPANSION_MODE=AIV, OMP_NUM_THREADS=1.
The inherited VLLM_ASCEND_ENABLE_FLASHCOMM=1 name is not the runtime's recognized
VLLM_ASCEND_ENABLE_FLASHCOMM1 switch; no FlashComm configuration was changed.

## Scope correction: generic investigation deferred

The generic determinism branch is deferred following user scope steering. The prepared
diagnostic worker/settings have NOT been launched. Existing observations remain evidence,
not a requirement to repair generic vLLM repeatability before any mainline work.

Do not invalidate historical performance/accuracy solely because repeated generated token
sequences differ. Do not accept packing merely because baseline also varies. Complete
change-specific numerical/state/graph checks, then collect diagnostic whole-model profiles
and profiler-disabled paired performance, retaining the no-pruning same-optimization control.
If meaningful comparisons remain impossible, report that limitation and request direction
before expanding scope. Final success still requires reproducible10%+ throughput gain and
no accuracy decline on the same configuration. No LoRA, thinking-mode tests or pushes.
