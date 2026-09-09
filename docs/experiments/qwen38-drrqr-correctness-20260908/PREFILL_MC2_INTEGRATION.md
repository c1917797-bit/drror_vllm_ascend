# Prefill-only MC2 plugin integration

Date:2026-09-08. Candidate code, not an end-to-end performance/quality result.

## Implementation

New opt-in: VLLM_ASCEND_DRRQR_PREFILL_MC2=1 (default0).
It is independent of pruning, so the same optimization can be applied to Dk128
and Dk64 for fair attribution. Capture mode is excluded. Source release is0.1.8.
Installation and live activation are recorded in the follow-up below.

The new patches/prefill_mc2.py adapter:

- Verifies the frozen Ascend row-linear source and existing Qwen/TP4/BF16 model contract.
- Validates all128 targets before binding:64 MLP down projections plus48 GDN output
  projections and16 full-attention output projections. No checkpoint weights are changed.
- Uses the existing TP HCCL communicator and native torch_npu.npu_mm_all_reduce_base.
- Requires a pure-prefill, unpadded batch, explicit valid metadata from every attention
  entry, no graph capture/dummy-profile run, and runtime graph mode NONE.
- Uses min_rows2048 for down_proj and8192 for attention output projections, based
  on the completed exploratory micro screen. Smaller and all mixed/decode batches
  preserve the original row-linear computation/communication implementation.
- Resolves the phase once around NPUModelRunner._model_forward, and resets its
  ContextVar in finally, including exceptional exits.
- Records all prepared targets and first actual fused invocation per target.
  Preparation evidence alone is not proof that real requests used fusion.

## Compilation correction

The frozen service log confirms CompilationMode.VLLM_COMPILE together with
FULL_DECODE_ONLY. FULL_DECODE_ONLY does NOT mean Python prefill code is uncompiled.
A plain Python branch in the layer forward could be specialized from a dummy/profile
batch or rejected by Dynamo. Before installation, dispatch was moved into an opaque
registered custom operator with a fake shape implementation. The phase is read at
execution; the outer compiled forward does not inspect the ContextVar.

A CPU fullgraph/dynamic torch.compile test first compiles with the profile marker,
then reuses the compiled callable for pure prefill and a mixed batch. It verifies
original -> fused -> original routing. This proves the CPU Dynamo routing contract,
NOT native Ascend compiler, TP model, graph replay or state/accuracy parity.

## Verification and incidents

The TP4 native micro screen passes24 rank-cases; see MATMUL_ALLREDUCE_MICRO_SCREEN.md.
Final CPU regression operation00e5:68 tests pass, including6 adapter tests.

Intermediate CPU operation00e1 caught a stale0.1.7 release assertion after the
intentional0.1.8 source version bump; the assertion was updated to0.1.8.
A subsequent patch-hunk ordering error left test indentation invalid (00e4).
The file was read back, corrected through apply_patch, and the full68-test suite
rerun successfully. No NPU model or benchmark was launched with either test failure.

## Historical checkpoint before installation

Offline build completed (00e6) and independently verified (00e7): all18 packaged
Python modules match the repository snapshot; ZIP integrity and0.1.8 metadata pass.
Build root: /drrqr-results/prefill-mc2-e2e-v1/build

- Wheel: wheels/drror_vllm_ascend_plugin-0.1.8-py3-none-any.whl
- Wheel SHA256:76d677ff2ebc43f0d90debc21b99014136b9b2d73ee2578221138629c269ea63
- Build manifest SHA256:455c762e4ee0e9fc01c269889e26ac8856459abd9dded86d201bf4f53cc3f588
- Adapter SHA256:23a7d07e32d9d33f8d9f095dd8db7931ada1d18978d12472f116caf85e558ab4

Installed purelib distribution is still0.1.7; upstream files still match the frozen
baseline manifest. All NPU work from this turn is terminal and physical4–7 were
confirmed free after the micro screen. No0.1.8 service has been launched.

1. Verify the offline0.1.8 wheel/build manifest and install only that local artifact
   into the dedicated experiment container; keep upstream sources unchanged.
2. Extend the existing provenance-locked launcher with an explicit0.1.8 selection
   and explicit MC2 opt-in. The current launcher still requires0.1.7; do not bypass it.
3. Verify TP4 preparation AND real-request fused-call coverage, then a bounded
   compiled/native graph/sequence smoke test. If any route is silently unused,
   fix that integration issue before interpreting performance.
4. One profiler-disabled Dk64 MC2-off/on screen at the existing fixed-output protocol.
   Keep convolution layout fixed on both sides; do not mix NZ into this intervention.
5. If promising, run the unpruned same-optimization control and expand paired
   performance/quality evaluation. Final quality uses max_tokens2048 both sides.

No LoRA, thinking-mode tests, generic determinism investigation, upstream edits,
commit or push. The final joint10%+ performance / no-accuracy-decline goal is still open.

## Follow-up: installed and live activation passed

Operation00ec installed the verified offline0.1.8 wheel into the dedicated container.
All18 installed package modules match the build/repository hashes, and the frozen
upstream reference hashes are unchanged. The0.1.7 wheel is retained for rollback.
Receipt: /drrqr-results/prefill-mc2-e2e-v1/install-receipt.json.

The launcher now explicitly supports0.1.8 and the MC2 flag, and records competing
runtime environment settings. CPU regression00eb passes71 tests, including the new
live-evidence validator. The final launcher environment guard is also exercised by
the successful live launch below.

ON service00ee passes native compilation, graph capture and the activation smoke00ef:
four prepared workers (PIDs42746–42749), all128 targets on every worker, and512 actual
first32K pure-prefill fused calls. Three requests complete with exactly32/16/16 outputs;
decode graph replay is observed. The smoke takes334.388seconds including startup.
This proves integration/activation, not full-model numerical or task-quality parity.

- Root: /drrqr-results/prefill-mc2-screen-v1/dk64-on
- Service manifest SHA256:5f7b698cef9bd87145ad1f421baae2baf8825d6ca77deeb9957307dd86a8dde7
- Evidence SHA256 at smoke:5680f95312baa7670037b2fe005288e9c86625ada5bf2902ad0af30ae531620f
- Protocol: PREFILL_MC2_SCREEN.md (ON then OFF, same0.1.8/Dk64/packed layout).

Both performance screens are now complete (ON00f5, OFF0107). Throughput is
134.514702159 versus134.606909753 tok/s: ON/OFF -0.068501%, no material gain
observed in this single pair. Both services exit0 and physical NPU4–7 are free.
PREFILL_MC2_SCREEN_RESULT.md and prefill-mc2-screen-v1-result.json are the completed
paired-screen result. No task-quality parity or final joint success is claimed.
