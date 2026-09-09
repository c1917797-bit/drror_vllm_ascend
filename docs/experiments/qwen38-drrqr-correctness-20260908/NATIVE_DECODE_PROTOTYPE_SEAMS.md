# Native Ascend decode prototype seams

Status: source audit and proposed gates, 2026-09-09. No native prototype was
implemented, built, or executed during this subtask; no new NPU numerical,
graph, throughput, or accuracy result is claimed. All reference inspection was
read-only. No cannbot skill was read or applied on behalf of the main agent.

## Evidence scope and source identities

The execution source is the dedicated container
`drrqr_qwen38_energy_preflight_20260908`, image ID
`sha256:660ce23a83574fdcd28ba126b07615829c06770c06fbdf6533f03cc4d90acdb1`.
Container source roots:

- Ascend runtime: `/vllm-workspace/vllm-ascend`.
- vLLM: `/vllm-workspace/vllm`.
- Installed custom vendor:
  `/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer`.

User-highlighted references are `/cache/cch/vllm-ascend` and
`/cache/cch/ops-transformer`. They are source references, not proof of the
executed implementation. The exact v0.23 reference recorded elsewhere is
`/cache/austinov/src/vllm-ascend`, commit
`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`; this audit used container files
for execution-path claims.

Current inspected source SHA256 values:

| Source | SHA256 |
| --- | --- |
| container `vllm_ascend/ops/gdn.py` | `d6ec29919268178f5bf6e70e689c1d273d04b1cb1d84dc94efa7bbbc35490816` |
| container `csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h` | `96e6d9667af1e3457316ffce92a8a983fcb90096df08e3efd13ec0b35b8b1c5b` |
| container `csrc/attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_def.cpp` | `c388285948566df9e106bca2dc895aa6dad57506c9da5216cde67b3cc16276d0` |
| container and `/cache/cch/vllm-ascend/csrc/utils/inc/fallback.h` (equal hashes) | `edda7ea481ca5711ccdd1667934bcd3a099e800461968f77a3ae1b850d0803c3` |
| `/cache/cch/ops-transformer/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.cpp` | `b1b6d72db33eec0a832e927e297718992c2711beae83368dbf8912f1ae60ca8d` |
| same reference op `op_host/CMakeLists.txt` | `493399679ef6825c2ecdda97769cdf19442d4dd32eae85010aedfaae106b42f6` |
| same reference op `op_api/aclnn_recurrent_gated_delta_rule.cpp` | `02c53e2f0d40ed429ed6c19839184d38aed4356289385b5378ea61e84c6c2ce9` |

Read-only `git rev-parse` for the two user-highlighted reference repositories
returned Git's dubious-ownership error. No global safe-directory setting was
changed; this audit does not assert newly verified reference HEADs. The file
hashes above bind the inspected content.

## Actual execution and useful boundaries

Container `vllm_ascend/ops/gdn.py`:

- Line 313 computes `DeviceOperator.fused_gdn_gating(A_log, a, b, dt_bias)`.
- Decode lines 423–424 call independent `l2norm_fwd` for Q and K.
- Line 427 calls `torch.ops._C_ascend.npu_recurrent_gated_delta_rule`.
- Prefill uses the chunk path, including its own in-kernel Q/K normalization.
  A first native decode prototype should leave prefill and mixed prefill/decode
  dispatch on their admitted existing paths.

This is source-backed separation of gating, normalization, and recurrent
execution. It is not the generic packed Triton decode implementation previously
misidentified as the running Ascend path.

Container `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`,
`rearrange_mixed_qkv` around lines 812–844, splits the convolution output
`[T, Qwidth + Kwidth + Vwidth]`, flattens each component, then concatenates
them into one planar buffer. Q/K/V are views into that result. Its source
comment describes a possible torch.compile copy fusion; that comment does not
prove the actual Ascend graph's copy count or timing. Do not report this source
as three already-measured standalone copy kernels.

The native adapter is
`csrc/attention/recurrent_gated_delta_rule/recurrent_gated_delta_rule_torch_adpt.h`.
It allocates BF16 output with value's shape and calls
`EXEC_NPU_CMD(aclnnRecurrentGatedDeltaRule, ...)`.
The schema at `csrc/torch_binding.cpp:2269` marks state as `Tensor(a!)` and
registers the kernel under PrivateUse1. Preserve that mutation declaration in
any new namespace.

The ordinary 910B entry is
`csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.cpp`.
Its non-arch35 branch includes `recurrent_gated_delta_rule.h` and instantiates
`RGDR<bfloat16_t, bfloat16_t, DTYPE_STATE>`. The useful local seam in that
header is `CopyInQKV`, lines 201–247:

- Separate GM bases and token strides load Q/K/V.
- BF16 Q/K/V are cast into FP32 UB tensors.
- Q then receives the explicit `scale`.
- `ProcessHead` maps each value head to its Q/K head through
  `head_i / (NV / NK)`.

The first storage optimization can retain this arithmetic while changing only
the source-layout reader. For example, retain separately normalized Q/K and
allow V to be read from the original packed convolution row via explicit base,
row stride, and offset. This requires an adapter/op contract that does not
silently make V contiguous. It cannot by itself remove all Q/K unpacking:
the current normalization kernel expects contiguous rows. Removing that
preparation belongs to the L2 fusion stage below.

## Build and registration seam

The frozen custom-op build is an ops-transformer-derived project under container
`csrc`. Its `CMakeLists.txt` exposes `ASCEND_OP_NAME`,
`ASCEND_COMPUTE_UNIT` (default `ascend910b`), and `VENDOR_NAME`, and drives
`cmake/opbuild.cmake`, `cmake/custom_build.cmake`, and
`gen_aclnn_with_opdef()`.

For the existing recurrent op,
`csrc/attention/recurrent_gated_delta_rule/op_host/CMakeLists.txt` calls:

- `add_op_to_compiled_list()`;
- `add_ops_compile_options(OP_NAME RecurrentGatedDeltaRule ...)`;
- `add_modules_sources(OPTYPE recurrent_gated_delta_rule ACLNNTYPE aclnn_exclude)`.

Its OpDef uses `OP_ADD(RecurrentGatedDeltaRule)`, and its tiling source uses
`REGISTER_OPS_TILING_TEMPLATE(RecurrentGatedDeltaRule, ...)`.

The current ops-transformer reference is not a drop-in equivalent:
its op-host CMake uses `add_modules_sources_with_soc` with
`OP_API_INDEPENDENT ON` and `../op_api`; its ordinary kernel explicitly
includes `arch22/recurrent_gated_delta_rule.h` and instantiates a two-parameter
`RGDR`. Use the frozen implementation as the initial semantic control and
consult the reference for organization and APIs; do not transplant the newer
reference implementation and call it an unchanged baseline.

A concrete plugin-owned prototype layout can contain:

- `native/<prototype>/op_host`, `op_kernel`, explicit op API, and renamed
  torch adapter;
- a standalone build entry and pinned copy manifest for the minimum required
  custom-op build support;
- a unique vendor, for example `drrqr_research`, and unique OpDef/tiling/kernel/
  ACLNN symbols such as `DrrqrRecurrentV1` and `aclnnDrrqrRecurrentV1`;
- a separate torch namespace, for example `drrqr_ascend::recurrent_v1`;
- an explicit opt-in configuration and an absolute-path loader.

These are proposed names, not existing implementation. Do not register a
replacement under `_C_ascend.npu_recurrent_gated_delta_rule` or overwrite the
original vendor. Keep build source, build scripts, source hashes, and artifact
identities under this plugin repository. The current read-only container repo
mount means the eventual build needs an explicitly scoped writable plugin
build/artifact mount; do not solve that by making a reference tree or original
image writable.

The root vLLM-Ascend CMake links its wrapper to torch libraries, `torch_npu`,
`ascendcl`, `tiling_api`, `register`, `platform`, `ascendalog`, `dl`,
and `opapi`. This identifies dependencies, not a validated standalone build
command. Before building, pin the actual compiler/CANN/torch ABI paths and
review the minimal build recipe. Retain original copyright and license
headers and provenance when copying source; additions remain in this repo.

## Dtypes and arithmetic boundaries

The frozen OpDef and tiling source accept BF16 and FP32 state, with BF16
Q/K/V/beta and FP32 g. The frozen kernel templates state type independently.
The installed 910B kernel configuration also contains two state variants.
By contrast, the inspected current ops-transformer `op_api` explicitly lists
only BF16 in `STATE_TYPE_SUPPORT_LIST`. These are distinct contracts.

The previous heterogeneous arena v4 evidence proves the tested BF16 state
configuration only. An older failed FP32 initialization probe does not prove
that the properly initialized frozen FP32 variant is impossible. If a
prototype or actual model uses FP32 state, it needs its own correctly
initialized native control, multi-step parity, and graph evidence. Do not
silently cast state to BF16 or reuse BF16 correctness as FP32 evidence.

In container `vllm/model_executor/layers/fla/ops/l2norm.py:79–150`,
the current kernel2 loads BF16 into FP32, computes a sum of squares plus
`eps=1e-6`, multiplies by reciprocal square root, and stores into the original
dtype by default. The recurrent kernel then casts that stored BF16 Q/K back to
FP32 and scales Q by `Dk**-0.5`. A fused implementation must preserve the
BF16 rounding boundary before recurrent arithmetic unless a separately
declared numerical-change experiment justifies removing it. It must also
match epsilon, head grouping, reduction axis, and scale placement.

The frozen `fused_gdn_gating` sources specify:
`g = -exp(A_log) * softplus(float(a) + dt_bias)` and
`beta = sigmoid(float(b))`, with beta stored back into the input dtype.
The adapter defaults are softplus beta 1 and threshold 20, returns FP32 g,
and requires A_log and dt_bias to share their parameter dtype. The recurrent
kernel's `CopyInGamaBeta` casts BF16 beta into FP32 and computes `exp(g)`.
A gating fusion must preserve these cast boundaries and actual parameter
dtypes, not merely match the symbolic expression.

## Three bounded gates before a model performance screen

All gates are proposed; none has run for a new independently named operator.
Revalidate physical NPU4–7, container mappings, current loads, and unique output
directories before any execution. Retain all failure receipts. Initialization
must follow the previously validated native probe order; do not introduce
determinism switches or the separately deferred determinism worker.

1. **Copy/layout only.** Establish the unique vendor and torch/ACLNN names
   with the unchanged native recurrent arithmetic as control. The first
   layout change should consume exactly the same normalized Q/K, beta/g, and
   state values, with only an explicitly described packed-value/stride reader
   or QKV unpacker changed. Require bitwise-identical output and entire state
   against the frozen native path after every step. Verify all arena guards,
   unselected slots, shapes, addresses, and graph-replay outputs. Accept only
   supported layouts; reject unknown strides rather than insert an unreported
   contiguous copy. This stage establishes build/ABI/storage correctness,
   not a forecast of throughput gain.

2. **Packed QKV plus L2 normalization and recurrent.** Read the post-convolution
   raw packed rows directly, replace the Q/K preparation and L2 materialization,
   and normalize at the CopyInQKV/FP32 UB seam before the preserved BF16 round
   and Q scale. Keep gating external for this gate. Compare candidate and
   frozen independent L2+native recurrent on identical inputs and initial
   states, and also against the existing FP32 recurrence oracle. Before first
   execution, freeze the justified arithmetic error budget in the probe.
   The existing golden decode budget is rtol=0.003, atol=0.01
   (`tools/gdn_golden_parity.py`); it is an outer golden check, not permission
   to spend that entire error on this incremental change. Report candidate
   versus native and each versus oracle, including maximum/mean absolute and
   relative error, exact equality, finite status, and first diverging step.
   Freeze any tighter change-specific criterion before running.

3. **Add gating to the admitted packed/L2/recurrent implementation.** Consume
   a/b/A_log/dt_bias directly, preserving the softplus threshold and BF16 beta
   round before recurrent FP32 computation. Compare both intermediate g/beta
   semantics in a diagnostic mode and complete output/state against stage 2
   plus native gating. Test saturation, the threshold boundary, near-zero
   norms, negative/positive inputs, and extreme finite values. Use a separate
   predeclared arithmetic budget and retain failures without loosening it.

Common first matrix: TP4-local HQK=4, HV=12, Dv=128; Dk=32/64/128;
batch 1 and 16; 8 changing-input steps; slots=batch+3 with changing valid
indices and sentinel guards; eager and NPUGraph replay with stable buffers.
Support Dk96/112 only after explicit new coverage if a selected layer plan
uses them. Dk104/88 remain rejected by the existing alignment gate. Test
zero/padded inactive graph slots only according to the actual frozen
metadata/sentinel contract, not an invented index convention.

For graphs, complete all symbol resolution and required workspace preparation
before capture, record workspace size/allocation, and prove that replay with
changed inputs/indices updates the intended state without reallocation or
stale metadata. Record the captured custom operator name and bound source;
an eager pass does not imply capture/replay correctness.

After native numerical/graph admission, use the normal worker path for one
representative model integration gate, then disable diagnostics/profiling for
one end-to-end screen. Apply any common native optimization to both dense and
the selected DRRQR candidate. Report TTFT and TPOT separately with fixed
1024 output tokens. The representative 4-request/32-token integration
trajectory and its token/logprob criteria are not a performance or final
accuracy test. Promising candidates still require the full matched
three-round performance and thinking-off, max_tokens=2048, normal-EOS
LongBench-v2/GSM8K acceptance.

## Loaded-symbol and binary provenance

The existing v4 arena report
`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/heterogeneous-state-abi-v4/report.json`
records both of these mapped libraries:

- `/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant/op_api/lib/libcust_opapi.so`,
  SHA256 `ff82d965a8a08bfa2862567022a04a1fd04df56d142f4393f1fb527592111069`;
- custom_transformer `op_api/lib/libcust_opapi.so`,
  SHA256 `b77ec1ec0c5e8b8c31f29dd6763979a7f9442a288c7812988655d273a41e2855`.

This shows simultaneous mapping, not which library supplied each symbol.
The identical-hash frozen/reference `fallback.h:80–139` uses
`dlopen("libcust_opapi.so", RTLD_LAZY)`, caches a handle, then calls
`dlsym`, with fallbacks to other libraries. Blindly reusing this resolver
does not prove a new plugin call reaches its own vendor.

The plugin loader should resolve its uniquely named ACLNN entry and
GetWorkspaceSize entry from an explicitly selected absolute library, record
`dladdr` for both actual function pointers, and fail closed if they are not
the expected artifact. Record torch extension path/hash, dispatcher schema,
vendor/proto/tiling libraries, ordered custom-OPP search paths, and all actual
resolved library identities. Preserve original search behavior for existing
operators.

The installed 910B kernel config is
`<custom_transformer>/op_impl/ai_core/tbe/kernel/config/ascend910b/recurrent_gated_delta_rule.json`,
SHA256 `983e46e61ccd6b62eb0813e2db6453d5e20fc85470c13b4421413c3ef9cf6d8f`.
Its state-specific binInfo entries lead under
`op_impl/ai_core/tbe/kernel/ascend910b/recurrent_gated_delta_rule/` to:

| State dtype | Binary file | SHA256 |
| --- | --- | --- |
| BF16 | `RecurrentGatedDeltaRule_85669048de2f4ae7785fd8d81b12f70b.o` | `17258834f08703bdfeef2f808739702b2f107cb3a5ad79952779a845cdb32e34` |
| FP32 | `RecurrentGatedDeltaRule_37ef20e52a126f008ab9edfe51ac2143.o` | `d3d0ef136d259b542e7a75fea91d17759ed7d958bf16e03a1954e835de95d050` |

Also retain each selected binary's adjacent JSON, op-config matching key,
compiler options/version, tiling identity, and build manifest. These inspected
files identify available artifacts; they do not by themselves prove the
runtime selected one. For a new unique op, bind its actual dispatch/schema
and selected config to the generated binary. Device binaries need not appear
as ordinary mapped host libraries, so `/proc/<pid>/maps` alone is insufficient.

No kernel-duration sum or assumed launch-cost saving is used here to estimate
end-to-end throughput. The next authorized implementation is a plugin-owned,
independently named copy/layout control plus its exact-state NPU gate, followed
by the L2 and gating arithmetic stages if that control passes.
