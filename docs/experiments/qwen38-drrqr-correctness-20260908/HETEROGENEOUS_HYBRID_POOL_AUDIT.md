# Heterogeneous hybrid shared-pool byte-layout audit

Date: 2026-09-09
Status: source-proven shared-pool incompatibility; the earlier GDN-only probe does not admit heterogeneous hybrid serving.

## Scope and evidence boundary

This source audit traced the uncommitted layerwise manifest and model-constructor integration through the installed v0.23 runtime. No model, NPU numerical experiment, performance run or accuracy evaluation was started by this audit. The byte examples below are direct arithmetic consequences of allocation/reshape code, not measured failures.

The passing HETEROGENEOUS_RUNNER_ALLOCATION.md experiment supplied GDN-only specs. Its heterogeneous case used one UniformTypeKVCacheSpecs group with independent per-layer tensors. Full-attention plus GDN uses a different allocator branch which can share one raw tensor across different groups; the earlier exact NPU results do not qualify it.

## Construction and binding

- vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:434 takes config, vllm_config, prefix and gqa_interleaved_layout, then builds projection/convolution widths and loaders from the instance config. The isolated-config wrapper has this signature.
- vllm_ascend/patch/worker/patch_qwen3_5.py replaces GDN state-shape/backend/forward/core/warmup methods at its end, but not __init__. Installing the constructor wrapper at top-level model construction is consistent with this order.
- vllm/model_executor/layers/mamba/abstract.py:44 constructs each MambaSpec using self.get_state_shape() and self.get_state_dtype().
- Ascend worker/model_runner_v1.py:3964 resolves each layer's own spec. Around line 4835 it groups metadata builders by backend and the full layer spec.
- vllm/v1/worker/utils.py:462 binds each cache using its exact layer name in static_forward_context. No dimension flattening was found here.

These observations show that per-layer dimensions reach the cache planner; they do not prove the shared bytes are safe.

## Source-proven collision

Inspected source roots inside the dedicated container:
- /vllm-workspace/vllm/vllm
- /vllm-workspace/vllm-ascend/vllm_ascend

1. v1/core/kv_cache_utils.py:1137 groups hybrid layers by full spec; Dk32/64/128 become different groups. The general allocator at lines 1302–1330 creates one KVCacheTensor per group column, with shared_by containing one layer from each participating group. Its comment relies on different block tables using disjoint portions of a tensor.
2. Ascend worker/model_runner_v1.py:4100–4133 allocates one raw int8 tensor and assigns that same tensor to all layers in shared_by.
3. Its Mamba reshape at lines 4682–4715 uses each individual layer's shape, laying out all convolution slots followed by all recurrent slots using contiguous views. Both the recurrent segment offset and slot strides vary with Dk, even when padded page bytes agree.
4. Different global block IDs only isolate compatible byte mappings. They cannot isolate different strides.

For real TP4 local geometry, Q/K heads=4, V heads=12, Dv=128, convolution history=3. BF16 convolution bytes per slot are:

C(Dk) = (2 * 4 * Dk + 12 * 128) * 3 * 2.

C64=12,288 and C32=10,752. In one shared raw tensor with at least 9 slots:

| View and distinct block ID | Byte interval relative to raw tensor |
| --- | --- |
| Dk64 convolution, block 7 | [86,016, 98,304) |
| Dk32 convolution, block 8 | [86,016, 96,768) |

Different nonzero block IDs overlap by 10,752 bytes. This does not depend on recurrent dtype, arithmetic, graphs or allocator alignment.

For BF16 recurrent state and 64 slots, S64=196,608 and S32=98,304:
- Dk64 recurrent block 1 starts at 64*C64 + S64 = 983,040.
- Dk32 recurrent block 3 starts at 64*C32 + 3*S32 = 983,040.

Their recurrent ranges also overlap.

## Allocation-time admission guard

The actual field names are kv_cache_config.kv_cache_tensors and KVCacheTensor.shared_by. Specs can be resolved using runner._get_layer_kv_cache_specs. Before original allocation, reject a pool containing Mamba specs with different (shapes, dtypes). This preserves the all-Dk64 control while leaving genuinely independent heterogeneous pools available for research.

Checking multiple Mamba shapes alone is insufficient generally. A pool can contain one smaller Mamba shape and full attention sized from the larger shared config. Attention reshape at approximately lines 4575–4585 skips the global convolution-padding prefix and uses the common K/V stride; the smaller Mamba view uses its own smaller convolution prefix and recurrent stride. Such cross-type sharing must also be rejected unless the Mamba layout matches the audited common layout, or an explicit block-to-byte isolation check proves compatibility. An isolated Mamba-only pool does not need this cross-type check.

This guard admits an existing layout; it does not implement safe heterogeneous sharing. Runtime evidence should include group membership, raw storage identity, byte offset and strides. Shape agreement alone cannot expose this bug.

## Implementation direction and deciding checks

A plugin-owned allocator could partition each pool by compatible physical layout or use independent compact GDN buffers. It must recompute memory accounting and attainable block/request capacity. Duplicating buffers without reducing the global block budget can exceed admitted HBM. The scheduler may retain global block IDs if each partition safely supports every admissible ID.

Qualify any new allocator with actual mixed full-attention/GDN planning and shared_by relations, distinct live block IDs, request admission/release/reuse, convolution and recurrent state, and NPUGraph replay. Preserve native contiguous-state requirements.

When changing pool construction, also account for hybrid_with_attn_and_mamba: the current allocator updates this runner-wide flag while traversing pools, whereas reshape uses it globally. A mixture of private attention pools and hybrid pools must not allocate and reshape tensors under inconsistent formats.

No throughput, TTFT, TPOT or task-accuracy claim follows from this audit. Joint acceptance remains open.

## Bounded next prototype: compact request-state storage

This section proposes the next implementation boundary, not a validated runtime. Prefer preserving the current scheduler group list for the first prototype, separating all GDN storage from attention storage, and translating only GDN global block IDs into a fixed request-state slot domain. That avoids both incompatible byte sharing and a state allocation proportional to the entire attention block-ID domain. Group consolidation can be considered after this path is qualified.

### Actual service dtype and state budget

The four current legacy64 worker reports were read directly:

/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/layerwise-model-control-v1/legacy64/worker-probe/rank{0,1,2,3}.cache.json

All four say passed, 48 GDN layers and 925 global blocks. Every GDN convolution buffer is BF16 with shape [925,3,2048], and every recurrent buffer is FP32 with shape [925,12,128,64]. They expose 16 distinct raw pools and GDN group IDs 1,2,3. These reports validate attributes/allocation/binding; they explicitly do not read cache values or claim forward correctness.

The BF16 recurrent example earlier in this document is an ABI-probe variant, not the dtype of this model service. The convolution overlap proof is unaffected. Budget implementation must always use actual per-layer shapes and dtype sizes. For the current BF16 convolution/FP32 recurrent TP4 service:

C_l = 48*Dk_l + 9,216 bytes; R_l = 6,144*Dk_l bytes.

The complete GDN bytes for one request slot are:

H = sum_l(C_l + R_l) = 6,192*sum_l(Dk_l) + 9,216*L_GDN.

For 48 layers and sum Dk=3,072, H=19,464,192 bytes. This is identical for the proposed same-sum heterogeneous dimensions and all-Dk64, before allocation alignment. With S fixed slots, reserve S*H plus explicit alignment, LUT and metadata bytes before deciding attention capacity. Reserve local slot 0 outside live request allocation for the currently supported NULL_BLOCK_ID=0 contract; it is not a replacement for PAD_SLOT_ID=-1.

### Planner and physical buffers

The existing KVCacheCoordinator constructs one BlockPool for all groups (kv_cache_coordinator.py:90). Merely partitioning the existing shared_by list by layout while allocating the same num_blocks for every partition is safe only with corrected accounting, and can squander the intended reduction by reserving GDN state for every attention-owned global ID.

A concrete first planner can retain all current group IDs and layer membership:

1. Keep scheduler Mamba specs, no-prefix semantics, block_size=max_model_len and zero speculative blocks. All layers remain bound to their original group IDs.
2. Preserve sharing only among compatible attention layouts across groups. Let P_A be the number of physical attention pools and A_p be the physical K+V bytes per global block of pool p, excluding GDN padding. Allocate N blocks for every attention pool.
3. Allocate each GDN layer independent contiguous convolution and recurrent buffers with S slots. All GDN groups for one request may map their different global block IDs to the same local request slot because the layer buffers are independent.
4. For worker r, choose the largest N_r satisfying N_r*sum_p(A_p) + S*H_r + E_r <= available_memory_r, where E_r explicitly includes alignment, lookup tables and auxiliary storage. Use N=min_r(N_r), then allocate each worker using this common N. A user block override must pass the same physical-memory inequality.
5. Check usable scheduler capacity, including its null block: for admitted requests, sum_requests(sum_attention_groups ceil(tokens/B_g) + number_of_Mamba_groups) <= N-1. A full-length request must fit. Do not infer capacity from total tensor bytes alone.

The root integration seam is v1/core/kv_cache_utils.py:get_kv_cache_configs, called through an imported binding in v1/engine/core.py:268. Its original body performs memory admission using old pooled layouts, then linearly shrinks every KVCacheTensor.size to the cross-rank minimum num_blocks. Fixed S-slot state buffers must not pass through that proportional shrink. Use a narrowly gated plugin planner with explicit variable attention/fixed state records, including equivalent max-length/override/rank-agreement checks; patch and verify the actual engine binding as well as the defining module.

The plugin's runner initialization must consume that exact physical plan and bind its returned cache dictionary by unchanged layer names. It cannot feed compact state buffers into the original Mamba reshape, which derives N from raw bytes/common padded pages. Likewise, unpadded attention tensors must not be presented to original reshape checks as padded-size tensors.

The relevant global format flag has only these direct runner uses: initialized at 4104, accumulated at 4112, read for allocation at 4118 and reshape around 4495/4569. If every attention pool is separate from every GDN pool, hybrid_with_attn_and_mamba can consistently remain false. use_hybrid_blocks remains true because it describes multiple logical groups and controls block-table/kernel splitting; do not conflate these flags. Explicit per-pool format descriptors avoid traversal-order dependence if shared hybrid pools are reintroduced later.

### Slot identity, lifecycle and asynchronous execution

Scheduler max_num_running_reqs equals max_num_seqs (scheduler.py:104); admission stops at that limit (567), and an assertion enforces it (875). A request-state pool with S=max_num_seqs+1 is therefore plausible for current prefix-off serving, provided lifecycle handling follows actual scheduler ownership, not the current input-batch rows.

The relevant ownership evidence is concrete:
- SchedulerOutput carries finished_req_ids and preempted_req_ids (scheduler.py:940–945).
- _preempt_request frees all KV blocks and resets computed tokens to zero (974 onward); recompute must receive a fresh/zeroed local state epoch.
- The worker retains unscheduled requests' cached state while removing them from the current batch (gpu_model_runner.py:1159–1180). Being absent from one batch is not permission to free a state slot.
- A finished request can be resubmitted with the same ID in the same update (1143 onward). The mapping needs an epoch, with finished/preempted release processed before new admission.
- Resume replaces the request's global block-ID lists (1364 onward). The map must replace stale global-ID bindings and verify all GDN groups map to the request's current slot.

Maintain a deterministic request-epoch-to-slot map and group/global-block-to-slot map on every TP rank. Order release/admission deterministically, compare rank evidence during qualification, and retain slots for temporarily unscheduled live requests. Never use a movable input-batch row index as persistent state identity.

Ascend execute_model calls _update_states inside synchronize_input_prep (model_runner_v1.py:2026–2049). The inherited synchronization waits for reuse of prior CPU input buffers (gpu_model_runner.py:3703); its comment does not establish completion of previous GDN state writes. CPU release may happen before all prior NPU work completes. Any slot zeroing/remapping consumed by the device must be ordered after the previous state writer and before the next model execution on the actual execution stream. Same-stream ordering may suffice after the execution path is verified; otherwise use an explicit event/stream dependency rather than a blanket per-step device synchronization. Cross-rank deterministic maps do not replace this device-ordering requirement.

The inherited _update_states also calls _zero_block_ids with scheduler-global IDs before processing admissions. Ascend initializes AscendKVBlockZeroer at model_runner_v1.py:5136. Its inspected implementation in vllm_ascend/worker/utils.py:60 processes only FullAttentionSpec and explicitly skips Mamba. Preserve that exclusion and initialize compact GDN state through the local-slot map. The existing global zeroer therefore cannot be relied upon to initialize new local GDN slots; extending it blindly to compact tensors would make global IDs unsafe.

### Metadata and graph seam

Ascend GDN builder build at gdn_attn_builder.py:553 consumes the group's common block table; at 600–601 it derives both recurrent state indices and convolution cache indices from that table. Translate the GDN block table before the original builder runs, using a preallocated device lookup/output buffer. This ensures prefill, convolution and recurrent consumers derive consistent local indices. Leave attention block tables and scheduler-visible global IDs unchanged.

The runner constructs separate shallow-copied common metadata for each group (model_runner_v1.py:3274 onward). A plugin wrapper can replace only a GDN builder's common block-table view. It must cover both normal build and build_for_cudagraph_capture. The existing builder's fixed non-spec graph inputs at lines 346–365 copy real indices and fill dummy rows with NULL_BLOCK_ID, so retain slot 0, stable buffer addresses, fixed shapes/strides and unchanged query padding. Capture before any live request must not allocate or associate ordinary request slots.

The manifest hash, exact layer dimensions/dtypes, grouping, N/S capacities and allocator version must be immutable for the lifetime of captured graphs. Changing any of them requires a new runtime instance and recapture; no hot plan replacement is admitted.

### Exact null and padding semantics

The frozen source uses three distinct concepts even where integer values coincide:

- Scheduler global null block: block_pool.py:162–177 constructs blocks in ID order from 0, pops the first as null_block, and marks it is_null. Thus global ID 0 is reserved in this implementation; it is not a live request block.
- GDN metadata constants: vllm/v1/attention/backends/utils.py:44–45 defines PAD_SLOT_ID=-1 and NULL_BLOCK_ID=0. Ascend non-spec graph padding uses NULL_BLOCK_ID at gdn_attn_builder.py:347 and repeated terminal query offsets at 352–364. Spec reset uses PAD_SLOT_ID at 375; that speculative route remains out of scope.
- Proposed local slot 0: a physical slot withheld from live compact-state allocation so the current zero-valued null metadata remains in bounds. This is a plugin allocation decision, not proof that all sentinel values are interchangeable.

The native non-spec recurrent kernel is especially important: csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h:160–190 skips a sequence when its actual length is nonpositive, before reading its state index. For positive-length sequences it assigns the index to uint64_t stateOffset and performs address arithmetic (approximately 443–476), without treating -1 or 0 as a universal no-op sentinel. Therefore current dummy safety depends on preserved zero-length query/actual-length metadata as well as NULL_BLOCK_ID=0. A real row with index 0 could write the reserved slot; a real row with index -1 is not admitted.

The actual native convolution consumer receives pad_slot_id=PAD_SLOT_ID from gdn.py:286–303. Its ResolveSeqCacheIndex in csrc/moe/causal_conv1d/op_kernel/causal_conv1d.h:925–943 skips the configured pad ID and rejects other negative/out-of-range indices. Those checks must not be assumed to exist in the recurrent consumer.

The proposed translator must classify values before lookup: map only valid owned positive global IDs, preserve supported sentinel semantics in each metadata field, retain zero-length dummy rows, and reject unresolved IDs on real rows. Never use -1 as a lookup-table index or blindly convert every -1 to local 0. Qualification must include exact dummy metadata, unchanged reserved-slot state where the zero-length contract applies, and both native convolution and recurrent consumers.

### Deciding tests before model performance

Implement only after the all64 full-model control is established. The first prototype should run actual hybrid planning/allocation, real scheduler admission/release/reuse and NPU state updates against independent per-layer references. Cover mixed prefill/decode, reorder, a temporarily unscheduled live request, preemption/recompute, abort-and-resubmit with the same ID, reused global block IDs, graph padding, and asynchronous slot reuse. Require unchanged graph pointers, untouched other-request states, exact storage-transform outputs/full state, and deterministic TP rank mappings.

Admission evidence must include actual HBM/tensor allocation within the reserved budget and enough attention blocks for the same workload. Numerical equivalence and fitting a single request do not establish end-to-end throughput or accuracy benefit.
