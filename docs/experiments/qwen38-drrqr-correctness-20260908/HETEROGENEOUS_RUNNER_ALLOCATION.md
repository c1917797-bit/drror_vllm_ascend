# Heterogeneous GDN cache allocation through the Ascend runner

Date: 2026-09-09
Status: NPU allocation/reshape/recurrent graph gate passed; serving integration pending.

## Hypothesis and bounded execution

After the v4 arena ABI gate passed, the next hypothesis was that the installed
vLLM cache grouping and Ascend runner allocation/reshape implementations can
preserve per-layer unpadded GDN state sizes and numerical behavior.
This is a GDN-only subsystem test, not a full hybrid model or scheduler test.

One NPU matrix was run on physical NPU4 (container logical0), after checking
NPU4–7 were idle and their container mappings. AWMCP operation
exec-000000000000039c completed with exit code0 in33.615 seconds.
No retries or relaxed numerical criteria were used.

Code: tools/heterogeneous_cache_allocator_probe.py
SHA256: 640e2ba0b7c0dcbc8322c9c3e33414d37f8a8cfae01872250ad1c2c5b3776a91

The probe directly invokes unmodified:
- get_kv_cache_groups and get_kv_cache_config_from_groups
- NPUModelRunner._allocate_kv_cache_tensors
- NPUModelRunner._reshape_kv_cache_tensors
- native npu_recurrent_gated_delta_rule and NPUGraph replay

It supplies a minimal runner context and GDN-only specs, not a loaded model.
The v4 numerical helper is source-hash pinned. No service monkeypatch is
installed by this probe.

## NPU results

| Layout | Batch / state slots | Exact changing-input steps |
| --- | --- | --- |
| 64/64/64/64 | 1 / 4 | 8/8 |
| 64/64/64/64 | 16 / 19 | 8/8 |
| 128/32/32/64 | 1 / 4 | 8/8 |
| 128/32/32/64 | 16 / 19 | 8/8 |

For all cases, eager and graph outputs and entire recurrent state were finite
and bitwise identical to independently allocated native state. Adjacent entire
convolution buffers retained their sentinel values. All tracked state/input
pointers were unchanged, and state addresses were512-byte aligned.
Unlike the previous arena probe, there are no additional outer canary regions.

At19 slots, both layouts allocated15,876,096 bytes including convolution
state. The heterogeneous state was not padded uniformly toDk128:
per-layer page bytes were408,576 /109,056 /109,056 /208,896.
A hypothetical max-layer-padded layout would use31,051,776 bytes, but that is
NOT a measured production baseline. Do not advertise this comparison as an
end-to-end memory or throughput gain. Allocator-reserved memory/HBM consumption
may exceed tensor storage bytes and was not compared.

The uniform layout used MambaSpec grouping; heterogeneous used
UniformTypeKVCacheSpecs. All untyped storage byte counts matched the specified
logical allocation, without global Mamba page padding.

## Report and source provenance

Report:
/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/heterogeneous-runner-allocation-v1/report.json

SHA256:
d076ecf7d75917307f891c88924e4c3e67c9a8fb3b23a0ade23a363f292b774e

Installed source hashes:
- vllm_ascend.worker.model_runner_v1:
  94d75dbeb5d23ab5b383cdc69114167392b967994bb21342cf748b733ba9a968
- vllm.v1.core.kv_cache_utils:
  b89f9afa7f95fcc30412ae2031d5370a23e95f35d068fa14e5824dce1c900ee5
- vllm.v1.kv_cache_interface:
  8d6be98bbf75b3d55007b64ccf5afef047ca01790a432522f70a196412c103d9

The container remained running with only its original bash after this probe.
Reference runtime sources and the image were not modified.

## Remaining integration work

The full hybrid path is not equivalent to this GDN-only grouping:
1. MambaBase.get_kv_cache_spec uses one global mamba_page_size_padded.
2. The Ascend runner's get_kv_cache_spec aligns full-attention specs using a
   Mamba page value assigned while visiting layers. Variable layer sizes need
   an explicit contract, not reliance on whichever layer is visited last.
3. get_kv_cache_groups falls through to uniform-page-size grouping for ordinary
   full-attention+GDN hybrids. The multi-UniformType construction helper is
   specialized for MLA/sliding-window MLA and cannot simply be reused as a
   generic hybrid planner.
4. generate_scheduler_kv_cache_config unwraps each UniformType group to a
   representative spec. Every layer in such a group must share scheduling
   behavior, including block size, cache mode and speculative-block count.
5. Splitting into private per-type pools can waste state allocations or reduce
   request capacity despite smaller layer tensors. Compare actual total
   allocation and attainable request capacity against the current hybrid pool.
6. The current DRRQR plan and model loader still assume a global Dk.
   Layerwise widths need validated slicing, constructor config isolation and
   cache/graph metadata consistency before a service launch.

Next work package: implement an explicit plugin-owned layerwise-Dk contract
and hybrid grouping/allocation prototype. First validate its all-Dk64 identity
control, then heterogeneous cases, including request admission/release and
state-index isolation. Advance to model-level NPU checks and paired endpoint
performance only after these integration invariants hold.

No throughput or task-accuracy update was produced by this experiment.
The best endpoint evidence remains one +4.752041% paired screen; the current
1024-token LongBench-v2 preflight remains39.370%, versus45.407% historical
dense mean. Final2048 accuracy and joint acceptance remain pending.
