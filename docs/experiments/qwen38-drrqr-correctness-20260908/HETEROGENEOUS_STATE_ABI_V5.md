# Heterogeneous state ABI v5: BF16 and FP32 qualification

## Result and scope

Both fresh v5 native recurrent-state matrices passed: **2 state dtypes × 2 layouts × 2 batch sizes = 8 cases**, each with 8 consecutive changing-input steps. All 64 case-steps passed all 11 recorded checks. The main agent subsequently confirmed both processes completed with exit code 0, without timeout or cancellation.

This closes the **FP32 recurrent-state dtype coverage gap** for this native ABI/storage experiment. It is not a vLLM heterogeneous hybrid-cache integration result, an end-to-end speedup, or a task-accuracy result. The independently allocated reference invokes the same native operator; this experiment tests storage/replay equivalence, not that operator's arithmetic against an independent mathematical oracle.

## Frozen evidence

All paths below are on chicaihong01. Reports were independently read through the target-validated direct host route and their SHA256 values checked with host `sha256sum`.

| Evidence | SHA256 |
| --- | --- |
| `tools/heterogeneous_state_probe.py` | `d455dfbb86cf24e61e41ed5fd423b2b18bd009d6a824b03a4cbf60f82c8cdf71` |
| `heterogeneous-state-abi-v5-bf16/report.json` | `531f603fff8efe3cdfb5089155ab1b4829a00511ece15a515d109236f2a20ccd` |
| `heterogeneous-state-abi-v5-fp32/report.json` | `682116368a6b61fc7aee980eed3d3aa8145362c3996fb80a4fa71522515b4db5` |

Report root: `/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight`.

The full machine-readable digest is [heterogeneous-state-abi.v5.summary.json](heterogeneous-state-abi.v5.summary.json). The raw reports retain every step's checks, actual tensor shapes/dtypes/pointers, operator schema, runtime identity and mapped-library hashes; the digest does not replace them.

| State dtype | Operation | Process status | Duration |
| --- | --- | --- | --- |
| BF16 | `exec-00000000000004d4` | completed, exit 0 | 34,001 ms |
| FP32 | `exec-00000000000004e3` | completed, exit 0 | 35,112 ms |

Operation status above comes from the main agent's final operation-receipt verification. These are whole probe process durations, **not latency or throughput measurements**. Report `phase=complete` and `status=native_heterogeneous_state_abi_pass` were checked separately from process exit.

The main agent's postflight checks found only the original bash process (PID 262195) in the dedicated container (`exec-00000000000004f3`) and no running workload on physical NPU4–7 (`exec-00000000000004f5`). This describes the end of these probes, not a continuing reservation or future device-idleness guarantee.

## Bounded protocol and observed matrix

One invocation per dtype, no automatic retry. The operator runs on logical NPU0, externally verified as physical NPU4, inside `drrqr_qwen38_energy_preflight_20260908`. This is a single worker with TP4-local shapes, not a four-rank distributed run.

- Q/K heads = 4, value heads = 12, Dv = 128; state layout `[slots, 12, 128, Dk]`.
- Q/K/V and beta are BF16, g is FP32, and output is BF16 in both state-dtype matrices.
- `uniform64=[64,64,64,64]`; `heterogeneous=[128,32,32,64]`. Each has total Dk 256.
- Batch sizes 1 and 16 use 4 and 19 slots, respectively. Inputs and selected state indices change over 8 steps.
- Independent native reference, one contiguous eager arena, and one contiguous NPUGraph arena are compared. All layer-state pointers are 512-byte aligned.
- Each boundary guard is 512 bytes: 256 BF16 elements or 128 FP32 elements. This v5 protocol supersedes the original unrun draft's 64-FP32-element guard description.

| State dtype | Layout | Batch / slots | Steps passed | Logical state bytes per path |
| --- | --- | --- | --- | --- |
| BF16 | uniform64 | 1 / 4 | 8 / 8 | 3,145,728 |
| BF16 | uniform64 | 16 / 19 | 8 / 8 | 14,942,208 |
| BF16 | heterogeneous | 1 / 4 | 8 / 8 | 3,145,728 |
| BF16 | heterogeneous | 16 / 19 | 8 / 8 | 14,942,208 |
| FP32 | uniform64 | 1 / 4 | 8 / 8 | 6,291,456 |
| FP32 | uniform64 | 16 / 19 | 8 / 8 | 29,884,416 |
| FP32 | heterogeneous | 1 / 4 | 8 / 8 | 6,291,456 |
| FP32 | heterogeneous | 16 / 19 | 8 / 8 | 29,884,416 |

Logical state bytes exclude guards, inputs, outputs, the other comparison paths and runtime allocations; they are not full-process HBM usage.

The exactness predicate requires the same shape and dtype, rejects nonfinite values, and compares contiguous logical tensor bytes through a uint8 view. It distinguishes signed zero. Eager and graph outputs and **full states** match the independently allocated path byte-for-byte.

All guards remained intact. All 46 tracked tensors per case retained shape, stride, dtype, device, data pointer, storage pointer and storage offset. Reference, eager and graph each independently retained every unselected slot unchanged against the preceding CPU snapshot; agreement with the reference alone was not used as evidence for this property.

The reviewed graph path performs sanity/warmup/capture, restores initial graph state in place, and then updates inputs/indices in place before replay. Failure/progress receipts and graph cleanup handling remain part of the v5 harness; this passing run did not exercise every failure branch.

## Actual service FP32 relevance

The existing all64 model cache-binding reports at

`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/layerwise-model-control-v1/legacy64/worker-probe/rank{0,1,2,3}.cache.json`

all reported passed, 48 GDN layers and 925 blocks. Actual conv state was BF16 `[925,3,2048]`; actual recurrent state was **FP32** `[925,12,128,64]`. Thus the FP32 v5 matrix now exercises the actual service's recurrent-state dtype instead of relying on the prior BF16-only ABI variant.

Those model reports qualify load/cache binding, and v5 qualifies synthetic native state storage/replay. Neither evidence source turns the heterogeneous arena into a qualified model cache-manager implementation.

## Historical predicate and helper compatibility

v4 checked finite `torch.equal` element equality, which did not distinguish positive and negative zero. Its predicate therefore did not establish strict byte equality in that corner case. **No actual v4 bit mismatch is asserted or evidenced by this caveat.** Historical outputs and reports are preserved; v5 provides new stricter evidence.

The existing `tools/heterogeneous_cache_allocator_probe.py:50–51` pins the v4 helper SHA256 `5e1a726710f5a3ef3da81f651dbe4a61da833e97fd6e9061a65c97047be27982` and raises `reviewed v4 numerical helper changed` when it differs. The current v5 helper has a different identity, so that guard still rejects it. No allocator-helper code or historical allocator result was changed, and this v5 result must not be represented as a new allocator qualification.

## Runtime identity and limits

Both reports recorded torch `2.10.0+cpu`, torch_npu `2.10.0.post4`, the same native operator schema and Ascend GDN source SHA256 `d6ec29919268178f5bf6e70e689c1d273d04b1cb1d84dc94efa7bbbc35490816`. The torch version's suffix is not the experiment device: recorded state/input tensors and operations are on `npu:0`.

The same six library paths/hashes were recorded in both processes, including the custom-transformer libraries and an external `batch_invariant/libcust_opapi.so`. These are **process-maps observations plus file hashes**, not proof of symbol resolution, selection of one vendor's operator implementation, or an exhaustive identification of the executed kernel binary. Full entries are retained in the raw reports and JSON digest.

Existing hybrid-pool guards remain necessary. This experiment has no attention/Mamba shared pools, scheduler block ownership/reuse, prefix-cache identity or full-model graph-key validation. It does not override the source-proven mixed-layout aliasing described in [HETEROGENEOUS_HYBRID_POOL_AUDIT.md](HETEROGENEOUS_HYBRID_POOL_AUDIT.md).

Next integration evidence must qualify actual mixed attention/Mamba allocation and lifecycle before a full-model numerical candidate and a bounded performance screen. There are no new throughput, TTFT, TPOT, LongBench-v2 or GSM8K numbers here.
