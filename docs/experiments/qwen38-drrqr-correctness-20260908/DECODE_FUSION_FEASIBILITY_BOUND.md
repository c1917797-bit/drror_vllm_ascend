# Decode fusion feasibility bound

> Superseded on 2026-09-09: the user authorized extended heterogeneous-cache/operator
> research and push. See [EXTENDED_RUNTIME_RESEARCH.md](EXTENDED_RUNTIME_RESEARCH.md).
> The packed-Triton hot-path inference and broad fusion/layerwise impossibility
> conclusions below are withdrawn; retained here as historical audit evidence.
> Measurements are unchanged. Current status is research authorized, not success.

Date: 2026-09-09

## Decision

Do not start a custom decode-fusion implementation or another NPU screen under
the current no-training, at-least-10%-throughput phase. The installed runtime
already uses a packed recurrent decode kernel, and the remaining unfused work
does not have enough credible candidate-specific headroom to close the locked
performance gap.

This is a feasibility rejection, not a claim that no faster kernel can ever be
written. A future custom-kernel project would be a separately approved,
multi-day research phase with its own baseline and numerical gates.

## Locked performance gap

The profiler-disabled matched endpoint screen measured:

| Metric | Common-optimized dense | Dk64 |
| --- | ---: | ---: |
| Output throughput | 135.142470 tok/s | 141.564496 tok/s |
| Change versus dense | - | +4.752041% |

The locked 10% target is 148.656717 tok/s. Dk64 therefore needs another
5.009887% throughput over its current result.

For an intentionally optimistic bound, assume throughput scales inversely with
the profiled Dk64 decode kernel envelope and that every saved kernel millisecond
lies on the critical path. The observed 1350.836910 ms envelope would need to
fall to 1286.390219 ms, a reduction of 64.446691 ms or 4.770871%.

## What the runtime already fuses

The installed vLLM core is commit
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.

The non-spec decode path in
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` performs two
kernel-level stages:

1. `causal_conv1d_update` on the packed mixed-QKV tensor.
2. `fused_recurrent_gated_delta_rule_packed_decode`.

The packed recurrent Triton kernel directly computes Q/K/V offsets without
materialized Q/K/V splits, performs Q/K L2 normalization and Q scaling,
computes softplus decay and sigmoid beta, reads and updates recurrent state,
and writes output and final state. These are not remaining standalone
Slice/Concat/normalization/gating kernels available for a new fusion win.

Its tiling is `BK = next_power_of_2(K)`,
`BV = min(next_power_of_2(V), 32)`, with grid `(ceil(V/BV), B*HV)`.
For V=128, the FP32 state tile is 8 KiB at Dk64 and 16 KiB at Dk128. This
explains why reducing K can help the recurrent kernel, but it does not reveal
an omitted fusion boundary.

Source fingerprints:

| Source | SHA256 |
| --- | --- |
| `qwen_gdn_linear_attn.py` | `885c56f5e3584ab60bb4ce4b42cf8259e5e92c86ba13aa6128e6ed7a2a20d2b0` |
| `fused_recurrent.py` | `3a2a3c5245acba127af326f0b3f287b352776a8eb1e1f6e40c24a767fae3c2ef` |

## Profile-derived upper bound

The current-stack paired profile used four ranks, 30 stable decode steps,
concurrency 16, and the same frozen workload for dense and Dk64.

| Dk64 decode component | Aggregated rank-mean kernel time |
| --- | ---: |
| RecurrentGatedDeltaRule | 55.046810 ms |
| CausalConv1d | 27.899320 ms |
| Combined | 82.946130 ms |

Consequences:

- Eliminating the recurrent kernel completely would save only 55.046810 ms,
  still 9.399881 ms short of the optimistic 64.446691 ms requirement.
- Even after adding all causal-convolution time, a combined kernel would have
  to eliminate 77.697044% of the pair's total time.
- A real fusion can remove launches and some intermediate memory traffic, but
  it cannot remove the convolution, recurrent-state read/update/write, Q/K
  reductions, or output math. The required 77.7% elimination is therefore not
  a credible screen hypothesis.
- The dense pair is larger (63.584425 + 28.158165 = 91.742590 ms). A generally
  applicable fusion must also be applied to dense and cannot be credited as a
  DRRQR-specific gain; equal proportional improvement would save more absolute
  time on dense.

The projection shape audit is consistent with this bound. Across the same four
ranks and 30 decode steps, the GDN projection mean fell from 63.644 us at
Dk128 (`16x5120 * 4096x5120`) to about 57.709 us at Dk64
(`16x5120 * 3584x5120`). Dk64 already receives the expected benefit from the
smaller projection; there is no hidden Q/K projection left outside it.

Kernel sums are not additive critical-path attribution. Treating them as fully
additive above deliberately favors the hypothesis, so failing even this
optimistic bound is a strong rejection.

## Why the reference native operator is not the next action

The read-only reference tree
`/cache/cch/ops-transformer/attention/recurrent_gated_delta_rule` is commit
`0684247214bae5abb5cb6dc7d90b34a6ad401cdd`. Its native operator is not
called by the current service path, which calls the packed Triton function
above. Editing or tuning that reference operator would have no effect unless a
new integration layer were designed, implemented, compiled and numerically
qualified.

Per project policy, the reference tree remains unmodified. Any future operator
implementation must live in `/cache/cch/drror_vllm_ascend` and be loaded
explicitly by the plugin.

## Reproducibility pointers

- Endpoint evidence:
  `prefill-mc2-mixed-v1/dk64-vs-dense-common-optimized.json`,
  SHA256
  `d222aa3a39fa0933c8f2c72d0fba405aae6aa32fcc95e5823d3911092ffdffbd`.
- Paired profile:
  `prefill-mc2-mixed-v1/profile-current-v1/profile-comparison.v1.json`,
  SHA256
  `9cca1cd11a9eecd12c6e45988505d767c92ac6aa41fe3a857445a50d5b64115b`.
- The profile report itself records driver, prompt, service-manifest,
  per-variant input and output hashes.

No NPU run was performed for this feasibility audit. It used existing frozen
artifacts and read-only source inspection.
