# Layerwise mixed-Dk energy-budget audit

Date: 2026-09-09

## Question

Could an accuracy-aware allocation keep sensitive linear-attention layers wide,
prune robust layers more aggressively, and preserve the same average Dk64
compute budget?

This is an offline algorithm screen. It does not claim downstream accuracy or
endpoint performance.

## Inputs and comparability

The audit uses the frozen energy-kernel plans for Dk32 and Dk64. The tool
requires the plans to have identical checkpoint hashes, capture-manifest hash,
layer topology, covered layers, and per-layer observation counts.

| Input | SHA256 |
| --- | --- |
| Dk32 plan | `47ca4d1eada4f31f1a9bf1e07b8f1ff050fec6451ae37579add9a9c004ca3e79` |
| Dk64 plan | `b1fcb727d3e58be66cc24148e30a5a1722d3bdc0914784f01cad02f11f741b8c` |
| Shared capture manifest | `3a6394c30c00a69f0daffb38a00e494089b4faf752b48937b9b12583a13638b0` |

Both plans cover all 48 linear-attention layers, with 16
rank/head observations per layer.

## Method

`tools/analyze_layerwise_energy_budget.py` performs exact dynamic programming
over per-layer choices Dk32, Dk64 and Dk128 while holding the total dimension
budget at `48 * 64 = 3072`. Dk128 has retained-energy fraction 1 by
definition. The score is the mean retained normalized Q/K energy across
layers.

The 1.0 percentage-point prototype threshold is an engineering admission
threshold, not a preregistered statistical threshold: an exploratory
calculation preceded the formal reproducible tool run. The result must
therefore be interpreted by its effect size and runtime constraints, not by a
claim of prospective hypothesis testing.

## Result

| Allocation | Mean retained-energy proxy |
| --- | ---: |
| Uniform Dk32 | 0.761766404 |
| Uniform Dk64 | 0.879619113 |
| Exact best mixed allocation at average Dk64 | 0.880589854 |

The exact optimum improves the proxy by only **0.097074 percentage points**.
It assigns 42 layers to Dk64, four layers (8,22,38,58) to Dk32 and two layers
(50,53) to Dk128. Only 6/48 layers change.

Formal report:

`/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/layerwise-energy-budget-v1/report.json`

Report SHA256:

`6d09689fe9088534a10ec10f12ccfaa9c3ccdfe3d84421891b141d288ecc63b6`

## Runtime feasibility

The current plugin and vLLM configuration contract have one global
`target_head_k_dim`:

- plan validation requires every layer to keep
  `num_key_heads * target_head_k_dim` coordinates;
- the reduced Hugging Face text configuration has one
  `linear_key_head_dim`;
- recurrent-cache shape derivation is patched from that single global value.

Therefore the mixed allocation is not directly loadable. Supporting it would
require a new per-layer model/cache/runtime contract rather than a selector-only
change.

It also has no arithmetic performance advantage over uniform Dk64: the summed
Q/K dimension across the 48 layers is exactly the same. It adds three kernel
shape classes and includes two Dk128 layers. Existing profiles already show
Dk32 does not improve endpoint throughput over Dk64. Thus this allocation
cannot credibly supply the additional 5.009887% throughput needed for the
locked target.

## Decision

Reject a mixed-layer runtime prototype in the current 1-3 day no-training
phase:

1. the best possible gain in the available selector proxy is only 0.097074 pp;
2. the historical Dk64 LongBench deficit is approximately 9.449 pp, although
   proxy and downstream score are not directly interchangeable;
3. the proposal requires a new heterogeneous-cache/runtime design;
4. it preserves total pruned arithmetic and has no credible path to the
   missing throughput.

This does not prove that layerwise sensitivity is useless after training. It
proves that the frozen no-training evidence does not justify implementing and
benchmarking this branch now.

## Verification

- Analyzer SHA256:
  `83641c7730d5da930d6cfe922bb54af78b21be100eb075fd18904a3d8f41b4db`.
- Unit tests: `3 passed in0.05s`.
- Full CPU suite after integration: `90 passed,14 warnings in11.23s`.
- The first test attempt contained an invalid two-layer budget example and
  failed as expected. The fixture was corrected to the exact identity
  `128 +32 +32 =3 *64`; the algorithm was unchanged.
- No NPU run was performed.
