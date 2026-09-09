# Dk64 operator feasibility bound

Date: 2026-09-09. Status: bounded negative decision for the current
performance-first operator branch. The final performance × accuracy target is
not achieved.

## Admission and fairness rule

The current same-stack, profiler-disabled screen is:

- common-optimized dense Dk128: 135.142469917 output tok/s;
- Dk64: 141.564496153 output tok/s, +4.752041%;
- final 10% boundary: 148.656716909 output tok/s;
- missing candidate throughput: +5.009887% over the current Dk64 result.

At fixed work, that last step would require an idealized 4.770871% candidate
duration reduction if the dense control did not improve. Applied to the
1350.83691 ms Dk64 30-step profile envelope, this is 64.446696 ms. This is an
optimistic screening bound, not an assertion that profile envelope time maps
one-to-one to endpoint duration.

Any Qwen runtime optimization compatible with the unpruned model must also be
enabled for the dense control. Only the differential benefit can close the
DRRQR gap. A candidate-only switch is not a fair DRRQR result.

## Existing profile headroom

The validated current-stack pair is
`prefill-mc2-mixed-v1/profile-current-v1/profile-comparison.v1.json`,
SHA256
`9cca1cd11a9eecd12c6e45988505d767c92ac6aa41fe3a857445a50d5b64115b`.

In Dk64 decode, recurrent GDN is 55.04681 ms and causal convolution is
27.89932 ms, together 82.94613 ms or 6.140% of the envelope. Reaching the
64.446696 ms bound from only those kernels would require removing 77.70% of
their observed time before overlap and service overhead. The reference
`fused_causal_conv1d` and `inplace_fused_causal_conv1d` implementations in
ops-transformer explicitly support Ascend 950 only, not the Ascend 910B4 used
here. The installed recurrent and causal paths are already native Ascend
operators. Copying those sources into the plugin is therefore not a supported
or sufficiently large mechanism.

Rank-0 decode shape accounting confirms the two Qwen3.5 GDN input projections:

| Variant | QKVZ shape/count/sum | BA shape/count/sum |
| --- | --- | --- |
| dense | M16,K5120,N4096; 1440; 91.607280 ms | M16,K5120,N24; 1440; 17.196860 ms |
| Dk64 | M16,K5120,N3584 is mixed with 480 attention calls; estimated GDN-only 83.135500 ms | M16,K5120,N24; 1440; 17.239720 ms |

The Dk64 estimate subtracts the dense trace's 480-call N3584 attention sum
(27.804980 ms) from the Dk64 1920-call N3584 total (110.940480 ms). It is
diagnostic shape accounting, not critical-path additivity.

## Paired QKVZBA fusion micro screen

An older 910B4 branch contained a single-QKVZBA projection experiment, but its
Ascend patch depends on a separate vLLM model-construction and weight-loader
change. It cannot be copied independently into this plugin. More importantly,
the optimization is valid for both dense and Dk64, so its differential benefit
had to be measured before integration.

The frozen probe used physical NPU4 only (container logical0), BF16, M=16,
K=5120, BA N=24, and QKVZ N=4096/3584. Each route used 20 warmups, five
rotated NPU-graph timing blocks and 100 replays per block. Numerical acceptance
was finite output, relative L2 <= 2^-7 and normalized max absolute error <=
2^-6. The first invocation stopped before NPU work because the explicit
logical-device environment variable was absent; the preserved corrected
invocation added only `ASCEND_RT_VISIBLE_DEVICES=0`.

| Variant | Separate median | Fused median | Pair speedup | Projected 48×30 saving |
| --- | ---: | ---: | ---: | ---: |
| dense | 40.648398 us | 30.697401 us | 1.324164× | 14.329437 ms |
| Dk64 | 39.016199 us | 29.499600 us | 1.322601× | 13.703903 ms |

Both numerical cases passed. Dense relative L2 was 3.2521e-5 and Dk64 was
4.7757e-5. The Dk64-minus-dense projected saving is **-0.625534 ms**, versus
the predeclared required **+64.446696 ms**. The isolated timing is more
cache-friendly than the model trace and cannot predict endpoint speedup, but
its paired differential decisively shows that fusion is generic and slightly
favours dense in this screen.

Artifacts:

- report:
  `/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/gdn-projection-fusion-paired-micro-v1/report.json`
  (SHA256 `e2dc2780e4fac017ae694978b3bac9aec2087fd8df2c24c23d2bf08ea78fbf1d`);
- protocol SHA256:
  `9044ab7b3108721e38ff89ec10e0e248b79a09a0101338dbbffd2f299c46c136`;
- probe: `tools/gdn_projection_fusion_probe.py`
  (SHA256 `3073af24c769f90fe6ca9d0f51b2e6c85d6fdbd38abf66e69709d42e4151d245`).

## Decision

Reject QKVZBA fusion as a DRRQR gap-closing candidate. Do not implement its
model/weight-loader monkeypatch, build another wheel, or run endpoint tests for
it. It may be useful as a generic Qwen optimization in a different project,
but an equally optimized dense control removes that attribution.

Together with the rejected Dk32 boundary, NZ-weight screen, unsupported 950-only
causal-conv alternative, and the already-common mixed/prefill MC2 path, the
current profile exposes no credible Dk64-specific operator intervention that
can supply the missing approximately 5% endpoint throughput within the bounded
1–3 day plan. The scientifically valid result is therefore negative for the
current operator/pruning branch: retain Dk64 +4.752041% as the best same-stack
single-screen candidate, do not promote it to three-round/final accuracy, and
do not manufacture 10% by withholding generic optimizations from dense.

This closes only the current bounded branch. A new attempt needs a separately
declared algorithmic mechanism with a new headroom argument; it must not reopen
the rejected parameter sweep or change the final joint acceptance criteria.
