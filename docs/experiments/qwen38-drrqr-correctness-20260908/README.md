# Qwen3.8 DRRQR GDN correctness gate

This is a new scientific iteration. It does not edit or reopen the rejected
qwen38-drrqr-20260908 decision.

The first gate checks the exact TP4-local Qwen3.8 GDN shapes against CPU
float32 references:

- query/key heads: 4
- value heads: 12
- value dimension: 128
- key dimensions: 128, 104, 88 and 64
- prefill and decode are tested separately
- prefill lengths 63, 64 and 65 straddle the 64-token chunk boundary
- decode reuses recurrent state for eight consecutive steps

Physical NPU4 is mapped to container NPU0 for this single-card diagnostic.
Physical NPU0-3 are outside the resource contract. NPU5-7 remain free until
TP4 model-level evidence is necessary.

The protocol was recorded in the worktree before any result was observed.
If Dk128 fails, the harness is invalid. If Dk128 and Dk64 pass while Dk104 or
Dk88 fails, the result is direct evidence of a reduced-shape numerical defect.
If every dimension passes, diagnosis advances to the packed projection,
causal-convolution, rearrange and cache path.

Artifacts are written outside git under:

    /cache/cch/state-reduction-qwen38-drrqr-correctness-20260908/

The runner is:

    python tools/gdn_golden_parity.py --help

The completed result and its interpretation are in `result-v2.json` and
`FINDINGS.md`. The full per-case report remains outside git at the artifact
path above and is bound by SHA-256 in `result-v2.json`.
