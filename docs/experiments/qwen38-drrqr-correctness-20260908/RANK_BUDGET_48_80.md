# Dk48/Dk80 rank-budget numerical admission

Date: 2026-09-09. Status: native eager numerical matrix passed; no performance or task-accuracy claim.

## Why this experiment

Mainline is DRRQR algorithm adaptation. The next layerwise hypothesis uses less extreme widths than {32,64,128}: retaining 80 coordinates in sensitive layers and 48 in others can keep an average Dk64 budget. This is not yet a selected or deployed plan. Equal dimension totals do not guarantee equal runtime cost or accuracy.

The single Dk32 negative endpoint measurement did not test Dk48/Dk80 or a mixed-layer allocation. Earlier blanket closure of every rank-budget option is superseded by the user's extended research authorization. This experiment does not launch a broad pruning-ratio sweep.

## Execution and result

One run, no retries, 600-second budget, physical NPU4 (logical 0). The dedicated container mappings, image, mounts, processes, port and host memory were checked before launch. Plugin disabled; original Ascend operators unmodified. Dk128/Dk64 are controls; Dk48/Dk80 are the new widths. One seed (42), batch 1/16, 8 decode steps, plus packed prefill lengths 63/64/65. FP32 recurrent state and BF16 Q/K/V/beta; TP4-local Hqk=4, Hv=12, Dv=128.

All 68 cases passed the unchanged golden gate. Decode uses rtol=0.003/atol=0.01 and prefill rtol=0.01/atol=0.01, on both complete state and output; finite checks also required. Raw report and per-width error magnitudes are bound in the adjacent summary. This is a representative screen, not exhaustive qualification.

- Dk48 decode: state max absolute 2.2351741790771484e-8; output max absolute 3.814697265625e-6.
- Dk80 decode: state max absolute 1.4901161193847656e-8; output max absolute 1.9073486328125e-6.
- Completed operation exec-00000000000005df in 72.193 seconds, exit 0.
- Raw report SHA256: 3e0ff767fcb28996a2e25309d3adcd99fff67432a08b1a3538dd7641b7ab6698.
- Raw report: /cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/rank-budget-48-80-golden-v1/report.json.

The raw plugin-version field is 0.1.6 from a repository-cwd metadata lookup. A separate /tmp-cwd lookup returned installed version 0.1.11 and site-packages origin; the correction is explicit in the summary and the raw report is not rewritten. This run disables the DRRQR plugin and does not establish installed-plugin model behavior. The original operator import source hash remains the frozen GDN hash.

After completion the container had only its bash process and physical NPU4-7 had no running processes. The CANN/HDK 32-padding allocator warning is preserved in the operation output; it did not cause a numerical gate failure.

## Admission boundary and next work

Dk48/Dk80 now have representative eager native numerical evidence. They are not yet admitted for model-performance claims: graph replay, actual projection/conv/cache integration, a bound selector plan, and endpoint testing are still required. The failed full-model all-Dk64 layerwise logprob control is not waived by this result.

The other direct algorithm hypothesis under examination is retained-key-energy beta compensation at Dk64. For a constant per-head retained energy r, a state-coordinate rescaling maps the projected recurrence to beta'=r*beta. This does not restore the omitted-state prediction term, and per-token r changes are not handled by beta scaling alone. A separate small NPU implementation gate is being prepared; no accuracy improvement is assumed.

Joint acceptance remains unmet: latest accepted single endpoint pair is +4.752041%, and no new LongBench-v2 or GSM8K result was produced here.
