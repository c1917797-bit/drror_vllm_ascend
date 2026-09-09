# Retained-key beta compensation: bounded algorithm hypothesis

Date: 2026-09-09. Experimental energy-kernel/DRRQR-family adaptation, not paper-exact StrongRRQR and not a result of training.

## Motivation and exact boundary

The actual Ascend core normalizes reduced Q/K then uses scale=Dk**-0.5. Merely changing that output scale is not an established repair: the actual Ascend forward reshapes to individual value-head rows, applies self.norm(core_attn_out,z), then flattens and projects. The Qwen module constructs RMSNormGated(head_v_dim, eps=layer_norm_epsilon, norm_before_gate=True). Positive per-head scaling cancels only when epsilon is zero and before cross-head mixing; finite epsilon and quantization preclude exact invariance in general.

A separate mechanism is recurrence update strength after retained keys are renormalized. In a projected recurrence that ignores the deleted-state prediction term, write k_ret=sqrt(rk)*u with rk>0 constant per head. With T=sqrt(rk)*S_ret,

```text
S_ret_next = alpha*S_ret + beta*(v-alpha*S_ret@k_ret)*k_ret^T
T_next     = alpha*T     + (rk*beta)*(v-alpha*T@u)*u^T
```

Thus a standard recurrent operator can implement this projected state representation with beta'=rk*beta. Nonzero initial state must be rescaled consistently. For q_ret=sqrt(rq)*q_unit, the projected output is sqrt(rq/rk)*(T_next@q_unit), additionally accounting for the runtime attention-scale ratio.

This is NOT dense recovery. The actual dense residual also contains alpha*S_deleted@k_deleted, which this transformation omits. If rk varies by token, the transformed cache needs an extra sqrt(rk_t/rk_previous) scale in alpha; beta-only compensation is then approximate. CPU unit tests include an explicit two-token counterexample and finite-epsilon RMSNorm non-invariance.

## Real-capture assumption check

One bounded device-free run examined the existing Dk64 energy plan on six layers (0,12,25,38,50,62), four ranks and all 16 calibration requests: 384 files, 96 heads, 786432 captured token rows. Each read is checked against source/calibration/rank/layer/token metadata and its file hash is saved. This is an offline analysis of actual NPU-origin post-convolution Q/K captures, not an NPU execution or an actual recurrence replay.

- Mean across head-mean K retention: 0.8611182769513729.
- Head-mean K retention range: 0.6374316819399106 to 0.98735366918817.
- Within-head token K-retention CV: median 0.05525046939842109, P95 0.15765783913140283.
- Per-head held-out token absolute relative-deviation P95 from the training scalar: median 0.1036790164042864, P95 across heads 0.2959072099467396.

These observations contradict an exact constant-r interpretation of the real captures. They do not prove the proposed approximation useful or useless. The fixed coordinate plan itself used all 16 requests, so the scalar 12/4 split is NOT held-out selector validation or generalization evidence.

Raw report SHA256: 8882d60948ed773bb200303a1878d43441bda105bac4090e776b34ad684ceb7b.
Raw path: /cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/retained-key-stats-v1/run-v2/report.json.

First launch failed before data reading because Torch auto-loaded an NPU backend in a device-free container. The failure is preserved. The corrected second launch set TORCH_DEVICE_BACKEND_AUTOLOAD=0, without adding drivers or NPU devices; it completed in 12.014 seconds.

## Current implementation gate

tools/gdn_retained_energy_probe.py tests synthetic Dk64 input/state only. Four cases combine batch1/16 with r=1 or rk=[0.5,0.75,0.875,1], expanded to the 12 value heads. Inside NPUGraph it computes beta_base(BF16)->FP32 multiply r->BF16, then calls the unchanged native recurrent operator.

Admission requires the predeclared golden rtol/atol, relative-L2<=2^-7 and normalized max<=2^-6 for output and full state; finite eager/graph byte equality; unchanged unselected slots and captured storage metadata; exact intermediate beta agreement; and r=1 equivalence to the direct native operator. One 8-step matrix, no retries, physical4 only, 600-second budget. CPU tests passed 25 cases before NPU launch.

Result: all four cases and all 32 measured steps passed. Operation exec-000000000000060f completed in 28.488 seconds, exit 0. Maximum output relative-L2 was 4.343870273029215e-5, and maximum full-state relative-L2 was 6.322533191095341e-8. Every declared byte, slot, beta-intermediate and storage-metadata check passed. This is implementation correctness for the stated synthetic decode workload only.

Raw NPU report SHA256: d548c37e94594082819628e0052acaca8e5df94f304dd3969e74c56523dc6baf.
Raw path: /cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/retained-key-beta-npu-v1/report.json.
The adjacent retained-key-beta-npu.v1.summary.json includes actual mapped library hashes, source hashes, schema and per-case errors. Mapped library identities are not an ACLNN symbol-resolution proof. No DRRQR patch modules were loaded. After completion the container held only bash and physical NPU4-7 had no running processes.

## Mainline next decision

Even a passing synthetic gate does not admit this as an accuracy-preserving model candidate. Before an endpoint claim, collect or replay actual V/g/beta/state on disjoint non-evaluation calibration requests and test the changed recurrence's effect, then check consistent prefill/decode application. Only a justified model candidate should pay the model-screening cost. A generic beta multiply may add overhead; a fused implementation is an option only if it serves this concrete DRRQR candidate and passes its own NPU numerical gate.

No new throughput, LongBench-v2 or GSM8K result has been produced. The joint >=10% same-stack throughput/no-accuracy-decline objective is unchanged.
