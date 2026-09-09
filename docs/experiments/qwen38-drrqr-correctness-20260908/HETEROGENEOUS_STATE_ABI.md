# Heterogeneous state arena native ABI probe

Date: 2026-09-09
Status: stopped at a native dtype precondition; heterogeneous arena compatibility remains untested.

## Declared experiment

This was the first bounded work package from `EXTENDED_RUNTIME_RESEARCH.md`.
The hypothesis was that layer-local state tensors with widths
`[128, 32, 32, 64]` could be views into one continuous arena while preserving
the behavior of independently allocated state under eager execution and
`torch.npu.NPUGraph` replay. The control widths were `[64, 64, 64, 64]`.
Both layouts have the same aggregate Dk.

The predeclared matrix used batches 1 and 16, eight changing decode steps, and
required finite bitwise-identical outputs and full state, intact inter-layer
guards, and unchanged storage pointers. It was limited to one matrix with no
automatic retry. This is an ABI/storage probe only, not cache-manager
integration, throughput evidence, or task-accuracy evidence.

## Environment verification

- AWMCP route decision: `peer_direct`
- Target peer: `lgw_53d59c2f73ef4b81ba3cc0e00c2f8d35`
- Container: `drrqr_qwen38_energy_preflight_20260908`
- Image ID: `sha256:660ce23a83574fdcd28ba126b07615829c06770c06fbdf6533f03cc4d90acdb1`
- Physical devices 4, 5, 6, 7 mapped to container devices 0, 1, 2, 3
- Physical NPU4–7 had no running processes before launch.
- Repository and model mounts were read-only; `/drrqr-results` was writable.
- Python: `/usr/local/python3.12.13/bin/python3`, version 3.12.13
- Probe SHA256: `8efa6e0b8be89cd17b1c4f94cfe7b6dc5945a56c0722bc6b367c2ad4c1cda113`

## Result

The sole operation was `exec-0000000000000325`. It exited after 15.237 seconds
during the first warm-up invocation, before any layout/batch case completed.
The native tiler rejected the declared FP32 state:

```text
beta dtype and state dtype should be bfloat16.
Invalid dtypes.
```

The saved report therefore has `status: failed`, an empty `cases` array, and
must not be interpreted as a uniform or heterogeneous arena numerical failure.
It establishes that the FP32-state premise in the handed-off probe does not
match this loaded 910B native operator ABI. No criterion was relaxed and the
matrix was not retried.

Saved report:

```text
/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/heterogeneous-state-abi-v1/report.json
SHA256 cce5d919a16705f13744b8633452b76f176d45d1a1b244459a2f2e0f29fc8083
```

The report records the operator schema, Ascend GDN source hash, custom OPP
path, and hashes for the actually loaded custom op API, tiling/proto,
`libvllm_ascend_kernels.so`, and Python extension libraries.

After termination, physical NPU4–7 again showed no running processes and
`docker top` showed only the pre-existing bash process.

## Consequence

A BF16-state probe would be a corrected, separately declared experiment, not a
retry of this failed matrix. Before running it, confirm the serving path's
actual state and beta dtypes and update the arena byte accounting and report
labels accordingly. Passing such a probe would still prove only native
ABI/storage compatibility, not heterogeneous vLLM cache integration or an
end-to-end gain.

## Corrected BF16 v2 experiment

Source and runtime inspection after v1 showed that Qwen3 Next derives both
GDN cache dtypes from the BF16 model dtype when the mamba cache dtype settings
are `auto`. The loaded native tiler also rejected FP32 state. The probe was
therefore changed to BF16 state and beta, its schema was advanced to v2, and
the guard sentinel was changed to an exactly representable BF16 value.

The v2 matrix was declared as a separate experiment with the same four cases,
eight-step exactness criteria, and one-run/no-retry budget. Its sole operation
was `exec-0000000000000345`. It passed dtype tiling but failed during the first
uniform64/batch1 warm-up synchronization with error 507035: the vector-core
instruction read or wrote UB out of bounds. Four layer calls had been queued
before that synchronization, so v2 cannot identify the individual call.
No case completed and heterogeneous widths were not reached.

Saved v2 report:

```text
/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/heterogeneous-state-abi-v2/report.json
SHA256 68f278f97c39c29caea56fe7db52362f30c997b2eaff985831f8f487605fe63c
```

This is a uniform-control failure, not evidence that heterogeneous Dk is
invalid. A concrete probe defect is that changing the arena from FP32 to BF16
also changed the 64-element leading guard from 256 to 128 bytes. Every state
body in the tested shapes is a multiple of 512 bytes, so every arena view
retained that 128-byte base-pointer phase instead of the stronger alignment of
an independent NPU allocation. A future v3 must preserve allocation-grade
alignment, synchronize each diagnostic warm-up call separately, and record
pointer alignment before launching the native kernel. It is a new experiment,
not a retry of v2.

There is also source/binary behavior drift worth preserving: the exact v0.23
reference `recurrent_gated_delta_rule_tiling.cpp` accepts BF16 or FP32 state,
whereas the actually loaded tiler reported that beta and state must both be
BF16. Runtime behavior and loaded-library hashes are authoritative for this
experiment.

## Diagnostic v3 and passing v4

The v3 probe added 512-byte state-view alignment and per-call synchronization.
It still failed, but the new diagnostic output localized the failure to the
first independently allocated uniform64/batch1 Dk64 call, before any arena
view was used. This disproved the arena-alignment explanation for v2.

The decisive difference from the previously passing `gdn_golden_parity.py`
was custom-op initialization order. The golden harness sets compile mode,
loads the custom operator, sets the NPU device, and initializes Triton device
properties in that order. Earlier heterogeneous probes set the device before
custom-op loading and omitted Triton device-property initialization.

V4 preserved the BF16 service dtype and 512-byte arena alignment, adopted the
golden initialization order, and retained the stronger independent-sanity and
per-layer synchronization diagnostics. The single v4 matrix operation
`exec-000000000000035e` completed successfully in 34.189 seconds.

All four cases passed:

- uniform64, batch 1: 8/8 exact steps
- uniform64, batch 16: 8/8 exact steps
- heterogeneous `[128, 32, 32, 64]`, batch 1: 8/8 exact steps
- heterogeneous `[128, 32, 32, 64]`, batch 16: 8/8 exact steps

For every case, independently allocated state, arena eager state, and arena
NPUGraph state had zero pointer remainder modulo 512. Outputs and full state
were finite and bitwise identical across all eight changing-input/state-index
steps; guards remained intact and all tracked storage pointers were unchanged.
Uniform and heterogeneous layouts had identical logical state bytes because
both have aggregate Dk 256.

Passing v4 report:

```text
/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/heterogeneous-state-abi-v4/report.json
SHA256 af09119cd9ca7400dd0346378098c5f31d431cbec90deffe2b2fee745685fe45
```

Probe SHA256:

```text
5e1a726710f5a3ef3da81f651dbe4a61da833e97fd6e9061a65c97047be27982
```

This pass admits heterogeneous BF16 arena storage to the next integration
stage. It does not prove vLLM cache allocation/grouping, scheduler or graph-key
correctness, prefix-cache identity, end-to-end throughput, or task accuracy.
