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
