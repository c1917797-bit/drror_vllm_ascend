# DRRQR vLLM Ascend plugin

Opt-in, fail-closed DRRQR state-reduction monkeypatch for the official
`Qwen/Qwen3.8-27B` checkpoint on vLLM Ascend.

The package is installed after vLLM and vLLM Ascend. It does not edit either
installed project:

```bash
cd /cache/cch/drror_vllm_ascend
git fetch --prune origin codex/qwen38-drrqr-v023-abi-20260907
git switch --detach FETCH_HEAD
PLUGIN_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain=v1)"
pip install --no-deps -e .
```

Plugin discovery is inert by default. Enabling it additionally requires an
immutable, hashed DRRQR plan.

The official Qwen3.8-27B configuration declares the reused
`Qwen3_5ForConditionalGeneration` architecture interface. The plugin accepts
only plans labelled `Qwen/Qwen3.8-27B` and validates the original checkpoint
configuration and safetensors-index hashes; it does not accept a Qwen3.5
checkpoint. The current immutable model contract is official revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

See [the piercing command runbook](docs/PIERCING_COMMANDS.md). No performance or
quality claim is made until the NPU A/B gates in that runbook pass.

The current target is `/cache/austinov/Qwen3.8-27B` on the audited vLLM and
vLLM-Ascend 0.23.0 image. This adapter verifies the target GDN/model source
hashes, observes the actual module-global prefill function, and independently
records a completed decode branch. Calibration uses a fresh, eager token-ID
capture with runtime-verified model and request provenance. The frozen v0.23
AscendC decode state copy-out transfers float32 rows and requires 32-byte row
alignment, so executable plans use Dk104, Dk88, and Dk64: exact reductions of
18.75%, 31.25%, and 50%. Nominal Dk102/Dk89 plans are retained only as
non-runnable paper-ratio evidence and fail closed before NPU execution.
