# DRRQR vLLM Ascend plugin

Opt-in, fail-closed DRRQR state-reduction monkeypatch for the official
`Qwen/Qwen3.8-27B` checkpoint on vLLM Ascend.

The package is installed after vLLM and vLLM Ascend. It does not edit either
installed project:

```bash
cd /cache/cch/drror_vllm_ascend
git pull --ff-only origin main
pip install --no-deps -e .
```

Plugin discovery is inert by default. Enabling it additionally requires an
immutable, hashed DRRQR plan.

The official Qwen3.8-27B configuration declares the reused
`Qwen3_5ForConditionalGeneration` architecture interface. The plugin accepts
only plans labelled `Qwen/Qwen3.8-27B` and validates the original checkpoint
configuration hash; it does not accept a Qwen3.5 checkpoint.

See [the piercing command runbook](docs/PIERCING_COMMANDS.md). No performance or
quality claim is made until the NPU A/B gates in that runbook pass.
