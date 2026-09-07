"""Install DRRQR after the separately installed vLLM Ascend package."""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
from pathlib import Path

from ..diagnostics import emit_evidence
from ..envs import DrrqrConfig
from ..plan import file_sha256

EXPECTED_GDN_SHA256 = "d6ec29919268178f5bf6e70e689c1d273d04b1cb1d84dc94efa7bbbc35490816"
EXPECTED_QWEN_SHA256 = "04f3de5372973770d71c5c05f2e8a5e221ec52b62cca7a1e302d9201b61b8adc"
EXPECTED_RUNNER_SHA256 = "94d75dbeb5d23ab5b383cdc69114167392b967994bb21342cf748b733ba9a968"

logger = logging.getLogger(__name__)


class DrrqrRuntimeUnsupported(RuntimeError):
    pass


def _ensure_ascend_global_patch() -> None:
    try:
        import vllm_ascend
    except Exception as error:  # pragma: no cover - requires runtime installation
        raise DrrqrRuntimeUnsupported(
            "VLLM_ASCEND_DRRQR_ENABLE=1 requires vllm-ascend to be installed first"
        ) from error
    ensure = getattr(vllm_ascend, "_ensure_global_patch", None)
    if ensure is not None:
        ensure()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def apply_patches(config: DrrqrConfig) -> None:
    _ensure_ascend_global_patch()
    versions = {name: _version(name) for name in ("vllm", "vllm-ascend")}
    if any(value.split("+")[0] != "0.23.0" for value in versions.values()):
        raise DrrqrRuntimeUnsupported(f"DRRQR: this adapter requires vLLM/vLLM-Ascend 0.23.0: {versions}")
    try:
        # The official Qwen3.8-27B config declares
        # architecture=Qwen3_5ForConditionalGeneration. vLLM therefore serves
        # this Qwen3.8 checkpoint through Qwen3_5Model; the immutable plan and
        # config hash enforce the actual checkpoint identity before loading.
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
        from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
        from vllm_ascend.ops import gdn
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    except Exception as error:  # pragma: no cover - requires pinned runtime
        raise DrrqrRuntimeUnsupported(
            "DRRQR requires vLLM's official Qwen3.8-27B model path and Ascend GDN modules"
        ) from error

    required = (
        hasattr(gdn, "chunk_gated_delta_rule"),
        hasattr(gdn, "get_forward_context"),
        hasattr(gdn, "AscendGatedDeltaNetAttention"),
        hasattr(gdn.AscendGatedDeltaNetAttention, "_forward_core"),
        hasattr(QwenGatedDeltaNetAttention, "rearrange_mixed_qkv"),
        hasattr(NPUModelRunner, "_model_forward"),
    )
    if config.require_runtime_hooks and not all(required):
        raise DrrqrRuntimeUnsupported("DRRQR required Ascend GDN hooks are missing")
    runtime_sources = {}
    for name, source, expected in (
        ("ascend_gdn", inspect.getsourcefile(gdn), EXPECTED_GDN_SHA256),
        ("qwen_model", inspect.getsourcefile(Qwen3_5Model), EXPECTED_QWEN_SHA256),
        ("ascend_model_runner", inspect.getsourcefile(NPUModelRunner), EXPECTED_RUNNER_SHA256),
    ):
        if source is None or file_sha256(Path(source)) != expected:
            raise DrrqrRuntimeUnsupported(f"DRRQR: {name} source differs from the audited v0.23 image")
        runtime_sources[name] = {"path": source, "sha256": expected}

    def verify_worker():
        # General plugins run before Ascend worker patches. Check at model
        # construction, after the worker installs its real Qwen forward/core.
        core = QwenGatedDeltaNetAttention._forward_core
        core = getattr(core, "_drror_core_original", core)
        if core is not gdn.AscendGatedDeltaNetAttention._forward_core:
            raise DrrqrRuntimeUnsupported("DRRQR: Ascend Qwen worker core is not installed or has been replaced")
        if core.__globals__.get("chunk_gated_delta_rule") is not gdn.chunk_gated_delta_rule:
            raise DrrqrRuntimeUnsupported("DRRQR: Ascend core does not call the expected prefill module binding")
        emit_evidence(
            config,
            "worker_dispatch_verified",
            component="plugin",
            plan_sha256=config.plan_sha256,
            core_module=core.__module__,
            core_source=inspect.getsourcefile(core),
            qwen_class=QwenGatedDeltaNetAttention.__module__ + "." + QwenGatedDeltaNetAttention.__name__,
            runtime_sources=runtime_sources,
        )

    if config.capture_enable:
        from .capture import install_capture_patch
        from .capture_binding import (
            CaptureBinding,
            install_capture_model_patch,
            install_capture_runner_patch,
        )

        binding = CaptureBinding(config)
        install_capture_runner_patch(NPUModelRunner, config, binding)
        install_capture_model_patch(Qwen3_5Model, config, binding, verify_worker=verify_worker)
        install_capture_patch(QwenGatedDeltaNetAttention, gdn, config, binding=binding)
        emit_evidence(
            config,
            "runtime_patches_installed",
            component="plugin",
            mode="capture",
            source_config_sha256=config.source_config_sha256,
            calibration_sha256=config.calibration_sha256,
            vllm_version=_version("vllm"),
            vllm_ascend_version=_version("vllm-ascend"),
        )
        logger.info(
            "DRRQR capture monkeypatch installed: source=%s calibration=%s",
            config.source_config_sha256,
            config.calibration_sha256,
        )
        return

    from .gdn import install_decode_observer, install_prefill_patch
    from .model import install_model_patch

    install_prefill_patch(gdn, config)

    def prepare_runtime(plan):
        verify_worker()
        install_decode_observer(QwenGatedDeltaNetAttention, gdn, config)

    install_model_patch(Qwen3_5Model, config, prepare_runtime=prepare_runtime)
    emit_evidence(
        config,
        "runtime_patches_installed",
        component="plugin",
        mode="treatment",
        plan_sha256=config.plan_sha256,
        vllm_version=_version("vllm"),
        vllm_ascend_version=_version("vllm-ascend"),
    )
    logger.info(
        "DRRQR runtime monkeypatches installed: plan=%s vllm=%s vllm-ascend=%s",
        config.plan_sha256,
        _version("vllm"),
        _version("vllm-ascend"),
    )
