"""Install DRRQR after the separately installed vLLM Ascend package."""

from __future__ import annotations

import importlib.metadata
import logging

from ..diagnostics import emit_evidence
from ..envs import DrrqrConfig

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
    except Exception as error:  # pragma: no cover - requires pinned runtime
        raise DrrqrRuntimeUnsupported(
            "DRRQR requires vLLM's official Qwen3.8-27B model path and Ascend GDN modules"
        ) from error

    required = (
        hasattr(gdn, "chunk_gated_delta_rule"),
        hasattr(gdn, "get_forward_context"),
        hasattr(gdn, "AscendGatedDeltaNetAttention"),
        hasattr(gdn.AscendGatedDeltaNetAttention, "_chunk_gated_delta_rule_fused"),
    )
    if config.require_runtime_hooks and not all(required):
        raise DrrqrRuntimeUnsupported("DRRQR required Ascend GDN hooks are missing")

    if config.capture_enable:
        from .capture import install_capture_patch

        install_capture_patch(QwenGatedDeltaNetAttention, gdn, config)
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

    from .gdn import install_fused_shape_guard
    from .model import install_model_patch

    install_fused_shape_guard(gdn, config)
    install_model_patch(Qwen3_5Model, config)
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
