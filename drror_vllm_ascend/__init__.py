"""vLLM general-plugin entry point for the DRRQR Ascend adapter."""

from __future__ import annotations

import logging

from .diagnostics import configure_package_logging, emit_evidence
from .envs import DrrqrConfig, get_config

__version__ = "0.1.1"

logger = logging.getLogger(__name__)

__all__ = ["DrrqrConfig", "__version__", "get_config", "register"]


def register() -> None:
    """Install DRRQR monkeypatches when explicitly enabled.

    vLLM imports general plugins in its processes. Package discovery remains
    side-effect free unless VLLM_ASCEND_DRRQR_ENABLE is true.
    """

    config = get_config()
    if not config.enable and not config.capture_enable:
        return

    configure_package_logging()
    emit_evidence(
        config,
        "plugin_build_active",
        component="plugin",
        version=__version__,
        module=__file__,
        mode="treatment" if config.enable else "capture",
    )
    logger.info(
        "DRRQR plugin build active: version=%s module=%s plan=%s",
        __version__,
        __file__,
        config.plan_sha256 or "capture",
    )

    from .patches.bootstrap import apply_patches

    apply_patches(config)
