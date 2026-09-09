"""vLLM general-plugin entry point for the DRRQR Ascend adapter."""

from __future__ import annotations

import logging

from .diagnostics import configure_package_logging, emit_evidence
from .envs import DrrqrConfig, get_config

__version__ = "0.1.11"

logger = logging.getLogger(__name__)

__all__ = ["DrrqrConfig", "__version__", "get_config", "register"]


def register() -> None:
    """Install DRRQR monkeypatches when explicitly enabled.

    vLLM imports general plugins in its processes. Package discovery remains
    side-effect free unless treatment, capture, or a performance opt-in is true.
    """

    config = get_config()
    if not config.enable and not config.capture_enable and not config.conv_layout and not config.prefill_mc2:
        return

    configure_package_logging()
    emit_evidence(
        config,
        "plugin_build_active",
        component="plugin",
        version=__version__,
        module=__file__,
        mode="treatment" if config.enable else ("capture" if config.capture_enable else "performance"),
        conv_layout=config.conv_layout,
        prefill_mc2=config.prefill_mc2,
        prefill_mc2_mixed=config.prefill_mc2_mixed,
    )
    logger.info(
        "DRRQR plugin build active: version=%s module=%s plan=%s",
        __version__,
        __file__,
        config.plan_sha256 or ("capture" if config.capture_enable else "performance"),
    )

    from .patches.bootstrap import apply_patches

    apply_patches(config)
