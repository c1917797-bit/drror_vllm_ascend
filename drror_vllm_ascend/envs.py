"""Environment contract for the DRRQR vLLM Ascend plugin."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off", ""}
HASH = re.compile(r"^[0-9a-f]{64}$")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class DrrqrConfig:
    enable: bool = False
    capture_enable: bool = False
    strict: bool = True
    require_runtime_hooks: bool = True
    plan_path: str = ""
    plan_sha256: str = ""
    evidence_file: str = ""
    capture_dir: str = ""
    capture_max_tokens: int = 2048
    capture_max_per_layer: int = 16
    source_config_sha256: str = ""
    calibration_sha256: str = ""


def get_config() -> DrrqrConfig:
    config = DrrqrConfig(
        enable=_bool_env("VLLM_ASCEND_DRRQR_ENABLE", False),
        capture_enable=_bool_env("VLLM_ASCEND_DRRQR_CAPTURE_ENABLE", False),
        strict=_bool_env("VLLM_ASCEND_DRRQR_STRICT", True),
        require_runtime_hooks=_bool_env(
            "VLLM_ASCEND_DRRQR_REQUIRE_RUNTIME_HOOKS",
            True,
        ),
        plan_path=os.getenv("VLLM_ASCEND_DRRQR_PLAN_PATH", "").strip(),
        plan_sha256=os.getenv("VLLM_ASCEND_DRRQR_PLAN_SHA256", "").strip(),
        evidence_file=os.getenv("VLLM_ASCEND_DRRQR_EVIDENCE_FILE", "").strip(),
        capture_dir=os.getenv("VLLM_ASCEND_DRRQR_CAPTURE_DIR", "").strip(),
        capture_max_tokens=_positive_int_env(
            "VLLM_ASCEND_DRRQR_CAPTURE_MAX_TOKENS", 2048
        ),
        capture_max_per_layer=_positive_int_env(
            "VLLM_ASCEND_DRRQR_CAPTURE_MAX_PER_LAYER", 16
        ),
        source_config_sha256=os.getenv(
            "VLLM_ASCEND_DRRQR_SOURCE_CONFIG_SHA256", ""
        ).strip(),
        calibration_sha256=os.getenv(
            "VLLM_ASCEND_DRRQR_CALIBRATION_SHA256", ""
        ).strip(),
    )
    if config.enable and config.capture_enable:
        raise ValueError("DRRQR treatment and capture modes are mutually exclusive")
    if not config.enable and not config.capture_enable:
        return config
    if config.evidence_file and not Path(config.evidence_file).is_absolute():
        raise ValueError("VLLM_ASCEND_DRRQR_EVIDENCE_FILE must be absolute")
    if config.capture_enable:
        if not config.capture_dir or not Path(config.capture_dir).is_absolute():
            raise ValueError("VLLM_ASCEND_DRRQR_CAPTURE_DIR must be absolute")
        for name, value in (
            ("VLLM_ASCEND_DRRQR_SOURCE_CONFIG_SHA256", config.source_config_sha256),
            ("VLLM_ASCEND_DRRQR_CALIBRATION_SHA256", config.calibration_sha256),
        ):
            if not HASH.fullmatch(value):
                raise ValueError(f"{name} must be an explicit lowercase SHA256")
        return config
    if not config.plan_path or not Path(config.plan_path).is_absolute():
        raise ValueError(
            "VLLM_ASCEND_DRRQR_PLAN_PATH must be an absolute path visible to every worker"
        )
    if not HASH.fullmatch(config.plan_sha256):
        raise ValueError(
            "VLLM_ASCEND_DRRQR_PLAN_SHA256 must be an explicit lowercase SHA256"
        )
    return config
