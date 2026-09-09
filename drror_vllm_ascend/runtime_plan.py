"""Load either the legacy uniform plan or explicit layerwise manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Union

from .envs import DrrqrConfig
from .layerwise import LayerwiseDrrqrPlan, SCHEMA as LAYERWISE_SCHEMA
from .plan import DrrqrPlan, HASH, SCHEMA as UNIFORM_SCHEMA

RuntimePlan = Union[DrrqrPlan, LayerwiseDrrqrPlan]


def load_bound_plan(config: DrrqrConfig) -> RuntimePlan:
    path = Path(config.plan_path)
    if not path.is_absolute() or not HASH.fullmatch(config.plan_sha256):
        raise ValueError("DRRQR: absolute plan path and explicit SHA256 required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.plan_sha256:
        raise ValueError("DRRQR: plan hash mismatch")
    document = json.loads(raw)
    schema = document.get("schema") if isinstance(document, dict) else None
    if schema == UNIFORM_SCHEMA:
        plan: RuntimePlan = DrrqrPlan.load(config.plan_path, config.plan_sha256)
    elif schema == LAYERWISE_SCHEMA:
        plan = LayerwiseDrrqrPlan.load(config.plan_path, config.plan_sha256)
    else:
        raise ValueError("DRRQR: unsupported runtime plan schema")
    return plan


def load_runtime_plan(config: DrrqrConfig, vllm_config) -> RuntimePlan:
    plan = load_bound_plan(config)
    plan.validate_runtime(vllm_config)
    return plan
