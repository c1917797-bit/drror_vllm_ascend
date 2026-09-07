"""Qwen3.8-27B hybrid-model construction and packed-weight monkeypatch."""

from __future__ import annotations

import functools
import inspect
import logging

from ..diagnostics import emit_evidence
from ..envs import DrrqrConfig
from ..plan import load_runtime_plan

logger = logging.getLogger(__name__)


def install_model_patch(
    model_cls,
    config: DrrqrConfig,
    *,
    prepare_runtime=None,
) -> None:
    current = model_cls.__init__
    installed = getattr(current, "_drror_drrqr_config", None)
    identity = (config.plan_path, config.plan_sha256)
    if installed is not None:
        if installed != identity:
            raise RuntimeError("DRRQR: model class already patched with another plan")
        return
    if getattr(current, "_ascend_drrqr_wrapper", False):
        raise RuntimeError("DRRQR: conflicting in-tree monkeypatch is already installed")
    original_init = current
    original_load = model_cls.load_weights
    if "vllm_config" not in inspect.signature(original_init).parameters:
        raise RuntimeError("DRRQR: Qwen model constructor interface changed")
    if "weights" not in inspect.signature(original_load).parameters:
        raise RuntimeError("DRRQR: Qwen weight-loader interface changed")

    @functools.wraps(original_init)
    def initialize(self, *, vllm_config, prefix=""):
        plan = load_runtime_plan(config, vllm_config)
        if prepare_runtime is not None:
            prepare_runtime(plan)
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        self._drror_drrqr_plan = plan
        emit_evidence(
            config,
            "model_configured",
            component="model",
            plan_sha256=plan.digest,
            old_head_k_dim=plan.old_head_k_dim,
            target_head_k_dim=plan.target_head_k_dim,
            linear_layer_count=len(plan.keep_indices),
        )
        logger.info(
            "DRRQR model configured: plan=%s Dk=%d->%d layers=%d",
            plan.digest,
            plan.old_head_k_dim,
            plan.target_head_k_dim,
            len(plan.keep_indices),
        )

    @functools.wraps(original_load)
    def load_weights(self, weights):
        plan = getattr(self, "_drror_drrqr_plan", None)
        if plan is None:
            raise RuntimeError("DRRQR: model initialization did not bind a plan")
        transformed = plan.transform_weights(weights)
        result = original_load(self, transformed)
        sentinel = object()
        if next(transformed, sentinel) is not sentinel:
            raise RuntimeError("DRRQR: original loader did not consume all weights")
        target_count = 2 * len(plan.keep_indices)
        emit_evidence(
            config,
            "weight_load_complete",
            component="model",
            plan_sha256=plan.digest,
            target_tensor_count=target_count,
        )
        logger.info(
            "DRRQR weight loading complete: plan=%s target_tensors=%d",
            plan.digest,
            target_count,
        )
        return result

    initialize._drror_drrqr_config = identity
    model_cls.__init__ = initialize
    model_cls.load_weights = load_weights
