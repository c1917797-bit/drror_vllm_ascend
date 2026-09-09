"""Per-layer Qwen GDN constructor isolation for heterogeneous Dk plans."""

from __future__ import annotations

import functools

from ..layerwise import LayerwiseDrrqrPlan


def install_layerwise_constructor_patch(attention_cls, plan, identity) -> None:
    if not isinstance(plan, LayerwiseDrrqrPlan):
        return
    current = attention_cls.__init__
    installed = getattr(current, "_drror_layerwise_config", None)
    if installed is not None:
        if installed != identity:
            raise RuntimeError("DRRQR: layerwise constructor already uses another plan")
        return
    original = current

    @functools.wraps(original)
    def initialize(
        self,
        config,
        vllm_config,
        prefix="",
        gqa_interleaved_layout=False,
    ):
        local_config = plan.isolated_layer_config(config, prefix)
        original(
            self,
            local_config,
            vllm_config,
            prefix=prefix,
            gqa_interleaved_layout=gqa_interleaved_layout,
        )

    initialize._drror_layerwise_config = identity
    initialize._drror_layerwise_original = original
    attention_cls.__init__ = initialize


def validate_shared_mamba_layouts(tensors, layer_specs, mamba_spec_type, *, expected_max_dk=None) -> None:
    """Shared block IDs isolate states only when their byte layout agrees.

    The Ascend runner places all convolution slots before all recurrent slots.
    Different compact widths change both that offset and the per-slot stride.
    A global padded page size alone does not prevent cross-group overlap.
    """
    for tensor in tensors:
        names = [name for name in tensor.shared_by
                 if isinstance(layer_specs[name], mamba_spec_type)]
        layouts = {(tuple(layer_specs[name].shapes), tuple(layer_specs[name].dtypes))
                   for name in names}
        if len(layouts) > 1:
            raise RuntimeError(
                "DRRQR: heterogeneous GDN layers share an incompatible raw cache "
                "pool; private/grouped allocation must pass block-byte isolation "
                f"before serving: {names}"
            )
        if (names and len(names) != len(tensor.shared_by)
                and expected_max_dk is not None
                and any(layer_specs[name].shapes[-1][-1] != expected_max_dk for name in names)):
            raise RuntimeError(
                "DRRQR: compact GDN state shares an attention pool sized for a "
                "different maximum Dk; block-byte isolation is not established"
            )


def install_layerwise_cache_guard(runner_cls, config, plan) -> None:
    if not isinstance(plan, LayerwiseDrrqrPlan):
        return
    current = runner_cls._allocate_kv_cache_tensors
    identity = (config.plan_path, config.plan_sha256)
    installed = getattr(current, "_drror_layerwise_cache_identity", None)
    if installed is not None:
        if installed != identity:
            raise RuntimeError("DRRQR: cache guard already uses another plan")
        return
    original = current

    @functools.wraps(original)
    def allocate(self, kv_cache_config):
        from vllm.v1.kv_cache_interface import MambaSpec
        specs = self._get_layer_kv_cache_specs(kv_cache_config)
        validate_shared_mamba_layouts(kv_cache_config.kv_cache_tensors, specs, MambaSpec,
                                      expected_max_dk=plan.max_head_k_dim)
        return original(self, kv_cache_config)

    allocate._drror_layerwise_cache_identity = identity
    runner_cls._allocate_kv_cache_tensors = allocate
