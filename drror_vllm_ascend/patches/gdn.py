"""Reduced-shape prefill dispatch for the Ascend GDN implementation."""

from __future__ import annotations

import contextvars
import functools
import inspect

from ..diagnostics import emit_evidence
from ..envs import DrrqrConfig

SUPPORTED_REDUCED_KEY_DIMS = (64, 88, 104)


def install_prefill_patch(gdn_module, config: DrrqrConfig) -> None:
    """Observe the actual v0.23 module-global prefill call without changing it.

    v0.23 has no _chunk_gated_delta_rule_fused. Its Ascend core calls this
    module binding directly with state already in [N,Nv,Dk,Dv] layout and
    prebuilt per-request metadata. Preserve both the inputs and outputs.
    """
    original = gdn_module.chunk_gated_delta_rule
    identity = (config.plan_path, config.plan_sha256)
    installed = getattr(original, "_drror_prefill_config", None)
    if installed is not None:
        if installed != identity:
            raise RuntimeError("DRRQR: prefill already patched with another plan")
        return
    signature = inspect.signature(original)
    required = {"q", "k", "v", "initial_state", "cu_seqlens", "prebuilt_meta", "use_qk_l2norm_in_kernel"}
    if not required.issubset(signature.parameters):
        raise RuntimeError("DRRQR: v0.23 chunk_gated_delta_rule ABI changed")
    reported = False

    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        nonlocal reported
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        q, k, v = values["q"], values["k"], values["v"]
        if q.shape[-1] == 128:
            return original(*args, **kwargs)
        if q.shape[-1] not in SUPPORTED_REDUCED_KEY_DIMS or k.shape[-1] != q.shape[-1] or v.shape[-1] != 128:
            raise RuntimeError("DRRQR: unexpected reduced prefill dimensions")
        starts, prebuilt = values["cu_seqlens"], values["prebuilt_meta"]
        state = values["initial_state"]
        if (
            starts is None
            or prebuilt is None
            or state is None
            or tuple(state.shape[-2:]) != (q.shape[-1], v.shape[-1])
            or state.shape[0] != starts.shape[0] - 1
            or values.get("head_first", False)
            or not values["use_qk_l2norm_in_kernel"]
        ):
            raise RuntimeError("DRRQR: prefill requires original state layout, normalization and chunk metadata")
        # No transposition, Q/K normalization, scale change, tensor copy, or
        # fallback is introduced by this v0.23 evidence hook.
        result = original(*args, **kwargs)
        if not reported:
            emit_evidence(
                config,
                "reduced_gdn_hot_path",
                component="gdn",
                plan_sha256=config.plan_sha256,
                key_head_dim=int(q.shape[-1]),
                value_head_dim=int(v.shape[-1]),
                sequence_count=int(starts.shape[0] - 1),
                token_count=int(q.shape[1]),
                backend="chunk_gated_delta_rule",
                query_shape=list(q.shape),
                key_shape=list(k.shape),
                value_shape=list(v.shape),
                initial_state_shape=list(state.shape),
                prebuilt_metadata_forwarded=True,
                state_layout="N,Nv,Dk,Dv",
                completed_python_call=True,
                note="Kernel return observed; asynchronous device completion requires request success",
            )
            reported = True
        return result

    dispatch._drror_prefill_config = identity
    gdn_module.chunk_gated_delta_rule = dispatch


def install_decode_observer(gdn_cls, gdn_module, config: DrrqrConfig) -> None:
    """Observe an unchanged v0.23 decode branch after worker patches install.

    Do not replace torch.ops: retain the original core/operator and graph path.
    Q/K shapes come from the original rearrange result. The recorded branch is
    the audited non-speculative AscendC decode branch after its core returns.
    """
    original_core = gdn_cls._forward_core
    identity = (config.plan_path, config.plan_sha256)
    installed = getattr(original_core, "_drror_decode_config", None)
    if installed is not None:
        if installed != identity:
            raise RuntimeError("DRRQR: decode observer already bound to another plan")
        return
    original_rearrange = gdn_cls.rearrange_mixed_qkv
    active = contextvars.ContextVar("drrqr_decode_shape_observation", default=None)
    reported = False

    @functools.wraps(original_rearrange)
    def rearrange(self, mixed_qkv):
        if reported:
            return original_rearrange(self, mixed_qkv)
        result = original_rearrange(self, mixed_qkv)
        observation = active.get()
        if observation is not None and result[0] is not None:
            q, k, v = result
            if q.shape[1] == observation["decode_tokens"]:
                observation.update(query_shape=list(q.shape), key_shape=list(k.shape), value_shape=list(v.shape))
        return result

    @functools.wraps(original_core)
    def core(self, mixed_qkv, b, a, core_attn_out):
        nonlocal reported
        if reported:
            return original_core(self, mixed_qkv, b, a, core_attn_out)
        metadata = getattr(gdn_module.get_forward_context(), "attn_metadata", None)
        candidate = metadata.get(self.prefix) if isinstance(metadata, dict) else metadata
        if candidate is None or getattr(candidate, "num_decodes", 0) <= 0:
            return original_core(self, mixed_qkv, b, a, core_attn_out)
        if getattr(candidate, "spec_sequence_masks", None) is not None:
            raise RuntimeError("DRRQR: speculative decode is outside the bound runtime contract")
        observation = {"decode_tokens": int(candidate.num_decode_tokens)}
        token = active.set(observation)
        try:
            result = original_core(self, mixed_qkv, b, a, core_attn_out)
        finally:
            active.reset(token)
        state = self.kv_cache[1]
        qshape = observation.get("query_shape")
        if not qshape or qshape[-1] not in SUPPORTED_REDUCED_KEY_DIMS or state.shape[-1] != qshape[-1]:
            raise RuntimeError("DRRQR: completed decode did not expose expected reduced Q/K/state shapes")
        emit_evidence(
            config,
            "reduced_gdn_decode_branch",
            component="gdn",
            plan_sha256=config.plan_sha256,
            backend="npu_recurrent_gated_delta_rule",
            prefix=str(self.prefix),
            query_shape=observation["query_shape"],
            key_shape=observation["key_shape"],
            value_shape=observation["value_shape"],
            state_shape=list(state.shape),
            state_layout="N,Nv,Dv,Dk",
            key_head_dim=int(qshape[-1]),
            value_head_dim=int(state.shape[-2]),
            token_count=observation["decode_tokens"],
            completed_python_call=True,
            observation="unchanged audited v0.23 core decode branch returned; torch.ops was not replaced",
        )
        reported = True
        return result

    core._drror_decode_config = identity
    core._drror_core_original = original_core
    gdn_cls._forward_core = core
    gdn_cls.rearrange_mixed_qkv = rearrange
