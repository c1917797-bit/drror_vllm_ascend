"""Reduced-shape prefill dispatch for the Ascend GDN implementation."""

from __future__ import annotations

import functools

from ..diagnostics import emit_evidence
from ..envs import DrrqrConfig


def install_fused_shape_guard(gdn_module, config: DrrqrConfig) -> None:
    """Keep the 128x128 fused path and route reduced shapes with metadata."""
    target = gdn_module.AscendGatedDeltaNetAttention
    original = target._chunk_gated_delta_rule_fused
    if getattr(original, "_drror_drrqr_wrapper", False):
        installed = getattr(original, "_drror_drrqr_config", None)
        identity = (config.plan_path, config.plan_sha256)
        if installed != identity:
            raise RuntimeError("DRRQR: GDN already patched with another plan")
        return

    hot_path_reported = False

    @functools.wraps(original)
    def dispatch(q, k, v, g, beta, initial_state, cu_seqlens, scale):
        nonlocal hot_path_reported
        if q.shape[-1] == 128 and v.shape[-1] == 128:
            return original(q, k, v, g, beta, initial_state, cu_seqlens, scale)

        context = gdn_module.get_forward_context()
        metadata = getattr(context, "attn_metadata", None)
        candidates = metadata.values() if isinstance(metadata, dict) else (metadata,)
        prebuilt_meta = None
        for candidate in candidates:
            if getattr(candidate, "prefill_query_start_loc", None) is not cu_seqlens:
                continue
            prefill = getattr(candidate, "non_spec_prefill_metadata", None)
            chunk = getattr(prefill, "chunk", None)
            if chunk is None:
                raise RuntimeError("DRRQR: matching prefill has no chunk metadata")
            if prebuilt_meta is not None and prebuilt_meta is not chunk:
                raise RuntimeError("DRRQR: ambiguous prefill chunk metadata")
            prebuilt_meta = chunk
        if prebuilt_meta is None:
            raise RuntimeError("DRRQR: no matching prebuilt prefill chunk metadata")

        if not hot_path_reported:
            emit_evidence(
                config,
                "reduced_gdn_hot_path",
                component="gdn",
                plan_sha256=config.plan_sha256,
                key_head_dim=int(q.shape[-1]),
                value_head_dim=int(v.shape[-1]),
                sequence_count=int(cu_seqlens.shape[0] - 1),
                token_count=int(q.shape[1]),
                backend="chunk_gated_delta_rule",
            )
            hot_path_reported = True

        output, final_state = gdn_module.chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state.transpose(-1, -2).contiguous(),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            prebuilt_meta=prebuilt_meta,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            scale=scale,
        )
        return output, final_state.transpose(-1, -2).contiguous()

    dispatch._drror_drrqr_wrapper = True
    dispatch._drror_drrqr_config = (config.plan_path, config.plan_sha256)
    target._chunk_gated_delta_rule_fused = staticmethod(dispatch)
