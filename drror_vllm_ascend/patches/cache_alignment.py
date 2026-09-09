"""Bounded hybrid-cache alignment support for reduced Qwen3.8 GDN state."""

from __future__ import annotations

import functools
import math
from pathlib import Path

from ..diagnostics import emit_evidence
from ..envs import DrrqrConfig
from ..layerwise import LayerwiseDrrqrPlan
from ..plan import file_sha256
from ..runtime_plan import load_bound_plan

EXPECTED_ASCEND_MAMBA_CONFIG_SHA256 = (
    "ee38c68c27357a7d18f7e368e03a352bb312f79c3576431a64f688ce953716ac"
)
NATIVE_ALIGNMENT_ERROR = "Cannot align ssm_page_size and attn_page_size."


def calculate_padded_layout(
    *,
    ssm_page_size: int,
    conv_page_size: int,
    attn_single_token_k_page_size: int,
    attn_token_page_size: int,
    current_block_size: int | None,
    kernel_block_size: int = 128,
) -> dict[str, int | bool]:
    """Mirror the audited 910B layout, allowing its documented padding case."""

    values = (
        ssm_page_size,
        conv_page_size,
        attn_single_token_k_page_size,
        attn_token_page_size,
        kernel_block_size,
    )
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("DRRQR: cache page sizes and alignment must be positive integers")
    if current_block_size is not None and (
        type(current_block_size) is not int or current_block_size <= 0
    ):
        raise ValueError("DRRQR: current cache block size must be positive")
    aligned_block_size = kernel_block_size * math.ceil(
        ssm_page_size
        / (kernel_block_size * attn_single_token_k_page_size)
    )
    block_size = max(current_block_size or 0, aligned_block_size)
    attn_k_page_size = attn_single_token_k_page_size * block_size
    if attn_k_page_size < ssm_page_size:
        raise RuntimeError("DRRQR: rounded attention K page is smaller than the SSM state")
    attn_page_size = attn_token_page_size * block_size
    mamba_page_size_padded = attn_page_size + conv_page_size
    return {
        "kernel_block_size": kernel_block_size,
        "aligned_block_size": aligned_block_size,
        "block_size": block_size,
        "ssm_page_size": ssm_page_size,
        "conv_page_size": conv_page_size,
        "attn_single_token_k_page_size": attn_single_token_k_page_size,
        "attn_token_page_size": attn_token_page_size,
        "attn_k_page_size": attn_k_page_size,
        "attn_page_size": attn_page_size,
        "mamba_page_size_padded": mamba_page_size_padded,
        "native_exact_alignment": attn_k_page_size == ssm_page_size,
    }


def install_hybrid_cache_alignment_patch(config: DrrqrConfig) -> None:
    """Install a fail-closed continuation for the audited v0.23 910B assertion.

    The native Ascend function already rounds the attention block size up to a
    128-token boundary and documents padding the Mamba page when it is larger.
    It nevertheless asserts that the rounded K page exactly equals the SSM
    state. Of the supported aligned candidates, Dk112 exercises the documented
    padding case, while Dk96 and Dk64 are natively exact.
    This wrapper only continues that exact assertion for the pinned model and
    serving contract; all other errors and configurations still fail closed.
    """

    if not config.enable:
        return
    plan = load_bound_plan(config)

    import vllm.model_executor.models.config as model_configs
    from vllm.model_executor.models import ModelRegistry
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
    from vllm_ascend.patch.platform import patch_mamba_config

    source = getattr(patch_mamba_config, "__file__", None)
    if (
        source is None
        or file_sha256(Path(source)) != EXPECTED_ASCEND_MAMBA_CONFIG_SHA256
    ):
        raise RuntimeError(
            "DRRQR: Ascend hybrid-cache source differs from the audited v0.23 image"
        )

    target = model_configs.HybridAttentionMambaModelConfig
    current = target.verify_and_update_config
    installed = getattr(current, "_drror_cache_alignment_config", None)
    identity = (config.plan_path, config.plan_sha256)
    if installed is not None:
        if installed != identity:
            raise RuntimeError("DRRQR: cache alignment already patched with another plan")
        return
    original = current

    @functools.wraps(original)
    def verify_and_update(cls, vllm_config) -> None:
        padded = False
        try:
            original(vllm_config)
        except AssertionError as error:
            if str(error) != NATIVE_ALIGNMENT_ERROR:
                raise
            padded = True

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        plan.validate_runtime(vllm_config)
        if (
            model_config.architecture != "Qwen3_5ForConditionalGeneration"
            or str(model_config.dtype) not in ("torch.bfloat16", "bfloat16")
            or model_config.use_mla
            or parallel_config.tensor_parallel_size != 4
            or cache_config.enable_prefix_caching
            or cache_config.mamba_cache_mode != "none"
            or vllm_config.kv_transfer_config is not None
        ):
            raise RuntimeError(
                "DRRQR: cache alignment support is limited to the pinned "
                "Qwen3.8 BF16 TP4 no-prefix/no-KV-transfer contract"
            )

        model_cls, _ = ModelRegistry.resolve_model_cls(
            model_config.architecture,
            model_config=model_config,
        )
        shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
        dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
        if len(shapes) != 2 or len(dtypes) != 2:
            raise RuntimeError("DRRQR: expected Qwen GDN conv and temporal states")
        pages = [math.prod(shape) * get_dtype_size(dtype) for shape, dtype in zip(shapes, dtypes)]
        ssm_page_size, conv_page_size = max(pages), min(pages)
        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        attn_head_size = model_config.get_head_size()
        attn_single_token_k_page_size = (
            attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        )
        layout = calculate_padded_layout(
            ssm_page_size=ssm_page_size,
            conv_page_size=conv_page_size,
            attn_single_token_k_page_size=attn_single_token_k_page_size,
            attn_token_page_size=2 * attn_single_token_k_page_size,
            current_block_size=cache_config.block_size,
        )

        if padded:
            cache_config.block_size = layout["block_size"]
            cache_config.mamba_page_size_padded = layout["mamba_page_size_padded"]
        elif not layout["native_exact_alignment"]:
            raise RuntimeError(
                "DRRQR: native cache verifier returned without exact alignment "
                "or the audited padding assertion"
            )
        if (
            cache_config.block_size != layout["block_size"]
            or cache_config.mamba_page_size_padded
            != layout["mamba_page_size_padded"]
        ):
            raise RuntimeError("DRRQR: hybrid-cache layout differs from the audited calculation")

        emit_evidence(
            config,
            "hybrid_cache_alignment_verified",
            component="cache",
            plan_sha256=plan.digest,
            target_head_k_dim=plan.target_head_k_dim,
            layer_head_k_dims=(
                dict(plan.layer_dims)
                if isinstance(plan, LayerwiseDrrqrPlan)
                else None
            ),
            alignment_mode="ceil-pad" if padded else "native-exact",
            source_path=source,
            source_sha256=EXPECTED_ASCEND_MAMBA_CONFIG_SHA256,
            **layout,
        )

    verify_and_update._drror_cache_alignment_config = identity
    target.verify_and_update_config = classmethod(verify_and_update)
