"""Experimental load-time layout preparation for the audited Ascend GDN core.

No automatic installation: callers must run this after final weight loading and
before any graph capture. It preserves values and Parameter identity, but changes
storage. Never call it while inference, graph replay, training or offload is active.
"""
from __future__ import annotations


def prepare_conv_weight_layout(conv_module):
    """Make the existing core's weight.view(C, 4).transpose(0, 1) contiguous.

    This is not a projection, quantization or pruning operation. Only the physical
    layout changes from channel-major to tap-major. A separate integration gate
    must verify the model/loader ABI and all 48 intended GDN layer targets.
    """
    import torch

    weight = conv_module.weight
    if not isinstance(weight, torch.nn.Parameter):
        raise ValueError("expected a registered Parameter")
    if weight.requires_grad or weight.grad is not None:
        raise ValueError("layout preparation is inference-only")
    if weight.ndim != 3 or weight.shape[1:] != (1, 4):
        raise ValueError("expected Qwen GDN conv weight [channels, 1, 4]")
    if weight.shape[0] not in (1792, 2048, 2304, 2432, 2560):
        raise ValueError("expected audited TP4-local aligned Qwen GDN channels")
    if weight.dtype != torch.bfloat16 or weight.device.type not in ("cpu", "npu"):
        raise ValueError("expected CPU/NPU bfloat16 weight")
    if weight.storage_offset() != 0 or weight.untyped_storage().nbytes() != weight.numel() * weight.element_size():
        raise ValueError("weight must own an unoffset, exactly sized storage")
    before = list(weight.stride())
    existing_view = weight.view(weight.size(0), weight.size(2)).transpose(0, 1)
    if existing_view.is_contiguous():
        return {"changed": False, "shape": list(weight.shape), "before_stride": before,
                "after_stride": before, "operator_weight_contiguous": True}
    if not weight.is_contiguous():
        raise ValueError("unknown original weight layout")
    with torch.no_grad():
        packed = existing_view.contiguous().transpose(0, 1).unsqueeze(1)
        # Keep loader metadata, parameter registration and object identity intact.
        # Storage replacement is allowed only before graph capture.
        weight.data = packed
    operator_view = weight.view(weight.size(0), weight.size(2)).transpose(0, 1)
    if not operator_view.is_contiguous():
        raise RuntimeError("failed to prepare contiguous Ascend operator weight")
    return {"changed": True, "shape": list(weight.shape), "before_stride": before,
            "after_stride": list(weight.stride()), "operator_weight_contiguous": True}
