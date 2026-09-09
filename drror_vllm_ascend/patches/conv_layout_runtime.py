"""Opt-in load-time convolution layout hook for the frozen Qwen Ascend runtime."""
from __future__ import annotations

import functools
import inspect
from pathlib import Path
import re

from ..diagnostics import emit_evidence
from ..plan import file_sha256, validate_official_checkpoint_hashes
from .conv_layout import prepare_conv_weight_layout

LINEAR_LAYERS = frozenset(i for i in range(64) if (i + 1) % 4)


def validate_config(vllm_config):
    import torch

    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    text = model.hf_text_config
    expected = {"linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                "hidden_size": 5120, "num_hidden_layers": 64}
    if any(getattr(text, key, None) != value for key, value in expected.items()):
        raise RuntimeError("conv layout: unsupported Qwen dimensions")
    if text.linear_key_head_dim not in (32, 64, 96, 112, 128):
        raise RuntimeError("conv layout: unsupported key dimension")
    expected_types = ["linear_attention" if i in LINEAR_LAYERS else "full_attention" for i in range(64)]
    if list(text.layer_types) != expected_types:
        raise RuntimeError("conv layout: unsupported hybrid layer topology")
    if parallel.tensor_parallel_size != 4 or parallel.pipeline_parallel_size != 1:
        raise RuntimeError("conv layout: requires TP4/PP1")
    if model.dtype != torch.bfloat16 or model.quantization is not None:
        raise RuntimeError("conv layout: requires unquantized bfloat16")
    if vllm_config.lora_config is not None or vllm_config.speculative_config is not None:
        raise RuntimeError("conv layout: LoRA/speculative decoding is not validated")
    if model.enable_sleep_mode:
        raise RuntimeError("conv layout: sleep mode is not validated")
    source = Path(model.model)
    validate_official_checkpoint_hashes(
        file_sha256(source / "config.json"),
        file_sha256(source / "model.safetensors.index.json"))
    return text.linear_key_head_dim


def prepare_model(model, gdn_cls, *, expected_dk, require_npu=True):
    """Validate every target before repacking, then prove exact parameter values."""
    import torch

    targets = []
    seen = set()
    for name, module in model.named_modules():
        if not isinstance(module, gdn_cls):
            continue
        match = re.search(r"(?:^|\.)layers\.(\d+)\.linear_attn$", name)
        if match is None:
            raise RuntimeError(f"conv layout: unrecognized GDN target {name}")
        layer = int(match[1])
        weight = module.conv1d.weight
        expected_shape = (2 * 4 * expected_dk + 12 * 128, 1, 4)
        if layer in seen or layer not in LINEAR_LAYERS:
            raise RuntimeError("conv layout: duplicate or unexpected layer")
        if (module.tp_size != 4 or module.num_k_heads != 16 or module.num_v_heads != 48
                or module.head_k_dim != expected_dk or module.head_v_dim != 128
                or tuple(weight.shape) != expected_shape or weight.dtype != torch.bfloat16):
            raise RuntimeError(f"conv layout: unexpected local shape at {name}")
        if require_npu and weight.device.type != "npu":
            raise RuntimeError("conv layout: weights are not resident on NPU")
        if weight.requires_grad or weight.grad is not None:
            raise RuntimeError("conv layout: training parameter is not supported")
        if not weight.is_contiguous():
            raise RuntimeError("conv layout: target was already repacked or has an unknown layout")
        seen.add(layer)
        targets.append((name, layer, module))
    if seen != LINEAR_LAYERS or len(targets) != 48:
        raise RuntimeError("conv layout: expected all 48 GDN layers before mutation")
    if len({m.conv1d.weight.data_ptr() for _, _, m in targets}) != 48:
        raise RuntimeError("conv layout: aliased convolution parameters")
    records = []
    for name, layer, module in targets:
        parameter = module.conv1d.weight
        reference = parameter.detach().cpu().clone()
        if not bool(torch.isfinite(reference).all()):
            raise RuntimeError(f"conv layout: nonfinite loaded weight at {name}")
        result = prepare_conv_weight_layout(module.conv1d)
        actual = parameter.detach().cpu()
        if parameter is not module.conv1d.weight or not torch.equal(reference, actual):
            raise RuntimeError(f"conv layout: loaded values changed at {name}")
        records.append({"name": name, "layer": layer, **result, "exact_values_verified": True})
    return records


def install_runner_patch(runner_cls, model_cls, gdn_cls, config, verify_worker):
    original = runner_cls.load_model
    identity = (config.enable, config.plan_sha256, config.evidence_file)
    installed = getattr(original, "_drror_conv_layout_config", None)
    if installed is not None:
        if installed != identity:
            raise RuntimeError("conv layout: runner already bound to another configuration")
        return
    if list(inspect.signature(original).parameters) != ["self"]:
        raise RuntimeError("conv layout: runner load_model ABI changed")

    @functools.wraps(original)
    def load_model(self):
        from vllm.model_executor.offloader.base import NoopOffloader, get_offloader

        if getattr(self, "_drror_conv_layout_ready", False):
            raise RuntimeError("conv layout: model reload after preparation is not validated")
        dk = validate_config(self.vllm_config)
        if type(get_offloader()) is not NoopOffloader:
            raise RuntimeError("conv layout: weight offloading is not validated")
        result = original(self)
        if type(get_offloader()) is not NoopOffloader:
            raise RuntimeError("conv layout: loader enabled weight offloading")
        verify_worker()
        model = self.get_model()
        if not isinstance(model, model_cls):
            raise RuntimeError("conv layout: runner did not load the audited Qwen model")
        records = prepare_model(model, gdn_cls, expected_dk=dk)
        self._drror_conv_layout_ready = True
        emit_evidence(
            config, "conv_weight_layout_prepared", component="performance",
            optimization="conv-weight-tap-major-v1", pruning_enabled=config.enable,
            plan_sha256=config.plan_sha256 if config.enable else None,
            key_head_dim=dk, layer_count=len(records), layers=records,
            stage="after_runner_load_model_before_graph_capture",
            integration_source=__file__, integration_sha256=file_sha256(Path(__file__)),
            layout_source=inspect.getsourcefile(prepare_conv_weight_layout),
            layout_sha256=file_sha256(Path(inspect.getsourcefile(prepare_conv_weight_layout))),
        )
        return result

    load_model._drror_conv_layout_config = identity
    runner_cls.load_model = load_model
