"""Large-prefill MatMul/AllReduce fusion with an explicit mixed-batch opt-in."""
from __future__ import annotations

import contextvars
import functools
import inspect
from pathlib import Path
import re

import torch

from ..diagnostics import emit_evidence
from ..plan import file_sha256

# Integer row count only; no tensors or global runtime state escape a forward call.
_PREFILL_ROWS = contextvars.ContextVar("drrqr_prefill_mc2_rows", default=0)
_PREFILL_PHASE = contextvars.ContextVar("drrqr_prefill_mc2_phase", default="none")
LINEAR_SOURCE_SHA256 = "3f14a92019a6612158ba41a26c358c8f164cbda66b743c518a5e4019694b5dc3"
TARGET = re.compile(r"(?:^|\.)layers\.(\d+)\.(mlp\.down_proj|linear_attn\.out_proj|self_attn\.o_proj)$")


def prefill_rows(context, rows, *, allow_mixed=False):
    """Allow large unpadded prefill-containing matrices; pure decode is excluded."""
    if type(rows) is not int or rows < 2048:
        return 0
    if context is None or getattr(context, "in_profile_run", False) or getattr(context, "capturing", False):
        return 0
    if getattr(getattr(context, "cudagraph_runtime_mode", None), "name", None) != "NONE":
        return 0
    metadata = getattr(context, "attn_metadata", None)
    if not isinstance(metadata, dict) or not metadata:
        return 0
    for item in metadata.values():
        counts = [getattr(item, name, None) for name in
                  ("num_prefills", "num_decodes", "num_decode_tokens", "num_actual_tokens")]
        if not all(type(value) is int for value in counts):
            return 0
        if not (counts[0] > 0 and counts[3] == rows):
            return 0
        if allow_mixed:
            # Frozen no-speculation runtime: each decode sequence contributes one row.
            if not (0 <= counts[1] == counts[2] < rows and counts[0] <= rows - counts[2]):
                return 0
        elif counts[1] != 0 or counts[2] != 0:
            return 0
    return rows


def collect_targets(model, linear_cls, *, require_npu=True, get_format=None):
    """Validate all128 row-parallel targets before attaching any bindings."""
    import torch

    found = {}
    for name, layer in model.named_modules():
        match = TARGET.search(name)
        if match is None:
            continue
        index, kind = int(match[1]), match[2]
        key = (index, kind)
        expected_kind = "self_attn.o_proj" if (index + 1) % 4 == 0 else "linear_attn.out_proj"
        if index not in range(64) or kind not in ("mlp.down_proj", expected_kind) or key in found:
            raise RuntimeError("prefill MC2: unexpected or duplicate target " + name)
        k = 4352 if kind == "mlp.down_proj" else 1536
        if not isinstance(layer, linear_cls):
            raise RuntimeError("prefill MC2: target is not the audited Ascend row linear")
        weight = layer.weight
        if (tuple(weight.shape) != (5120, k) or weight.dtype != torch.bfloat16
                or not weight.is_contiguous() or weight.requires_grad or weight.grad is not None
                or layer.tp_size != 4 or not layer.input_is_parallel or not layer.reduce_results
                or layer.bias is not None or layer.custom_op is not None
                or getattr(layer, "out_dtype", None) is not None):
            raise RuntimeError("prefill MC2: unsupported row-linear configuration at " + name)
        if require_npu and (weight.device.type != "npu" or get_format(weight) != 2):
            raise RuntimeError("prefill MC2: expected resident BF16 ND weight")
        if hasattr(layer, "_drrqr_prefill_mc2"):
            raise RuntimeError("prefill MC2: target already bound")
        found[key] = (name, layer, k, 2048 if kind == "mlp.down_proj" else 8192)
    expected = {(i, "mlp.down_proj") for i in range(64)} | {
        (i, "self_attn.o_proj" if (i + 1) % 4 == 0 else "linear_attn.out_proj") for i in range(64)}
    if set(found) != expected:
        raise RuntimeError("prefill MC2: expected all64 down projections and64 attention output projections")
    return list(found.values())


def install_dispatch(runner_cls, linear_cls, config, *, get_context, fused, make_opaque=None):
    """Install inert-until-bound hooks; dependency injection supports CPU routing tests."""
    identity = (config.enable, config.plan_sha256, config.evidence_file, config.prefill_mc2_mixed)
    original_forward = linear_cls.forward
    prior = getattr(original_forward, "_drrqr_prefill_mc2_identity", None)
    if prior is not None:
        if prior != identity:
            raise RuntimeError("prefill MC2: another configuration is installed")
        return original_forward._drrqr_prefill_mc2_bindings
    bindings = {}
    original_model_forward = runner_cls._model_forward
    signature = inspect.signature(original_model_forward)
    if "num_tokens_padded" not in signature.parameters:
        raise RuntimeError("prefill MC2: runner ABI changed")

    @functools.wraps(original_model_forward)
    def model_forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        context = get_context()
        rows = prefill_rows(context, bound.arguments["num_tokens_padded"], allow_mixed=config.prefill_mc2_mixed)
        mixed = bool(rows) and any(item.num_decode_tokens > 0 for item in context.attn_metadata.values())
        token = _PREFILL_ROWS.set(rows)
        phase_token = _PREFILL_PHASE.set("mixed_prefill" if mixed else ("pure_prefill" if rows else "none"))
        try:
            return original_model_forward(self, *args, **kwargs)
        finally:
            _PREFILL_PHASE.reset(phase_token)
            _PREFILL_ROWS.reset(token)

    def dispatch(input_, weight, key: str):
        # This body is opaque to Dynamo; the phase must be read at execution,
        # not specialized from the profile/dummy batch used for compilation.
        self, binding = bindings[key]
        if input_.shape[0] < binding["min_rows"] or _PREFILL_ROWS.get() != input_.shape[0]:
            output = original_forward(self, input_)
            return output[0] if self.return_bias else output
        if input_.ndim != 2 or input_.shape[1] != binding["k"] or input_.dtype != self.weight.dtype:
            raise RuntimeError("prefill MC2: live input violates prepared tensor contract")
        if input_.device != self.weight.device or not input_.is_contiguous():
            output = original_forward(self, input_)
            return output[0] if self.return_bias else output
        output = fused(input_, weight.t(), binding["hcom"])
        if not binding["observed"]:
            emit_evidence(config, "prefill_mc2_called", component="performance",
                          name=binding["name"], rows=input_.shape[0], k=binding["k"],
                          weight_shape=list(self.weight.shape), phase=_PREFILL_PHASE.get(),
                          source=__file__, source_sha256=binding["source_sha256"])
            binding["observed"] = True
        if _PREFILL_PHASE.get() == "mixed_prefill" and not binding.get("observed_mixed", False):
            emit_evidence(config, "prefill_mc2_mixed_called", component="performance",
                          name=binding["name"], rows=input_.shape[0], k=binding["k"],
                          phase="mixed_prefill", source=__file__, source_sha256=binding["source_sha256"])
            binding["observed_mixed"] = True
        return output

    opaque = make_opaque(dispatch) if make_opaque is not None else dispatch

    @functools.wraps(original_forward)
    def forward(self, input_, **kwargs):
        binding = getattr(self, "_drrqr_prefill_mc2", None)
        if binding is None or kwargs:
            return original_forward(self, input_, **kwargs)
        output = opaque(input_, self.weight, binding["name"])
        return (output, None) if self.return_bias else output

    forward._drrqr_prefill_mc2_identity = identity
    forward._drrqr_prefill_mc2_bindings = bindings
    model_forward._drrqr_prefill_mc2_identity = identity
    linear_cls.forward = forward
    runner_cls._model_forward = model_forward
    return bindings


def install_runtime_patch(runner_cls, linear_cls, model_cls, config, verify_worker):
    import torch
    import torch_npu
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
    from .conv_layout_runtime import validate_config

    if file_sha256(Path(inspect.getsourcefile(linear_cls))) != LINEAR_SOURCE_SHA256:
        raise RuntimeError("prefill MC2: Ascend linear source changed")
    original_load = runner_cls.load_model
    identity = (config.enable, config.plan_sha256, config.evidence_file, config.prefill_mc2_mixed)
    prior = getattr(original_load, "_drrqr_prefill_mc2_load", None)
    if prior is not None:
        if prior != identity:
            raise RuntimeError("prefill MC2: another model binding is installed")
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    def make_opaque(dispatch):
        def real(input_: torch.Tensor, weight: torch.Tensor, key: str) -> torch.Tensor:
            return dispatch(input_, weight, key)

        def fake(input_: torch.Tensor, weight: torch.Tensor, key: str) -> torch.Tensor:
            return input_.new_empty((input_.shape[0], weight.shape[0]))

        direct_register_custom_op(
            op_name="drrqr_prefill_mc2_linear", op_func=real, fake_impl=fake,
            mutates_args=[], dispatch_key="PrivateUse1",
        )
        return torch.ops.vllm.drrqr_prefill_mc2_linear

    bindings = install_dispatch(runner_cls, linear_cls, config, get_context=get_forward_context,
                                fused=torch_npu.npu_mm_all_reduce_base, make_opaque=make_opaque)

    @functools.wraps(original_load)
    def load_model(self):
        from vllm.model_executor.offloader.base import NoopOffloader, get_offloader

        if getattr(self, "_drrqr_prefill_mc2_ready", False):
            raise RuntimeError("prefill MC2: reload is not supported")
        validate_config(self.vllm_config)  # Same frozen Qwen/TP4/BF16/no-training contract.
        parallel = self.vllm_config.parallel_config
        if any(getattr(parallel, key, 1) != 1 for key in (
                "prefill_context_parallel_size", "decode_context_parallel_size", "data_parallel_size")):
            raise RuntimeError("prefill MC2: context/data parallelism not covered by this TP4 adapter")
        if type(get_offloader()) is not NoopOffloader:
            raise RuntimeError("prefill MC2: weight offloading is not supported")
        result = original_load(self)
        if type(get_offloader()) is not NoopOffloader:
            raise RuntimeError("prefill MC2: loader enabled weight offloading")
        verify_worker()
        model = self.get_model()
        if not isinstance(model, model_cls):
            raise RuntimeError("prefill MC2: unexpected model")
        targets = collect_targets(model, linear_cls, get_format=torch_npu.get_npu_format)
        if any(type(layer.quant_method) is not AscendUnquantizedLinearMethod for _, layer, _, _ in targets):
            raise RuntimeError("prefill MC2: unexpected quantization method")
        group = get_tp_group()
        if group.world_size != 4:
            raise RuntimeError("prefill MC2: wrong communication group size")
        hcom = group.device_group._get_backend(torch.device("npu")).get_hccl_comm_name(group.rank)
        source_sha = file_sha256(Path(__file__))
        for name, layer, k, min_rows in targets:
            layer._drrqr_prefill_mc2 = {
                "name": name, "k": k, "min_rows": min_rows, "hcom": hcom,
                "observed": False, "observed_mixed": False, "source_sha256": source_sha,
            }
            bindings[name] = (layer, layer._drrqr_prefill_mc2)
        self._drrqr_prefill_mc2_ready = True
        emit_evidence(config, "prefill_mc2_prepared", component="performance",
                      layer_count=len(targets),
                      phase="large_prefill_including_mixed" if config.prefill_mc2_mixed else "pure_prefill_only",
                      allow_mixed=config.prefill_mc2_mixed,
                      fallback="original small/decode/padded/graph/profile path; mixed also original unless opted in",
                      layers=[dict(name=name, k=k, min_rows=min_rows) for name, _, k, min_rows in targets],
                      source=__file__, source_sha256=source_sha,
                      linear_source_sha256=LINEAR_SOURCE_SHA256,
                      weights_modified=False, kernel="torch_npu.npu_mm_all_reduce_base")
        return result

    load_model._drrqr_prefill_mc2_load = identity
    runner_cls.load_model = load_model
