"""Bind calibration tensors to the actual source model and forwarded tokens."""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
from pathlib import Path

from ..calibration import read_calibration, token_sha256
from ..diagnostics import emit_evidence
from ..plan import file_sha256, validate_official_checkpoint_hashes
from ..prepare import validate_source


class CaptureBinding:
    def __init__(self, config):
        self.config = config
        self.verified = False
        self.rows = []
        self.source_model_path = ""
        self.source_index_sha256 = ""
        self.current_request = contextvars.ContextVar("drrqr_capture_request", default=None)

    def bind(self, vllm_config):
        self.verified = False
        model = vllm_config.model_config
        root = Path(model.model).resolve()
        config_sha = file_sha256(root / "config.json")
        index_sha = file_sha256(root / "model.safetensors.index.json")
        validate_official_checkpoint_hashes(config_sha, index_sha)
        if config_sha != self.config.source_config_sha256:
            raise ValueError("DRRQR: loaded capture model differs from declared source hash")
        source = json.loads((root / "config.json").read_text(encoding="utf-8"))
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        parallel = vllm_config.parallel_config
        text, _ = validate_source(root, source, index, 64, parallel.tensor_parallel_size)
        for name in (
            "model_type",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "hidden_size",
            "num_hidden_layers",
            "layer_types",
        ):
            if getattr(model.hf_text_config, name, None) != text[name]:
                raise ValueError(f"DRRQR: actual capture runtime overrides source {name}")
        if str(model.dtype) not in ("torch.bfloat16", "bfloat16") or model.quantization is not None:
            raise ValueError("DRRQR: capture requires the original unquantized BF16 model")
        if not getattr(model, "enforce_eager", False):
            raise ValueError("DRRQR: bound token capture requires --enforce-eager")
        if getattr(vllm_config, "lora_config", None) or getattr(vllm_config, "speculative_config", None):
            raise ValueError("DRRQR: capture does not support LoRA or speculation")
        if any(
            getattr(parallel, name, 1) != 1
            for name in (
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        ):
            raise ValueError("DRRQR: capture requires PP/context-parallel size one")
        rows = read_calibration(Path(self.config.calibration_jsonl), self.config.calibration_sha256)
        if len(rows) != self.config.capture_max_per_layer:
            raise ValueError("DRRQR: calibration row count must equal CAPTURE_MAX_PER_LAYER")
        if any(len(row["input_ids"]) > self.config.capture_max_tokens for row in rows):
            raise ValueError("DRRQR: each calibration request must fit CAPTURE_MAX_TOKENS")
        self.rows = rows
        self.source_model_path = str(root)
        self.source_index_sha256 = index_sha
        self.verified = True
        emit_evidence(
            self.config,
            "capture_source_bound",
            component="capture",
            source_model_path=str(root),
            source_config_sha256=config_sha,
            source_index_sha256=index_sha,
            calibration_sha256=self.config.calibration_sha256,
            calibration_row_count=len(rows),
            tensor_parallel_size=parallel.tensor_parallel_size,
        )

    def request(self, input_ids):
        if not self.verified:
            raise RuntimeError("DRRQR: capture model has not passed source validation")
        if input_ids is None or input_ids.ndim != 1:
            raise RuntimeError("DRRQR: capture requires one complete token-ID request per forward")
        digest = token_sha256(input_ids.detach().cpu().tolist())
        matches = [row for row in self.rows if row["input_ids_sha256"] == digest]
        if len(matches) != 1:
            raise RuntimeError("DRRQR: actual forward tokens do not match a complete calibration row")
        return matches[0]


def install_capture_model_patch(model_cls, config, binding, *, verify_worker=None):
    current = model_cls.__init__
    identity = (config.capture_dir, config.source_config_sha256, config.calibration_sha256)
    installed = getattr(current, "_drror_capture_model", None)
    if installed is not None:
        if installed != identity:
            raise RuntimeError("DRRQR: capture model already bound to another source")
        return
    if getattr(current, "_drror_drrqr_config", None) is not None:
        raise RuntimeError("DRRQR: treatment and capture model patches cannot coexist")
    original_forward = model_cls.forward
    init_signature = inspect.signature(current)
    forward_signature = inspect.signature(original_forward)
    if "vllm_config" not in init_signature.parameters or "input_ids" not in forward_signature.parameters:
        raise RuntimeError("DRRQR: capture model ABI changed")

    @functools.wraps(current)
    def initialize(self, *, vllm_config, prefix=""):
        if verify_worker is not None:
            verify_worker()
        binding.bind(vllm_config)
        current(self, vllm_config=vllm_config, prefix=prefix)

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        if not (Path(config.capture_dir) / ".armed").is_file():
            return original_forward(self, *args, **kwargs)
        bound = forward_signature.bind(self, *args, **kwargs)
        request = binding.current_request.get()
        if request is None:
            raise RuntimeError("DRRQR: capture runner did not bind the forwarded token IDs")
        input_ids = bound.arguments.get("input_ids")
        if input_ids is not None:
            forwarded = binding.request(input_ids)
            if forwarded["input_ids_sha256"] != request["input_ids_sha256"]:
                raise RuntimeError("DRRQR: runner and model token-ID bindings differ")
        return original_forward(self, *args, **kwargs)

    initialize._drror_capture_model = identity
    model_cls.__init__ = initialize
    model_cls.forward = forward


def install_capture_runner_patch(runner_cls, config, binding):
    """Bind the v0.23 scheduler token buffer before multimodal embedding."""

    current = runner_cls._model_forward
    identity = (config.capture_dir, config.source_config_sha256, config.calibration_sha256)
    installed = getattr(current, "_drror_capture_runner", None)
    if installed is not None:
        if installed != identity:
            raise RuntimeError("DRRQR: capture runner already bound to another source")
        return
    signature = inspect.signature(current)
    if not {"num_tokens_padded", "input_ids"}.issubset(signature.parameters):
        raise RuntimeError("DRRQR: Ascend model-runner ABI changed")

    @functools.wraps(current)
    def model_forward(self, *args, **kwargs):
        if not (Path(config.capture_dir) / ".armed").is_file():
            return current(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        num_tokens = bound.arguments["num_tokens_padded"]
        input_ids = bound.arguments.get("input_ids")
        if input_ids is None:
            token_buffer = getattr(getattr(self, "input_ids", None), "gpu", None)
            if token_buffer is None:
                raise RuntimeError("DRRQR: Ascend runner token-ID buffer is unavailable")
            input_ids = token_buffer[:num_tokens]
        request = binding.request(input_ids)
        token = binding.current_request.set(request)
        try:
            return current(self, *args, **kwargs)
        finally:
            binding.current_request.reset(token)

    model_forward._drror_capture_runner = identity
    runner_cls._model_forward = model_forward
