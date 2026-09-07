"""Prepare an immutable Qwen3.8-27B DRRQR load-time plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from .plan import (
    SCHEMA,
    TARGET_ARCHITECTURE,
    TARGET_MODEL_ID,
    TARGET_OUTER_MODEL_TYPE,
    TARGET_TEXT_MODEL_TYPE,
)

SOURCE_PREFIXES = (
    "layers.",
    "model.layers.",
    "model.language_model.layers.",
    "language_model.model.layers.",
)
TARGET_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.linear_attn\."
    r"(in_proj_qkv|conv1d)\.(weight|bias)$"
)
SCOPE = (
    "Inference-only, load-time structured Q/K selection for the official "
    "Qwen/Qwen3.8-27B BF16 packed GDN path. The original checkpoint is not "
    "rewritten. Capture and input hashes establish identity but do not by "
    "themselves establish quality, performance, or paper-exact reproduction."
)


def _safe_open(path: Path, **kwargs: Any) -> Any:
    """Import safetensors only when checkpoint metadata is inspected."""

    from safetensors import safe_open

    return safe_open(path, **kwargs)


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"DRRQR: {key} must be a positive integer")
    return value


def _target_dim(old_dim: int, ratio: float) -> int:
    if not isinstance(ratio, float) or not 0.0 < ratio < 1.0:
        raise ValueError("DRRQR: pruning ratio must be strictly between zero and one")
    return max(1, math.floor(old_dim * (1.0 - ratio)))


def validate_source(
    source: Path,
    config: dict[str, Any],
    index: dict[str, Any],
    new_dim: int,
    tp_size: int,
) -> tuple[dict[str, Any], list[int]]:
    """Validate exact Qwen3.8-27B architecture and target tensor metadata."""

    if (
        config.get("architectures") != [TARGET_ARCHITECTURE]
        or config.get("model_type") != TARGET_OUTER_MODEL_TYPE
        or config.get("language_model_only") is not False
    ):
        raise ValueError("DRRQR: source is not the official Qwen3.8-27B architecture")
    text = config.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != TARGET_TEXT_MODEL_TYPE:
        raise ValueError("DRRQR: Qwen3.8-27B text configuration is missing")
    for candidate in (config, text):
        if candidate.get("quantization_config") not in (None, {}):
            raise ValueError("DRRQR: quantized checkpoints are unsupported")
        if candidate.get("num_experts", 0):
            raise ValueError("DRRQR: MoE checkpoints are unsupported")
    dtype = text.get(
        "dtype",
        text.get("torch_dtype", config.get("dtype", config.get("torch_dtype"))),
    )
    if dtype not in ("bfloat16", "torch.bfloat16"):
        raise ValueError("DRRQR: source must be BF16")

    old_dim = _positive_int(text, "linear_key_head_dim")
    num_heads = _positive_int(text, "linear_num_key_heads")
    num_value_heads = _positive_int(text, "linear_num_value_heads")
    value_dim = _positive_int(text, "linear_value_head_dim")
    conv_dim = _positive_int(text, "linear_conv_kernel_dim")
    hidden_size = _positive_int(text, "hidden_size")
    num_layers = _positive_int(text, "num_hidden_layers")
    exact = {
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "hidden_size": 5120,
        "num_hidden_layers": 64,
    }
    for name, expected in exact.items():
        if text.get(name) != expected:
            raise ValueError(
                f"DRRQR: Qwen3.8-27B {name} must be {expected}, got {text.get(name)!r}"
            )
    if type(new_dim) is not int or not 0 < new_dim < old_dim:
        raise ValueError("DRRQR: target Dk must be between zero and 128")
    if (
        type(tp_size) is not int
        or tp_size <= 0
        or num_heads % tp_size
        or num_value_heads % tp_size
    ):
        raise ValueError("DRRQR: TP must divide Q/K and V head counts")
    if num_value_heads % num_heads:
        raise ValueError("DRRQR: V head count must be divisible by Q/K heads")

    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != num_layers:
        raise ValueError("DRRQR: layer_types must describe all 64 layers")
    expected_types = [
        "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
        for layer in range(64)
    ]
    if layer_types != expected_types:
        raise ValueError("DRRQR: source does not have the Qwen3.8-27B 3:1 hybrid topology")
    linear_layers = [
        layer
        for layer, kind in enumerate(layer_types)
        if kind == "linear_attention"
    ]

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("DRRQR: a nonempty safetensors weight map is required")
    targets: dict[tuple[int, str], tuple[str, str]] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise TypeError("DRRQR: weight map names and shards must be strings")
        match = TARGET_RE.search(name)
        if match is None or not name.startswith(SOURCE_PREFIXES):
            continue
        layer, component, kind = int(match[1]), match[2], match[3]
        if kind == "bias":
            raise ValueError(f"DRRQR: packed bias is unsupported: {name}")
        if layer not in linear_layers:
            raise ValueError(f"DRRQR: linear tensor occurs in a full-attention layer: {name}")
        key = (layer, component)
        if key in targets:
            raise ValueError(f"DRRQR: duplicate target tensor {key}")
        targets[key] = (name, shard)
    expected_targets = {
        (layer, component)
        for layer in linear_layers
        for component in ("in_proj_qkv", "conv1d")
    }
    if set(targets) != expected_targets:
        raise ValueError(
            "DRRQR: target tensor coverage mismatch "
            f"missing={sorted(expected_targets - set(targets))}"
        )

    packed_width = 2 * num_heads * old_dim + num_value_heads * value_dim
    shapes = {
        "in_proj_qkv": (packed_width, hidden_size),
        "conv1d": (packed_width, 1, conv_dim),
    }
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for (_, component), (name, shard) in targets.items():
        by_shard.setdefault(shard, []).append((component, name))
    root = source.resolve()
    for shard, items in by_shard.items():
        shard_path = (source / shard).resolve()
        if not shard_path.is_relative_to(root) or not shard_path.is_file():
            raise ValueError(f"DRRQR: source shard missing or outside checkpoint: {shard}")
        with _safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for component, name in items:
                if name not in keys:
                    raise ValueError(f"DRRQR: indexed tensor missing from shard: {name}")
                tensor_slice = handle.get_slice(name)
                if tuple(tensor_slice.get_shape()) != shapes[component]:
                    raise ValueError(f"DRRQR: source tensor shape mismatch: {name}")
                if tensor_slice.get_dtype() != "BF16":
                    raise ValueError(f"DRRQR: source tensor must be BF16: {name}")
    return text, linear_layers


def _write_exclusive_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def prepare_plan(
    source: Path,
    capture_dir: Path,
    calibration_jsonl: Path,
    official_rrqr: Path,
    output: Path,
    *,
    pruning_ratio: float,
    tp_size: int = 4,
    captures_per_rank: int = 16,
) -> dict[str, Any]:
    from . import selection

    source = source.resolve()
    capture_dir = capture_dir.resolve()
    calibration_jsonl = calibration_jsonl.resolve()
    official_rrqr = official_rrqr.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"DRRQR: plan already exists: {output}")
    if output.is_relative_to(source):
        raise ValueError("DRRQR: plan output must be outside the source checkpoint")
    if selection.file_sha256(official_rrqr) != selection.OFFICIAL_RRQR_SHA256:
        raise ValueError("DRRQR: official RRQR source hash drift")
    if not calibration_jsonl.is_file() or not calibration_jsonl.stat().st_size:
        raise ValueError("DRRQR: nonempty calibration JSONL is required")
    if not capture_dir.is_dir():
        raise ValueError("DRRQR: capture directory does not exist")
    if type(captures_per_rank) is not int or captures_per_rank <= 0:
        raise ValueError("DRRQR: captures_per_rank must be positive")

    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    config_bytes = config_path.read_bytes()
    index_bytes = index_path.read_bytes()
    config = json.loads(config_bytes)
    index = json.loads(index_bytes)
    if not isinstance(config, dict) or not isinstance(index, dict):
        raise TypeError("DRRQR: source metadata must be JSON objects")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise TypeError("DRRQR: Qwen3.8 text_config is missing")
    target_dim = _target_dim(_positive_int(text, "linear_key_head_dim"), pruning_ratio)
    text, linear_layers = validate_source(
        source,
        config,
        index,
        target_dim,
        tp_size,
    )
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    index_sha = hashlib.sha256(index_bytes).hexdigest()
    calibration_sha = selection.file_sha256(calibration_jsonl)
    keep_maps, selection_evidence = selection.load_keep_maps(
        capture_dir,
        expected_layer_ids=linear_layers,
        tp_size=tp_size,
        num_heads=text["linear_num_key_heads"],
        old_dim=text["linear_key_head_dim"],
        new_dim=target_dim,
        captures_per_rank=captures_per_rank,
        source_config_sha256=config_sha,
        calibration_sha256=calibration_sha,
    )
    if set(keep_maps) != set(linear_layers):
        raise ValueError("DRRQR: selection layer coverage mismatch")
    keep_indices: dict[str, list[int]] = {}
    for layer, tensor in keep_maps.items():
        values = tensor.tolist()
        expected_width = text["linear_num_key_heads"] * target_dim
        if len(values) != expected_width or len(set(values)) != len(values):
            raise ValueError(f"DRRQR: invalid selection width for layer {layer}")
        for head in range(text["linear_num_key_heads"]):
            selected = values[head * target_dim : (head + 1) * target_dim]
            lower = head * text["linear_key_head_dim"]
            upper = (head + 1) * text["linear_key_head_dim"]
            if selected != sorted(selected) or any(
                index < lower or index >= upper
                for index in selected
            ):
                raise ValueError(
                    f"DRRQR: selection crosses head boundary in layer {layer}"
                )
        keep_indices[str(layer)] = values
    if (
        selection.file_sha256(config_path) != config_sha
        or selection.file_sha256(index_path) != index_sha
        or selection.file_sha256(calibration_jsonl) != calibration_sha
    ):
        raise RuntimeError("DRRQR: immutable input changed during preparation")

    plan = {
        "schema": SCHEMA,
        "target_model_id": TARGET_MODEL_ID,
        "source_config_sha256": config_sha,
        "source_index_sha256": index_sha,
        "old_head_k_dim": text["linear_key_head_dim"],
        "target_head_k_dim": target_dim,
        "pruning_ratio": pruning_ratio,
        "num_key_heads": text["linear_num_key_heads"],
        "num_value_heads": text["linear_num_value_heads"],
        "head_v_dim": text["linear_value_head_dim"],
        "conv_kernel_dim": text["linear_conv_kernel_dim"],
        "hidden_size": text["hidden_size"],
        "layer_types": text["layer_types"],
        "model_type": text["model_type"],
        "keep_indices": keep_indices,
        "provenance": {
            "method": "DRRQR",
            "official_commit": selection.OFFICIAL_COMMIT,
            "official_rrqr_sha256": selection.OFFICIAL_RRQR_SHA256,
            "calibration_sha256": calibration_sha,
            "selection": selection_evidence,
            "scope": SCOPE,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive_json(output, plan)
    plan_sha = selection.file_sha256(output)
    return {
        "plan_path": str(output),
        "plan_sha256": plan_sha,
        "target_model_id": TARGET_MODEL_ID,
        "pruning_ratio": pruning_ratio,
        "target_head_k_dim": target_dim,
        "hf_overrides": {
            "text_config": {"linear_key_head_dim": target_dim}
        },
        "environment": {
            "VLLM_ASCEND_DRRQR_ENABLE": "1",
            "VLLM_ASCEND_DRRQR_PLAN_PATH": str(output),
            "VLLM_ASCEND_DRRQR_PLAN_SHA256": plan_sha,
        },
        "scope": SCOPE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--official-rrqr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pruning-ratio",
        type=float,
        required=True,
        choices=(0.2, 0.3, 0.5),
    )
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--captures-per-rank", type=int, default=16)
    args = parser.parse_args()
    result = prepare_plan(**vars(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
