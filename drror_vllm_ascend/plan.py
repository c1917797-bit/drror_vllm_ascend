"""Validated DRRQR plan and load-time packed-weight transformation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .envs import DrrqrConfig

SCHEMA = "ascend-drrqr-plan/v1"
TARGET_MODEL_ID = "Qwen/Qwen3.8-27B"
TARGET_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
TARGET_OUTER_MODEL_TYPE = "qwen3_5"
TARGET_TEXT_MODEL_TYPE = "qwen3_5_text"
OFFICIAL_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
OFFICIAL_CONFIG_SHA256 = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
OFFICIAL_INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
OFFICIAL_COMMIT = "919d8667d951c385e08510bc1267c2e7049a4f56"
OFFICIAL_RRQR_SHA256 = "fa4bacf516011ba1c88f2e0e92957bbe4513cc1bb6634912fdb0d06ba7821a62"
WEIGHT_NAME = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.linear_attn\."
    r"(in_proj_qkv|conv1d)\.weight$"
)
HASH = re.compile(r"^[0-9a-f]{64}$")
ASCEND_DECODE_KEY_ALIGNMENT = 16


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"DRRQR: {name} must be a positive integer")
    return value


def validate_official_checkpoint_hashes(
    config_sha256: str,
    index_sha256: str,
) -> None:
    if (
        config_sha256 != OFFICIAL_CONFIG_SHA256
        or index_sha256 != OFFICIAL_INDEX_SHA256
    ):
        raise ValueError(
            "DRRQR: checkpoint metadata does not match official "
            f"Qwen/Qwen3.8-27B revision {OFFICIAL_MODEL_REVISION}"
        )


def validate_ascend_decode_alignment(target_head_k_dim: int) -> None:
    """Reject Dk values the frozen v0.23 AscendC decode kernel cannot copy out."""

    if target_head_k_dim % ASCEND_DECODE_KEY_ALIGNMENT:
        raise ValueError(
            "DRRQR: target Dk is unsafe for the v0.23 AscendC decode state "
            "copy-out because its local state row is padded to a multiple of "
            f"{ASCEND_DECODE_KEY_ALIGNMENT} while CopyOutState assumes compact "
            f"rows; use a multiple of {ASCEND_DECODE_KEY_ALIGNMENT}"
        )


@dataclass(frozen=True)
class DrrqrPlan:
    digest: str
    source_config_sha256: str
    source_index_sha256: str
    old_head_k_dim: int
    target_head_k_dim: int
    num_key_heads: int
    num_value_heads: int
    head_v_dim: int
    conv_kernel_dim: int
    hidden_size: int
    layer_types: tuple[str, ...]
    keep_indices: tuple[tuple[int, tuple[int, ...]], ...]

    @classmethod
    def load(cls, path: str, expected_sha256: str) -> DrrqrPlan:
        if not isinstance(expected_sha256, str) or not HASH.fullmatch(expected_sha256):
            raise ValueError("DRRQR: an explicit lowercase plan SHA256 is required")
        plan_path = Path(path)
        if not plan_path.is_absolute():
            raise ValueError("DRRQR: plan path must be absolute")
        raw = plan_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("DRRQR: plan hash mismatch")
        data = json.loads(raw)
        if (
            not isinstance(data, dict)
            or data.get("schema") != SCHEMA
            or data.get("target_model_id") != TARGET_MODEL_ID
            or data.get("model_type") != TARGET_TEXT_MODEL_TYPE
        ):
            raise ValueError(
                "DRRQR: plan must target the official Qwen/Qwen3.8-27B checkpoint"
            )
        for name in ("source_config_sha256", "source_index_sha256"):
            if not isinstance(data.get(name), str) or not HASH.fullmatch(data[name]):
                raise ValueError(f"DRRQR: invalid {name}")
        validate_official_checkpoint_hashes(
            data["source_config_sha256"],
            data["source_index_sha256"],
        )
        provenance = data.get("provenance", {})
        if (
            not isinstance(provenance, dict)
            or provenance.get("method") != "DRRQR"
            or provenance.get("official_commit") != OFFICIAL_COMMIT
            or provenance.get("official_rrqr_sha256") != OFFICIAL_RRQR_SHA256
            or not isinstance(provenance.get("calibration_sha256"), str)
            or not HASH.fullmatch(provenance["calibration_sha256"])
        ):
            raise ValueError("DRRQR: missing pinned algorithm/calibration provenance")
        names = (
            "old_head_k_dim",
            "target_head_k_dim",
            "num_key_heads",
            "num_value_heads",
            "head_v_dim",
            "conv_kernel_dim",
            "hidden_size",
        )
        dims = {name: _positive_int(data.get(name), name) for name in names}
        old = dims["old_head_k_dim"]
        new = dims["target_head_k_dim"]
        heads = dims["num_key_heads"]
        if new >= old or dims["num_value_heads"] % heads:
            raise ValueError("DRRQR: invalid key reduction or value/key head ratio")
        layer_types = data.get("layer_types")
        if (
            not isinstance(layer_types, list)
            or not layer_types
            or any(not isinstance(kind, str) for kind in layer_types)
            or set(layer_types) != {"linear_attention", "full_attention"}
        ):
            raise ValueError("DRRQR: expected an explicit dense hybrid layer list")
        layer_ids = {
            index
            for index, kind in enumerate(layer_types)
            if kind == "linear_attention"
        }
        keeps = data.get("keep_indices")
        if not isinstance(keeps, dict) or set(keeps) != {
            str(index) for index in layer_ids
        }:
            raise ValueError("DRRQR: selection must cover all linear-attention layers")
        validated = []
        for layer in sorted(layer_ids):
            keep = keeps[str(layer)]
            if not isinstance(keep, list) or len(keep) != heads * new:
                raise ValueError(f"DRRQR: incorrect selection width in layer {layer}")
            if any(type(index) is not int for index in keep) or len(set(keep)) != len(keep):
                raise ValueError(f"DRRQR: duplicate/non-integer indices in layer {layer}")
            for head in range(heads):
                selected = keep[head * new : (head + 1) * new]
                if any(
                    index < head * old or index >= (head + 1) * old
                    for index in selected
                ):
                    raise ValueError(
                        f"DRRQR: indices cross head boundaries in layer {layer}"
                    )
            validated.append((layer, tuple(keep)))
        return cls(
            digest=expected_sha256,
            source_config_sha256=data["source_config_sha256"],
            source_index_sha256=data["source_index_sha256"],
            layer_types=tuple(layer_types),
            keep_indices=tuple(validated),
            **dims,
        )

    def validate_source(self, model_path: str) -> None:
        root = Path(model_path)
        if not root.is_dir():
            raise ValueError("DRRQR: use the original local checkpoint")
        for filename, digest in (
            ("config.json", self.source_config_sha256),
            ("model.safetensors.index.json", self.source_index_sha256),
        ):
            if file_sha256(root / filename) != digest:
                raise ValueError(f"DRRQR: original checkpoint {filename} hash mismatch")
        source = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if (
            source.get("architectures") != [TARGET_ARCHITECTURE]
            or source.get("model_type") != TARGET_OUTER_MODEL_TYPE
            or source.get("language_model_only") is not False
        ):
            raise ValueError(
                "DRRQR: source is not the official Qwen3.8-27B architecture contract"
            )
        self.validate_text_config(source.get("text_config", source), reduced=False)
        if source.get("quantization_config"):
            raise ValueError("DRRQR: quantized checkpoints are unsupported")

    def validate_text_config(self, config: Any, *, reduced: bool) -> None:
        def get(name: str, default: Any = None) -> Any:
            if isinstance(config, dict):
                return config.get(name, default)
            return getattr(config, name, default)

        expected = {
            # Qwen3.8-27B intentionally publishes this architecture identifier.
            # It names the reused implementation contract, not a Qwen3.5 weight.
            "model_type": TARGET_TEXT_MODEL_TYPE,
            "linear_key_head_dim": (
                self.target_head_k_dim if reduced else self.old_head_k_dim
            ),
            "linear_num_key_heads": self.num_key_heads,
            "linear_num_value_heads": self.num_value_heads,
            "linear_value_head_dim": self.head_v_dim,
            "linear_conv_kernel_dim": self.conv_kernel_dim,
            "hidden_size": self.hidden_size,
        }
        for name, value in expected.items():
            if get(name) != value:
                raise ValueError(
                    f"DRRQR: {name} must be {value!r}, got {get(name)!r}"
                )
        if tuple(get("layer_types", ())) != self.layer_types:
            raise ValueError("DRRQR: hybrid topology differs from the plan")
        if get("quantization_config") or get("num_experts", 0):
            raise ValueError("DRRQR: only unquantized dense Qwen GDN is supported")

    def validate_runtime(self, vllm_config: Any) -> None:
        validate_ascend_decode_alignment(self.target_head_k_dim)
        model = vllm_config.model_config
        self.validate_source(model.model)
        self.validate_text_config(model.hf_text_config, reduced=True)
        if (
            str(model.dtype) not in ("torch.bfloat16", "bfloat16")
            or model.quantization is not None
        ):
            raise ValueError("DRRQR: initial support is unquantized bfloat16 only")
        if getattr(vllm_config, "lora_config", None) or getattr(
            vllm_config,
            "speculative_config",
            None,
        ):
            raise ValueError("DRRQR: LoRA and speculative/MTP are unsupported")
        parallel = vllm_config.parallel_config
        tp_size = parallel.tensor_parallel_size
        if (
            type(tp_size) is not int
            or tp_size < 1
            or self.num_key_heads % tp_size
            or self.num_value_heads % tp_size
            or parallel.pipeline_parallel_size != 1
            or getattr(parallel, "prefill_context_parallel_size", 1) != 1
            or getattr(parallel, "decode_context_parallel_size", 1) != 1
        ):
            raise ValueError("DRRQR: unsupported TP/PP/context parallelism")
        load_format = getattr(
            getattr(vllm_config, "load_config", None),
            "load_format",
            "auto",
        )
        load_format = getattr(load_format, "value", load_format)
        if str(load_format).lower() not in ("auto", "safetensors"):
            raise ValueError("DRRQR: use the original packed safetensors loader")

    def transform_weights(self, weights: Any):
        """Select Q/K and matching convolution rows before mapping/TP sharding.

        This transformer is intentionally bound to the top-level
        Qwen3_5ForConditionalGeneration loader.  Its input is the one complete
        Hugging Face checkpoint stream, whose Qwen3.8 language weights retain
        the ``model.language_model`` prefix.  Inner Qwen model loaders may be
        called repeatedly by AutoWeightsLoader and therefore cannot prove
        whole-checkpoint target coverage.
        """
        import torch

        keeps = dict(self.keep_indices)
        required = {
            (layer, kind)
            for layer in keeps
            for kind in ("in_proj_qkv", "conv1d")
        }
        seen = set()
        qk_width = self.num_key_heads * self.old_head_k_dim
        packed_width = 2 * qk_width + self.num_value_heads * self.head_v_dim
        for name, weight in weights:
            match = WEIGHT_NAME.fullmatch(name)
            if match is None:
                if ".linear_attn." in name and any(
                    part in name for part in (".in_proj_qkv", ".conv1d.")
                ):
                    raise ValueError(f"DRRQR: unsupported packed weight name {name}")
                yield name, weight
                continue
            layer = int(match[1])
            kind = match[2]
            key = (layer, kind)
            if key not in required or key in seen:
                raise ValueError(f"DRRQR: unexpected or repeated target {name}")
            if kind == "in_proj_qkv":
                expected_shapes = {(packed_width, self.hidden_size)}
            else:
                expected_shapes = {
                    (packed_width, 1, self.conv_kernel_dim),
                    (packed_width, self.conv_kernel_dim),
                }
            if (
                tuple(weight.shape) not in expected_shapes
                or not weight.is_floating_point()
            ):
                raise ValueError(
                    f"DRRQR: unexpected source shape/dtype for {name}: {tuple(weight.shape)}"
                )
            indices = torch.tensor(
                keeps[layer],
                dtype=torch.long,
                device=weight.device,
            )
            reduced = torch.cat(
                (
                    weight[:qk_width].index_select(0, indices),
                    weight[qk_width : 2 * qk_width].index_select(0, indices),
                    weight[2 * qk_width :],
                ),
                dim=0,
            ).contiguous()
            seen.add(key)
            yield name, reduced
        if seen != required:
            missing = sorted(required - seen)
            raise ValueError(f"DRRQR: incomplete target coverage: {missing}")


def load_runtime_plan(config: DrrqrConfig, vllm_config: Any) -> DrrqrPlan:
    plan = DrrqrPlan.load(config.plan_path, config.plan_sha256)
    plan.validate_runtime(vllm_config)
    return plan
