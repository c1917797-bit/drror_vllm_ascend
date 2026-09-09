"""Hashed experimental per-layer Dk contract for the opt-in Ascend runtime."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .plan import DrrqrPlan, HASH, validate_ascend_decode_alignment

SCHEMA = "ascend-drrqr-layerwise-manifest/v1"
PREFIX = re.compile(r"(?:^|\.)layers\.(\d+)\.linear_attn$")
SHARED_FIELDS = (
    "source_config_sha256", "source_index_sha256", "old_head_k_dim",
    "num_key_heads", "num_value_heads", "head_v_dim", "conv_kernel_dim",
    "hidden_size", "layer_types",
)


@dataclass(frozen=True)
class LayerwiseDrrqrPlan:
    reference: DrrqrPlan
    layer_dims: tuple[tuple[int, int], ...]
    keep_indices: tuple[tuple[int, tuple[int, ...]], ...]
    manifest_digest: str = ""

    @classmethod
    def compose(cls, plans: dict[int, DrrqrPlan], layer_dims: dict[int, int]):
        if not plans:
            raise ValueError("at least one bound reduced plan is required")
        reference = next(iter(plans.values()))
        layer_ids = {i for i, kind in enumerate(reference.layer_types)
                     if kind == "linear_attention"}
        if any(type(i) is not int for i in layer_dims) or set(layer_dims) != layer_ids:
            raise ValueError("layer dimensions must cover exactly the linear layers")
        if any(type(dk) is not int or not 0 < dk <= reference.old_head_k_dim
               for dk in layer_dims.values()):
            raise ValueError("invalid layer Dk")
        for dk in layer_dims.values():
            validate_ascend_decode_alignment(dk)
        required = set(layer_dims.values()) - {reference.old_head_k_dim}
        if set(plans) != required:
            raise ValueError("source plans must exactly match the reduced dimensions used")
        maps = {}
        for dk, plan in plans.items():
            if dk != plan.target_head_k_dim:
                raise ValueError("source plan Dk/key mismatch")
            if any(getattr(plan, field) != getattr(reference, field) for field in SHARED_FIELDS):
                raise ValueError("source plans have different model contracts")
            maps[dk] = dict(plan.keep_indices)
            if set(maps[dk]) != layer_ids:
                raise ValueError("source plan layer coverage mismatch")
        keeps = []
        for layer, dk in sorted(layer_dims.items()):
            if dk == reference.old_head_k_dim:
                selected = tuple(range(reference.num_key_heads * reference.old_head_k_dim))
            else:
                selected = maps[dk][layer]
            if len(selected) != reference.num_key_heads * dk or len(set(selected)) != len(selected):
                raise ValueError("source selection width/uniqueness mismatch")
            for head in range(reference.num_key_heads):
                lo = head * reference.old_head_k_dim
                if any(type(i) is not int or not lo <= i < lo + reference.old_head_k_dim
                       for i in selected[head * dk:(head + 1) * dk]):
                    raise ValueError("source selection crosses a key-head boundary")
            keeps.append((layer, selected))
        return cls(reference, tuple(sorted(layer_dims.items())), tuple(keeps))

    @classmethod
    def load(cls, path: str, expected_sha256: str):
        if not Path(path).is_absolute() or not isinstance(expected_sha256, str) or not HASH.fullmatch(expected_sha256):
            raise ValueError("absolute manifest path and SHA256 required")
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("layerwise manifest hash mismatch")
        document = json.loads(raw)
        if not isinstance(document, dict) or document.get("schema") != SCHEMA:
            raise ValueError("wrong layerwise manifest schema")
        widths = document.get("layer_head_k_dims")
        refs = document.get("source_plans")
        if not isinstance(widths, dict) or not isinstance(refs, dict):
            raise ValueError("manifest requires explicit layer dimensions and source plans")
        for keys in (widths, refs):
            if any(not isinstance(k, str) or not k.isdigit() or str(int(k)) != k for k in keys):
                raise ValueError("manifest keys must be canonical nonnegative integers")
        plans, calibration, objective, capture = {}, set(), set(), set()
        for key, ref in refs.items():
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
                raise ValueError("each source reference requires only path and sha256")
            # Read once for both parsing and verified provenance. A changed file
            # cannot contribute unchecked metadata after DrrqrPlan.load.
            plan = DrrqrPlan.load(ref["path"], ref["sha256"])
            source = Path(ref["path"]).read_bytes()
            if hashlib.sha256(source).hexdigest() != ref["sha256"]:
                raise ValueError("source plan changed during manifest load")
            data = json.loads(source)
            prov = data["provenance"]
            calibration.add(prov["calibration_sha256"])
            objective.add(data.get("selection_objective", "official-raw"))
            selection = prov.get("selection")
            capture_hash = selection.get("capture_manifest_sha256") if isinstance(selection, dict) else None
            if not isinstance(capture_hash, str) or not HASH.fullmatch(capture_hash):
                raise ValueError("source plan requires an explicit capture manifest hash")
            capture.add(capture_hash)
            plans[int(key)] = plan
        if len(calibration) != 1 or len(objective) != 1 or len(capture) != 1:
            raise ValueError("source plans must share calibration/selector/capture provenance")
        return replace(
            cls.compose(plans, {int(k): v for k, v in widths.items()}),
            manifest_digest=expected_sha256,
        )

    @property
    def digest(self) -> str:
        if not self.manifest_digest:
            raise ValueError("layerwise plan is not bound to an immutable manifest")
        return self.manifest_digest

    def layer_dim(self, layer: int) -> int:
        try:
            return dict(self.layer_dims)[layer]
        except KeyError as error:
            raise ValueError("layer is not in the bound linear topology") from error

    @property
    def max_head_k_dim(self) -> int:
        return max(dk for _, dk in self.layer_dims)

    @property
    def target_head_k_dim(self) -> int:
        """Compatibility value for diagnostics and conservative cache sizing."""
        return self.max_head_k_dim

    def isolated_layer_config(self, config: Any, prefix: str):
        match = PREFIX.search(prefix)
        if match is None:
            raise ValueError("not a canonical GDN layer prefix")
        layer = int(match[1])
        dk = self.layer_dim(layer)
        result = deepcopy(config)
        if isinstance(result, dict):
            result["linear_key_head_dim"] = dk
        else:
            result.linear_key_head_dim = dk
        return result

    def validate_runtime(self, vllm_config: Any) -> None:
        """Use the largest layer as the shared cache-sizing envelope.

        The launcher sets this envelope once. Per-layer constructor copies
        carry their actual Dk without mutating the shared HF config. A uniform64
        control therefore retains the legacy64 attention block and page sizes.
        """
        ref = self.reference
        model = vllm_config.model_config
        ref.validate_source(model.model)
        replace(ref, target_head_k_dim=self.max_head_k_dim).validate_text_config(
            model.hf_text_config, reduced=True
        )
        if (
            str(model.dtype) not in ("torch.bfloat16", "bfloat16")
            or model.quantization is not None
        ):
            raise ValueError("DRRQR: initial support is unquantized bfloat16 only")
        if getattr(vllm_config, "lora_config", None) or getattr(
            vllm_config, "speculative_config", None
        ):
            raise ValueError("DRRQR: LoRA and speculative/MTP are unsupported")
        parallel = vllm_config.parallel_config
        tp_size = parallel.tensor_parallel_size
        if (
            type(tp_size) is not int
            or tp_size < 1
            or ref.num_key_heads % tp_size
            or ref.num_value_heads % tp_size
            or parallel.pipeline_parallel_size != 1
            or getattr(parallel, "prefill_context_parallel_size", 1) != 1
            or getattr(parallel, "decode_context_parallel_size", 1) != 1
        ):
            raise ValueError("DRRQR: unsupported TP/PP/context parallelism")
        load_format = getattr(
            getattr(vllm_config, "load_config", None), "load_format", "auto"
        )
        load_format = getattr(load_format, "value", load_format)
        if str(load_format).lower() not in ("auto", "safetensors"):
            raise ValueError("DRRQR: use the original packed safetensors loader")

    def transform_weights(self, weights):
        # The existing transformer uses explicit per-layer coordinate lists;
        # it does not derive the output width from target_head_k_dim.
        delegate = replace(self.reference, keep_indices=self.keep_indices)
        yield from delegate.transform_weights(weights)

    def layer_shapes(self, layer: int, tp_size: int = 4):
        ref, dk = self.reference, self.layer_dim(layer)
        if type(tp_size) is not int or tp_size <= 0 or ref.num_key_heads % tp_size or ref.num_value_heads % tp_size:
            raise ValueError("invalid TP partition")
        packed = 2 * ref.num_key_heads * dk + ref.num_value_heads * ref.head_v_dim
        return {
            "global_qkv_rows": packed,
            "local_qkv_rows": packed // tp_size,
            "local_state_shape": (ref.num_value_heads // tp_size, ref.head_v_dim, dk),
            "local_conv_elements": (packed // tp_size) * (ref.conv_kernel_dim - 1),
        }
