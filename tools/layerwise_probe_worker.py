"""One-shot model/cache qualification for the frozen Ascend TP4 control.

Use --worker-cls layerwise_probe_worker.LayerwiseProbeWorker with this tools
directory on PYTHONPATH, and set DRRQR_LAYERWISE_PROBE_DIR to a new, pre-created
absolute result directory. Each rank exclusively creates two JSON reports.
This worker does not wrap forward, replace operators, or change determinism.
Parameter hashes are diagnostic startup work; disable this worker for timing.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

import torch
from vllm_ascend.worker.worker import NPUWorker

SCHEMA = "ascend-drrqr-layerwise-worker-probe/v1"
PREFIX = re.compile(r"(?:^|\.)layers\.(\d+)\.linear_attn$")
FROZEN_SOURCES = {
    "vllm_ascend.worker.worker":
        "c392da56095b255d47fe5ad7ffdfbedd269fe69851690e89a8d2ed44c073f6c3",
    "vllm_ascend.worker.model_runner_v1":
        "94d75dbeb5d23ab5b383cdc69114167392b967994bb21342cf748b733ba9a968",
    "vllm_ascend.ops.gdn":
        "d6ec29919268178f5bf6e70e689c1d273d04b1cb1d84dc94efa7bbbc35490816",
}


def require(condition, message):
    if not condition:
        raise RuntimeError("DRRQR layerwise probe: " + message)


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def source_identity(obj):
    result = {
        "symbol": f"{obj.__module__}.{obj.__qualname__}",
        **file_identity(inspect.getsourcefile(obj)),
    }
    unwrapped = inspect.unwrap(obj)
    if unwrapped is not obj:
        result["unwrapped"] = {
            "symbol": f"{unwrapped.__module__}.{unwrapped.__qualname__}",
            **file_identity(inspect.getsourcefile(unwrapped)),
        }
    return result


def tensor_metadata(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "data_ptr": tensor.data_ptr(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
        "storage_offset": tensor.storage_offset(),
        "storage_nbytes": tensor.untyped_storage().nbytes(),
        "contiguous": tensor.is_contiguous(),
    }


def parameter_identity(name, parameter):
    before = tensor_metadata(parameter)
    inference = parameter.is_inference()
    version = None if inference else parameter._version
    # BF16 cannot be converted directly to NumPy. Hash exact numeric storage
    # bytes after a single CPU transfer, with only one parameter copy alive.
    cpu = parameter.detach().to(device="cpu").contiguous()
    octets = cpu.reshape(-1).view(torch.uint8).numpy()
    digest = hashlib.sha256(memoryview(octets).cast("B")).hexdigest()
    require(tensor_metadata(parameter) == before, f"{name}: metadata changed while hashing")
    require(parameter.is_inference() == inference, f"{name}: inference flag changed while hashing")
    if not inference:
        require(parameter._version == version, f"{name}: version changed while hashing")
    return {"name": name, **before, "sha256": digest, "version": version,
            "inference_tensor": inference}


def spec_identity(spec):
    return {
        "class": f"{type(spec).__module__}.{type(spec).__qualname__}",
        "shapes": [list(shape) for shape in spec.shapes],
        "dtypes": [str(dtype) for dtype in spec.dtypes],
        "block_size": spec.block_size,
        "page_size_padded": spec.page_size_padded,
        "page_size_bytes": spec.page_size_bytes,
        "mamba_type": str(spec.mamba_type),
        "mamba_cache_mode": spec.mamba_cache_mode,
        "num_speculative_blocks": spec.num_speculative_blocks,
    }


def layer_dimensions(plan, layer):
    ref = getattr(plan, "reference", plan)
    dk = plan.layer_dim(layer) if hasattr(plan, "layer_dim") else plan.target_head_k_dim
    return ref, dk


def check_module(module, plan, layer, rank):
    ref, dk = layer_dimensions(plan, layer)
    expected = {
        "layer_idx": layer, "tp_size": 4, "tp_rank": rank,
        "head_k_dim": dk, "head_v_dim": ref.head_v_dim,
        "num_k_heads": ref.num_key_heads, "num_v_heads": ref.num_value_heads,
        "hidden_size": ref.hidden_size, "conv_kernel_size": ref.conv_kernel_dim,
        "key_dim": ref.num_key_heads * dk,
        "value_dim": ref.num_value_heads * ref.head_v_dim, "num_spec": 0,
    }
    for name, value in expected.items():
        require(getattr(module, name) == value, f"layer {layer}: {name} differs from plan")
    packed = (2 * ref.num_key_heads * dk + ref.num_value_heads * ref.head_v_dim) // 4
    local_value = ref.num_value_heads * ref.head_v_dim // 4
    shapes = tuple(tuple(s) for s in module.get_state_shape())
    dtypes = tuple(module.get_state_dtype())
    require(len(shapes) == len(dtypes) == 2, f"layer {layer}: expected two states")
    require(
        shapes[1] == (ref.num_value_heads // 4, ref.head_v_dim, dk),
        f"layer {layer}: recurrent state shape differs from plan",
    )
    require(
        shapes[0] in ((packed, ref.conv_kernel_dim - 1),
                      (ref.conv_kernel_dim - 1, packed)),
        f"layer {layer}: convolution state shape differs from plan",
    )
    require(
        tuple(module.conv1d.weight.shape) == (packed, 1, ref.conv_kernel_dim),
        f"layer {layer}: local convolution weight shape differs from plan",
    )
    qkvz = getattr(module, "in_proj_qkvz", None)
    if qkvz is not None:
        require(
            tuple(qkvz.weight.shape) == (packed + local_value, ref.hidden_size),
            f"layer {layer}: local packed QKVZ weight shape differs from plan",
        )
        projection_layout = "qkvz"
    else:
        require(
            tuple(module.in_proj_qkv.weight.shape) == (packed, ref.hidden_size)
            and tuple(module.in_proj_z.weight.shape) == (local_value, ref.hidden_size),
            f"layer {layer}: local QKV/Z weight shape differs from plan",
        )
        projection_layout = "qkv_and_z"
    return {
        "expected_attributes": expected,
        "local_qkv_rows": packed,
        "projection_layout": projection_layout,
        "state_shapes": [list(shape) for shape in shapes],
        "state_dtypes": [str(dtype) for dtype in dtypes],
    }


@contextmanager
def phase_report(directory, rank, local_rank, phase):
    root = Path(directory)
    require(root.is_absolute() and root.is_dir(), "probe directory must already exist and be absolute")
    path = root / f"rank{rank}.{phase}.json"
    report = {
        "schema": SCHEMA, "phase": phase, "rank": rank, "local_rank": local_rank,
        "pid": os.getpid(), "started_unix_ns": time.time_ns(), "status": "running",
        "scope": "one-shot GDN load and cache-binding qualification; no forward evidence",
        "worker_source": file_identity(__file__), "layers": [],
    }
    # An existing report (including a failed/partial one) is never overwritten.
    with path.open("x", encoding="utf-8") as stream:
        try:
            yield report
            report["status"] = "passed"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            report["traceback"] = traceback.format_exc()
            raise
        finally:
            report["finished_unix_ns"] = time.time_ns()
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


class LayerwiseProbeWorker(NPUWorker):
    """Read-only startup observer; the inherited execution path is untouched."""

    def _report(self, phase):
        directory = os.environ.get("DRRQR_LAYERWISE_PROBE_DIR", "")
        require(bool(directory), "DRRQR_LAYERWISE_PROBE_DIR is required")
        return phase_report(directory, self.rank, self.local_rank, phase)

    def _admit(self, report):
        config = self.vllm_config
        parallel = config.parallel_config
        require(
            parallel.tensor_parallel_size == 4 and parallel.pipeline_parallel_size == 1
            and getattr(parallel, "data_parallel_size", 1) == 1
            and getattr(parallel, "prefill_context_parallel_size", 1) == 1
            and getattr(parallel, "decode_context_parallel_size", 1) == 1,
            "only TP4/PP1 without context parallelism is admitted",
        )
        require(self.rank in range(4) and self.local_rank in range(4), "invalid TP4 rank")
        require(config.model_config.dtype == torch.bfloat16, "BF16 model required")
        require(not config.cache_config.enable_prefix_caching, "prefix caching must be off")
        for name in ("speculative_config", "kv_transfer_config", "weight_transfer_config", "lora_config"):
            require(getattr(config, name, None) is None, name + " must be absent")
        report["protocol"] = {
            "tensor_parallel_size": 4, "pipeline_parallel_size": 1,
            "enable_prefix_caching": False, "speculative_config": None,
            "kv_transfer_config": None, "weight_transfer_config": None,
            "dtype": str(config.model_config.dtype),
        }
        import importlib
        report["runtime_sources"] = {}
        for name, digest in FROZEN_SOURCES.items():
            identity = file_identity(importlib.import_module(name).__file__)
            report["runtime_sources"][name] = identity
            require(identity["sha256"] == digest, f"frozen runtime source changed: {name}")
        report["parent_worker"] = source_identity(NPUWorker)

    def _plan_and_layers(self):
        from drror_vllm_ascend.envs import get_config
        from drror_vllm_ascend.runtime_plan import load_runtime_plan
        from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention

        config = get_config()
        require(config.enable and not config.capture_enable, "an enabled bound DRRQR plan is required")
        plan = load_runtime_plan(config, self.vllm_config)
        ref = getattr(plan, "reference", plan)
        expected = {i for i, kind in enumerate(ref.layer_types) if kind == "linear_attention"}
        require(len(expected) == 48, "the model must have exactly 48 GDN layers")
        layers = {}
        for name, module in self.model_runner.get_model().named_modules():
            if not isinstance(module, GatedDeltaNetAttention):
                continue
            match = PREFIX.search(module.prefix)
            require(match is not None, f"noncanonical GDN prefix {module.prefix}")
            layer = int(match[1])
            require(layer not in layers, f"duplicate GDN layer {layer}")
            layers[layer] = (name, module)
        require(set(layers) == expected, "actual GDN layers differ from the plan topology")
        return plan, layers

    def load_model(self):
        require(not getattr(self, "_layerwise_probe_load_called", False), "load_model may run only once")
        self._layerwise_probe_load_called = True
        with self._report("load") as report:
            self._admit(report)
            super().load_model()
            plan, layers = self._plan_and_layers()
            report["plan_sha256"] = plan.digest
            report["plan_class"] = type(plan).__name__
            report["model_class"] = source_identity(type(self.model_runner.get_model()))
            for layer, (name, module) in sorted(layers.items()):
                record = {
                    "layer": layer, "module_name": name, "prefix": module.prefix,
                    "module_class": source_identity(type(module)),
                    **check_module(module, plan, layer, self.rank), "parameters": [],
                }
                report["layers"].append(record)
                parameters = list(module.named_parameters(recurse=True, remove_duplicate=False))
                require(bool(parameters), f"layer {layer}: no local parameters")
                for parameter_name, parameter in parameters:
                    require(
                        parameter.device.type == "npu" and parameter.device.index == self.local_rank,
                        f"layer {layer}: {parameter_name} is not resident on this rank's NPU",
                    )
                    record["parameters"].append(parameter_identity(parameter_name, parameter))
                require(
                    [(n, id(p)) for n, p in module.named_parameters(recurse=True, remove_duplicate=False)]
                    == [(n, id(p)) for n, p in parameters],
                    f"layer {layer}: parameter registration changed while hashing",
                )
                for method in ("forward", "_forward_core", "get_state_shape", "get_state_dtype"):
                    if hasattr(module, method):
                        record.setdefault("method_sources", {})[method] = source_identity(getattr(module, method))
            report["layer_count"] = len(layers)
            report["parameter_bytes_hashed"] = sum(
                p["numel"] * p["element_size"] for r in report["layers"] for p in r["parameters"]
            )
        self._layerwise_probe_load_passed = True

    def initialize_from_config(self, kv_cache_config):
        require(not getattr(self, "_layerwise_probe_cache_called", False), "cache binding may run only once")
        self._layerwise_probe_cache_called = True
        with self._report("cache") as report:
            require(getattr(self, "_layerwise_probe_load_passed", False), "load qualification must pass first")
            self._admit(report)
            super().initialize_from_config(kv_cache_config)
            from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

            plan, layers = self._plan_and_layers()
            report["plan_sha256"] = plan.digest
            report["plan_class"] = type(plan).__name__
            runner = self.model_runner
            config = runner.kv_cache_config
            report["num_blocks"] = config.num_blocks
            specs = {}
            for group_index, group in enumerate(config.kv_cache_groups):
                for prefix in group.layer_names:
                    require(prefix not in specs, f"duplicate cache spec for {prefix}")
                    spec = group.kv_cache_spec
                    if isinstance(spec, UniformTypeKVCacheSpecs):
                        spec = spec.kv_cache_specs[prefix]
                    specs[prefix] = (group_index, spec)
            context = self.vllm_config.compilation_config.static_forward_context
            for layer, (name, module) in sorted(layers.items()):
                record = {
                    "layer": layer, "module_name": name, "prefix": module.prefix,
                    **check_module(module, plan, layer, self.rank), "states": [],
                }
                report["layers"].append(record)
                require(context.get(module.prefix) is module, f"layer {layer}: static context has a different module")
                require(module.prefix in specs, f"layer {layer}: cache spec is absent")
                group_index, spec = specs[module.prefix]
                require(isinstance(spec, MambaSpec), f"layer {layer}: cache spec is not MambaSpec")
                require(spec == module.get_kv_cache_spec(self.vllm_config), f"layer {layer}: assigned spec differs from module")
                record["cache_group_index"] = group_index
                record["mamba_spec"] = spec_identity(spec)
                cache = module.kv_cache
                require(isinstance(cache, (list, tuple)) and len(cache) == 2, f"layer {layer}: wrong cache tuple")
                slots = [i for i, entry in enumerate(runner.kv_caches) if entry is cache]
                require(len(slots) == 1, f"layer {layer}: runner cache identity is not unique")
                record["runner_cache_index"] = slots[0]
                for index, (tensor, shape, dtype) in enumerate(zip(cache, spec.shapes, spec.dtypes)):
                    before = tensor_metadata(tensor)
                    require(tensor.device.type == "npu", f"layer {layer}: state {index} is not on NPU")
                    require(tensor.device.index == self.local_rank, f"layer {layer}: state {index} is on another NPU")
                    require(tuple(tensor.shape[1:]) == tuple(shape), f"layer {layer}: state {index} shape mismatch")
                    require(tensor.shape[0] >= config.num_blocks > 0, f"layer {layer}: insufficient cache blocks")
                    require(tensor.dtype == dtype, f"layer {layer}: state {index} dtype mismatch")
                    require(tensor.data_ptr() != 0, f"layer {layer}: null cache pointer")
                    require(tensor.numel() == tensor.shape[0] * math.prod(shape), f"layer {layer}: state size mismatch")
                    require(context[module.prefix].kv_cache[index] is tensor, f"layer {layer}: state binding identity mismatch")
                    require(tensor_metadata(tensor) == before, f"layer {layer}: cache metadata changed during probe")
                    record["states"].append({"index": index, **before})
                record["binding_identity_verified"] = True
            report["layer_count"] = len(layers)
            report["cache_values_read"] = False
        self._layerwise_probe_cache_passed = True
