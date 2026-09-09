#!/usr/bin/env python3
"""NPU test of real Ascend runner allocation for unpadded per-layer GDN specs.

This deliberately isolates GDN cache allocation. It does not patch a serving
model or claim that mixed full-attention/GDN scheduler integration is complete.
"""
import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace, MethodType
from unittest.mock import patch
import time

import heterogeneous_state_probe as probe

torch = probe.torch
torch_npu = probe.torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if Path("/drrqr-results") not in out.parents or out.exists():
        parser.error("fresh directory below /drrqr-results required")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3":
        parser.error("requires externally audited physical4-7 to logical0-3 mapping")

    from vllm_ascend.utils import enable_custom_op
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    torch_npu.npu.set_compile_mode(jit_compile=False)
    if not enable_custom_op():
        raise RuntimeError("native custom op not loaded")
    torch.npu.set_device(0)
    init_device_properties_triton()

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups, get_kv_cache_config_from_groups

    runner_source = inspect.getsourcefile(NPUModelRunner)
    if probe.checksum(runner_source) != "94d75dbeb5d23ab5b383cdc69114167392b967994bb21342cf748b733ba9a968":
        raise RuntimeError("runner source hash changed; re-audit")
    if probe.checksum(Path(__file__).with_name("heterogeneous_state_probe.py")) != "5e1a726710f5a3ef3da81f651dbe4a61da833e97fd6e9061a65c97047be27982":
        raise RuntimeError("reviewed v4 numerical helper changed")

    report = {
        "schema": "drrqr-heterogeneous-runner-allocation/v1",
        "status": "running", "created_unix_ns": time.time_ns(),
        "script_sha256": probe.checksum(__file__),
        "source_hashes": {name: probe.checksum(inspect.getsourcefile(importlib.import_module(name)))
                          for name in ("vllm_ascend.worker.model_runner_v1",
                                       "vllm.v1.core.kv_cache_utils", "vllm.v1.kv_cache_interface")},
        "protocol": {
            "layouts": probe.LAYOUTS, "batches": [1, 16], "steps": probe.STEPS,
            "state_dtype": "bfloat16", "prefix_cache": False, "mamba_cache_mode": "none",
            "allocation": "unmodified NPUModelRunner allocation and reshape methods",
            "grouping": "unmodified get_kv_cache_groups on GDN-only specs without global padding",
            "criterion": "finite bitwise state/output equality; unchanged conv sentinel and pointers",
            "budget": "one four-case matrix; stop on first failure",
        },
        "limits": "GDN-only allocation/reshape and recurrent graph probe using a minimal runner context; not a model, scheduler, TP service, performance, or accuracy test.",
        "allocations": [], "cases": [],
    }
    out.mkdir(parents=True)
    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def allocate(dks, slots):
        config = SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks_override=slots, mamba_cache_mode="none",
                                         enable_prefix_caching=False),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            kv_transfer_config=None)
        specs = {}
        for layer, dk in enumerate(dks):
            shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(4, 16, 48, dk, 128, 4, 0)
            specs[f"model.layers.{layer}.linear_attn"] = MambaSpec(
                block_size=128, shapes=tuple(shapes),
                dtypes=(torch.bfloat16, torch.bfloat16), page_size_padded=None,
                mamba_type=MambaAttentionBackendEnum.GDN_ATTN, mamba_cache_mode="none")
        groups = get_kv_cache_groups(config, specs)
        expected_bytes = sum(s.page_size_bytes for s in specs.values()) * slots
        cache_config = get_kv_cache_config_from_groups(config, groups, expected_bytes)
        if cache_config.num_blocks != slots:
            raise RuntimeError("unexpected block count")
        runner = SimpleNamespace(vllm_config=config, device=torch.device("npu:0"),
                                 runner_only_attn_layers=set(), use_compress=False)
        runner._get_layer_kv_cache_specs = MethodType(NPUModelRunner._get_layer_kv_cache_specs, runner)
        # Only provide the metadata grouping used by the real reshape routine.
        # Each layer keeps its real MambaSpec; no compute or allocation is mocked.
        attn_groups = [SimpleNamespace(backend=None, kv_cache_spec=spec, layer_names=[name])
                       for name, spec in specs.items()]
        runner._kv_cache_spec_attn_group_iterator = lambda: iter(attn_groups)
        raw = NPUModelRunner._allocate_kv_cache_tensors(runner, cache_config)
        cache = NPUModelRunner._reshape_kv_cache_tensors(runner, cache_config, raw)
        torch.npu.synchronize()
        states, sentinels = [], []
        for name, spec in specs.items():
            conv, state = cache[name]
            if tuple(state.shape) != (slots, *spec.shapes[1]) or not state.is_contiguous():
                raise RuntimeError("layer state shape/contiguity mismatch")
            if raw[name].numel() != slots * spec.page_size_bytes:
                raise RuntimeError("unexpected allocation padding")
            conv.fill_(probe.SENTINEL)
            sentinels.append(conv)
            states.append(state)
        unique_bytes = {x.untyped_storage().data_ptr(): x.untyped_storage().nbytes() for x in raw.values()}
        if sum(unique_bytes.values()) != expected_bytes:
            raise RuntimeError("storage allocation differs from logical cache budget")
        allocation = {
            "dks": dks, "slots": slots,
            "group_types": [type(g.kv_cache_spec).__name__ for g in groups],
            "page_bytes_by_layer": {n: s.page_size_bytes for n, s in specs.items()},
            "allocated_storage_bytes": sum(unique_bytes.values()), "expected_bytes": expected_bytes,
            "naive_max_layer_padding_bytes": max(s.page_size_bytes for s in specs.values()) * len(specs) * slots,
            "state_shapes": [list(s.shape) for s in states],
            "state_pointer_mod_512": [s.data_ptr() % 512 for s in states],
            "guards": "entire adjacent convolution states retain exact sentinel; no extra outer canaries",
        }
        report["allocations"].append(allocation)
        save()
        # The numerical helper tracks every state pointer. Views retain all raw
        # storages; the first raw tensor is an additional tracked owner.
        return next(iter(raw.values())), states, sentinels

    save()
    try:
        with patch.object(probe, "arena", allocate):
            for name, dks in probe.LAYOUTS.items():
                for batch in (1, 16):
                    result = probe.case(name, dks, batch)
                    report["cases"].append(result)
                    save()
                    print(json.dumps({"layout": name, "batch": batch, "passed": result["passed"]}), flush=True)
                    if not result["passed"]:
                        raise RuntimeError("native runner allocation parity failed")
        report["status"] = "native_runner_gdn_allocation_pass"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["completed_unix_ns"] = time.time_ns()
        save()
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
