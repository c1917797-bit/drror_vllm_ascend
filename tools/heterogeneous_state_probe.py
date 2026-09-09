#!/usr/bin/env python3
"""NPU ABI probe for BF16 heterogeneous GDN state arenas; not a serving integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import torch_npu

H_QK, H_V, D_V = 4, 12, 128
STATE_DTYPE = torch.bfloat16
ALIGNMENT_BYTES = 512
GUARD = ALIGNMENT_BYTES // torch.tensor([], dtype=STATE_DTYPE).element_size()
SENTINEL = 123.0
STEPS = 8
LAYOUTS = {"uniform64": (64, 64, 64, 64), "heterogeneous": (128, 32, 32, 64)}


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    return bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.equal(a, b))


def arena(dks, slots):
    sizes = [slots * H_V * D_V * dk for dk in dks]
    storage = torch.full((sum(sizes) + GUARD * (len(sizes) + 1),),
                         SENTINEL, dtype=STATE_DTYPE, device="npu:0")
    views, guards = [], [storage[:GUARD]]
    offset = GUARD
    for dk, size in zip(dks, sizes):
        views.append(storage.narrow(0, offset, size).view(slots, H_V, D_V, dk))
        offset += size
        guards.append(storage[offset:offset + GUARD])
        offset += GUARD
    return storage, views, guards


def operation(inputs, state):
    return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=inputs["q"], key=inputs["k"], value=inputs["v"],
        g=inputs["g"], beta=inputs["beta"], state=state,
        scale=inputs["q"].shape[-1] ** -0.5,
        actual_seq_lengths=inputs["lengths"], ssm_state_indices=inputs["indices"])


def case(name, dks, batch):
    slots = batch + 3
    generator = torch.Generator().manual_seed(20260909 + batch)
    inputs, initial = [], []
    for dk in dks:
        initial.append((torch.randn((slots, H_V, D_V, dk), generator=generator) * 0.01).to(STATE_DTYPE))
        inputs.append({
            "q": torch.zeros((batch, H_QK, dk), dtype=torch.bfloat16, device="npu:0"),
            "k": torch.zeros((batch, H_QK, dk), dtype=torch.bfloat16, device="npu:0"),
            "v": torch.zeros((batch, H_V, D_V), dtype=torch.bfloat16, device="npu:0"),
            "g": torch.full((batch, H_V), -0.15, dtype=torch.float32, device="npu:0"),
            "beta": torch.full((batch, H_V), 0.5, dtype=torch.bfloat16, device="npu:0"),
            "lengths": torch.cat((torch.zeros(1, dtype=torch.int32),
                                  torch.ones(batch, dtype=torch.int32))).to("npu:0"),
            "indices": torch.arange(batch, dtype=torch.int32).to("npu:0"),
        })
    reference = [x.to("npu:0") for x in initial]
    eager_storage, eager, eager_guards = arena(dks, slots)
    graph_storage, graphed, graph_guards = arena(dks, slots)
    for states in (eager, graphed):
        for state, value in zip(states, initial):
            state.copy_(value)
    pointer_alignment = {
        "reference": [state.data_ptr() % ALIGNMENT_BYTES for state in reference],
        "eager": [state.data_ptr() % ALIGNMENT_BYTES for state in eager],
        "graph": [state.data_ptr() % ALIGNMENT_BYTES for state in graphed],
    }
    if any(value != 0 for values in pointer_alignment.values() for value in values):
        raise RuntimeError(f"state pointer alignment failed: {pointer_alignment}")
    for layer, (data, state) in enumerate(zip(inputs, reference)):
        print(json.dumps({"phase": "independent_sanity", "layout": name,
                          "batch": batch, "layer": layer, "dk": dks[layer],
                          "pointer_mod_512": state.data_ptr() % ALIGNMENT_BYTES}), flush=True)
        operation(data, state)
        torch.npu.synchronize()
    for state, value in zip(reference, initial):
        state.copy_(value)
    torch.npu.synchronize()
    for warmup in range(3):
        for layer, (data, state) in enumerate(zip(inputs, graphed)):
            print(json.dumps({"phase": "arena_warmup", "layout": name,
                              "batch": batch, "warmup": warmup, "layer": layer,
                              "dk": dks[layer],
                              "pointer_mod_512": state.data_ptr() % ALIGNMENT_BYTES}), flush=True)
            operation(data, state)
            torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_outputs = [operation(data, state) for data, state in zip(inputs, graphed)]
    torch.npu.synchronize()
    # Restore values after warmup/capture without rebinding captured storage.
    for state, value in zip(graphed, initial):
        state.copy_(value)
    torch.npu.synchronize()
    tracked = [eager_storage, graph_storage, *eager, *graphed]
    tracked += [tensor for data in inputs for tensor in data.values()]
    pointers = [tensor.data_ptr() for tensor in tracked]
    records = []
    try:
        for step in range(STEPS):
            for dk, data in zip(dks, inputs):
                for key in ("q", "k"):
                    x = torch.randn((batch, H_QK, dk), generator=generator)
                    data[key].copy_(F.normalize(x, dim=-1).to(torch.bfloat16))
                data["v"].copy_((torch.randn((batch, H_V, D_V), generator=generator) * 0.1).to(torch.bfloat16))
                data["g"].copy_(F.logsigmoid(torch.randn((batch, H_V), generator=generator) * 0.25 + 2.0))
                data["beta"].copy_(torch.sigmoid(torch.randn((batch, H_V), generator=generator)).to(torch.bfloat16))
                # Cyclic unique slots change in place, exercising state indirection.
                data["indices"].copy_((torch.arange(batch, dtype=torch.int32) + step) % slots)
            refs = [operation(data, state) for data, state in zip(inputs, reference)]
            outs = [operation(data, state) for data, state in zip(inputs, eager)]
            graph.replay()
            torch.npu.synchronize()
            checks = {
                "eager_output_exact": all(exact(a, b) for a, b in zip(outs, refs)),
                "graph_output_exact": all(exact(a, b) for a, b in zip(graph_outputs, refs)),
                "eager_full_state_exact": all(exact(a, b) for a, b in zip(eager, reference)),
                "graph_full_state_exact": all(exact(a, b) for a, b in zip(graphed, reference)),
                "guards_intact": all(bool((g.cpu() == SENTINEL).all()) for g in eager_guards + graph_guards),
                "pointers_unchanged": pointers == [tensor.data_ptr() for tensor in tracked],
            }
            records.append({"step": step + 1, **checks})
            if not all(checks.values()):
                break
    finally:
        graph.reset()
    return {"layout": name, "dks": dks, "batch": batch, "slots": slots,
            "state_layout": "N,Nv,Dv,Dk", "state_dtype": "bfloat16",
            "alignment_bytes": ALIGNMENT_BYTES, "pointer_alignment": pointer_alignment,
            "logical_state_bytes": sum(x.numel() * x.element_size() for x in eager),
            "steps": records,
            "passed": len(records) == STEPS and all(
                all(v for k, v in row.items() if k != "step") for row in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if Path("/drrqr-results") not in out.parents or out.exists():
        parser.error("a fresh directory below /drrqr-results is required")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3":
        parser.error("requires externally audited physical4-7 to logical0-3 mapping")
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.utils import enable_custom_op
    torch_npu.npu.set_compile_mode(jit_compile=False)
    if not enable_custom_op():
        raise RuntimeError("native custom operators unavailable")
    torch.npu.set_device(0)
    init_device_properties_triton()
    source = Path("/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py")
    if checksum(source) != "d6ec29919268178f5bf6e70e689c1d273d04b1cb1d84dc94efa7bbbc35490816":
        raise RuntimeError("Ascend GDN source changed; re-audit")
    out.mkdir(parents=True)
    report = {
        "schema": "drrqr-heterogeneous-state-probe/v4", "status": "running",
        "created_unix_ns": time.time_ns(), "script_sha256": checksum(__file__),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "device": "logical npu:0; externally verified physical NPU4",
        "operator_schema": str(torch.ops._C_ascend.npu_recurrent_gated_delta_rule.default._schema),
        "ascend_gdn_sha256": checksum(source),
        "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "protocol": {"layouts": LAYOUTS, "batches": [1, 16], "steps": STEPS,
                     "state_dtype": "bfloat16", "beta_dtype": "bfloat16",
                     "arena_alignment_bytes": ALIGNMENT_BYTES,
                     "diagnostics": "independent sanity then per-layer synchronized arena warmup",
                     "initialization": "golden order: compile mode, custom op, device, triton properties",
                     "criterion": "finite bitwise equality; full state and output, intact guards",
                     "budget": "one matrix, no retries", "graph": "torch.npu.NPUGraph"},
        "limits": "Synthetic native ABI/storage probe only, not model cache-manager integration, throughput or task accuracy.",
        "cases": [],
    }
    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        for name, dks in LAYOUTS.items():
            for batch in (1, 16):
                result = case(name, dks, batch)
                report["cases"].append(result)
                save()
                print(json.dumps({"layout": name, "batch": batch, "passed": result["passed"]}), flush=True)
                if not result["passed"]:
                    raise RuntimeError("exact NPU parity failed; stop without relaxing criterion")
        report["status"] = "native_heterogeneous_state_abi_pass"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        # Record actual process-loaded custom shared objects, not just intended paths.
        libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if "/" in line and ("/vllm_ascend/" in line or "libcust_opapi" in line)})
        report["loaded_libraries"] = [{"path": p, "sha256": checksum(p)}
                                     for p in libraries if Path(p).is_file()]
        report["completed_unix_ns"] = time.time_ns()
        save()
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
