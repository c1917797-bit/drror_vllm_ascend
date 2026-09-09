#!/usr/bin/env python3
"""NPU ABI probe for BF16/FP32 heterogeneous GDN state arenas.

This is not a serving integration or performance test. Historical v4 used
finite torch.equal element equality, which did not distinguish signed zero;
v5 requires equal dtype/shape and exact finite logical tensor bytes. This
does not assert that any v4 result actually contained a bit mismatch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
import torch_npu

H_QK, H_V, D_V = 4, 12, 128
STATE_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
STATE_DTYPE = STATE_DTYPES["bfloat16"]  # Legacy helper API default.
ALIGNMENT_BYTES = 512
GUARD = ALIGNMENT_BYTES // 2  # Legacy BF16 guard; typed paths use guard_elements().
SENTINEL = 123.0
STEPS = 8
LAYOUTS = {"uniform64": (64, 64, 64, 64), "heterogeneous": (128, 32, 32, 64)}
ASCEND_GDN_SHA256 = "d6ec29919268178f5bf6e70e689c1d273d04b1cb1d84dc94efa7bbbc35490816"


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return False
    return bool(torch.equal(a.reshape(-1).view(torch.uint8),
                            b.reshape(-1).view(torch.uint8)))


def guard_elements(state_dtype):
    element_size = torch.empty((), dtype=state_dtype).element_size()
    if ALIGNMENT_BYTES % element_size:
        raise ValueError("guard alignment is not divisible by state element size")
    return ALIGNMENT_BYTES // element_size


def tensor_metadata(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device),
            "data_ptr": tensor.data_ptr(),
            "storage_ptr": tensor.untyped_storage().data_ptr(),
            "storage_offset": tensor.storage_offset()}


def unchanged_slots(before, after, selected):
    """Compare unselected slots in already materialized CPU state snapshots."""
    if before.shape != after.shape or before.dtype != after.dtype:
        return False
    if before.device.type != "cpu" or after.device.type != "cpu":
        raise ValueError("unselected-slot comparison requires CPU snapshots")
    slots = before.shape[0]
    if any(type(i) is not int or not 0 <= i < slots for i in selected):
        raise ValueError("selected state indices are outside the slot range")
    unselected = [i for i in range(slots) if i not in set(selected)]
    return exact(before[unselected], after[unselected])


def reset_graph(graph, record, original_error):
    if graph is None:
        return
    try:
        graph.reset()
    except BaseException as error:
        record["graph_cleanup_error"] = repr(error)
        if original_error is None:
            record.update(status="failed", passed=False, failed_phase="graph_reset")
            raise


def arena(dks, slots, state_dtype=STATE_DTYPE):
    guard = guard_elements(state_dtype)
    sizes = [slots * H_V * D_V * dk for dk in dks]
    element_size = torch.empty((), dtype=state_dtype).element_size()
    if any(size * element_size % ALIGNMENT_BYTES for size in sizes):
        raise ValueError("state sizes do not preserve the required arena alignment")
    storage = torch.full((sum(sizes) + guard * (len(sizes) + 1),),
                         SENTINEL, dtype=state_dtype, device="npu:0")
    views, guards = [], [storage[:guard]]
    offset = guard
    for dk, size in zip(dks, sizes):
        views.append(storage.narrow(0, offset, size).view(slots, H_V, D_V, dk))
        offset += size
        guards.append(storage[offset:offset + guard])
        offset += guard
    return storage, views, guards


def operation(inputs, state):
    return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=inputs["q"], key=inputs["k"], value=inputs["v"],
        g=inputs["g"], beta=inputs["beta"], state=state,
        scale=inputs["q"].shape[-1] ** -0.5,
        actual_seq_lengths=inputs["lengths"], ssm_state_indices=inputs["indices"])


def case(name, dks, batch, state_dtype=None, *, record=None, progress=None):
    record = {} if record is None else record
    legacy_api = state_dtype is None
    state_dtype = STATE_DTYPE if legacy_api else state_dtype
    slots = batch + 3
    dtype_name = str(state_dtype).removeprefix("torch.")
    record.update(layout=name, dks=dks, batch=batch, slots=slots,
                  state_layout="N,Nv,Dv,Dk", state_dtype=dtype_name,
                  alignment_bytes=ALIGNMENT_BYTES,
                  guard_elements=guard_elements(state_dtype),
                  state_element_size=torch.empty((), dtype=state_dtype).element_size(),
                  status="running", passed=False, steps=[])
    graph = None

    def phase(value, **details):
        record["phase"] = value
        record["active"] = details
        if progress is not None:
            progress()

    try:
        phase("allocation")
        generator = torch.Generator().manual_seed(20260909 + batch)
        inputs, initial = [], []
        for dk in dks:
            initial.append((torch.randn((slots, H_V, D_V, dk), generator=generator) * 0.01).to(state_dtype))
            inputs.append({
                "q": torch.zeros((batch, H_QK, dk), dtype=torch.bfloat16, device="npu:0"),
                "k": torch.zeros((batch, H_QK, dk), dtype=torch.bfloat16, device="npu:0"),
                "v": torch.zeros((batch, H_V, D_V), dtype=torch.bfloat16, device="npu:0"),
                "g": torch.full((batch, H_V), -0.15, dtype=torch.float32, device="npu:0"),
                "beta": torch.full((batch, H_V), 0.5, dtype=torch.bfloat16, device="npu:0"),
                # Frozen native ABI: initial offset, then per-sequence lengths.
                "lengths": torch.cat((torch.zeros(1, dtype=torch.int32),
                                      torch.ones(batch, dtype=torch.int32))).to("npu:0"),
                "indices": torch.arange(batch, dtype=torch.int32).to("npu:0"),
            })
        reference = [x.to("npu:0") for x in initial]
        # The older allocator probe supplies a two-argument BF16 arena factory.
        arena_args = (dks, slots) if legacy_api else (dks, slots, state_dtype)
        eager_storage, eager, eager_guards = arena(*arena_args)
        graph_storage, graphed, graph_guards = arena(*arena_args)
        for states in (eager, graphed):
            for state, value in zip(states, initial):
                state.copy_(value)
        state_groups = {"reference": reference, "eager": eager, "graph": graphed}
        pointer_alignment = {
            kind: [state.data_ptr() % ALIGNMENT_BYTES for state in states]
            for kind, states in state_groups.items()
        }
        record["pointer_alignment"] = pointer_alignment
        record["actual_state_dtypes"] = {
            kind: [str(state.dtype) for state in states]
            for kind, states in state_groups.items()
        }
        if any(state.dtype != state_dtype for states in state_groups.values() for state in states):
            raise RuntimeError("allocated state dtype differs from requested dtype")
        if any(value != 0 for values in pointer_alignment.values() for value in values):
            raise RuntimeError(f"state pointer alignment failed: {pointer_alignment}")
        for layer, (data, state) in enumerate(zip(inputs, reference)):
            phase("independent_sanity", layer=layer, dk=dks[layer])
            print(json.dumps({"phase": "independent_sanity", "layout": name,
                              "batch": batch, "layer": layer, "dk": dks[layer],
                              "state_dtype": dtype_name,
                              "pointer_mod_512": state.data_ptr() % ALIGNMENT_BYTES}), flush=True)
            operation(data, state)
            torch.npu.synchronize()
        for state, value in zip(reference, initial):
            state.copy_(value)
        torch.npu.synchronize()
        for warmup in range(3):
            for layer, (data, state) in enumerate(zip(inputs, graphed)):
                phase("arena_warmup", warmup=warmup, layer=layer, dk=dks[layer])
                print(json.dumps({"phase": "arena_warmup", "layout": name,
                                  "batch": batch, "warmup": warmup, "layer": layer,
                                  "dk": dks[layer], "state_dtype": dtype_name,
                                  "pointer_mod_512": state.data_ptr() % ALIGNMENT_BYTES}), flush=True)
                operation(data, state)
                torch.npu.synchronize()
        phase("graph_capture")
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_outputs = [operation(data, state) for data, state in zip(inputs, graphed)]
        torch.npu.synchronize()
        # Restore values after warmup/capture without rebinding captured storage.
        for state, value in zip(graphed, initial):
            state.copy_(value)
        torch.npu.synchronize()
        tracked = {"eager.storage": eager_storage, "graph.storage": graph_storage}
        for kind, states in state_groups.items():
            tracked.update({f"{kind}.state.{i}": state for i, state in enumerate(states)})
        for layer, data in enumerate(inputs):
            tracked.update({f"input.{layer}.{key}": tensor for key, tensor in data.items()})
        tracked.update({f"graph.output.{i}": tensor for i, tensor in enumerate(graph_outputs)})
        metadata = {name: tensor_metadata(tensor) for name, tensor in tracked.items()}
        record["tracked_tensor_metadata"] = metadata
        record["logical_state_bytes"] = sum(x.numel() * x.element_size() for x in eager)
        previous_states = {
            kind: [state.detach().cpu() for state in states]
            for kind, states in state_groups.items()
        }
        for step in range(STEPS):
            phase("step_inputs", step=step + 1)
            selected = ((torch.arange(batch, dtype=torch.int32) + step) % slots).tolist()
            for dk, data in zip(dks, inputs):
                for key in ("q", "k"):
                    x = torch.randn((batch, H_QK, dk), generator=generator)
                    data[key].copy_(F.normalize(x, dim=-1).to(torch.bfloat16))
                data["v"].copy_((torch.randn((batch, H_V, D_V), generator=generator) * 0.1).to(torch.bfloat16))
                data["g"].copy_(F.logsigmoid(torch.randn((batch, H_V), generator=generator) * 0.25 + 2.0))
                data["beta"].copy_(torch.sigmoid(torch.randn((batch, H_V), generator=generator)).to(torch.bfloat16))
                # Cyclic unique slots change in place, exercising state indirection.
                data["indices"].copy_(torch.tensor(selected, dtype=torch.int32))
            phase("step_execute", step=step + 1)
            refs = [operation(data, state) for data, state in zip(inputs, reference)]
            outs = [operation(data, state) for data, state in zip(inputs, eager)]
            graph.replay()
            torch.npu.synchronize()
            phase("step_compare", step=step + 1)
            current_states = {
                kind: [state.detach().cpu() for state in states]
                for kind, states in state_groups.items()
            }
            observed = {name: tensor_metadata(tensor) for name, tensor in tracked.items()}
            checks = {
                "eager_output_exact": all(exact(a, b) for a, b in zip(outs, refs)),
                "graph_output_exact": all(exact(a, b) for a, b in zip(graph_outputs, refs)),
                "eager_full_state_exact": all(exact(a, b) for a, b in zip(current_states["eager"], current_states["reference"])),
                "graph_full_state_exact": all(exact(a, b) for a, b in zip(current_states["graph"], current_states["reference"])),
                "guards_intact": all(bool((g.cpu() == SENTINEL).all()) for g in eager_guards + graph_guards),
                "pointers_unchanged": all(metadata[name]["data_ptr"] == value["data_ptr"] for name, value in observed.items()),
                "tensor_metadata_unchanged": metadata == observed,
                "output_dtype_bfloat16": all(x.dtype == torch.bfloat16 for x in [*refs, *outs, *graph_outputs]),
            }
            for kind in state_groups:
                checks[f"{kind}_unselected_slots_unchanged"] = all(
                    unchanged_slots(before, after, selected)
                    for before, after in zip(previous_states[kind], current_states[kind])
                )
            record["steps"].append({"step": step + 1, **checks})
            record["output_dtypes"] = {
                "reference": [str(x.dtype) for x in refs],
                "eager": [str(x.dtype) for x in outs],
                "graph": [str(x.dtype) for x in graph_outputs],
            }
            if progress is not None:
                progress()
            if not all(checks.values()):
                record["failed_checks"] = [key for key, value in checks.items() if not value]
                record["changed_tensor_metadata"] = {
                    name: {"before": metadata[name], "after": value}
                    for name, value in observed.items() if metadata[name] != value
                }
                raise RuntimeError("exact NPU parity or storage invariant failed")
            previous_states = current_states
        record.update(status="passed", passed=True)
        phase("complete")
        return record
    except BaseException as error:
        record.update(status="failed", passed=False, error=repr(error),
                      failed_phase=record.get("phase"), error_type=type(error).__name__)
        raise
    finally:
        reset_graph(graph, record, sys.exc_info()[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--state-dtype", choices=tuple(STATE_DTYPES), default="bfloat16",
                        help="one state dtype per process; use distinct new output directories")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if Path("/drrqr-results") not in out.parents or out.exists():
        parser.error("a fresh directory below /drrqr-results is required")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3":
        parser.error("requires externally audited physical4-7 to logical0-3 mapping")
    state_dtype = STATE_DTYPES[args.state_dtype]
    source = Path("/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py")
    out.mkdir(parents=True)
    report = {
        "schema": "drrqr-heterogeneous-state-probe/v5", "status": "running",
        "phase": "initializing", "created_unix_ns": time.time_ns(),
        "script_sha256": checksum(__file__),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "device": "logical npu:0; externally verified physical NPU4",
        "operator_schema": None, "ascend_gdn_sha256": None, "custom_opp_path": None,
        "protocol": {"layouts": LAYOUTS, "batches": [1, 16], "steps": STEPS,
                     "state_dtype": args.state_dtype, "beta_dtype": "bfloat16",
                     "qkv_dtype": "bfloat16", "g_dtype": "float32",
                     "arena_alignment_bytes": ALIGNMENT_BYTES,
                     "guard_elements": guard_elements(state_dtype),
                     "diagnostics": "independent sanity then per-layer synchronized arena warmup",
                     "initialization": "golden order: compile mode, custom op, device, triton properties",
                     "criterion": "same shape/dtype, finite logical tensor bytes exactly equal; full state/output, intact guards and unselected slots",
                     "metadata_criterion": "shape/stride/dtype/device/data_ptr/storage_ptr/storage_offset unchanged",
                     "budget": "one single-dtype matrix, no retries", "graph": "torch.npu.NPUGraph"},
        "historical_predicate": "v4 checked finite torch.equal element equality, which did not distinguish signed zero; no actual v4 bit mismatch is asserted",
        "limits": "Synthetic native ABI/storage probe only, not model cache-manager integration, throughput or task accuracy.",
        "cases": [],
    }
    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def phase(value):
        report["phase"] = value
        save()

    save()
    try:
        phase("source_check")
        report["ascend_gdn_sha256"] = checksum(source)
        if report["ascend_gdn_sha256"] != ASCEND_GDN_SHA256:
            raise RuntimeError("Ascend GDN source changed; re-audit")
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
        from vllm_ascend.utils import enable_custom_op
        phase("compile_mode")
        torch_npu.npu.set_compile_mode(jit_compile=False)
        phase("enable_custom_op")
        if not enable_custom_op():
            raise RuntimeError("native custom operators unavailable")
        phase("set_device")
        torch.npu.set_device(0)
        phase("triton_properties")
        init_device_properties_triton()
        report["operator_schema"] = str(torch.ops._C_ascend.npu_recurrent_gated_delta_rule.default._schema)
        report["custom_opp_path"] = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        for name, dks in LAYOUTS.items():
            for batch in (1, 16):
                record = {"layout": name, "dks": dks, "batch": batch,
                          "state_dtype": args.state_dtype, "status": "running",
                          "phase": "pending", "passed": False, "steps": []}
                report["cases"].append(record)
                phase("case")
                case(name, dks, batch, state_dtype, record=record, progress=save)
                save()
                print(json.dumps({"layout": name, "batch": batch,
                                  "state_dtype": args.state_dtype,
                                  "passed": record["passed"]}), flush=True)
                if not record["passed"]:
                    raise RuntimeError("exact NPU parity failed; stop without relaxing criterion")
        report["status"] = "native_heterogeneous_state_abi_pass"
        report["phase"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        report["error_type"] = type(error).__name__
        raise
    finally:
        # Record mapped custom objects; this alone does not prove symbol selection.
        try:
            libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                if "/" in line and ("/vllm_ascend/" in line or "libcust_opapi" in line)})
            report["loaded_libraries"] = [{"path": p, "sha256": checksum(p)}
                                         for p in libraries if Path(p).is_file()]
        except Exception as error:
            report["library_evidence_error"] = repr(error)
            report["status"] = "failed"
            # Preserve any original probe exception; a clean run with incomplete
            # evidence is rejected below after its failed receipt is saved.
        report["completed_unix_ns"] = time.time_ns()
        save()
    if report["status"] != "native_heterogeneous_state_abi_pass":
        raise RuntimeError("native probe evidence collection failed")
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
