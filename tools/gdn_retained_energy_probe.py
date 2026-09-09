#!/usr/bin/env python3
"""Synthetic Dk64 retained-key beta implementation gate, not model accuracy.

This synthetic probe does not consume real retained-energy distributions or
model V/g/beta/state captures. It is neither dense-recovery evidence nor real
recurrent replay; model V/g/beta/state captures are still unavailable.
The only NPU execution is one fixed four-case matrix on logical npu:0.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F

H_QK, H_V, D_V, D_K = 4, 12, 128, 64
BATCHES, STEPS = (1, 16), 8
R_CASES = {"identity": (1.0, 1.0, 1.0, 1.0),
           "retained_key": (0.5, 0.75, 0.875, 1.0)}
RELATIVE_L2_LIMIT, NORMALIZED_MAX_LIMIT = 2.0 ** -7, 2.0 ** -6
RESULTS_ROOT = Path("/drrqr-results")
GOLDEN_SOURCE = Path(__file__).with_name("gdn_golden_parity.py")
GOLDEN_SHA256 = "1a2d1a61040a58a6f0c1d90cdbf0fc17058f3273eaef8b3d1b80116c97d92cb9"
GDN_SOURCE = Path("/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py")
GDN_SHA256 = "d6ec29919268178f5bf6e70e689c1d273d04b1cb1d84dc94efa7bbbc35490816"
DISABLED_FLAGS = (
    "VLLM_ASCEND_DRRQR_ENABLE", "VLLM_ASCEND_DRRQR_CAPTURE_ENABLE",
    "VLLM_ASCEND_DRRQR_CONV_LAYOUT", "VLLM_ASCEND_DRRQR_PREFILL_MC2",
    "VLLM_ASCEND_DRRQR_PREFILL_MC2_MIXED",
)
SOURCE_FILES = (
    GDN_SOURCE,
    Path("/vllm-workspace/vllm-ascend/vllm_ascend/utils.py"),
    Path("/vllm-workspace/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/"
         "recurrent_gated_delta_rule_torch_adpt.h"),
    Path("/vllm-workspace/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/"
         "op_kernel/recurrent_gated_delta_rule.cpp"),
    Path("/vllm-workspace/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/"
         "op_kernel/recurrent_gated_delta_rule.h"),
)


def checksum(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


# This module's independent CPU recurrence is reused, not reimplemented.
_spec = importlib.util.spec_from_file_location("retained_energy_golden_reference", GOLDEN_SOURCE)
_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_golden)
decode_golden = _golden._decode_golden
DECODE_RTOL, DECODE_ATOL = _golden.DECODE_RTOL, _golden.DECODE_ATOL


def exact_finite(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return False
    return bool(torch.equal(a.reshape(-1).view(torch.uint8),
                            b.reshape(-1).view(torch.uint8)))


def error_metrics(actual, reference):
    result = {"shape_matches": actual.shape == reference.shape,
              "dtype_matches": actual.dtype == reference.dtype,
              "finite": False, "allclose": False, "max_abs": None,
              "relative_l2": None, "normalized_max": None, "passed": False}
    if not result["shape_matches"] or not result["dtype_matches"]:
        return result
    actual, reference = actual.detach().cpu().float(), reference.detach().cpu().float()
    result["finite"] = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not result["finite"]:
        return result
    difference = actual - reference
    maximum = float(difference.abs().max()) if difference.numel() else 0.0
    reference_max = float(reference.abs().max()) if reference.numel() else 0.0
    epsilon = torch.finfo(torch.float32).eps
    relative_l2 = float(torch.linalg.vector_norm(difference)) / max(
        float(torch.linalg.vector_norm(reference)), epsilon)
    normalized_max = maximum / max(reference_max, epsilon)
    result.update(max_abs=maximum, relative_l2=relative_l2, normalized_max=normalized_max,
                  allclose=bool(torch.allclose(actual, reference,
                                              rtol=DECODE_RTOL, atol=DECODE_ATOL)))
    result["passed"] = (result["allclose"] and relative_l2 <= RELATIVE_L2_LIMIT
                        and normalized_max <= NORMALIZED_MAX_LIMIT)
    return result


def value_head_energy(r_key):
    if (len(r_key) != H_QK
            or any(not math.isfinite(float(r)) or not 0.0 < float(r) <= 1.0 for r in r_key)):
        raise ValueError("this fixed probe requires four finite retained energies in (0,1]")
    return torch.tensor(r_key, dtype=torch.float32).repeat_interleave(H_V // H_QK)


def selected_slots(batch, step):
    if batch not in BATCHES or not 0 <= step < STEPS:
        raise ValueError("outside the fixed batch/step matrix")
    return (torch.arange(batch, dtype=torch.int32) + step) % (batch + 3)


def unselected_unchanged(before, after, indices):
    if (before.device.type != "cpu" or after.device.type != "cpu"
            or before.shape != after.shape or before.dtype != after.dtype):
        return False
    chosen = indices.tolist()
    if (len(chosen) != len(set(chosen))
            or any(type(i) is not int or not 0 <= i < before.shape[0] for i in chosen)):
        raise ValueError("state indices must be unique valid slots")
    keep = [i for i in range(before.shape[0]) if i not in set(chosen)]
    return exact_finite(before[keep], after[keep])


def metadata(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device),
            "data_ptr": tensor.data_ptr(), "storage_ptr": tensor.untyped_storage().data_ptr(),
            "storage_offset": tensor.storage_offset()}


def cpu_inputs(batch, step, generator, r_key):
    return {
        "q": F.normalize(torch.randn((batch, H_QK, D_K), generator=generator, dtype=torch.float32),
                         dim=-1).to(torch.bfloat16),
        "k": F.normalize(torch.randn((batch, H_QK, D_K), generator=generator, dtype=torch.float32),
                         dim=-1).to(torch.bfloat16),
        "v": (torch.randn((batch, H_V, D_V), generator=generator, dtype=torch.float32) * 0.1).to(torch.bfloat16),
        "g": F.logsigmoid(torch.randn((batch, H_V), generator=generator, dtype=torch.float32) * 0.25 + 2.0),
        "beta_base": torch.sigmoid(
            torch.randn((batch, H_V), generator=generator, dtype=torch.float32)).to(torch.bfloat16),
        "r": value_head_energy(r_key),
        # Native ABI is offset followed by sequence lengths, NOT cumulative sums.
        "lengths": torch.cat((torch.zeros(1, dtype=torch.int32),
                              torch.ones(batch, dtype=torch.int32))),
        "indices": selected_slots(batch, step),
    }


def quantized_beta(beta_base, r_value):
    # This exact function executes on NPU tensors inside NPUGraph capture/replay.
    if (beta_base.dtype != torch.bfloat16 or r_value.dtype != torch.float32
            or beta_base.ndim != 2 or beta_base.shape[-1] != H_V
            or tuple(r_value.shape) != (H_V,) or beta_base.device != r_value.device):
        raise ValueError("beta must be BF16 [batch,Hv], r FP32 [Hv], on the same device")
    product_fp32 = beta_base.to(torch.float32) * r_value
    return product_fp32, product_fp32.to(torch.bfloat16)


def native_recurrent(inputs, state, beta):
    return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=inputs["q"], key=inputs["k"], value=inputs["v"], g=inputs["g"],
        beta=beta, state=state, scale=D_K ** -0.5,
        actual_seq_lengths=inputs["lengths"], ssm_state_indices=inputs["indices"])


def compensated_operation(inputs, state):
    product, beta = quantized_beta(inputs["beta_base"], inputs["r"])
    output = native_recurrent(inputs, state, beta)
    return output, product, beta


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


def run_case(name, r_key, batch, record, save):
    graph = None
    record.update(name=name, batch=batch, slots=batch + 3, r_key=list(r_key),
                  r_value=value_head_energy(r_key).tolist(), steps=[],
                  status="running", passed=False)

    def phase(value, **details):
        record.update(phase=value, active=details)
        save()

    try:
        phase("allocation")
        generator = torch.Generator(device="cpu").manual_seed(20260909 + batch)
        initial = torch.randn((batch + 3, H_V, D_V, D_K), generator=generator, dtype=torch.float32) * 0.01
        state_cpu = initial.clone()
        eager_state, graph_state = initial.to("npu:0"), initial.to("npu:0")
        direct_state = initial.to("npu:0") if name == "identity" else None
        # Warmup/capture inputs use r=1; actual case r is copied into the same
        # captured NPU buffer before step 1, and is then constant over its 8 steps.
        warmup = cpu_inputs(batch, 0, generator, R_CASES["identity"])
        inputs = {key: value.to("npu:0") for key, value in warmup.items()}
        for iteration in range(3):
            phase("warmup", iteration=iteration)
            compensated_operation(inputs, graph_state)
            torch.npu.synchronize()
        phase("graph_capture")
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_output, graph_product, graph_beta = compensated_operation(inputs, graph_state)
        torch.npu.synchronize()
        graph_state.copy_(initial)
        torch.npu.synchronize()
        tracked = {**{"input." + key: value for key, value in inputs.items()},
                   "eager_state": eager_state, "graph_state": graph_state,
                   "graph_output": graph_output, "graph_product_fp32": graph_product,
                   "graph_beta_bfloat16": graph_beta}
        if direct_state is not None:
            tracked["identity_direct_state"] = direct_state
        expected_metadata = {key: metadata(value) for key, value in tracked.items()}
        record["tracked_tensor_metadata"] = expected_metadata
        previous_eager, previous_graph = initial.clone(), initial.clone()
        previous_direct = initial.clone() if direct_state is not None else None

        for step in range(STEPS):
            phase("inputs", step=step + 1)
            data = cpu_inputs(batch, step, generator, r_key)
            for key, value in data.items():
                inputs[key].copy_(value)
            cpu_product, cpu_beta = quantized_beta(data["beta_base"], data["r"])
            cpu_output, next_cpu = decode_golden(
                data["q"], data["k"], data["v"], state_cpu, cpu_beta, D_K ** -0.5,
                torch.ones(batch, dtype=torch.int32), data["indices"], data["g"])
            phase("native_execute", step=step + 1)
            eager_output, eager_product, eager_beta = compensated_operation(inputs, eager_state)
            direct_output = (native_recurrent(inputs, direct_state, inputs["beta_base"])
                             if direct_state is not None else None)
            graph.replay()
            torch.npu.synchronize()
            phase("compare", step=step + 1)
            eager_cpu, graph_cpu = eager_state.cpu(), graph_state.cpu()
            direct_cpu = direct_state.cpu() if direct_state is not None else None
            output_metrics = error_metrics(eager_output, cpu_output)
            state_metrics = error_metrics(eager_cpu, next_cpu)
            checks = {
                "output_cpu_parity": output_metrics["passed"],
                "full_state_cpu_parity": state_metrics["passed"],
                "graph_eager_output_bytes_equal": exact_finite(graph_output, eager_output),
                "graph_eager_full_state_bytes_equal": exact_finite(graph_cpu, eager_cpu),
                "eager_beta_product_fp32_cpu_exact": exact_finite(eager_product, cpu_product),
                "graph_beta_product_fp32_cpu_exact": exact_finite(graph_product, cpu_product),
                "eager_beta_bfloat16_cpu_exact": exact_finite(eager_beta, cpu_beta),
                "graph_beta_bfloat16_cpu_exact": exact_finite(graph_beta, cpu_beta),
                "eager_unselected_slots_unchanged": unselected_unchanged(
                    previous_eager, eager_cpu, data["indices"]),
                "graph_unselected_slots_unchanged": unselected_unchanged(
                    previous_graph, graph_cpu, data["indices"]),
                "cpu_unselected_slots_unchanged": unselected_unchanged(
                    state_cpu, next_cpu, data["indices"]),
                "tracked_metadata_unchanged":
                    expected_metadata == {key: metadata(value) for key, value in tracked.items()},
            }
            if direct_state is not None:
                checks.update(
                    identity_quantized_beta_bytes_equal=exact_finite(cpu_beta, data["beta_base"]),
                    identity_native_output_bytes_equal=exact_finite(eager_output, direct_output),
                    identity_native_full_state_bytes_equal=exact_finite(eager_cpu, direct_cpu),
                    identity_native_unselected_slots_unchanged=unselected_unchanged(
                        previous_direct, direct_cpu, data["indices"]))
            result = {"step": step + 1, "state_indices": data["indices"].tolist(),
                      "output_vs_cpu": output_metrics, "full_state_vs_cpu": state_metrics,
                      "checks": checks, "passed": all(checks.values())}
            record["steps"].append(result)
            save()
            if not result["passed"]:
                record["failed_checks"] = [key for key, value in checks.items() if not value]
                raise RuntimeError("predeclared numerical/byte/state gate failed; no retry")
            state_cpu = next_cpu
            previous_eager, previous_graph, previous_direct = eager_cpu, graph_cpu, direct_cpu
        record.update(status="passed", passed=True, phase="complete")
        save()
    except BaseException as error:
        record.update(status="failed", passed=False, error=repr(error),
                      error_type=type(error).__name__, failed_phase=record.get("phase"),
                      traceback=traceback.format_exc())
        raise
    finally:
        reset_graph(graph, record, sys.exc_info()[1])


def disable_plugin():
    before = {key: os.environ.get(key) for key in DISABLED_FLAGS}
    for key in DISABLED_FLAGS:
        os.environ[key] = "0"
    # Disable general-plugin discovery for this isolated native-op process.
    os.environ["VLLM_PLUGINS"] = ""
    return {"original_activation_flags": before,
            "effective_activation_flags": {key: os.environ[key] for key in DISABLED_FLAGS},
            "VLLM_PLUGINS": ""}


def initialize_npu(report, phase):
    if checksum(GOLDEN_SOURCE) != GOLDEN_SHA256 or checksum(GDN_SOURCE) != GDN_SHA256:
        raise RuntimeError("audited golden/GDN source identity changed")
    report["source_sha256"] = {str(path): checksum(path)
                                for path in (Path(__file__), GOLDEN_SOURCE, *SOURCE_FILES)}
    import torch_npu
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.utils import enable_custom_op
    report["torch_npu"] = torch_npu.__version__
    phase("compile_mode")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    phase("enable_custom_op")
    if not enable_custom_op():
        raise RuntimeError("native custom operator unavailable")
    phase("set_device")
    torch.npu.set_device(0)
    phase("triton_properties")
    init_device_properties_triton()
    patch_modules = sorted(name for name in sys.modules
                           if name.startswith("drror_vllm_ascend.patches"))
    report["plugin_patch_modules_loaded"] = patch_modules
    if patch_modules or any(os.environ.get(key) != "0" for key in DISABLED_FLAGS):
        raise RuntimeError("DRRQR plugin isolation failed")
    report["operator_schema"] = str(
        torch.ops._C_ascend.npu_recurrent_gated_delta_rule.default._schema)
    report["custom_opp_path"] = os.environ.get("ASCEND_CUSTOM_OPP_PATH")


def loaded_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = fields[5]
        if (".so" in path and ("vllm_ascend" in path
                or any(name in path for name in ("libcust_opapi", "libopapi.so",
                                                "libcust_opmaster", "libcust_opsproto")))):
            if not Path(path).is_file():
                raise RuntimeError("mapped operator library is not a readable file: " + path)
            paths.add(path)
    return [{"path": path, "sha256": checksum(path)} for path in sorted(paths)]


def fresh_output(path):
    path = Path(path)
    resolved = path.resolve()
    if (not path.is_absolute() or path != resolved or RESULTS_ROOT not in resolved.parents
            or resolved.exists()):
        raise ValueError("a fresh absolute non-symlink directory below /drrqr-results is required")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") not in ("0", "0,1,2,3"):
        parser.error("requires externally audited physical4-7 mapping; uses logical0 only")
    out = fresh_output(args.output_dir)
    report = {
        "schema": "drrqr-retained-key-beta-probe/v1", "status": "running",
        "created_unix_ns": time.time_ns(), "phase": "initializing", "cases": [],
        "script_sha256": checksum(__file__), "golden_sha256": checksum(GOLDEN_SOURCE),
        "torch": torch.__version__, "python": sys.version,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "device": "logical npu:0 only; physical mapping requires external audit",
        "plugin_isolation": disable_plugin(),
        "protocol": {
            "Dk": D_K, "Hqk": H_QK, "Hv": H_V, "Dv": D_V,
            "state_dtype": "float32", "qkv_beta_base_dtype": "bfloat16", "g_dtype": "float32",
            "state_layout": "slots,Hv,Dv,Dk", "slots": "batch+3",
            "state_interpretation": "Synthetic operator state T, not recovered dense state S.",
            "batches": list(BATCHES), "steps": STEPS, "r_cases": R_CASES,
            "r_value_mapping": "FP32 key-head energies.repeat_interleave(3)",
            "beta_path": "NPU beta_base.to(float32)*r_value -> bfloat16 inside captured graph",
            "rtol": DECODE_RTOL, "atol": DECODE_ATOL,
            "relative_l2": "norm(actual-reference)/max(norm(reference), FP32 epsilon)",
            "relative_l2_limit": RELATIVE_L2_LIMIT,
            "normalized_max": "max(abs(actual-reference))/max(max(abs(reference)), FP32 epsilon)",
            "normalized_max_limit": NORMALIZED_MAX_LIMIT,
            "admission": "All declared metrics, finite exact eager/graph bytes, quantized beta, "
                         "identity control, unselected slots and metadata checks must pass.",
            "budget": "one four-case 8-step matrix, three fixed graph warmups per case, no retries",
        },
        "limits": "Synthetic implementation gate only. This probe does not consume real "
                  "retained-energy distributions or model V/g/beta/state captures; model "
                  "V/g/beta/state captures remain unavailable. Not real recurrent replay, dense accuracy recovery, "
                  "a model test, timing, throughput or task-accuracy screening.",
    }

    def save():
        temporary = out / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(out / "report.json")

    def phase(value):
        report["phase"] = value
        save()

    save()
    try:
        initialize_npu(report, phase)
        for name, r_key in R_CASES.items():
            for batch in BATCHES:
                record = {"name": name, "batch": batch, "passed": False, "phase": "pending"}
                report["cases"].append(record)
                phase("case")
                run_case(name, r_key, batch, record, save)
                print(json.dumps({"case": name, "batch": batch, "passed": record["passed"]}),
                      flush=True)
        report.update(status="synthetic_retained_key_beta_implementation_pass", phase="complete")
    except BaseException as error:
        report.update(status="failed", error=repr(error), error_type=type(error).__name__,
                      traceback=traceback.format_exc())
        raise
    finally:
        try:
            report["loaded_libraries"] = loaded_libraries()
            report["library_evidence_scope"] = (
                "Actually mapped operator objects and their file hashes; mapping alone does not "
                "prove which same-named ACLNN symbol was selected.")
            if report["status"] != "failed" and not any(
                    "libcust_opapi" in item["path"] for item in report["loaded_libraries"]):
                raise RuntimeError("successful run lacks mapped custom-op library hash evidence")
        except BaseException as error:
            report.update(status="failed", library_evidence_error=repr(error))
        report["completed_unix_ns"] = time.time_ns()
        save()
    if report["status"] != "synthetic_retained_key_beta_implementation_pass":
        raise RuntimeError("native probe evidence collection failed")
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}), flush=True)
    return report


if __name__ == "__main__":
    main()
