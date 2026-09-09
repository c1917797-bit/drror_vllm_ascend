#!/usr/bin/env python3
"""Bounded paired GDN projection-fusion micro screen on one Ascend device."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

import torch
import torch_npu


BF16_EPS = 2**-7
K = 5120
BA_N = 24
M = 16
LAYERS = 48
DECODE_STEPS = 30
REPEATS = 5
ITERATIONS = 100
WARMUP = 20
ORDER = ("separate", "fused", "fused", "separate")
VARIANTS = (("dense", 4096), ("dk64", 3584))
REQUIRED_INCREMENTAL_MS = 64.446696


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    actual_cpu = actual.float().cpu()
    reference_cpu = reference.float().cpu()
    delta = actual_cpu - reference_cpu
    reference_sq = float(reference_cpu.square().sum(dtype=torch.float64))
    reference_max = float(reference_cpu.abs().max())
    return {
        "finite": bool(torch.isfinite(actual_cpu).all() and torch.isfinite(reference_cpu).all()),
        "relative_l2": math.sqrt(float(delta.square().sum(dtype=torch.float64)) / max(reference_sq, 1e-30)),
        "max_abs_error": float(delta.abs().max()),
        "normalized_max_abs": float(delta.abs().max()) / max(reference_max, 1e-30),
    }


def numerical_pass(metrics: dict[str, float | bool]) -> bool:
    return (
        metrics["finite"] is True
        and math.isfinite(float(metrics["relative_l2"]))
        and float(metrics["relative_l2"]) <= BF16_EPS
        and math.isfinite(float(metrics["normalized_max_abs"]))
        and float(metrics["normalized_max_abs"]) <= 2 * BF16_EPS
    )


def time_graph(graph: torch_npu.npu.NPUGraph) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    start.record()
    for _ in range(ITERATIONS):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / ITERATIONS


def run_case(name: str, qkvz_n: int) -> dict:
    generator = torch.Generator().manual_seed(20260908 + qkvz_n)
    x = torch.randn((M, K), dtype=torch.bfloat16, generator=generator).npu()
    qkvz_weight = (
        torch.randn((qkvz_n, K), dtype=torch.bfloat16, generator=generator) / math.sqrt(K)
    ).npu()
    ba_weight = (
        torch.randn((BA_N, K), dtype=torch.bfloat16, generator=generator) / math.sqrt(K)
    ).npu()
    fused_weight = torch.cat((qkvz_weight, ba_weight), dim=0).contiguous()

    for _ in range(WARMUP):
        separate_qkvz = torch.nn.functional.linear(x, qkvz_weight)
        separate_ba = torch.nn.functional.linear(x, ba_weight)
        fused_all = torch.nn.functional.linear(x, fused_weight)
    torch.npu.synchronize()

    separate_graph = torch.npu.NPUGraph()
    with torch.npu.graph(separate_graph):
        separate_qkvz = torch.nn.functional.linear(x, qkvz_weight)
        separate_ba = torch.nn.functional.linear(x, ba_weight)
    fused_graph = torch.npu.NPUGraph()
    with torch.npu.graph(fused_graph):
        fused_all = torch.nn.functional.linear(x, fused_weight)
    separate_graph.replay()
    fused_graph.replay()
    torch.npu.synchronize()

    metrics = error_metrics(
        fused_all,
        torch.cat((separate_qkvz, separate_ba), dim=-1),
    )
    if not numerical_pass(metrics):
        separate_graph.reset()
        fused_graph.reset()
        raise RuntimeError(f"{name}: numerical screen failed: {metrics}")

    graphs = {"separate": separate_graph, "fused": fused_graph}
    timings = {"separate": [], "fused": []}
    for repeat in range(REPEATS):
        rotated = ORDER[repeat % len(ORDER) :] + ORDER[: repeat % len(ORDER)]
        for route in rotated[:2]:
            timings[route].append(time_graph(graphs[route]))

    separate_us = median(timings["separate"])
    fused_us = median(timings["fused"])
    savings_us = separate_us - fused_us
    record = {
        "name": name,
        "m": M,
        "k": K,
        "n_qkvz": qkvz_n,
        "n_ba": BA_N,
        "n_fused": qkvz_n + BA_N,
        "numerical": metrics,
        "timing_us_per_projection_pair": timings,
        "median_us": {"separate": separate_us, "fused": fused_us},
        "projection_pair_speedup": separate_us / fused_us,
        "savings_us": savings_us,
        "projected_savings_ms_for_48_layers_x_30_steps": savings_us * LAYERS * DECODE_STEPS / 1000.0,
    }
    separate_graph.reset()
    fused_graph.reset()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if Path("/drrqr-results") not in output.parents or output.exists():
        parser.error("fresh directory under /drrqr-results required")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0":
        parser.error("requires container logical NPU0, mapped to authorized physical NPU4")

    protocol = {
        "phase": "performance-first operator feasibility",
        "hypothesis": (
            "single QKVZBA projection has a Dk64-specific differential benefit large enough "
            "to close the remaining 5.0099% endpoint-throughput gap after an equally optimized dense control"
        ),
        "cases": [{"name": name, "m": M, "k": K, "n_qkvz": n, "n_ba": BA_N} for name, n in VARIANTS],
        "dtype": "bfloat16",
        "seed": "20260908 + n_qkvz",
        "numeric_criteria": {
            "finite": True,
            "relative_l2_max": BF16_EPS,
            "normalized_max_abs_max": 2 * BF16_EPS,
        },
        "timing": {
            "mode": "NPU graph replay with NPU events; profiler disabled",
            "warmup": WARMUP,
            "iterations_per_block": ITERATIONS,
            "blocks_per_route": REPEATS,
            "rotated_order": True,
        },
        "decision": (
            "advance only if Dk64 projected decode-window savings minus dense projected savings "
            f"is at least {REQUIRED_INCREMENTAL_MS:.6f} ms; otherwise classify as generic and reject"
        ),
        "budget": "one two-shape matrix on physical NPU4; no model service and no automatic retry",
        "limitations": (
            "synthetic fixed-address graph replay; establishes operator feasibility only, "
            "not full-model throughput, LongBench/GSM8K, or final acceptance"
        ),
    }
    output.mkdir(parents=True)
    report = {
        "schema": "drrqr-gdn-projection-fusion-paired-micro/v1",
        "status": "running",
        "device": "container logical0 = physical NPU4",
        "script_sha256": digest(Path(__file__)),
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "protocol": protocol,
        "cases": [],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    try:
        torch.set_num_threads(1)
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        with torch.inference_mode():
            for name, qkvz_n in VARIANTS:
                record = run_case(name, qkvz_n)
                report["cases"].append(record)
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(record), flush=True)
        by_name = {case["name"]: case for case in report["cases"]}
        differential = (
            by_name["dk64"]["projected_savings_ms_for_48_layers_x_30_steps"]
            - by_name["dense"]["projected_savings_ms_for_48_layers_x_30_steps"]
        )
        report["decision_metrics"] = {
            "required_incremental_ms": REQUIRED_INCREMENTAL_MS,
            "observed_projected_incremental_ms": differential,
        }
        report["status"] = (
            "advance_to_plugin_integration"
            if differential >= REQUIRED_INCREMENTAL_MS
            else "rejected_generic_optimization"
        )
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], **report["decision_metrics"]}), flush=True)


if __name__ == "__main__":
    main()
