#!/usr/bin/env python3
"""Bounded ND/NZ BF16 linear screen at observed TP4 shapes; not model acceptance."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

# (name, output features, input features), from the frozen Dk64 rank-0 trace.
SHAPES = (
    ("out_proj", 5120, 1536), ("down_proj", 5120, 4352),
    ("gate", 24, 5120), ("in_proj", 3584, 5120),
    ("lm_head", 62080, 5120), ("gate_up_proj", 8704, 5120),
)
BF16_EPS = 2 ** -7
ORDER = ("nd", "nz", "nz", "nd") * 2
RUNTIME = Path("/vllm-workspace/vllm-ascend/vllm_ascend")
torch = None
torch_npu = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix():
    return [
        {"phase": phase, "m": m, "name": name, "n": n, "k": k}
        for phase, batches in (("decode", (1, 16)), ("prefill", (32768,)))
        for m in batches for name, n, k in SHAPES
        if phase == "decode" or name not in ("gate", "lm_head")
    ]


def empty_metrics():
    return {"finite": True, "bitwise_equal": True, "error_sq": 0.0,
            "reference_sq": 0.0, "max_abs_error": 0.0, "reference_max_abs": 0.0,
            "elements": 0}


def accumulate(metrics, actual, reference):
    # CPU chunks bound memory; no full prefill FP32 output materialization.
    metrics["finite"] &= bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    metrics["bitwise_equal"] &= bool(torch.equal(actual, reference))
    actual, reference = actual.float(), reference.float()
    delta = actual - reference
    metrics["error_sq"] += float(delta.square().sum(dtype=torch.float64))
    metrics["reference_sq"] += float(reference.square().sum(dtype=torch.float64))
    metrics["max_abs_error"] = max(metrics["max_abs_error"], float(delta.abs().max()))
    metrics["reference_max_abs"] = max(metrics["reference_max_abs"], float(reference.abs().max()))
    metrics["elements"] += actual.numel()


def finish_metrics(metrics):
    metrics["relative_l2"] = math.sqrt(metrics["error_sq"] / max(metrics["reference_sq"], 1e-30))
    metrics["normalized_max_abs"] = metrics["max_abs_error"] / max(metrics["reference_max_abs"], 1e-30)
    return metrics


def numerical_pass(record):
    checks = record["numerical"]
    return (
        record["weight_roundtrip_exact"] is True
        and record["weight_formats"] == {"nd": 2, "nz": 29}
        and set(checks) == {"nz_vs_nd_all_elements", "nd_vs_cpu_fp32_sample", "nz_vs_cpu_fp32_sample"}
        and set(record["graph_matches_eager"]) == ({"nd", "nz"} if record["phase"] == "decode" else set())
        and all(item["finite"] is True and
                math.isfinite(item["relative_l2"]) and item["relative_l2"] <= BF16_EPS and
                math.isfinite(item["normalized_max_abs"]) and item["normalized_max_abs"] <= 2 * BF16_EPS
                for item in checks.values())
        and all(value is True for value in record["graph_matches_eager"].values())
    )


def validate(records):
    expected = {(c["phase"], c["m"], c["name"], c["n"], c["k"]) for c in matrix()}
    keys = [(c["phase"], c["m"], c["name"], c["n"], c["k"]) for c in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing or duplicate shape cases")
    for record in records:
        if not numerical_pass(record):
            raise ValueError("numerical screen failed")
        for variant in ("nd", "nz"):
            values = record["timing_ms_per_call"][variant]
            if len(values) != 4 or not all(math.isfinite(x) and x > 0 for x in values):
                raise ValueError("incomplete or invalid timing samples")
    return "micro_screen_complete"


def case(spec):
    m, n, k = (spec[key] for key in ("m", "n", "k"))
    generator = torch.Generator().manual_seed(20260908 + m + n + k)
    x_cpu = torch.randn((m, k), dtype=torch.bfloat16, generator=generator)
    w_cpu = torch.randn((n, k), dtype=torch.bfloat16, generator=generator) / math.sqrt(k)
    x = x_cpu.to("npu:0")
    nd = w_cpu.to("npu:0")
    nz = torch_npu.npu_format_cast(nd, 29)
    record = dict(spec, weight_formats={"nd": int(torch_npu.get_npu_format(nd)),
                                      "nz": int(torch_npu.get_npu_format(nz))},
                  weight_roundtrip_exact=bool(torch.equal(torch_npu.npu_format_cast(nz, 2).cpu(), w_cpu)),
                  graph_matches_eager={}, numerical={})
    weights = {"nd": nd, "nz": nz}
    outputs = {name: torch.nn.functional.linear(x, weight) for name, weight in weights.items()}
    torch.npu.synchronize()
    cross = empty_metrics()
    for start in range(0, m, 1024):
        accumulate(cross, outputs["nz"][start:start + 1024].cpu(),
                   outputs["nd"][start:start + 1024].cpu())
    record["numerical"]["nz_vs_nd_all_elements"] = finish_metrics(cross)
    ref_checks = {name: empty_metrics() for name in weights}
    # FP32 CPU reference for first up-to-eight rows, output-feature chunks.
    for start in range(0, n, 4096):
        reference = torch.nn.functional.linear(x_cpu[:8].float(), w_cpu[start:start + 4096].float())
        for name in weights:
            accumulate(ref_checks[name], outputs[name][:8, start:start + 4096].cpu(), reference)
    record["numerical"].update({name + "_vs_cpu_fp32_sample": finish_metrics(value)
                                for name, value in ref_checks.items()})
    graphs, replay_outputs = {}, {}
    for name in weights:
        for _ in range(3):
            torch.nn.functional.linear(x, weights[name])
        torch.npu.synchronize()
        if spec["phase"] == "decode":
            graphs[name] = torch.npu.NPUGraph()
            with torch.npu.graph(graphs[name]):
                replay_outputs[name] = torch.nn.functional.linear(x, weights[name])
            graphs[name].replay()
            torch.npu.synchronize()
            graph_cpu = replay_outputs[name].cpu()
            record["graph_matches_eager"][name] = bool(
                torch.isfinite(graph_cpu).all() and torch.equal(graph_cpu, outputs[name].cpu()))
    if not numerical_pass(record):
        for graph in graphs.values():
            graph.reset()
        return record
    repeats = 50 if spec["phase"] == "decode" else 5
    timings = {"nd": [], "nz": []}
    for name in ORDER:
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        start.record()
        for _ in range(repeats):
            if graphs:
                graphs[name].replay()
            else:
                result = torch.nn.functional.linear(x, weights[name])
        end.record()
        end.synchronize()
        timings[name].append(start.elapsed_time(end) / repeats)
    record.update(
        timing_mode="NPUGraph replay" if graphs else "eager",
        repeats_per_block=repeats, timing_ms_per_call=timings,
        median_ms={name: median(value) for name, value in timings.items()},
    )
    record["nz_latency_change_pct"] = (median(timings["nz"]) / median(timings["nd"]) - 1) * 100
    for graph in graphs.values():
        graph.reset()
    return record


def main():
    global torch, torch_npu
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if Path("/drrqr-results") not in output.parents or output.exists():
        parser.error("fresh directory under /drrqr-results required")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0":
        parser.error("requires logical 0 only in the audited physical4-7 container")
    protocol = {
        "phase": "performance-first exploration",
        "hypothesis": "persistent BF16 NZ weights reduce native F.linear cost at observed hotspots",
        "cases": matrix(), "seed": "20260908 + M + N + K", "dtype": "bfloat16",
        "runtime_route": "native weight_nz_mode=2 uses npu_format_cast(weight,29); F.linear",
        "numeric_criteria": {"weight_roundtrip": "bitwise exact", "finite": True,
            "relative_l2_max": BF16_EPS, "normalized_max_abs_max": 2 * BF16_EPS,
            "rationale": "one BF16 epsilon global L2 and two epsilon scaled max; exploratory arithmetic-kernel screen, not model quality",
            "coverage": "ND/NZ all output elements; both versus CPU FP32 first min(M,8) rows",
            "graph_vs_same_layout_eager": "bitwise exact"},
        "timing": {"order": ORDER, "decode": "50 replays/block", "prefill": "5 eager calls/block",
                   "warmup": 3, "clock": "NPU events; profiler disabled"},
        "budget": "one 16-case matrix, external timeout 900 seconds; stop on first numerical failure",
        "decision": "advance only if numerical screen passes and material gains occur in major shapes; no automatic full benchmark",
        "limitations": "synthetic data, one device, repeated weights may be cache-resident; no TP communication, full-model or accuracy proof",
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    report = {"schema": "drrqr-nz-weight-micro/v1", "status": "running", "protocol": protocol,
              "script_sha256": digest(__file__), "device": "container logical0 = physical NPU4",
              "runtime_sha256": {str(RUNTIME / p): digest(RUNTIME / p) for p in (
                  "utils.py", "ops/linear.py", "worker/model_runner_v1.py")},
              "cases": []}
    try:
        import torch
        import torch_npu
        torch.set_num_threads(1)
        torch.npu.set_device(0)
        # Same setting as the installed model_runner_v1.py:201.
        torch.npu.config.allow_internal_format = True
        report.update(torch_version=torch.__version__, torch_npu_version=torch_npu.__version__)
        with torch.inference_mode():
            for spec in matrix():
                record = case(spec)
                report["cases"].append(record)
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({**spec, "numerical_pass": numerical_pass(record),
                                  "nz_latency_change_pct": record.get("nz_latency_change_pct")}), flush=True)
                if not numerical_pass(record):
                    raise RuntimeError("numerical screen failed; retain evidence and stop")
        report["status"] = validate(report["cases"])
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
