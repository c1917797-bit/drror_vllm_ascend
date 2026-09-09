#!/usr/bin/env python3
"""One TP4 large-row MatMul+AllReduce numerical/performance micro screen."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
from statistics import median
import time

import nz_weight_probe as metrics

TP = 4
ORDER = ("separate", "fused", "fused", "separate") * 2
NUMERIC_NAMES = {"fused_vs_separate_all_elements", "separate_vs_cpu_fp32_sample", "fused_vs_cpu_fp32_sample"}
RUNTIME_FILES = (
    "/vllm-workspace/vllm-ascend/vllm_ascend/ops/linear_op.py",
    "/vllm-workspace/vllm-ascend/vllm_ascend/distributed/device_communicators/npu_communicator.py",
    "/vllm-workspace/vllm/vllm/distributed/device_communicators/base_device_communicator.py",
)


def matrix(rows=None, phase="prefill"):
    return [dict(phase=phase, m=m, n=5120, k=k, name=name)
            for m in ((2048, 8192, 32768) if rows is None else rows)
            for name, k in (("out_proj", 1536), ("down_proj", 4352))]


def numerical_pass(record):
    values = record["numerical"]
    return (
        record["weight_format"] == 2 and set(values) == NUMERIC_NAMES
        and all(v["finite"] is True and math.isfinite(v["relative_l2"])
                and 0 <= v["relative_l2"] <= metrics.BF16_EPS
                and math.isfinite(v["normalized_max_abs"])
                and 0 <= v["normalized_max_abs"] <= 2 * metrics.BF16_EPS
                for v in values.values())
    )


def validate_rank(report, rows=None):
    expected = [(c["m"], c["n"], c["k"]) for c in matrix(rows)]
    records = report["cases"]
    keys = [(c["m"], c["n"], c["k"]) for c in records]
    if report["status"] != "rank_complete" or keys != expected:
        raise ValueError("incomplete or duplicate rank cases")
    for case in records:
        if not numerical_pass(case):
            raise ValueError("numerical gate failed")
        for clock in ("device_ms", "wall_ms"):
            samples = case[clock]
            if set(samples) != {"separate", "fused"} or any(
                len(v) != 4 or not all(math.isfinite(x) and x > 0 for x in v)
                for v in samples.values()
            ):
                raise ValueError("incomplete timing blocks")


def summarize(reports, rows=None, phase="prefill"):
    if len(reports) != TP or sorted(r["rank"] for r in reports) != list(range(TP)):
        raise ValueError("missing or duplicate TP ranks")
    for report in reports:
        validate_rank(report, rows)
    summary_rows = []
    for index, spec in enumerate(matrix(rows, phase)):
        row = dict(spec)
        for clock in ("device_ms", "wall_ms"):
            # Compare slowest rank in each corresponding block; never cherry-pick a rank.
            values = {name: [max(r["cases"][index][clock][name][b] for r in reports)
                             for b in range(4)] for name in ("separate", "fused")}
            row[clock + "_max_rank_blocks"] = values
            row[clock + "_median_max_rank"] = {name: median(v) for name, v in values.items()}
            row[clock + "_fused_change_pct"] = (median(values["fused"]) / median(values["separate"]) - 1) * 100
        summary_rows.append(row)
    return summary_rows


def case(spec, rank, cpu_group, hcom, torch, torch_npu, dist):
    m, n, k = (spec[key] for key in ("m", "n", "k"))
    generator = torch.Generator().manual_seed(20260908 + 100003 * rank + m + k)
    x_cpu = torch.randn((m, k), dtype=torch.bfloat16, generator=generator)
    w_cpu = torch.randn((n, k), dtype=torch.bfloat16, generator=generator) / math.sqrt(k)
    device = "npu:" + str(rank)
    x, weight = x_cpu.to(device), w_cpu.to(device)
    # CPU FP32 product and cross-rank FP32 SUM are independent of BF16 HCCL/fused math.
    reference = torch.nn.functional.linear(x_cpu[:8].float(), w_cpu.float())
    dist.all_reduce(reference, group=cpu_group)

    def operation(name):
        if name == "fused":
            return torch_npu.npu_mm_all_reduce_base(x, weight.t(), hcom)
        output = torch.nn.functional.linear(x, weight)
        dist.all_reduce(output)
        return output

    outputs = {}
    for name in ("separate", "fused"):
        outputs[name] = operation(name)
        torch.npu.synchronize()
        dist.barrier(group=cpu_group)
    cross = metrics.empty_metrics()
    for start in range(0, m, 1024):
        metrics.accumulate(cross, outputs["fused"][start:start + 1024].cpu(),
                           outputs["separate"][start:start + 1024].cpu())
    values = {"fused_vs_separate_all_elements": metrics.finish_metrics(cross)}
    for name in outputs:
        value = metrics.empty_metrics()
        metrics.accumulate(value, outputs[name][:8].cpu(), reference)
        values[name + "_vs_cpu_fp32_sample"] = metrics.finish_metrics(value)
    record = dict(spec, weight_format=int(torch_npu.get_npu_format(weight)), numerical=values)
    all_pass = torch.tensor([int(numerical_pass(record))], dtype=torch.int32)
    dist.all_reduce(all_pass, op=dist.ReduceOp.MIN, group=cpu_group)
    if not bool(all_pass.item()):
        record["any_rank_numeric_failure"] = True
        return record
    # Warmup is workload-specific and outside all measured blocks, including lazy compilation.
    for name in ("separate", "fused"):
        for _ in range(10):
            operation(name)
        torch.npu.synchronize()
        dist.barrier(group=cpu_group)
    samples = {clock: {name: [] for name in ("separate", "fused")}
               for clock in ("device_ms", "wall_ms")}
    for name in ORDER:
        torch.npu.synchronize()
        dist.barrier(group=cpu_group)
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter()
        for _ in range(5):
            result = operation(name)
        end.record()
        end.synchronize()
        samples["wall_ms"][name].append((time.perf_counter() - wall_start) * 1000 / 5)
        samples["device_ms"][name].append(start.elapsed_time(end) / 5)
    record.update(samples)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-rows", type=int, nargs="+", default=None)
    parser.add_argument("--shape-context", choices=("prefill", "observed-mixed-linear"), default="prefill")
    parser.add_argument("--shape-evidence", type=Path)
    args = parser.parse_args()
    if args.matrix_rows is not None and (len(set(args.matrix_rows)) != len(args.matrix_rows)
            or not all(2048 <= value <= 40960 for value in args.matrix_rows)):
        parser.error("choose unique large-row shapes within the frozen40960-token budget")
    shape_evidence_sha = None
    if args.shape_context == "observed-mixed-linear":
        if args.shape_evidence is None or args.matrix_rows is None:
            parser.error("observed mixed shapes require the completed coverage report and explicit rows")
        evidence = json.loads(args.shape_evidence.read_text())
        if evidence["status"] != "coverage_complete" or evidence["ranks_match"] is not True:
            parser.error("incomplete coverage evidence")
        shape_evidence_sha = metrics.digest(args.shape_evidence)
        rank_path = args.shape_evidence.parent / "coverage-rank0.jsonl"
        if metrics.digest(rank_path) != evidence["rank_artifacts"]["0"]:
            parser.error("coverage rank artifact changed")
        observed = {row["actual_rows"] for line in rank_path.read_text().splitlines()
                    if (row := json.loads(line)).get("phase") == "mixed"}
        if not set(args.matrix_rows).issubset(observed):
            parser.error("requested row shape was not observed in mixed batches")
    output = args.output_dir.resolve()
    rank = int(os.environ.get("RANK", "-1"))
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3" or os.environ.get("WORLD_SIZE") != "4":
        parser.error("requires TP4 in audited physical4-7 to logical0-3 container")
    if rank not in range(TP) or Path("/drrqr-results") not in output.parents:
        parser.error("invalid rank or output root")
    protocol = {
        "phase": "performance-first exploration", "hypothesis": "overlap prefill GEMM with its TP AllReduce",
        "cases": matrix(args.matrix_rows, args.shape_context), "tp": TP, "physical_devices": [4, 5, 6, 7], "dtype": "bfloat16", "weight_format": "ND",
        "shape_context": args.shape_context, "shape_evidence_sha256": shape_evidence_sha,
        "baseline": "F.linear followed by in-place torch.distributed.all_reduce on the HCCL group",
        "candidate": "npu_mm_all_reduce_base(x,weight.t(),same HCCL comm name)",
        "restriction": "no pure-decode scenario; observed-mixed-linear is a standalone matrix-shape check, not mixed-model/state correctness",
        "numerical": {"finite": True, "relative_l2_max": metrics.BF16_EPS,
                      "normalized_max_abs_max": 2 * metrics.BF16_EPS,
                      "rationale": "BF16 kernel plus TP4 reduction rounding; one epsilon L2, two epsilon scaled max; exploration only",
                      "coverage": "all-element separate/fused comparison; both versus FP32 CPU product+Gloo sum for min(M,8) rows"},
        "timing": {"warmup_calls_per_path": 10, "order": ORDER, "calls_per_block": 5,
                   "clocks": ["NPU events", "host wall with final synchronization"], "report": "slowest rank per matched block"},
        "budget": "one explicit TP4 matrix; timeout900s; stop all ranks on first numerical failure",
        "limitations": "synthetic inputs/weights, standalone operators, no scheduler/TP model/state or task accuracy proof",
    }
    if rank == 0:
        if output.exists():
            parser.error("preserve prior experiment; choose fresh output directory")
        output.mkdir(parents=True)
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    report = {"schema": "drrqr-matmul-allreduce-rank/v1", "rank": rank, "status": "starting", "cases": [],
              "script_sha256": metrics.digest(__file__), "helper_sha256": metrics.digest(metrics.__file__),
              "runtime_sha256": {path: metrics.digest(path) for path in RUNTIME_FILES}}
    dist = None
    try:
        import torch
        import torch_npu
        import torch.distributed as dist
        metrics.torch = torch
        torch.set_num_threads(1)
        torch.npu.set_device(rank)
        torch.npu.config.allow_internal_format = True
        dist.init_process_group("hccl", timeout=timedelta(seconds=180))
        cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
        dist.barrier(group=cpu_group)
        hcom = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        report.update(status="running", torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                      environment={key: os.environ.get(key) for key in (
                          "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE", "HCCL_DETERMINISTIC", "LCCL_DETERMINISTIC")},
                      fused_schema=str(torch.ops.npu.npu_mm_all_reduce_base.default._schema))
        with torch.inference_mode():
            for spec in matrix(args.matrix_rows, args.shape_context):
                record = case(spec, rank, cpu_group, hcom, torch, torch_npu, dist)
                report["cases"].append(record)
                (output / ("rank" + str(rank) + ".json")).write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"rank": rank, **spec, "numeric_pass": numerical_pass(record)}), flush=True)
                if record.get("any_rank_numeric_failure"):
                    raise RuntimeError("some rank failed frozen numerical screen")
        report["status"] = "rank_complete"
        validate_rank(report, args.matrix_rows)
        (output / ("rank" + str(rank) + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        dist.barrier(group=cpu_group)
        if rank == 0:
            ranks = [json.loads((output / ("rank" + str(i) + ".json")).read_text()) for i in range(TP)]
            result = {"schema": "drrqr-matmul-allreduce-micro/v1", "status": "micro_screen_complete",
                      "protocol": protocol, "cases": summarize(ranks, args.matrix_rows, args.shape_context),
                      "rank_report_sha256": {str(i): metrics.digest(output / ("rank" + str(i) + ".json")) for i in range(TP)}}
            (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        dist.barrier(group=cpu_group)
    except Exception as error:
        report.update(status="failed", error=repr(error))
        if output.exists():
            (output / ("rank" + str(rank) + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
