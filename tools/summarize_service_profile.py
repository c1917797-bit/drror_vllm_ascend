#!/usr/bin/env python3
"""Validate all four ranks and summarize kernel timelines without adding streams.

Kernel sums identify hot operators, not end-to-end critical-path attribution.
Exact integer nanoseconds preserve sub-microsecond events at epoch timestamps.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


def ns(value):
    return int(Decimal(value.strip()) * 1000)


def union_ns(intervals):
    total = 0
    left = right = None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative interval")
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (right - left if left is not None else 0)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(directory, phase):
    output = directory / "ASCEND_PROFILER_OUTPUT"
    required = ["kernel_details.csv", "op_statistic.csv", "api_statistic.csv", "trace_view.json"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise ValueError(f"{directory.name}: missing {missing}")
    with (output / "kernel_details.csv").open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if len(fields) < 40 or not rows:
        raise ValueError("expected populated Level1 + PipeUtilization CSV")
    trace = json.loads((output / "trace_view.json").read_text())
    events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if not isinstance(events, list) or not events:
        raise ValueError("empty or invalid trace event list")
    del trace
    groups = defaultdict(list)
    streams = defaultdict(list)
    device_intervals, compute_intervals, communication_intervals = [], [], []
    for row in rows:
        duration = ns(row["Duration(us)"])
        start = ns(row["Start Time(us)"])
        interval = start, start + duration
        core = row["Accelerator Core"]
        groups[(row["Name"], row["Type"], core)].append((row, interval))
        streams[row["Stream ID"]].append(interval)
        # Keep CPU tasks visible in groups, but exclude them from NPU unions.
        if core not in ("AI_CPU", "AICPU"):
            device_intervals.append(interval)
            if core == "COMMUNICATION":
                communication_intervals.append(interval)
            else:
                compute_intervals.append(interval)
    full_start = min(x[0] for x in device_intervals)
    full_end = max(x[1] for x in device_intervals)
    wall = full_end - full_start
    active = union_ns(device_intervals)
    compute = union_ns(compute_intervals)
    comm = union_ns(communication_intervals)
    counts = defaultdict(int)
    top = []
    for (name, kind, core), entries in groups.items():
        durations = [end - start for _, (start, end) in entries]
        total = sum(durations)
        counts[kind] += len(entries)
        pipeline = {}
        for field in ("aic_mac_ratio", "aic_mte2_ratio", "aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"):
            valid = []
            for row, (start, end) in entries:
                try:
                    valid.append((float(row[field]), end - start))
                except (ValueError, KeyError):
                    pass
            denominator = sum(weight for _, weight in valid)
            pipeline[field] = sum(value * weight for value, weight in valid) / denominator if denominator else None
        top.append({
            "name": name, "type": kind, "core": core, "count": len(entries),
            "kernel_sum_ms": total / 1e6,
            "mean_kernel_us": total / len(entries) / 1000,
            "kernel_union_ms": union_ns([interval for _, interval in entries]) / 1e6,
            "input_shapes": sorted({row["Input Shapes"] for row, _ in entries}),
            "pipeline_duration_weighted_means": pipeline,
        })
    expected = {"CausalConv1d": 48, "FusedInferAttentionScore": 16}
    if phase == "decode":
        expected = {key: value * 30 for key, value in expected.items()}
        expected["RecurrentGatedDeltaRule"] = 48 * 30
    else:
        expected["RecurrentGatedDeltaRule"] = 0
        expected["ChunkGatedDeltaRuleFwdH"] = 48
    coverage = {name: {"expected": number, "observed": counts[name], "passed": counts[name] == number}
                for name, number in expected.items()}
    if not all(item["passed"] for item in coverage.values()):
        raise ValueError(f"{directory.name}: unexpected phase/step coverage {coverage}")
    return {
        "trace_dir": str(directory), "phase": phase,
        "rank": int(re.match(r"rank(\d+)_", directory.name)[1]),
        "kernel_columns": len(fields), "kernel_rows": len(rows),
        "trace_events": len(events), "coverage": coverage,
        "sha256": {name: file_sha256(output / name) for name in required},
        "kernel_envelope_ms": wall / 1e6,
        "npu_interval_union_ms": active / 1e6,
        "no_listed_npu_kernel_ms": (wall - active) / 1e6,
        "compute_interval_union_ms": compute / 1e6,
        "communication_interval_union_ms": comm / 1e6,
        "compute_communication_overlap_ms": (compute + comm - active) / 1e6,
        "communication_without_compute_ms": (active - compute) / 1e6,
        "streams": {stream: {"events": len(items), "union_ms": union_ns(items) / 1e6}
                    for stream, items in streams.items()},
        "top_operators": sorted(top, key=lambda row: row["kernel_sum_ms"], reverse=True),
        "interpretation": "Interval unions are observational. Communication without concurrent compute and kernel gaps are not automatically causal critical-path costs.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing reports; choose a new output")
    client = json.loads((args.service_dir / "profile-client.json").read_text())
    service = json.loads((args.service_dir / "service-manifest.json").read_text())
    if (client.get("status") != "requests_complete_activation_validated"
            or client.get("service_manifest_sha256")
            != file_sha256(args.service_dir / "service-manifest.json")):
        raise ValueError("profile client is not bound to this validated service manifest")
    records = []
    for phase in ("prefill", "decode"):
        paths = client["results"][phase]["trace_dirs"]
        if len(paths) != 4:
            raise ValueError("exactly four rank captures are required per phase")
        phase_records = [summarize(args.service_dir / path, phase) for path in paths]
        if {r["rank"] for r in phase_records} != {0, 1, 2, 3}:
            raise ValueError("rank coverage differs")
        records.extend(phase_records)
    report = {
        "schema": "drrqr-profile-timeline-summary/v1",
        "status": "eight_traces_validated",
        "summary_script_sha256": file_sha256(Path(__file__)),
        "client_sha256": file_sha256(args.service_dir / "profile-client.json"),
        "service_manifest_sha256": file_sha256(args.service_dir / "service-manifest.json"),
        "variant": service["variant"],
        "claim": "Single-variant diagnostic profile; not a baseline comparison or profiler-disabled benchmark",
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "records": [
        {key: r[key] for key in ("phase", "rank", "kernel_rows", "kernel_envelope_ms",
                                "npu_interval_union_ms", "communication_without_compute_ms",
                                "no_listed_npu_kernel_ms")} for r in records
    ]}, indent=2))


if __name__ == "__main__":
    main()
