#!/usr/bin/env python3
"""One fixed-work MC2 coverage diagnostic; timings are not performance evidence."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.request import urlopen

DRIVER_SHA = "cba2b1b463a1e436c818722d92147d93f8396b8d64f8da6582d7e67d2cb45822"
ROOT = Path("/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/mc2-coverage-v1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_records(records):
    if not records or [r["sequence"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("missing/duplicate coverage steps")
    if any(r["status"] != "complete" for r in records):
        raise ValueError("incomplete model forward")
    phases = {}
    for row in records:
        data = phases.setdefault(row["phase"], Counter())
        data["steps"] += 1
        for key in ("actual_rows", "prefill_rows", "decode_rows", "eligible_rows"):
            if type(row[key]) is int:
                data[key] += row[key]
            else:
                data["unknown_" + key + "_steps"] += 1
        for key in ("1536", "4352"):
            count = row["fused_calls_by_k"].get(key, 0)
            if count > 64 or count < 0 or (row["eligible_rows"] == 0 and count):
                raise ValueError("unexpected fused target count")
            data["fused_calls_k" + key] += count
            data["fused_row_calls_k" + key] += row["fused_row_calls_by_k"].get(key, 0)
    prompt_rows = sum(v.get("prefill_rows", 0) for v in phases.values())
    coverage = {key: sum(v.get("fused_row_calls_k" + key, 0) for v in phases.values()) / (64 * prompt_rows)
                if prompt_rows else None for key in ("1536", "4352")}
    return {"phases": {k: dict(v) for k, v in phases.items()},
            "observed_prefill_rows": prompt_rows,
            "fused_row_fraction_of_observed_prefill_by_k": coverage,
            "denominator": "64 projections of each K times metadata-derived prefill rows",
            "timing_claim": "none; counters perturb the workload"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", required=True, type=Path)
    parser.add_argument("--benchmark-driver", required=True, type=Path)
    args = parser.parse_args()
    directory = args.service_dir.resolve()
    output = directory / "coverage-report.json"
    marker = directory / "coverage-active"
    if ROOT not in directory.parents or output.exists() or marker.exists():
        parser.error("fresh coverage directory required")
    if digest(args.benchmark_driver) != DRIVER_SHA:
        raise RuntimeError("driver changed")
    deadline = time.monotonic() + 900
    while True:
        try:
            manifest = json.loads((directory / "service-manifest.json").read_text())
            with urlopen("http://127.0.0.1:6666/health", timeout=5) as response:
                if response.status == 200:
                    break
        except (OSError, ValueError):
            pass
        if (directory / "service-exit.json").exists():
            raise RuntimeError("service exited before readiness")
        if time.monotonic() >= deadline:
            raise TimeoutError("startup exceeded900s")
        time.sleep(5)
    diagnostic = manifest.get("diagnostic_mc2_coverage")
    if (not diagnostic or not manifest["prefill_mc2"] or manifest["plugin_version"] != "0.1.8"
            or manifest["variant"] != "dk64" or manifest["layout"] != "packed"
            or manifest["profiler_config"] is not None or manifest.get("diagnostic_determinism") is not None):
        raise RuntimeError("not the frozen MC2 coverage configuration")
    for rank in range(4):
        ready = json.loads((directory / ("coverage-rank" + str(rank) + ".jsonl")).read_text().splitlines()[0])
        if (ready["event"] != "coverage_ready" or ready["rank"] != rank or ready["target_count"] != 128
                or ready["worker_sha256"] != diagnostic["worker_sha256"]
                or ready["helper_sha256"] != diagnostic["helper_sha256"]
                or ready["adapter_sha256"] != manifest["module_sha256"]["patches/prefill_mc2.py"]):
            raise RuntimeError("coverage worker provenance mismatch")
    spec = importlib.util.spec_from_file_location("coverage_frozen_driver", args.benchmark_driver)
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    settings = SimpleNamespace(base_url="http://127.0.0.1:6666", model="qwen3.8",
                               input_tokens=32768, prefix_tokens=16384, output_tokens=1024,
                               requests=40, warmup_requests=2, concurrency=16, timeout=1800, respect_eos=False)
    report = {"schema": "drrqr-mc2-coverage/v1", "status": "running",
              "claim": "diagnostic batch/route coverage, not a performance or accuracy result",
              "service_manifest_sha256": digest(directory / "service-manifest.json"),
              "client_sha256": digest(Path(__file__)), "driver_sha256": DRIVER_SHA,
              "config": vars(settings), "warmup_smoke": [], "warmup_results": [],
              "results": [], "started_epoch_s": time.time()}
    try:
        # Match the earlier on/off service warmup, outside the diagnostic window.
        for length, count, tokens in ((32768, 1, 32), (8192, 2, 16)):
            short = SimpleNamespace(**{**vars(settings), "input_tokens": length, "prefix_tokens": length // 2,
                                       "requests": count, "warmup_requests": 0, "timeout": 180})
            prompts = driver._build_prompts(short)
            with ThreadPoolExecutor(max_workers=count) as pool:
                futures = [pool.submit(driver._stream_completion, short, i, prompt, None, tokens)
                           for i, prompt in enumerate(prompts)]
                for future in futures:
                    value = asdict(future.result())
                    report["warmup_smoke"].append(value)
                    if not value["ok"] or value["output_tokens"] != tokens or value["finish_reason"] != "length":
                        raise RuntimeError("short request failed")
        prompts = driver._build_prompts(settings)
        report["prompt_sha256"] = [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]
        warmup = driver._run_warmups(settings, prompts[:2])
        report["warmup_results"] = [asdict(v) for v in warmup]
        if len(warmup) != 2 or not all(v.ok and v.output_tokens == 1 for v in warmup):
            raise RuntimeError("warmup failed")
        with marker.open("x") as stream:
            stream.write("measured40\n")
        print("coverage window opened for40 fixed-work requests", flush=True)
        try:
            values, diagnostic_duration = driver._run_measured(settings, prompts[2:])
            report["results"] = [asdict(v) for v in values]
            report["diagnostic_duration_s_not_performance"] = diagnostic_duration
        finally:
            marker.unlink(missing_ok=True)
        rows = report["results"]
        if (len(rows) != 40 or sorted(v["index"] for v in rows) != list(range(40))
                or not all(v["ok"] and v["output_tokens"] == v["requested_output_tokens"] == 1024
                           and v["finish_reason"] == "length" for v in rows)):
            raise RuntimeError("fixed workload did not complete")
        normalized = []
        report["rank_artifacts"] = {}
        report["rank_summaries"] = {}
        for rank in range(4):
            path = directory / ("coverage-rank" + str(rank) + ".jsonl")
            events = [json.loads(line) for line in path.read_text().splitlines()]
            if len([v for v in events if v["event"] == "coverage_ready"]) != 1:
                raise RuntimeError("duplicate worker initialization")
            records = [{k: v for k, v in row.items() if k not in ("rank", "pid")}
                       for row in events if row["event"] == "model_forward"]
            report["rank_artifacts"][str(rank)] = digest(path)
            report["rank_summaries"][str(rank)] = summarize_records(records)
            normalized.append(records)
        if not all(records == normalized[0] for records in normalized[1:]):
            raise RuntimeError("TP rank coverage differs; do not pool inconsistent evidence")
        report["ranks_match"] = True
        report["expected_prompt_tokens"] = 40 * 32768
        report["status"] = "coverage_complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        marker.unlink(missing_ok=True)
        report["finished_epoch_s"] = time.time()
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in (
        "warmup_smoke", "warmup_results", "results", "prompt_sha256", "rank_summaries")}), flush=True)
    print(json.dumps(report["rank_summaries"]["0"]), flush=True)


if __name__ == "__main__":
    main()
