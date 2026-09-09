#!/usr/bin/env python3
"""Bounded live-service MC2 activation smoke, not a task-accuracy test."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
from urllib.request import urlopen

DRIVER_SHA = "cba2b1b463a1e436c818722d92147d93f8396b8d64f8da6582d7e67d2cb45822"
ROOT = Path("/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/prefill-mc2-screen-v1")
PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.(mlp\.down_proj|linear_attn\.out_proj|self_attn\.o_proj)$")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keys(names):
    result = []
    for name in names:
        match = PATTERN.search(name)
        if match is None:
            raise ValueError("unrecognized projection name")
        result.append((int(match[1]), match[2]))
    expected = {(i, "mlp.down_proj") for i in range(64)} | {
        (i, "self_attn.o_proj" if (i + 1) % 4 == 0 else "linear_attn.out_proj") for i in range(64)}
    if len(result) != 128 or set(result) != expected:
        raise ValueError("incomplete/duplicate projection coverage")


def validate_evidence(events, enabled, adapter_sha):
    conv = [e for e in events if e["event"] == "conv_weight_layout_prepared"]
    pids = {e["pid"] for e in conv}
    if len(conv) != 4 or len(pids) != 4:
        raise ValueError("expected four layout-prepared workers")
    for record in conv:
        layers = record["layers"]
        if len(layers) != 48 or {l["layer"] for l in layers} != {i for i in range(64) if (i + 1) % 4}:
            raise ValueError("convolution coverage differs")
        if not all(l["exact_values_verified"] is True for l in layers):
            raise ValueError("convolution weight check incomplete")
    builds = {e["pid"] for e in events if e["event"] == "plugin_build_active"
              and e.get("version") == "0.1.8" and e.get("prefill_mc2") is enabled}
    if not pids.issubset(builds):
        raise ValueError("worker plugin version/flag evidence missing")
    prepared = [e for e in events if e["event"] == "prefill_mc2_prepared"]
    calls = [e for e in events if e["event"] == "prefill_mc2_called"]
    if not enabled:
        if prepared or calls:
            raise ValueError("control unexpectedly uses MC2")
        return {"worker_pids": sorted(pids), "prepared": 0, "fused_calls": 0}
    if len(prepared) != 4 or {e["pid"] for e in prepared} != pids:
        raise ValueError("four MC2 preparations required")
    for e in prepared:
        if e.get("weights_modified") is not False or e.get("source_sha256") != adapter_sha:
            raise ValueError("MC2 preparation provenance mismatch")
        keys([l["name"] for l in e["layers"]])
    if len(calls) != 512 or {e["pid"] for e in calls} != pids:
        raise ValueError("real-request fused-call evidence is incomplete")
    for pid in pids:
        worker = [e for e in calls if e["pid"] == pid]
        keys([e["name"] for e in worker])
        if not all(e["phase"] == "pure_prefill" and e["rows"] == 32768 and e["source_sha256"] == adapter_sha for e in worker):
            raise ValueError("fused call does not match first32K smoke/provenance")
    return {"worker_pids": sorted(pids), "prepared": 4, "fused_calls": 512}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", type=Path, required=True)
    parser.add_argument("--benchmark-driver", type=Path, required=True)
    args = parser.parse_args()
    directory = args.service_dir.resolve()
    output = directory / "activation-smoke.json"
    if ROOT not in directory.parents or output.exists():
        parser.error("fresh smoke receipt under the MC2 screen directory required")
    if digest(args.benchmark_driver) != DRIVER_SHA:
        raise RuntimeError("frozen driver changed")
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
            raise RuntimeError("service exited before activation")
        if time.monotonic() >= deadline:
            raise TimeoutError("service startup exceeded900s")
        time.sleep(5)
    if (manifest["plugin_version"] != "0.1.8" or manifest["layout"] != "packed"
            or manifest["profiler_config"] is not None or manifest.get("diagnostic_determinism") is not None):
        raise RuntimeError("smoke requires the frozen profiler-disabled0.1.8 packed layout")
    spec = importlib.util.spec_from_file_location("mc2_frozen_driver", args.benchmark_driver)
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    report = {"schema": "drrqr-mc2-activation-smoke/v1", "status": "running",
              "service_manifest_sha256": digest(directory / "service-manifest.json"),
              "script_sha256": digest(Path(__file__)), "driver_sha256": DRIVER_SHA,
              "prefill_mc2": manifest["prefill_mc2"], "requests": [],
              "claim": "activation plus short native compiled/decode smoke, not numerical/task accuracy or throughput acceptance",
              "protocol": "one32K input/32output request, then two concurrent8K input/16output requests; raw completions, temperature0 seed0 ignore_eos"}
    try:
        for length, count, output_tokens in ((32768, 1, 32), (8192, 2, 16)):
            settings = SimpleNamespace(base_url="http://127.0.0.1:6666", model="qwen3.8",
                                       prefix_tokens=length // 2, input_tokens=length,
                                       requests=count, warmup_requests=0, timeout=180, respect_eos=False)
            prompts = driver._build_prompts(settings)
            with ThreadPoolExecutor(max_workers=count) as pool:
                futures = [pool.submit(driver._stream_completion, settings, i, prompt, None, output_tokens)
                           for i, prompt in enumerate(prompts)]
                for prompt, future in zip(prompts, futures):
                    row = asdict(future.result())
                    row.update(input_tokens=length, input_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest())
                    report["requests"].append(row)
                    if not row["ok"] or row["output_tokens"] != output_tokens or row["finish_reason"] != "length":
                        raise RuntimeError("native service response smoke failed")
        events = [json.loads(line) for line in (directory / "evidence.jsonl").read_text().splitlines()]
        report["activation"] = validate_evidence(events, manifest["prefill_mc2"],
                                                manifest["module_sha256"]["patches/prefill_mc2.py"])
        report["evidence_sha256_at_smoke"] = digest(directory / "evidence.jsonl")
        report["status"] = "activation_smoke_pass"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
