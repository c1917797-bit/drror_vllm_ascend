#!/usr/bin/env python3
"""Collect one bounded mixed-scheduler native sequence probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

DRIVER_SHA = "cba2b1b463a1e436c818722d92147d93f8396b8d64f8da6582d7e67d2cb45822"
ROOTS = (
    Path("/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/prefill-mc2-mixed-v1"),
    Path("/drrqr-results/prefill-mc2-mixed-v1"),
)
REQUESTS = 16
INPUT_TOKENS = 32768
OUTPUT_TOKENS = 16


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_driver(path):
    if digest(path) != DRIVER_SHA:
        raise RuntimeError("frozen prompt driver changed")
    spec = importlib.util.spec_from_file_location("mc2_mixed_frozen_driver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def request(base_url, prompt, index, barrier, *, input_tokens, output_tokens):
    payload = {"model": "qwen3.8", "prompt": prompt, "temperature": 0, "seed": 0,
               "max_tokens": output_tokens, "ignore_eos": True, "stream": False,
               "return_token_ids": True, "logprobs": 5,
               "vllm_xargs": {"hypic_segment_ends": f"{input_tokens // 2},{input_tokens}"}}
    if barrier is not None:
        barrier.wait(timeout=900)
    started = time.perf_counter()
    req = Request(base_url + "/v1/completions", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=1800) as response:
        data = json.load(response)
    elapsed = time.perf_counter() - started
    if data["usage"]["prompt_tokens"] != input_tokens or data["usage"]["completion_tokens"] != output_tokens:
        raise RuntimeError("response token accounting differs")
    if len(data.get("choices", [])) != 1:
        raise RuntimeError("expected one completion")
    choice = data["choices"][0]
    ids = choice.get("token_ids")
    logprobs = choice.get("logprobs") or {}
    selected = logprobs.get("token_logprobs")
    top = logprobs.get("top_logprobs")
    if (choice.get("index") != 0 or choice.get("finish_reason") != "length"
            or not isinstance(ids, list) or len(ids) != output_tokens
            or not all(type(value) is int for value in ids)
            or not isinstance(selected, list) or len(selected) != output_tokens
            or not all(type(value) in (int, float) and math.isfinite(value) for value in selected)
            or not isinstance(top, list) or len(top) != output_tokens
            or not all(isinstance(value, dict) and value
                       and all(type(score) in (int, float) and math.isfinite(score)
                               for score in value.values()) for value in top)):
        raise RuntimeError("completion token/logprob coverage differs")
    return {"index": index, "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
            "token_ids": ids, "selected_logprobs": selected, "top_logprobs": top,
            "duration_s_diagnostic": elapsed}


def validate_activation(events, manifest):
    builds = [e for e in events if e.get("event") == "plugin_build_active"
              and e.get("version") == manifest["plugin_version"]
              and e.get("prefill_mc2") is True
              and e.get("prefill_mc2_mixed") is manifest["prefill_mc2_mixed"]]
    prepared = [e for e in events if e.get("event") == "prefill_mc2_prepared"]
    pids = {e["pid"] for e in prepared}
    if len(prepared) != 4 or len(pids) != 4:
        raise RuntimeError("four MC2 preparation records required")
    if not pids.issubset({e["pid"] for e in builds}):
        raise RuntimeError("matching plugin activation missing from a prepared worker")
    if not all(e.get("allow_mixed") is manifest["prefill_mc2_mixed"]
               and e.get("source_sha256") == manifest["module_sha256"]["patches/prefill_mc2.py"]
               and e.get("weights_modified") is False for e in prepared):
        raise RuntimeError("MC2 preparation provenance differs")
    mixed = [e for e in events if e.get("event") == "prefill_mc2_mixed_called"]
    if manifest["prefill_mc2_mixed"]:
        if len(mixed) != 512 or {e["pid"] for e in mixed} != pids:
            raise RuntimeError("all128 targets on all4 workers must execute a mixed fused call")
        if not all(e.get("phase") == "mixed_prefill"
                   and e.get("source_sha256") == manifest["module_sha256"]["patches/prefill_mc2.py"]
                   for e in mixed):
            raise RuntimeError("mixed-call provenance/phase differs")
    elif mixed:
        raise RuntimeError("mixed-disabled control executed mixed fusion")
    return {"worker_pids": sorted(pids), "prepared": len(prepared), "mixed_calls": len(mixed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", required=True, type=Path)
    parser.add_argument("--benchmark-driver", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:6666")
    parser.add_argument("--variant", choices=("dk32", "dk64", "baseline"), default="dk64")
    args = parser.parse_args()
    directory = args.service_dir.resolve()
    output = directory / "mixed-sequence.json"
    if not any(root in directory.parents for root in ROOTS) or output.exists():
        parser.error("fresh receipt below the mixed experiment required")
    manifest_path = directory / "service-manifest.json"
    deadline = time.monotonic() + 900
    while True:
        try:
            manifest = json.loads(manifest_path.read_text())
            with urlopen(args.base_url + "/health", timeout=5) as response:
                if response.status == 200:
                    break
        except (OSError, ValueError):
            pass
        if (directory / "service-exit.json").exists():
            raise RuntimeError("service exited before readiness")
        if time.monotonic() >= deadline:
            raise TimeoutError("service startup exceeded900s")
        time.sleep(5)
    if (manifest.get("plugin_version") not in ("0.1.9", "0.1.10", "0.1.11")
            or manifest.get("variant") != args.variant
            or manifest.get("layout") != "packed" or manifest.get("prefill_mc2") is not True
            or type(manifest.get("prefill_mc2_mixed")) is not bool
            or manifest.get("profiler_config") is not None
            or manifest.get("diagnostic_mc2_coverage") is not None
            or manifest.get("diagnostic_determinism") is not None):
        raise RuntimeError("not a frozen compatible native sequence configuration")
    driver = load_driver(args.benchmark_driver)
    config = SimpleNamespace(base_url=args.base_url, model="qwen3.8", input_tokens=INPUT_TOKENS,
                             prefix_tokens=INPUT_TOKENS // 2, requests=REQUESTS,
                             warmup_requests=0, timeout=1800, respect_eos=False)
    prompts = driver._build_prompts(config)
    report = {"schema": "drrqr-mc2-mixed-sequence/v1", "status": "running",
              "service_manifest_sha256": digest(manifest_path), "driver_sha256": DRIVER_SHA,
              "script_sha256": digest(Path(__file__)), "prefill_mc2_mixed": manifest["prefill_mc2_mixed"],
              "variant": args.variant,
              "protocol": {"requests": REQUESTS, "concurrency": REQUESTS,
                           "input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
                           "temperature": 0, "seed": 0, "ignore_eos": True, "logprobs": 5},
              "claim": "mixed-scheduler activation and multi-step output trajectory; diagnostic, not throughput or task accuracy",
              "results": []}
    try:
        warm_config = SimpleNamespace(**{**vars(config), "input_tokens": 2048,
                                         "prefix_tokens": 1024, "requests": 1})
        request(args.base_url, driver._build_prompts(warm_config)[0], -1, None,
                input_tokens=2048, output_tokens=1)
        barrier = threading.Barrier(REQUESTS)
        with ThreadPoolExecutor(max_workers=REQUESTS) as pool:
            futures = {pool.submit(request, args.base_url, prompt, i, barrier,
                                   input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS): i
                       for i, prompt in enumerate(prompts)}
            for future in as_completed(futures):
                report["results"].append(future.result())
        report["results"].sort(key=lambda value: value["index"])
        events = [json.loads(line) for line in (directory / "evidence.jsonl").read_text().splitlines()]
        report["activation"] = validate_activation(events, manifest)
        report["evidence_sha256"] = digest(directory / "evidence.jsonl")
        report["status"] = "sequence_probe_pass"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "activation": report.get("activation"),
                      "results": len(report["results"])}), flush=True)


if __name__ == "__main__":
    main()
