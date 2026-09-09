#!/usr/bin/env python3
"""Drive bounded prefill/decode profiling using the frozen performance prompts.

Runs on the host. Uses built-in service profiling endpoints; does not patch
the model. SSE chunk counts are only a readiness signal, not token/step counts.
The server's audited max_iterations bounds the actual decode recording.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

from collect_mc2_mixed_sequence import validate_activation


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def post(url, payload, timeout):
    return urlopen(Request(url, data=json.dumps(payload).encode(),
                           headers={"Content-Type": "application/json"},
                           method="POST"), timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-driver", type=Path, required=True)
    parser.add_argument("--service-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:6666")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    output = args.service_dir / "profile-client.json"
    events_path = args.service_dir / "profile-client.events.jsonl"
    if output.exists() or events_path.exists():
        parser.error("client artifacts already exist; preserve old evidence")
    if not (args.service_dir / "service-manifest.json").is_file():
        parser.error("service manifest must exist")
    manifest = json.loads((args.service_dir / "service-manifest.json").read_text())
    if manifest["profiler_config"]["max_iterations"] != 30:
        parser.error("this frozen protocol requires 30 recorded decode steps")
    spec = importlib.util.spec_from_file_location("frozen_shared_prefix", args.benchmark_driver)
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    def event(kind, **detail):
        record = {"event": kind, "epoch_s": time.time(), **detail}
        with events_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    event("waiting_for_service")
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            with urlopen(args.base_url + "/health", timeout=5) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("service readiness timed out")
        time.sleep(5)
    event("service_ready")
    config = SimpleNamespace(
        base_url=args.base_url, model="qwen3.8", input_tokens=32768,
        prefix_tokens=16384, requests=16, warmup_requests=2,
        timeout=args.timeout, respect_eos=False,
    )
    prompts = bench._build_prompts(config)
    prompt_sha = [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]
    def checked(result, tokens):
        if not result.ok or result.output_tokens != tokens or result.finish_reason != "length":
            raise RuntimeError(f"request validity check failed: {asdict(result)}")
        return asdict(result)
    def control(path):
        with post(args.base_url + path, {}, args.timeout) as response:
            response.read()
            if response.status != 200:
                raise RuntimeError("profiler endpoint failed")
    def trace_dirs():
        return {str(p.relative_to(args.service_dir)) for p in (args.service_dir / "traces").glob("*ascend_pt")}
    results = {"warmup": [], "prefill": {}, "decode": {}}
    for i in range(2):
        results["warmup"].append(checked(
            bench._stream_completion(config, i, prompts[i], None, 64), 64
        ))
    event("single_request_warmup_complete")
    before = trace_dirs()
    control("/start_profile")
    event("prefill_profile_started", requests=1, input_tokens=32768, output_tokens=1)
    try:
        result = bench._stream_completion(config, 2, prompts[2], None, 1)
        # With one token requested, generation is prefill-only. EOS may label
        # the terminal reason differently; usage is the authoritative work.
        if not result.ok or result.output_tokens != 1:
            raise RuntimeError(f"prefill request failed: {asdict(result)}")
        results["prefill"]["request"] = asdict(result)
    finally:
        control("/stop_profile")
    results["prefill"]["trace_dirs"] = sorted(trace_dirs() - before)
    event("prefill_profile_stopped", trace_dirs=results["prefill"]["trace_dirs"])
    barrier = threading.Barrier(16)
    with ThreadPoolExecutor(max_workers=16) as pool:
        warm = [pool.submit(bench._stream_completion, config, i + 2, prompts[i + 2], barrier, 64)
                for i in range(16)]
        results["warmup_concurrent"] = [checked(f.result(), 64) for f in warm]
    event("concurrent_warmup_complete", concurrency=16)
    counts = [0] * 16
    done = [False] * 16
    condition = threading.Condition()
    barrier = threading.Barrier(16)
    def stream_request(index):
        payload = {
            "model": "qwen3.8", "prompt": prompts[index + 2],
            "max_tokens": 256, "ignore_eos": True, "temperature": 0, "seed": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "vllm_xargs": {"hypic_segment_ends": "16384,32768"},
        }
        barrier.wait(timeout=args.timeout)
        started = time.time()
        usage, finish, text = None, None, []
        try:
            with post(args.base_url + "/v1/completions", payload, args.timeout) as response:
                for raw in response:
                    if not raw.startswith(b"data:"):
                        continue
                    body = raw[5:].strip()
                    if body == b"[DONE]":
                        break
                    data = json.loads(body)
                    if data.get("usage"):
                        usage = data["usage"].get("completion_tokens")
                    for choice in data.get("choices", []):
                        finish = choice.get("finish_reason") or finish
                        value = choice.get("text") or ""
                        if value:
                            text.append(value)
                            with condition:
                                counts[index] += 1
                                condition.notify_all()
            if usage != 256 or finish != "length":
                raise RuntimeError(f"decode request {index}: usage={usage}, finish={finish}")
            return {"index": index, "output_tokens": usage, "finish_reason": finish,
                    "output_sha256": hashlib.sha256("".join(text).encode()).hexdigest(),
                    "started_epoch_s": started, "finished_epoch_s": time.time()}
        finally:
            with condition:
                done[index] = True
                condition.notify_all()
    before = trace_dirs()
    profile_started = False
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(stream_request, i) for i in range(16)]
        try:
            deadline = time.monotonic() + args.timeout
            last_report = 0.
            with condition:
                while min(counts) < 8:
                    if any(done):
                        raise RuntimeError("a request ended before every request entered stable decode")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("waiting for stable decode timed out")
                    if time.monotonic() - last_report > 10:
                        event("waiting_for_all_decode", nonempty_chunk_counts=list(counts))
                        last_report = time.monotonic()
                    condition.wait(timeout=1)
                if any(done) or max(counts) > 200:
                    raise RuntimeError("insufficient remaining generation for the profile window")
                results["decode"]["nonempty_chunks_at_trigger"] = list(counts)
            control("/start_profile")
            profile_started = True
            event("decode_profile_started", concurrency=16, worker_step_limit=30,
                  nonempty_chunks_at_trigger=results["decode"]["nonempty_chunks_at_trigger"])
            results["decode"]["requests"] = [future.result() for future in futures]
        finally:
            if profile_started:
                control("/stop_profile")
    results["decode"]["trace_dirs"] = sorted(trace_dirs() - before)
    event("decode_profile_stopped", trace_dirs=results["decode"]["trace_dirs"])
    evidence_path = args.service_dir / "evidence.jsonl"
    events = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    activation = validate_activation(events, manifest)
    report = {
        "schema": "drrqr-profile-client/v1", "status": "requests_complete_activation_validated",
        "service_manifest_sha256": sha256(args.service_dir / "service-manifest.json"),
        "benchmark_driver": str(args.benchmark_driver),
        "benchmark_driver_sha256": sha256(args.benchmark_driver),
        "client_sha256": sha256(Path(__file__)), "prompt_sha256": prompt_sha,
        "profile_steps": 30, "input_tokens": 32768, "prefix_tokens": 16384,
        "decode_concurrency": 16, "decode_output_tokens": 256,
        "prefill_concurrency": 1, "prefill_output_tokens": 1,
        "activation": activation, "evidence_sha256": sha256(evidence_path),
        "scope": "diagnostic only; final throughput benchmark retains fixed 1024 outputs",
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    event("client_complete", output=str(output))


if __name__ == "__main__":
    main()
