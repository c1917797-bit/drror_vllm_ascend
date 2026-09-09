#!/usr/bin/env python3
"""Collect deterministic output tokens for original/packed model comparison.

Synthetic benchmark prompts only. This is a runtime equivalence probe, not accuracy
evaluation or a throughput benchmark. Profile collection can follow after it succeeds.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

DRIVER_SHA = "cba2b1b463a1e436c818722d92147d93f8396b8d64f8da6582d7e67d2cb45822"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", type=Path, required=True)
    parser.add_argument("--benchmark-driver", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:6666")
    parser.add_argument("--profile-after", action="store_true")
    args = parser.parse_args()
    report_path = args.service_dir / "output-probe.json"
    events = args.service_dir / "output-probe.events.jsonl"
    if report_path.exists() or events.exists():
        parser.error("preserve existing output probes")
    manifest_path = args.service_dir / "service-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if digest(args.benchmark_driver) != DRIVER_SHA:
        raise RuntimeError("frozen benchmark prompt driver changed")
    def event(name, **detail):
        value = {"event": name, "epoch_s": time.time(), **detail}
        with events.open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)
    event("waiting_for_service")
    deadline = time.monotonic() + 900
    while True:
        try:
            with urlopen(args.base_url + "/health", timeout=5) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("service readiness timed out; inspect the existing service handle")
        time.sleep(5)
    event("service_ready")
    spec = importlib.util.spec_from_file_location("frozen_layout_prompt_driver", args.benchmark_driver)
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    cases = []
    for length, count, concurrency in ((2048, 16, 16), (32768, 2, 1)):
        config = SimpleNamespace(base_url=args.base_url, model="qwen3.8",
                                 input_tokens=length, prefix_tokens=length // 2,
                                 requests=count, warmup_requests=0, timeout=900,
                                 respect_eos=False)
        prompts = bench._build_prompts(config)
        if len(prompts) != count:
            raise RuntimeError("unexpected prompt count")
        for index, prompt in enumerate(prompts):
            cases.append({"id": f"input{length}-{index}", "prompt": prompt,
                          "input_tokens": length, "output_tokens": 128,
                          "concurrency": concurrency,
                          "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest()})
    def request(case, barrier=None):
        if barrier is not None:
            barrier.wait(timeout=900)
        payload = {"model": "qwen3.8", "prompt": case["prompt"],
                   "temperature": 0, "seed": 0, "max_tokens": case["output_tokens"],
                   "ignore_eos": True, "stream": False, "return_token_ids": True}
        req = Request(args.base_url + "/v1/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=900) as response:
            data = json.load(response)
        choice = data["choices"][0]
        tokens = choice.get("token_ids")
        if (data["usage"]["prompt_tokens"] != case["input_tokens"]
                or data["usage"]["completion_tokens"] != case["output_tokens"]
                or choice["finish_reason"] != "length"
                or not isinstance(tokens, list) or len(tokens) != case["output_tokens"]
                or not all(isinstance(t, int) for t in tokens)):
            raise RuntimeError(f"invalid output coverage for {case['id']}")
        return {"id": case["id"], "token_ids": tokens, "finish_reason": choice["finish_reason"],
                "text_sha256": hashlib.sha256(choice["text"].encode()).hexdigest()}
    rounds = []
    for repeat in range(2):
        result = []
        for case in [c for c in cases if c["concurrency"] == 1]:
            result.append(request(case))
        event("sequential_long_complete", repeat=repeat, requests=2)
        barrier = threading.Barrier(16)
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(request, c, barrier) for c in cases if c["concurrency"] == 16]
            result += [f.result() for f in futures]
        rounds.append(sorted(result, key=lambda r: r["id"]))
        event("round_complete", repeat=repeat, requests=len(result))
    stable = all(a["token_ids"] == b["token_ids"] for a, b in zip(*rounds))
    report = {"schema": "drrqr-layout-output-probe/v1",
              "status": "repeat_stable" if stable else "repeat_mismatch",
              "variant": manifest["variant"], "layout": manifest["layout"],
              "service_manifest_sha256": digest(manifest_path),
              "driver_sha256": DRIVER_SHA, "script_sha256": digest(Path(__file__)),
              "cases": [{k: v for k, v in c.items() if k != "prompt"} for c in cases],
              "rounds": rounds,
              "claim": "single-configuration output probe; cross-layout comparison remains required"}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    event("probe_complete", status=report["status"], output=str(report_path))
    if not stable:
        raise RuntimeError("same-layout repeated outputs differ; investigate before attributing layout differences")
    if args.profile_after:
        event("starting_profile_client")
        subprocess.run([sys.executable, str(Path(__file__).with_name("collect_service_profile.py")),
                        "--benchmark-driver", str(args.benchmark_driver),
                        "--service-dir", str(args.service_dir),
                        "--base-url", args.base_url], check=True)
        event("profile_client_complete")


if __name__ == "__main__":
    main()
