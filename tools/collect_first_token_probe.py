#!/usr/bin/env python3
"""Isolate prefill-output repeatability with one-token, serial requests.

Frozen synthetic inputs; no accuracy or performance claims. Run only after other clients
on the same service have completed. This does not change the service or enable thinking.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

DRIVER_SHA = "cba2b1b463a1e436c818722d92147d93f8396b8d64f8da6582d7e67d2cb45822"
CASE_IDS = ("input2048-0", "input2048-1", "input2048-12", "input2048-15", "input32768-0")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_response(data, input_tokens):
    if (data["usage"]["prompt_tokens"] != input_tokens
            or data["usage"]["completion_tokens"] != 1 or len(data["choices"]) != 1):
        raise ValueError("incorrect response coverage")
    c = data["choices"][0]
    if (c["index"] != 0 or c["finish_reason"] != "length"
            or not isinstance(c.get("token_ids"), list) or len(c["token_ids"]) != 1
            or type(c["token_ids"][0]) is not int):
        raise ValueError("incorrect first-token response")
    lp = c.get("logprobs")
    if (not lp or len(lp["top_logprobs"]) != 1 or len(lp["token_logprobs"]) != 1
            or not lp["top_logprobs"][0]
            or not all(math.isfinite(v) for v in lp["top_logprobs"][0].values())
            or not math.isfinite(lp["token_logprobs"][0])):
        raise ValueError("missing/nonfinite first-token probabilities")
    return {"token_id": c["token_ids"][0], "selected_logprob": lp["token_logprobs"][0],
            "top_logprobs": lp["top_logprobs"][0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", type=Path, required=True)
    parser.add_argument("--benchmark-driver", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:6666")
    args = parser.parse_args()
    target = args.service_dir / "first-token-probe.v1.json"
    events = args.service_dir / "first-token-probe.v1.events.jsonl"
    if target.exists() or events.exists():
        parser.error("preserve prior first-token diagnostics")
    manifest_path = args.service_dir / "service-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prior_path = args.service_dir / "output-probe.json"
    prior = json.loads(prior_path.read_text())
    if prior["service_manifest_sha256"] != digest(manifest_path):
        raise RuntimeError("service/probe provenance mismatch")
    if digest(args.benchmark_driver) != DRIVER_SHA:
        raise RuntimeError("frozen prompt driver changed")
    spec = importlib.util.spec_from_file_location("first_token_frozen_driver", args.benchmark_driver)
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    all_cases = {}
    expected = {c["id"]: c for c in prior["cases"]}
    for length, count in ((2048, 16), (32768, 2)):
        config = SimpleNamespace(base_url=args.base_url, model="qwen3.8", input_tokens=length,
                                 prefix_tokens=length // 2, requests=count, warmup_requests=0,
                                 timeout=900, respect_eos=False)
        for i, prompt in enumerate(bench._build_prompts(config)):
            name = f"input{length}-{i}"
            checksum = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            if expected[name]["prompt_sha256"] != checksum:
                raise RuntimeError("prompt hash differs")
            all_cases[name] = {"id": name, "prompt": prompt, "input_tokens": length,
                               "prompt_sha256": checksum}
    cases = [all_cases[name] for name in CASE_IDS]
    def event(name, **detail):
        value = {"event": name, "epoch_s": time.time(), **detail}
        with events.open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)
    def request(case):
        payload = {"model": "qwen3.8", "prompt": case["prompt"], "temperature": 0, "seed": 0,
                   "max_tokens": 1, "ignore_eos": True, "stream": False,
                   "return_token_ids": True, "logprobs": 5}
        req = Request(args.base_url + "/v1/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=900) as response:
            result = validate_response(json.load(response), case["input_tokens"])
        return {"id": case["id"], **result}
    event("started", variant=manifest["variant"], layout=manifest["layout"])
    for case in cases:
        request(case)
    event("warmup_complete")
    rounds = []
    for repeat in range(4):
        rounds.append([request(case) for case in cases])
        report = {"schema": "drrqr-first-token-probe/v1", "completed": False,
                  "variant": manifest["variant"], "layout": manifest["layout"],
                  "service_manifest_sha256": digest(manifest_path),
                  "prior_probe_sha256": digest(prior_path), "driver_sha256": DRIVER_SHA,
                  "script_sha256": digest(Path(__file__)), "output_tokens": 1,
                  "cases": [{k: v for k, v in c.items() if k != "prompt"} for c in cases],
                  "rounds": rounds,
                  "claim": "serial prefill-output diagnostic; no decode-chain, quality or speed acceptance"}
        target.write_text(json.dumps(report, indent=2) + "\n")
        event("round_complete", repeat=repeat)
    report["summary"] = [{"id": c["id"],
                          "distinct_first_tokens": sorted({row[i]["token_id"] for row in rounds}),
                          "selected_logprob_range": [
                              min(row[i]["selected_logprob"] for row in rounds),
                              max(row[i]["selected_logprob"] for row in rounds)],
                          "top5_exactly_repeated": all(row[i]["top_logprobs"] == rounds[0][i]["top_logprobs"]
                                                     for row in rounds)}
                         for i, c in enumerate(cases)]
    report["completed"] = True
    target.write_text(json.dumps(report, indent=2) + "\n")
    event("complete", summary=report["summary"])


if __name__ == "__main__":
    main()

