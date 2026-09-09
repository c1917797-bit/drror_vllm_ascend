#!/usr/bin/env python3
"""Separate output repeatability under serial and single-API prompt-batch submission.

Diagnostic only: neither accuracy nor performance acceptance. Existing failed probes
are preserved. A prompt-batch API request does not prove a fixed device batch schedule.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

DRIVER_SHA = "cba2b1b463a1e436c818722d92147d93f8396b8d64f8da6582d7e67d2cb45822"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rounds):
    expected = [r["id"] for r in rounds[0]]
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate case IDs")
    if any([r["id"] for r in row] != expected for row in rounds):
        raise ValueError("case coverage differs across rounds")
    result = []
    for rows in zip(*rounds):
        tokens = [r["token_ids"] for r in rows]
        if any(len(t) != 128 for t in tokens):
            raise ValueError("incomplete output")
        differences = [i for i in range(128) if len({t[i] for t in tokens}) > 1]
        result.append({"id": rows[0]["id"], "stable": not differences,
                       "first_different_position": differences[0] if differences else None,
                       "different_positions": len(differences)})
    return {"stable_cases": sum(r["stable"] for r in result),
            "total_cases": len(result), "cases": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", type=Path, required=True)
    parser.add_argument("--benchmark-driver", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:6666")
    args = parser.parse_args()
    report_path = args.service_dir / "output-isolation.v1.json"
    event_path = args.service_dir / "output-isolation.v1.events.jsonl"
    if report_path.exists() or event_path.exists():
        parser.error("preserve existing isolation results")
    manifest_path = args.service_dir / "service-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prior = json.loads((args.service_dir / "output-probe.json").read_text())
    if prior["service_manifest_sha256"] != digest(manifest_path):
        raise RuntimeError("prior probe targets a different service")
    if digest(args.benchmark_driver) != DRIVER_SHA:
        raise RuntimeError("frozen prompt driver changed")
    with urlopen(args.base_url + "/health", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("service not ready")
    def event(name, **detail):
        record = {"event": name, "epoch_s": time.time(), **detail}
        with event_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    spec = importlib.util.spec_from_file_location("frozen_isolation_prompt_driver", args.benchmark_driver)
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    config = SimpleNamespace(base_url=args.base_url, model="qwen3.8", input_tokens=2048,
                             prefix_tokens=1024, requests=16, warmup_requests=0,
                             timeout=900, respect_eos=False)
    prompts = bench._build_prompts(config)
    expected = {c["id"]: c["prompt_sha256"] for c in prior["cases"] if c["input_tokens"] == 2048}
    cases = [{"id": f"input2048-{i}", "prompt": p,
              "prompt_sha256": hashlib.sha256(json.dumps(p).encode()).hexdigest()}
             for i, p in enumerate(prompts)]
    if len(cases) != 16 or {c["id"]: c["prompt_sha256"] for c in cases} != expected:
        raise RuntimeError("prompts differ from preserved probe")

    def request(selected, batch):
        payload = {"model": "qwen3.8", "prompt": [c["prompt"] for c in selected] if batch else selected[0]["prompt"],
                   "temperature": 0, "seed": 0, "max_tokens": 128, "ignore_eos": True,
                   "stream": False, "return_token_ids": True, "logprobs": 5}
        req = Request(args.base_url + "/v1/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=900) as response:
            data = json.load(response)
        n = len(selected)
        choices = sorted(data["choices"], key=lambda c: c["index"])
        if (len(choices) != n or [c["index"] for c in choices] != list(range(n))
                or data["usage"]["prompt_tokens"] != n * 2048
                or data["usage"]["completion_tokens"] != n * 128):
            raise RuntimeError("invalid response coverage")
        result = []
        for case, choice in zip(selected, choices):
            tokens = choice.get("token_ids")
            if (choice["finish_reason"] != "length" or not isinstance(tokens, list)
                    or len(tokens) != 128 or not all(type(t) is int for t in tokens)):
                raise RuntimeError("invalid response token IDs")
            result.append({"id": case["id"], "token_ids": tokens, "logprobs": choice.get("logprobs"),
                           "text_sha256": hashlib.sha256(choice["text"].encode()).hexdigest()})
        return result

    event("isolation_started", variant=manifest["variant"], layout=manifest["layout"])
    request(cases[:1], False)
    event("warmup_complete")
    modes = {}
    for mode in ("serial", "prompt_batch"):
        rounds = []
        for repeat in range(2):
            result = request(cases, True) if mode == "prompt_batch" else [
                row for case in cases for row in request([case], False)]
            rounds.append(sorted(result, key=lambda r: r["id"]))
            event("round_complete", mode=mode, repeat=repeat, requests=len(result))
        modes[mode] = {"summary": summarize(rounds), "rounds": rounds}
        # Preserve completed modes even if a later request fails.
        report = {"schema": "drrqr-layout-isolation/v1",
                  "variant": manifest["variant"], "layout": manifest["layout"],
                  "service_manifest_sha256": digest(manifest_path),
                  "prior_probe_sha256": digest(args.service_dir / "output-probe.json"),
                  "script_sha256": digest(Path(__file__)), "driver_sha256": DRIVER_SHA,
                  "cases": [{k: v for k, v in c.items() if k != "prompt"} for c in cases],
                  "modes": modes, "completed": False,
                  "claim": "repeatability diagnostic; no quality, performance or cross-layout acceptance"}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        event("mode_complete", mode=mode, **{k: v for k, v in modes[mode]["summary"].items() if k != "cases"})
    report["completed"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    event("isolation_complete", output=str(report_path))


if __name__ == "__main__":
    main()

