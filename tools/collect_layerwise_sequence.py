#!/usr/bin/env python3
"""Bounded no-thinking model integration requests; no throughput/accuracy claim."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_receipt(path, report):
    """Publish complete JSON atomically without replacing any old receipt."""
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", required=True, type=Path)
    args = parser.parse_args()
    directory = args.service_dir.resolve()
    if Path("/drrqr-results") not in directory.parents:
        parser.error("service directory must be below /drrqr-results")
    output = directory / "integration-sequence.json"
    if output.exists():
        parser.error("a fresh request receipt is required")
    manifest_path = directory / "service-manifest.json"
    report = {
        "schema": "drrqr-layerwise-integration-sequence/v1", "status": "running",
        "service_manifest_sha256": None, "script_sha256": sha(Path(__file__)),
        "protocol": {"thinking": False, "temperature": 0, "seed": 0,
                     "requests": 4, "batches": [1, 3], "output_tokens": 32,
                     "ignore_eos": True, "logprobs": 5},
        "comparison_criteria": {
            "prompt_hashes_exact": True, "token_ids_exact": True,
            "selected_logprob_max_abs": 0.02, "selected_logprob_mean_abs": 0.005,
            "note": "Same representative endpoint tolerance as the prior native incremental-effect gate. Pure storage/output-state bitwise gates remain separate; report exact logprob equality as well.",
        },
        "claim": "model integration and representative output trajectory only",
        "results": [],
    }
    # Claim before health checks or requests. A second collector must neither
    # send duplicate requests nor stop the collector that owns this directory.
    claim = directory / "integration-sequence.claim.json"
    try:
        with claim.open("x", encoding="utf-8") as stream:
            json.dump({"status": "claimed", "pid": os.getpid(),
                       "started_unix_ns": time.time_ns(),
                       "script_sha256": report["script_sha256"]}, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        parser.error("this service directory is already claimed by a collector")
    try:
        deadline = time.monotonic() + 900
        while True:
            if (directory / "service-exit.json").exists():
                raise RuntimeError("service exited before requests")
            if manifest_path.exists():
                try:
                    with urlopen("http://127.0.0.1:6666/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError("900-second startup budget expired")
            time.sleep(3)
        report["service_manifest_sha256"] = sha(manifest_path)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/cache/austinov/Qwen3.8-27B",
                                                  trust_remote_code=True, local_files_only=True)
        tasks = [
            "Calculate 17 times 23. Give the answer and one sentence of explanation.",
            "A store has 83 apples, sells 29, and receives 16 more. How many apples remain?",
            "Explain the difference between a list and a tuple in Python in two sentences.",
            "What number is 25 percent of 960? Give the result and explain briefly.",
        ]
        repeats = [0, 200, 800, 1800]
        context = "The archive entry says the river is blue and the bridge has three arches. "
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": context * n + "\nTask: " + task}],
                tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False,
            )
            for n, task in zip(repeats, tasks)
        ]
        if any(not isinstance(p, list) or not p
               or any(type(token) is not int for token in p) for p in prompts):
            raise RuntimeError("tokenizer did not return flat integer token IDs")
        if any(len(p) + 32 > 40960 for p in prompts):
            raise RuntimeError("numerical prompt exceeds fixed model context")
        report["protocol"]["input_tokens"] = [len(p) for p in prompts]

        def request(index, barrier=None):
            prompt = prompts[index]
            payload = {"model": "qwen3.8", "prompt": prompt, "temperature": 0,
                       "seed": 0, "max_tokens": 32, "ignore_eos": True,
                       "return_token_ids": True, "logprobs": 5, "stream": False}
            if barrier is not None:
                barrier.wait(timeout=30)
            req = Request("http://127.0.0.1:6666/v1/completions",
                          data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=600) as response:
                result = json.load(response)
            choice = result["choices"][0]
            ids = choice.get("token_ids")
            scores = (choice.get("logprobs") or {}).get("token_logprobs")
            if (result["usage"]["prompt_tokens"] != len(prompt)
                    or result["usage"]["completion_tokens"] != 32
                    or not isinstance(ids, list) or len(ids) != 32
                    or any(type(i) is not int for i in ids)
                    or not isinstance(scores, list) or len(scores) != 32
                    or any(type(x) not in (float, int) or not math.isfinite(x) for x in scores)
                    or choice.get("finish_reason") != "length"):
                raise RuntimeError("response token/logprob coverage mismatch")
            return {"index": index, "input_tokens": len(prompt),
                    "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                    "token_ids": ids, "selected_logprobs": scores}
        report["results"].append(request(0))
        barrier = threading.Barrier(3)
        with ThreadPoolExecutor(max_workers=3) as executor:
            report["results"].extend(executor.map(lambda i: request(i, barrier), (1, 2, 3)))
        report["status"] = "requests_complete"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        try:
            publish_receipt(output, report)
        finally:
            (directory / "stop.request").touch(exist_ok=True)
    print(json.dumps({"status": report["status"], "output": str(output),
                      "input_tokens": report["protocol"].get("input_tokens")}), flush=True)


if __name__ == "__main__":
    main()
