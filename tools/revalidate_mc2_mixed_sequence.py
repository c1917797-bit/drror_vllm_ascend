#!/usr/bin/env python3
"""Revalidate the preserved control receipt rejected by the superseded PID rule."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", required=True, type=Path)
    parser.add_argument("--benchmark-driver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:6666")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous receipt")
    tool = Path(__file__).with_name("collect_mc2_mixed_sequence.py")
    spec = importlib.util.spec_from_file_location("mc2_mixed_validator", tool)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    manifest_path = args.service_dir / "service-manifest.json"
    evidence_path = args.service_dir / "evidence.jsonl"
    original_path = args.service_dir / "mixed-sequence.json"
    manifest = json.loads(manifest_path.read_text())
    original = json.loads(original_path.read_text())
    if (original.get("status") != "failed"
            or original.get("error") != "RuntimeError('four matching plugin workers required')"
            or original.get("service_manifest_sha256") != digest(manifest_path)
            or original.get("prefill_mc2_mixed") is not False):
        raise RuntimeError("not the preserved validator-only control failure")
    results = original.get("results")
    if len(results) != module.REQUESTS or [row.get("index") for row in results] != list(range(module.REQUESTS)):
        raise RuntimeError("request coverage incomplete")
    driver = module.load_driver(args.benchmark_driver)
    config = SimpleNamespace(base_url=args.base_url, model="qwen3.8", input_tokens=module.INPUT_TOKENS,
                             prefix_tokens=module.INPUT_TOKENS // 2, requests=module.REQUESTS,
                             warmup_requests=0, timeout=1800, respect_eos=False)
    prompts = driver._build_prompts(config)
    expected = [hashlib.sha256(json.dumps(prompt).encode()).hexdigest() for prompt in prompts]
    if [row.get("prompt_sha256") for row in results] != expected:
        raise RuntimeError("prompt hashes differ")
    for row in results:
        if (len(row.get("token_ids", [])) != module.OUTPUT_TOKENS
                or len(row.get("selected_logprobs", [])) != module.OUTPUT_TOKENS
                or len(row.get("top_logprobs", [])) != module.OUTPUT_TOKENS):
            raise RuntimeError("stored sequence coverage differs")
    events = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    activation = module.validate_activation(events, manifest)
    result = dict(original)
    result.update(status="sequence_probe_pass", error=None, activation=activation,
                  evidence_sha256=digest(evidence_path),
                  original_failed_receipt_sha256=digest(original_path),
                  revalidator_sha256=digest(Path(__file__)),
                  corrected_validator_sha256=digest(tool),
                  revalidation="post-run PID-scope correction only; no requests repeated")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "activation": activation,
                      "results": len(results)}), flush=True)


if __name__ == "__main__":
    main()
