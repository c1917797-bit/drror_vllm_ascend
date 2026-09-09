#!/usr/bin/env python3
"""Compare mixed-enabled and mixed-disabled native sequence probes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

MAX_SELECTED_LOGPROB_DELTA = 0.02
MEAN_SELECTED_LOGPROB_DELTA = 0.005


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control-report", default="mixed-sequence.json")
    parser.add_argument("--candidate-report", default="mixed-sequence.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve prior comparison")
    sides = {}
    manifests = {}
    for name, directory in (("control", args.control_dir), ("candidate", args.candidate_dir)):
        report_path = directory / (args.control_report if name == "control" else args.candidate_report)
        manifest_path = directory / "service-manifest.json"
        report = json.loads(report_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if (report["status"] != "sequence_probe_pass"
                or report["service_manifest_sha256"] != digest(manifest_path)
                or report.get("variant") != manifest.get("variant")
                or manifest["plugin_version"] not in ("0.1.9", "0.1.10", "0.1.11")
                or manifest["prefill_mc2_mixed"] is (name == "control")):
            raise RuntimeError(name + " provenance/configuration differs")
        sides[name] = report
        manifests[name] = manifest
    ignored_flags = {"VLLM_ASCEND_DRRQR_PREFILL_MC2_MIXED", "VLLM_ASCEND_DRRQR_EVIDENCE_FILE"}
    left_flags = {k: v for k, v in manifests["control"]["plugin_flags"].items() if k not in ignored_flags}
    right_flags = {k: v for k, v in manifests["candidate"]["plugin_flags"].items() if k not in ignored_flags}
    for field in ("variant", "layout", "plugin_version", "command", "profiler_config",
                  "module_sha256", "build_manifest_sha256", "wheel_sha256", "source_sha256",
                  "reference_manifest_sha256", "plan_sha256", "physical_npu_contract",
                  "container_npu_namespace", "inherited_runtime_environment"):
        if manifests["control"][field] != manifests["candidate"][field]:
            raise RuntimeError("service manifests differ beyond treatment: " + field)
    if left_flags != right_flags:
        raise RuntimeError("plugin flags differ beyond treatment/evidence path")
    control = sides["control"]["results"]
    candidate = sides["candidate"]["results"]
    if len(control) != 16 or [r["index"] for r in control] != list(range(16)):
        raise RuntimeError("control sequence coverage differs")
    if [r["index"] for r in candidate] != list(range(16)):
        raise RuntimeError("candidate sequence coverage differs")
    deltas = []
    exact = True
    prompt_match = True
    rows = []
    for left, right in zip(control, candidate):
        prompt_match &= left["prompt_sha256"] == right["prompt_sha256"]
        exact &= left["token_ids"] == right["token_ids"]
        values = [abs(a - b) for a, b in zip(left["selected_logprobs"], right["selected_logprobs"])]
        if len(values) != 16 or not all(math.isfinite(value) for value in values):
            raise RuntimeError("logprob step coverage differs")
        deltas.extend(values)
        rows.append({"index": left["index"], "prompt_match": left["prompt_sha256"] == right["prompt_sha256"],
                     "token_ids_exact": left["token_ids"] == right["token_ids"],
                     "selected_logprob_max_abs": max(values),
                     "selected_logprob_mean_abs": sum(values) / len(values)})
    maximum = max(deltas)
    mean = sum(deltas) / len(deltas)
    passed = prompt_match and exact and maximum <= MAX_SELECTED_LOGPROB_DELTA and mean <= MEAN_SELECTED_LOGPROB_DELTA
    result = {"schema": "drrqr-mc2-mixed-sequence-comparison/v1",
              "status": "pass" if passed else "failed",
              "criteria": {"prompt_hashes_exact": True, "token_ids_exact": True,
                           "selected_logprob_max_abs": MAX_SELECTED_LOGPROB_DELTA,
                           "selected_logprob_mean_abs": MEAN_SELECTED_LOGPROB_DELTA},
              "summary": {"prompt_hashes_exact": prompt_match, "token_ids_exact": exact,
                          "selected_logprob_max_abs": maximum, "selected_logprob_mean_abs": mean,
                          "token_steps": len(deltas)},
              "rows": rows,
              "artifacts": {"control": digest(args.control_dir / args.control_report),
                            "candidate": digest(args.candidate_dir / args.candidate_report)},
              "claim": "representative native multi-step incremental-effect gate; not full task accuracy"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
