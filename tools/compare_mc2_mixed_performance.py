#!/usr/bin/env python3
"""Validate and summarize the0.1.9 mixed MC2 single performance pair."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--sequence-comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous comparison")
    sequence = json.loads(args.sequence_comparison.read_text())
    if sequence.get("status") != "pass":
        raise RuntimeError("native sequence comparison did not pass")
    data = {}
    manifests = {}
    for side, directory in (("control", args.control_dir), ("candidate", args.candidate_dir)):
        manifest_path = directory / "service-manifest.json"
        screen_path = directory / "performance-screen.json"
        raw_path = directory / "performance.run1.json"
        manifest = json.loads(manifest_path.read_text())
        screen = json.loads(screen_path.read_text())
        raw = json.loads(raw_path.read_text())
        if (screen["status"] != "screen_complete"
                or screen["service_manifest_sha256"] != digest(manifest_path)
                or screen["report_sha256"] != digest(raw_path)
                or raw["summary"]["requests"] != 40 or raw["summary"]["successful"] != 40
                or raw["summary"]["fixed_output_verified"] is not True
                or raw["summary"]["output_tokens"] != 40960):
            raise RuntimeError(side + " screen provenance/work differs")
        if manifest["prefill_mc2_mixed"] is (side == "control"):
            raise RuntimeError(side + " treatment flag differs")
        data[side] = screen["summary"]
        manifests[side] = manifest
    ignored_flags = {"VLLM_ASCEND_DRRQR_PREFILL_MC2_MIXED", "VLLM_ASCEND_DRRQR_EVIDENCE_FILE"}
    left = {k: v for k, v in manifests["control"]["plugin_flags"].items() if k not in ignored_flags}
    right = {k: v for k, v in manifests["candidate"]["plugin_flags"].items() if k not in ignored_flags}
    if left != right:
        raise RuntimeError("plugin flags differ beyond treatment/evidence path")
    for field in ("variant", "layout", "plugin_version", "command", "profiler_config",
                  "module_sha256", "build_manifest_sha256", "wheel_sha256", "source_sha256",
                  "reference_manifest_sha256", "plan_sha256", "physical_npu_contract",
                  "container_npu_namespace", "inherited_runtime_environment"):
        if manifests["control"][field] != manifests["candidate"][field]:
            raise RuntimeError("manifests differ beyond treatment: " + field)
    metrics = {"output_tokens_per_second": True, "ttft_mean_ms": False,
               "tpot_mean_ms": False, "benchmark_duration_s": False}
    summary = {}
    for metric, higher_better in metrics.items():
        control = data["control"][metric]
        candidate = data["candidate"][metric]
        summary[metric] = {"control": control, "candidate": candidate,
                           "change_percent": (candidate / control - 1) * 100,
                           "improvement_percent": ((candidate / control - 1) if higher_better
                                                   else (1 - candidate / control)) * 100}
    result = {"schema": "drrqr-prefill-mc2-mixed-pair/v1", "status": "single_pair_complete",
              "summary": summary, "requests_per_side": 40, "output_tokens_per_side": 40960,
              "sequence_comparison_sha256": digest(args.sequence_comparison),
              "artifacts": {side: {"manifest": digest(directory / "service-manifest.json"),
                                   "screen": digest(directory / "performance-screen.json"),
                                   "raw": digest(directory / "performance.run1.json")}
                            for side, directory in (("control", args.control_dir),
                                                    ("candidate", args.candidate_dir))},
              "claim": "single Dk64 treatment screen; not repeatability, dense-baseline, or task-accuracy acceptance"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
