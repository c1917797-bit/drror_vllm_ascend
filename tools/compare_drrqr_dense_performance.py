#!/usr/bin/env python3
"""Validate one reduced-Dk versus dense screen with the same common optimizations."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_side(directory, expected_variant):
    manifest_path = directory / "service-manifest.json"
    screen_path = directory / "performance-screen.json"
    raw_path = directory / "performance.run1.json"
    sequence_path = directory / "mixed-sequence.json"
    manifest = json.loads(manifest_path.read_text())
    screen = json.loads(screen_path.read_text())
    raw = json.loads(raw_path.read_text())
    sequence = json.loads(sequence_path.read_text())
    if manifest["variant"] != expected_variant or screen["variant"] != expected_variant:
        raise RuntimeError(expected_variant + " variant identity differs")
    if (screen["status"] != "screen_complete"
            or screen["service_manifest_sha256"] != digest(manifest_path)
            or screen["report_sha256"] != digest(raw_path)
            or raw["summary"]["requests"] != 40
            or raw["summary"]["successful"] != 40
            or raw["summary"]["fixed_output_verified"] is not True
            or raw["summary"]["output_tokens"] != 40960):
        raise RuntimeError(expected_variant + " screen provenance/work differs")
    if (raw.get("config", {}).get("concurrency") not in (16, 32)
            or screen.get("concurrency", raw["config"]["concurrency"]) != raw["config"]["concurrency"]):
        raise RuntimeError(expected_variant + " concurrency provenance differs")
    if (sequence["status"] != "sequence_probe_pass"
            or sequence.get("variant", expected_variant) != expected_variant
            or sequence["service_manifest_sha256"] != digest(manifest_path)
            or sequence["activation"]["prepared"] != 4
            or sequence["activation"]["mixed_calls"] != 512):
        raise RuntimeError(expected_variant + " mixed activation evidence differs")
    return {
        "directory": directory,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "screen": screen,
        "screen_path": screen_path,
        "raw_path": raw_path,
        "sequence_path": sequence_path,
    }


def normalized_command(command, target_head_k_dim):
    command = list(command)
    if target_head_k_dim is not None:
        try:
            index = command.index("--hf-overrides")
        except ValueError as exc:
            raise RuntimeError("candidate command lacks --hf-overrides") from exc
        if index + 1 >= len(command):
            raise RuntimeError("candidate --hf-overrides lacks value")
        override = json.loads(command[index + 1])
        if override != {"text_config": {"linear_key_head_dim": target_head_k_dim}}:
            raise RuntimeError("candidate head-dimension treatment differs")
        del command[index:index + 2]
    elif "--hf-overrides" in command:
        raise RuntimeError("dense baseline unexpectedly has --hf-overrides")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", "--dk64-dir", dest="candidate_dir",
                        required=True, type=Path)
    parser.add_argument("--candidate-sequence-comparison", "--dk64-sequence-comparison",
                        dest="candidate_sequence_comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous comparison")

    dense = load_side(args.dense_dir, "baseline")
    candidate_manifest = json.loads((args.candidate_dir / "service-manifest.json").read_text())
    target_head_k_dim = candidate_manifest.get("target_head_k_dim")
    candidate_variant = candidate_manifest.get("variant")
    target_head_k_dim_source = "manifest"
    if target_head_k_dim is None and candidate_variant == "dk64":
        target_head_k_dim = 64
        target_head_k_dim_source = "legacy_variant_dk64"
    if (candidate_variant != f"dk{target_head_k_dim}"
            or type(target_head_k_dim) is not int
            or not 0 < target_head_k_dim < 128
            or target_head_k_dim % 16):
        raise RuntimeError("candidate dimension/variant contract differs")
    candidate = load_side(args.candidate_dir, candidate_variant)
    sequence_comparison = json.loads(args.candidate_sequence_comparison.read_text())
    if sequence_comparison.get("status") != "pass":
        raise RuntimeError("candidate native mixed numerical comparison did not pass")
    if json.loads(dense["raw_path"].read_text())["config"] != json.loads(
            candidate["raw_path"].read_text())["config"]:
        raise RuntimeError("dense and candidate workload configs differ")

    for field in (
        "layout", "plugin_version", "prefill_mc2", "prefill_mc2_mixed",
        "profiler_config", "module_sha256", "build_manifest_sha256",
        "wheel_sha256", "source_sha256", "reference_manifest_sha256",
        "physical_npu_contract", "container_npu_namespace",
        "inherited_runtime_environment",
    ):
        if dense["manifest"][field] != candidate["manifest"][field]:
            raise RuntimeError("common manifest field differs: " + field)
    if normalized_command(dense["manifest"]["command"], None) != normalized_command(
            candidate["manifest"]["command"], target_head_k_dim):
        raise RuntimeError("service commands differ beyond candidate head dimension")

    ignored_flags = {
        "VLLM_ASCEND_DRRQR_ENABLE",
        "VLLM_ASCEND_DRRQR_EVIDENCE_FILE",
        "VLLM_ASCEND_DRRQR_PLAN_PATH",
        "VLLM_ASCEND_DRRQR_PLAN_SHA256",
    }
    dense_flags = {key: value for key, value in dense["manifest"]["plugin_flags"].items()
                   if key not in ignored_flags}
    candidate_flags = {key: value for key, value in candidate["manifest"]["plugin_flags"].items()
                       if key not in ignored_flags}
    if dense_flags != candidate_flags:
        raise RuntimeError("plugin flags differ beyond candidate enable/evidence/plan path")
    if (dense["manifest"]["plugin_flags"]["VLLM_ASCEND_DRRQR_ENABLE"] != "0"
            or dense["manifest"]["plan_sha256"] is not None
            or candidate["manifest"]["plugin_flags"]["VLLM_ASCEND_DRRQR_ENABLE"] != "1"
            or candidate["manifest"]["plan_sha256"]
            != candidate["manifest"]["plugin_flags"]["VLLM_ASCEND_DRRQR_PLAN_SHA256"]):
        raise RuntimeError("candidate treatment contract differs")

    metrics = {
        "output_tokens_per_second": True,
        "ttft_mean_ms": False,
        "tpot_mean_ms": False,
        "benchmark_duration_s": False,
    }
    summary = {}
    for metric, higher_better in metrics.items():
        control = dense["screen"]["summary"][metric]
        treatment = candidate["screen"]["summary"][metric]
        summary[metric] = {
            "dense": control,
            candidate_variant: treatment,
            "change_percent": (treatment / control - 1) * 100,
            "improvement_percent": (
                (treatment / control - 1) if higher_better else (1 - treatment / control)
            ) * 100,
        }

    result = {
        "schema": "drrqr-reduced-dk-dense-common-optimized-screen/v1",
        "status": "single_pair_complete",
        "candidate_variant": candidate_variant,
        "target_head_k_dim": target_head_k_dim,
        "target_head_k_dim_source": target_head_k_dim_source,
        "summary": summary,
        "requests_per_side": 40,
        "output_tokens_per_side": 40960,
        "concurrency": json.loads(dense["raw_path"].read_text())["config"]["concurrency"],
        "candidate_sequence_comparison_sha256": digest(args.candidate_sequence_comparison),
        "artifacts": {
            side: {
                "manifest": digest(data["manifest_path"]),
                "screen": digest(data["screen_path"]),
                "raw": digest(data["raw_path"]),
                "sequence": digest(data["sequence_path"]),
            }
            for side, data in (("dense", dense), ("candidate", candidate))
        },
        "claim": (
            "single common-optimized reduced-Dk-versus-dense screen; "
            "not repeatability or task-accuracy acceptance"
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
