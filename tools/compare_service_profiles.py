#!/usr/bin/env python3
"""Compare validated diagnostic profiles without making benchmark claims."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_head_dim(manifest):
    if "target_head_k_dim" in manifest:
        return manifest["target_head_k_dim"]
    command = manifest["command"]
    if "--hf-overrides" not in command:
        return 128
    index = command.index("--hf-overrides")
    return json.loads(command[index + 1])["text_config"]["linear_key_head_dim"]


def normalized_command(command, expected_head_dim):
    result = list(command)
    if "--profiler-config" not in result:
        raise ValueError("profiler config missing from command")
    index = result.index("--profiler-config")
    del result[index:index + 2]
    if expected_head_dim != 128:
        if "--hf-overrides" not in result:
            raise ValueError("reduced head-dimension override missing")
        index = result.index("--hf-overrides")
        override = json.loads(result[index + 1])
        if override != {"text_config": {"linear_key_head_dim": expected_head_dim}}:
            raise ValueError("reduced head-dimension override differs")
        del result[index:index + 2]
    elif "--hf-overrides" in result:
        raise ValueError("dense baseline unexpectedly has an override")
    return result


def compare(root, control_variant="baseline", candidate_variant="dk64",
            required_time_reduction_percent=None):
    if control_variant == candidate_variant:
        raise ValueError("control and candidate variants must differ")
    variants = (control_variant, candidate_variant)
    clients = {v: read(root / v / "profile-client.json") for v in variants}
    manifests = {v: read(root / v / "service-manifest.json") for v in variants}
    summaries = {v: read(root / v / "profile-summary.v1.json") for v in variants}
    shared = ("benchmark_driver_sha256", "prompt_sha256", "profile_steps", "input_tokens",
              "prefix_tokens", "decode_concurrency", "decode_output_tokens", "prefill_concurrency", "prefill_output_tokens")
    for key in shared:
        if clients[control_variant][key] != clients[candidate_variant][key]:
            raise ValueError(f"unmatched workload: {key}")
    for v in variants:
        manifest_path = root / v / "service-manifest.json"
        client_path = root / v / "profile-client.json"
        if (manifests[v]["variant"] != v
                or clients[v]["status"] != "requests_complete_activation_validated"
                or clients[v]["service_manifest_sha256"] != sha(manifest_path)
                or clients[v]["activation"]["prepared"] != 4
                or clients[v]["activation"]["mixed_calls"] != 512
                or summaries[v]["service_manifest_sha256"] != sha(manifest_path)
                or summaries[v]["client_sha256"] != sha(client_path)):
            raise ValueError(f"{v}: provenance or mixed activation differs")
    for key in (
        "source_sha256", "module_sha256", "physical_npu_contract",
        "container_npu_namespace", "layout", "plugin_version", "prefill_mc2",
        "prefill_mc2_mixed", "build_manifest_sha256", "wheel_sha256",
        "reference_manifest_sha256", "inherited_runtime_environment",
    ):
        if manifests[control_variant][key] != manifests[candidate_variant][key]:
            raise ValueError(f"unmatched runtime: {key}")
    configs = [{k: val for k, val in manifests[v]["profiler_config"].items() if k != "torch_profiler_dir"} for v in variants]
    if (configs[0] != configs[1]
            or configs[0]["max_iterations"] != 30
            or normalized_command(manifests[control_variant]["command"],
                                  target_head_dim(manifests[control_variant]))
            != normalized_command(manifests[candidate_variant]["command"],
                                  target_head_dim(manifests[candidate_variant]))):
        raise ValueError("unmatched service or profiler settings")
    ignored_flags = {
        "VLLM_ASCEND_DRRQR_ENABLE",
        "VLLM_ASCEND_DRRQR_EVIDENCE_FILE",
        "VLLM_ASCEND_DRRQR_PLAN_PATH",
        "VLLM_ASCEND_DRRQR_PLAN_SHA256",
    }
    flags = [{k: val for k, val in manifests[v]["plugin_flags"].items()
              if k not in ignored_flags} for v in variants]
    if flags[0] != flags[1]:
        raise ValueError("non-treatment plugin flags differ")
    for v in variants:
        enabled = manifests[v]["plugin_flags"]["VLLM_ASCEND_DRRQR_ENABLE"]
        plan_sha = manifests[v]["plan_sha256"]
        if target_head_dim(manifests[v]) == 128:
            if enabled != "0" or plan_sha is not None:
                raise ValueError(f"{v}: dense treatment contract differs")
        elif (enabled != "1"
              or plan_sha != manifests[v]["plugin_flags"].get(
                  "VLLM_ASCEND_DRRQR_PLAN_SHA256")
              or not manifests[v]["plugin_flags"].get(
                  "VLLM_ASCEND_DRRQR_PLAN_PATH")):
            raise ValueError(f"{v}: reduced treatment contract differs")
    records = []
    for phase in ("prefill", "decode"):
        per_variant = {}
        for v in variants:
            if summaries[v]["status"] != "eight_traces_validated":
                raise ValueError("unvalidated summary")
            rs = [r for r in summaries[v]["records"] if r["phase"] == phase]
            if len(rs) != 4 or {r["rank"] for r in rs} != {0, 1, 2, 3}:
                raise ValueError("rank coverage differs")
            if not all(c["passed"] for r in rs for c in r["coverage"].values()):
                raise ValueError("coverage failed")
            operators = defaultdict(list)
            all_types = {o["type"] for r in rs for o in r["top_operators"]}
            for kind in all_types:
                operators[kind] = [sum(o["kernel_sum_ms"] for o in r["top_operators"] if o["type"] == kind) for r in rs]
            per_variant[v] = {
                "kernel_envelope_mean_ms": statistics.mean(r["kernel_envelope_ms"] for r in rs),
                "kernel_envelope_min_ms": min(r["kernel_envelope_ms"] for r in rs),
                "kernel_envelope_max_ms": max(r["kernel_envelope_ms"] for r in rs),
                "communication_without_compute_mean_ms": statistics.mean(r["communication_without_compute_ms"] for r in rs),
                "operator_kernel_sum_rank_mean_ms": {k: statistics.mean(values) for k, values in operators.items()},
            }
        base, candidate = (per_variant[v]["kernel_envelope_mean_ms"] for v in variants)
        operator_types = set(per_variant[control_variant][
            "operator_kernel_sum_rank_mean_ms"]) | set(per_variant[candidate_variant][
            "operator_kernel_sum_rank_mean_ms"])
        operator_deltas = []
        for kind in operator_types:
            control_ms = per_variant[control_variant][
                "operator_kernel_sum_rank_mean_ms"].get(kind, 0.0)
            candidate_ms = per_variant[candidate_variant][
                "operator_kernel_sum_rank_mean_ms"].get(kind, 0.0)
            operator_deltas.append({
                "type": kind,
                "control_ms": control_ms,
                "candidate_ms": candidate_ms,
                "candidate_minus_control_ms": candidate_ms - control_ms,
            })
        operator_deltas.sort(
            key=lambda value: abs(value["candidate_minus_control_ms"]),
            reverse=True)
        reduction = (1 - candidate / base) * 100
        record = {
            "phase": phase,
            "variants": per_variant,
            "diagnostic_envelope_reduction_percent": reduction,
            "candidate_minus_control_envelope_ms": candidate - base,
            "operator_kernel_sum_deltas_rank_mean_ms": operator_deltas,
        }
        if required_time_reduction_percent is not None:
            record["required_time_reduction_percent"] = required_time_reduction_percent
            record["meets_required_time_reduction"] = (
                reduction >= required_time_reduction_percent)
        records.append(record)
    return {
        "schema": "drrqr-paired-profile/v2", "status": "diagnostic_pair_validated",
        "control_variant": control_variant,
        "candidate_variant": candidate_variant,
        "shared_workload": {k: clients[control_variant][k] for k in shared},
        "file_sha256": {str(root / v / f): sha(root / v / f) for v in variants
                       for f in ("profile-client.json", "service-manifest.json", "profile-summary.v1.json")},
        "script_sha256": sha(Path(__file__)), "records": records,
        "limitations": [
            "One diagnostic capture per variant; rank spread is not independent-run uncertainty.",
            "Native warm decode at concurrency16; asynchronous scheduling means request decode positions are not identical.",
            "Compared dimensions have different allocated state slots and full-attention KV pool sizes under the same memory budget.",
            "Kernel sums are not additive critical-path attribution; no profiler-disabled throughput claim.",
            "Final joint performance and accuracy gates remain incomplete."
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-variant", default="baseline")
    parser.add_argument("--candidate-variant", default="dk64")
    parser.add_argument("--required-time-reduction-percent", type=float)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous reports")
    result = compare(
        args.root,
        control_variant=args.control_variant,
        candidate_variant=args.candidate_variant,
        required_time_reduction_percent=args.required_time_reduction_percent,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "records": [
        {"phase": r["phase"], "diagnostic_envelope_reduction_percent": r["diagnostic_envelope_reduction_percent"]}
        for r in result["records"]]}))


if __name__ == "__main__":
    main()
