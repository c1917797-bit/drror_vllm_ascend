#!/usr/bin/env python3
"""Independently audit the completed Qwen3.8 DRRQR experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/cache/cch/state-reduction-qwen38-drrqr-plugin-tp4-20260907")
OUTPUT = ROOT / "final-evidence-audit.json"
ARMS = {
    "baseline": {
        "performance": ROOT / "baseline",
        "quality": ROOT / "baseline/quality",
        "target_dk": 128,
        "plan": None,
    },
    "prune20_aligned": {
        "performance": ROOT / "prune20_aligned/performance",
        "quality": ROOT / "prune20_aligned/longbenchv2",
        "target_dk": 104,
        "plan": ROOT / "plans/prune20_aligned.json",
        "plan_sha256": "5bc3eabe4f98df880b28a287e28fe3c0fc9a8036b120c85f211d3cf41f7f9830",
    },
    "prune30_aligned": {
        "performance": ROOT / "prune30_aligned/performance",
        "quality": ROOT / "prune30_aligned/longbenchv2",
        "target_dk": 88,
        "plan": ROOT / "plans/prune30_aligned.json",
        "plan_sha256": "c75468decf7752388f3e6beb57b207bb6d426aa69b8aaf588431ceb22f7d6daf",
    },
    "prune50": {
        "performance": ROOT / "prune50/performance",
        "quality": ROOT / "prune50/longbenchv2",
        "target_dk": 64,
        "plan": ROOT / "plans/prune50.json",
        "plan_sha256": "3824e60fb9137fa1a1b134bff4594495014625583fdd62fdf5b9bc92f39b8bea",
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def audit_performance(name: str, spec: dict, errors: list[str]) -> dict:
    runs = []
    for run_id in range(1, 4):
        path = spec["performance"] / f"performance.run{run_id}.json"
        if not path.is_file():
            errors.append(f"{name}: missing {path}")
            continue
        item = load(path)
        rows = item.get("results", [])
        summary = item.get("summary", {})
        indexes = sorted(row.get("index") for row in rows)
        checks = {
            "requests_40": len(rows) == 40,
            "indexes_0_39": indexes == list(range(40)),
            "all_ok": all(row.get("ok") is True for row in rows),
            "all_1024_tokens": all(
                row.get("requested_output_tokens") == 1024
                and row.get("output_tokens") == 1024
                for row in rows
            ),
            "summary_counts": summary.get("requests") == 40
            and summary.get("successful") == 40
            and summary.get("failed") == 0
            and summary.get("fixed_output_requests") == 40
            and summary.get("fixed_output_verified") is True,
            "finite_metrics": all(
                finite(summary.get(key))
                for key in (
                    "output_tokens_per_second",
                    "ttft_mean_ms",
                    "tpot_mean_ms",
                    "benchmark_duration_s",
                )
            ),
        }
        for key, ok in checks.items():
            if not ok:
                errors.append(f"{name}: performance run{run_id} failed {key}")
        runs.append(
            {
                "run": run_id,
                "sha256": digest(path),
                "checks": checks,
                "output_tokens_per_second": summary.get("output_tokens_per_second"),
                "ttft_mean_ms": summary.get("ttft_mean_ms"),
                "tpot_mean_ms": summary.get("tpot_mean_ms"),
                "benchmark_duration_s": summary.get("benchmark_duration_s"),
            }
        )
    means = {}
    for key in (
        "output_tokens_per_second",
        "ttft_mean_ms",
        "tpot_mean_ms",
        "benchmark_duration_s",
    ):
        values = [run[key] for run in runs if finite(run.get(key))]
        means[key] = statistics.mean(values) if len(values) == 3 else None
    return {"runs": runs, "means": means}


def audit_quality(name: str, spec: dict, errors: list[str]) -> dict:
    runs = []
    correct_sets = []
    for run_id in range(1, 4):
        details = spec["quality"] / f"longbenchv2.run{run_id}.jsonl"
        summary_path = spec["quality"] / f"longbenchv2.run{run_id}.summary.json"
        if not details.is_file() or not summary_path.is_file():
            errors.append(f"{name}: missing quality run{run_id}")
            continue
        rows = [json.loads(line) for line in details.read_text(encoding="utf-8").splitlines()]
        summary = load(summary_path)
        indexes = sorted(row.get("index") for row in rows)
        correct = sum(row.get("correct") is True for row in rows)
        checks = {
            "requests_127": len(rows) == 127,
            "indexes_0_126": indexes == list(range(127)),
            "all_ok": all(row.get("ok") is True for row in rows),
            "summary_counts": summary.get("requests") == 127
            and summary.get("successful") == 127
            and summary.get("failed") == 0
            and summary.get("correct") == correct,
            "details_hash_bound": summary.get("details_sha256") == digest(details),
            "accuracy_recomputed": math.isclose(
                summary.get("accuracy_pct", float("nan")),
                100.0 * correct / 127,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
        }
        for key, ok in checks.items():
            if not ok:
                errors.append(f"{name}: quality run{run_id} failed {key}")
        correct_sets.append({row["sample_id"] for row in rows if row.get("correct") is True})
        runs.append(
            {
                "run": run_id,
                "details_sha256": digest(details),
                "summary_sha256": digest(summary_path),
                "checks": checks,
                "correct": correct,
                "accuracy_pct": 100.0 * correct / 127,
            }
        )
    accuracies = [run["accuracy_pct"] for run in runs]
    jaccard = []
    if len(correct_sets) == 3:
        for left, right in ((0, 1), (0, 2), (1, 2)):
            union = correct_sets[left] | correct_sets[right]
            jaccard.append(
                len(correct_sets[left] & correct_sets[right]) / len(union)
                if union
                else 1.0
            )
    return {
        "runs": runs,
        "mean_accuracy_pct": statistics.mean(accuracies) if len(accuracies) == 3 else None,
        "sample_sd_pp": statistics.stdev(accuracies) if len(accuracies) == 3 else None,
        "correct_set_jaccard": jaccard,
    }


def audit_runtime(name: str, spec: dict, errors: list[str]) -> dict:
    if name == "baseline":
        release_paths = [
            spec["performance"] / "resource-release.audit.json",
            spec["quality"] / "resource-release.audit.json",
        ]
        activation = None
    else:
        release_paths = [
            spec["performance"] / "resource-release.audit.json",
            spec["quality"] / "resource-release.audit.json",
        ]
        activation_path = spec["quality"] / "activation-audit.json"
        activation = load(activation_path)
        activation_ok = (
            activation.get("ok") is True
            and activation.get("errors") == []
            and activation.get("expected_workers") == 4
            and len(activation.get("observed_worker_pids", [])) == 4
            and activation.get("target_head_k_dim") == spec["target_dk"]
        )
        if not activation_ok:
            errors.append(f"{name}: activation audit failed")
        activation = {
            "path": str(activation_path),
            "sha256": digest(activation_path),
            "ok": activation_ok,
        }
    releases = []
    for path in release_paths:
        item = load(path)
        if "npu4_7_empty" in item:
            npu_empty = item["npu4_7_empty"] is True
        elif "npu4_7" in item:
            npu_empty = all(
                value.get("no_process") is True
                for value in item["npu4_7"].values()
            )
        else:
            npu_empty = all(
                value == "No process in device."
                for value in item.get("host_npu_4_7", {}).values()
            ) and len(item.get("host_npu_4_7", {})) == 4
        port_free = (
            item.get("port_free") is True
            or item.get("port8227_listener_absent") is True
            or item.get("port_8227_free") is True
        )
        container_record = item.get("container")
        oom_killed = item.get("oom_killed")
        if oom_killed is None and isinstance(container_record, dict):
            oom_killed = container_record.get("oom_killed")
        ok = (
            item.get("ok") is True
            and item.get("errors", []) == []
            and npu_empty
            and port_free
            and oom_killed is False
        )
        if not ok:
            errors.append(f"{name}: release audit failed: {path}")
        releases.append({"path": str(path), "sha256": digest(path), "ok": ok})
    plan = None
    if spec["plan"] is not None:
        item = load(spec["plan"])
        ok = (
            digest(spec["plan"]) == spec["plan_sha256"]
            and item.get("target_head_k_dim") == spec["target_dk"]
            and len(item.get("keep_indices", {})) == 48
        )
        if not ok:
            errors.append(f"{name}: plan audit failed")
        plan = {"path": str(spec["plan"]), "sha256": digest(spec["plan"]), "ok": ok}
    return {"activation": activation, "resource_releases": releases, "plan": plan}


def main() -> int:
    errors: list[str] = []
    result = {
        "schema": "paper-to-ascend-final-evidence-audit/v1",
        "experiment_id": "state-reduction-qwen38-drrqr-plugin-tp4-20260907",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": {},
        "errors": errors,
    }
    for name, spec in ARMS.items():
        result["arms"][name] = {
            "performance": audit_performance(name, spec, errors),
            "quality": audit_quality(name, spec, errors),
            "runtime": audit_runtime(name, spec, errors),
        }
    baseline = result["arms"]["baseline"]
    base_perf = baseline["performance"]["means"]["output_tokens_per_second"]
    base_quality = baseline["quality"]["mean_accuracy_pct"]
    for name in ("prune20_aligned", "prune30_aligned", "prune50"):
        arm = result["arms"][name]
        perf = arm["performance"]["means"]["output_tokens_per_second"]
        quality = arm["quality"]["mean_accuracy_pct"]
        arm["comparison"] = {
            "throughput_delta_pct": 100.0 * (perf / base_perf - 1.0),
            "quality_delta_pp": quality - base_quality,
        }
    result["decision"] = {
        "winner": None,
        "adoption": "reject",
        "gsm8k": "not-run-by-longbench-gate",
        "native_profile": "not-run-by-performance-and-quality-gate",
    }
    result["ok"] = not errors
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": result["ok"], "errors": errors, "output": str(OUTPUT)}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
