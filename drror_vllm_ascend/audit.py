"""Fail-closed audit for DRRQR treatment activation evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .patches.bootstrap import EXPECTED_GDN_SHA256, EXPECTED_QWEN_SHA256
from .patches.cache_alignment import EXPECTED_ASCEND_MAMBA_CONFIG_SHA256

HASH = re.compile(r"^[0-9a-f]{64}$")
WORKER_EVENTS = {
    "worker_dispatch_verified",
    "model_configured",
    "weight_load_complete",
    "reduced_gdn_hot_path",
    "reduced_gdn_decode_branch",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"DRRQR audit: invalid JSON on line {line_number}") from error
        if not isinstance(value, dict):
            raise TypeError(f"DRRQR audit: line {line_number} is not an object")
        records.append(value)
    if not records:
        raise ValueError("DRRQR audit: evidence file is empty")
    return records


def audit_activation(
    evidence: Path,
    *,
    plan_sha256: str,
    target_head_k_dim: int,
    expected_workers: int = 4,
) -> dict[str, Any]:
    if not HASH.fullmatch(plan_sha256):
        raise ValueError("DRRQR audit: explicit lowercase plan SHA256 is required")
    if type(target_head_k_dim) is not int or not 0 < target_head_k_dim < 128:
        raise ValueError("DRRQR audit: invalid target Dk")
    if type(expected_workers) is not int or expected_workers <= 0:
        raise ValueError("DRRQR audit: expected worker count must be positive")

    records = _read_jsonl(evidence)
    errors: list[str] = []
    schemas = {record.get("schema") for record in records}
    if schemas != {"drror-vllm-ascend-evidence/v1"}:
        errors.append(f"unexpected evidence schemas: {sorted(map(str, schemas))}")
    events = {record.get("event") for record in records}
    required_global = {"plugin_build_active", "runtime_patches_installed"}
    missing_global = sorted(required_global - events)
    if missing_global:
        errors.append(f"missing global events: {missing_global}")

    worker_records = [record for record in records if record.get("event") in WORKER_EVENTS]
    worker_pids = sorted({record.get("pid") for record in worker_records if type(record.get("pid")) is int})
    if len(worker_pids) != expected_workers:
        errors.append(f"expected {expected_workers} worker pids, observed {worker_pids}")
    per_worker = {}
    for pid in worker_pids:
        selected = [record for record in worker_records if record.get("pid") == pid]
        worker_events = {record.get("event") for record in selected}
        missing = sorted(WORKER_EVENTS - worker_events)
        if missing:
            errors.append(f"worker {pid} missing events: {missing}")
        per_worker[str(pid)] = sorted(str(event) for event in worker_events)
        for record in selected:
            event = record.get("event")
            if record.get("plan_sha256") != plan_sha256:
                errors.append(f"worker {pid} {event} has a different plan hash")
            if event == "worker_dispatch_verified":
                sources = record.get("runtime_sources") or {}
                if (
                    sources.get("ascend_gdn", {}).get("sha256") != EXPECTED_GDN_SHA256
                    or sources.get("qwen_model", {}).get("sha256") != EXPECTED_QWEN_SHA256
                ):
                    errors.append(f"worker {pid} does not bind the audited v0.23 source hashes")
            elif event == "model_configured":
                if (
                    record.get("old_head_k_dim") != 128
                    or record.get("target_head_k_dim") != target_head_k_dim
                    or record.get("linear_layer_count") != 48
                ):
                    errors.append(f"worker {pid} has invalid model dimensions")
            elif event == "weight_load_complete":
                if record.get("target_tensor_count") != 96:
                    errors.append(f"worker {pid} did not load all 96 target tensors")
            elif event == "reduced_gdn_hot_path" and (
                record.get("key_head_dim") != target_head_k_dim
                or record.get("value_head_dim") != 128
                or record.get("backend") != "chunk_gated_delta_rule"
                or type(record.get("token_count")) is not int
                or record.get("token_count") <= 0
                or record.get("prebuilt_metadata_forwarded") is not True
                or record.get("completed_python_call") is not True
                or record.get("query_shape", [])[-1:] != [target_head_k_dim]
                or record.get("key_shape", [])[-1:] != [target_head_k_dim]
                or record.get("initial_state_shape", [])[-2:] != [target_head_k_dim, 128]
            ):
                errors.append(f"worker {pid} has invalid reduced hot-path evidence")
            elif event == "reduced_gdn_decode_branch" and (
                record.get("key_head_dim") != target_head_k_dim
                or record.get("value_head_dim") != 128
                or record.get("backend") != "npu_recurrent_gated_delta_rule"
                or record.get("completed_python_call") is not True
                or type(record.get("token_count")) is not int
                or record.get("token_count") <= 0
                or record.get("query_shape", [])[-1:] != [target_head_k_dim]
                or record.get("key_shape", [])[-1:] != [target_head_k_dim]
                or record.get("state_shape", [])[-2:] != [128, target_head_k_dim]
            ):
                errors.append(f"worker {pid} has invalid reduced decode-branch evidence")

    treatment_patch_events = [
        record
        for record in records
        if record.get("event") == "runtime_patches_installed"
        and record.get("mode") == "treatment"
        and record.get("plan_sha256") == plan_sha256
    ]
    if not treatment_patch_events:
        errors.append("missing treatment runtime patch event for expected plan")
    cache_events = [
        record
        for record in records
        if record.get("event") == "hybrid_cache_alignment_verified"
        and record.get("plan_sha256") == plan_sha256
        and record.get("target_head_k_dim") == target_head_k_dim
        and record.get("source_sha256") == EXPECTED_ASCEND_MAMBA_CONFIG_SHA256
    ]
    expected_alignment = "native-exact" if target_head_k_dim == 64 else "ceil-pad"
    if not any(record.get("alignment_mode") == expected_alignment for record in cache_events):
        errors.append(
            "missing audited hybrid-cache alignment event "
            f"for mode {expected_alignment}"
        )
    return {
        "schema": "drror-activation-audit/v1",
        "ok": not errors,
        "evidence": str(evidence.resolve()),
        "plan_sha256": plan_sha256,
        "target_head_k_dim": target_head_k_dim,
        "expected_workers": expected_workers,
        "observed_worker_pids": worker_pids,
        "per_worker_events": per_worker,
        "record_count": len(records),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--target-head-k-dim", type=int, required=True)
    parser.add_argument("--expected-workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_activation(
        args.evidence,
        plan_sha256=args.plan_sha256,
        target_head_k_dim=args.target_head_k_dim,
        expected_workers=args.expected_workers,
    )
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
