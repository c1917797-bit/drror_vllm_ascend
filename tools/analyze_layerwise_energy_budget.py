#!/usr/bin/env python3
"""Bound a mixed-layer Dk idea using retained-energy records in frozen plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCHEMA = "ascend-drrqr-plan/v1"
OBJECTIVE = "energy-kernel"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_layer_energy(path: Path, expected_dim: int) -> tuple[dict[int, float], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported plan schema")
    if data.get("target_head_k_dim") != expected_dim:
        raise ValueError(f"{path}: expected target_head_k_dim={expected_dim}")
    if data.get("old_head_k_dim") != 128:
        raise ValueError(f"{path}: expected old_head_k_dim=128")
    selection = data.get("provenance", {}).get("selection", {})
    if selection.get("selection_objective") != OBJECTIVE:
        raise ValueError(f"{path}: expected selection_objective={OBJECTIVE}")
    layers = selection.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"{path}: missing retained-energy layer records")

    result: dict[int, float] = {}
    observations_per_layer: dict[int, int] = {}
    for raw_layer, layer_record in layers.items():
        values = []
        for rank_record in layer_record.get("ranks", {}).values():
            for head_record in rank_record.get("rrqr_by_local_head", []):
                value = head_record.get("retained_energy_fraction")
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{path}: invalid retained energy in layer {raw_layer}")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{path}: retained energy outside [0, 1]")
                values.append(float(value))
        if not values:
            raise ValueError(f"{path}: no head observations in layer {raw_layer}")
        layer = int(raw_layer)
        if layer in result:
            raise ValueError(f"{path}: duplicate layer {layer}")
        result[layer] = statistics.fmean(values)
        observations_per_layer[layer] = len(values)

    metadata = {
        "path": str(path),
        "sha256": sha256(path),
        "target_head_k_dim": expected_dim,
        "source_config_sha256": data.get("source_config_sha256"),
        "source_index_sha256": data.get("source_index_sha256"),
        "capture_manifest_sha256": selection.get("capture_manifest_sha256"),
        "layer_types": data.get("layer_types"),
        "observations_per_layer": observations_per_layer,
    }
    return result, metadata


def validate_pair(left: dict, right: dict, left_energy: dict, right_energy: dict) -> None:
    for field in (
        "source_config_sha256",
        "source_index_sha256",
        "capture_manifest_sha256",
        "layer_types",
    ):
        if left[field] != right[field]:
            raise ValueError(f"plan pair differs in {field}")
    if set(left_energy) != set(right_energy):
        raise ValueError("plan pair covers different layers")
    for layer in left_energy:
        if left["observations_per_layer"][layer] != right["observations_per_layer"][layer]:
            raise ValueError(f"plan pair differs in layer {layer} observation count")


def optimize_budget(
    energy32: dict[int, float],
    energy64: dict[int, float],
) -> tuple[float, list[dict[str, Any]]]:
    layers = sorted(energy64)
    target_budget = 64 * len(layers)
    # state: used dimension -> (summed retained-energy proxy, assignment)
    states: dict[int, tuple[float, tuple[tuple[int, int, float], ...]]] = {
        0: (0.0, ())
    }
    for layer in layers:
        next_states: dict[int, tuple[float, tuple[tuple[int, int, float], ...]]] = {}
        # Prefer the uniform choice when scores tie.
        for used, (score, assignment) in states.items():
            for dim, retained in (
                (64, energy64[layer]),
                (32, energy32[layer]),
                (128, 1.0),
            ):
                new_used = used + dim
                if new_used > target_budget:
                    continue
                candidate = (score + retained, assignment + ((layer, dim, retained),))
                current = next_states.get(new_used)
                if current is None or candidate[0] > current[0] + 1e-15:
                    next_states[new_used] = candidate
        states = next_states
    if target_budget not in states:
        raise RuntimeError("no exact mixed-dimension allocation for the target budget")
    score, assignment = states[target_budget]
    return score / len(layers), [
        {"layer": layer, "head_k_dim": dim, "retained_energy_fraction": retained}
        for layer, dim, retained in assignment
    ]


def analyze(
    dk32_path: Path,
    dk64_path: Path,
    *,
    prototype_gate_pp: float = 1.0,
) -> dict:
    if not math.isfinite(prototype_gate_pp) or prototype_gate_pp <= 0:
        raise ValueError("prototype gate must be positive and finite")
    energy32, metadata32 = load_layer_energy(dk32_path, 32)
    energy64, metadata64 = load_layer_energy(dk64_path, 64)
    validate_pair(metadata32, metadata64, energy32, energy64)
    optimized_mean, assignment = optimize_budget(energy32, energy64)
    uniform64_mean = statistics.fmean(energy64.values())
    proxy_gain_pp = 100.0 * (optimized_mean - uniform64_mean)
    counts = {
        str(dim): sum(item["head_k_dim"] == dim for item in assignment)
        for dim in (32, 64, 128)
    }
    changed = [item for item in assignment if item["head_k_dim"] != 64]
    decision = (
        "reject_mixed_dimension_runtime_prototype"
        if proxy_gain_pp < prototype_gate_pp
        else "proxy_promising_runtime_feasibility_still_required"
    )
    return {
        "schema": "drrqr-layerwise-energy-budget/v1",
        "method": (
            "Exact dynamic programming over per-layer Dk choices {32,64,128}; "
            "Dk128 retained energy is defined as 1.0."
        ),
        "claim_boundary": (
            "Retained normalized Q/K energy is a selector proxy, not downstream "
            "accuracy or endpoint performance. Mixed per-layer dimensions are not "
            "supported by the current global-Dk runtime contract."
        ),
        "inputs": {"dk32": metadata32, "dk64": metadata64},
        "layer_count": len(energy64),
        "dimension_budget": 64 * len(energy64),
        "uniform32_mean_retained_energy_fraction": statistics.fmean(energy32.values()),
        "uniform64_mean_retained_energy_fraction": uniform64_mean,
        "optimized_mean_retained_energy_fraction": optimized_mean,
        "optimized_gain_over_uniform64_pp": proxy_gain_pp,
        "prototype_gate_pp": prototype_gate_pp,
        "decision": decision,
        "assignment_counts": counts,
        "changed_layers": changed,
        "assignment": assignment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dk32-plan", type=Path, required=True)
    parser.add_argument("--dk64-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototype-gate-pp", type=float, default=1.0)
    args = parser.parse_args()
    report = analyze(
        args.dk32_plan,
        args.dk64_plan,
        prototype_gate_pp=args.prototype_gate_pp,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "optimized_gain_over_uniform64_pp": report[
            "optimized_gain_over_uniform64_pp"
        ],
        "assignment_counts": report["assignment_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
