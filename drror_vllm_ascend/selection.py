"""Pinned Strong RRQR selection for immutable plan preparation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

from .patches.capture import CAPTURE_SCHEMA

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
OFFICIAL_COMMIT = "919d8667d951c385e08510bc1267c2e7049a4f56"
OFFICIAL_RRQR_SHA256 = "fa4bacf516011ba1c88f2e0e92957bbe4513cc1bb6634912fdb0d06ba7821a62"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strong_rrqr_indices(
    activation: torch.Tensor,
    n_keep: int,
    *,
    f_param: float = 1.5,
    max_swaps: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """CPU adaptation of the pinned official Strong RRQR routine."""

    if activation.ndim != 2 or not activation.is_floating_point():
        raise ValueError("DRRQR: activation must be a floating two-dimensional tensor")
    rows, columns = activation.shape
    if rows < 1 or type(n_keep) is not int or not 0 < n_keep < columns:
        raise ValueError("DRRQR: n_keep must be strictly between zero and activation width")
    matrix = activation.detach().float().cpu().contiguous().numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("DRRQR: activation contains non-finite values")
    if matrix.shape[0] > 5000:
        generator = np.random.default_rng(seed)
        chosen = generator.choice(matrix.shape[0], 5000, replace=False)
        matrix = matrix[chosen]

    _, r_matrix, permutation = scipy.linalg.qr(
        matrix,
        pivoting=True,
        mode="economic",
    )
    permutation = permutation.copy()
    for _ in range(max_swaps):
        a_block = r_matrix[:n_keep, :n_keep]
        b_block = r_matrix[:n_keep, n_keep:]
        if np.abs(np.diag(a_block)).min() < 1e-9:
            break
        w_matrix = scipy.linalg.solve_triangular(a_block, b_block, lower=False)
        a_inverse = scipy.linalg.solve_triangular(
            a_block,
            np.eye(n_keep),
            lower=False,
        )
        omega = 1.0 / (np.linalg.norm(a_inverse, axis=1) + 1e-12)
        if r_matrix.shape[0] > n_keep and r_matrix.shape[1] > n_keep:
            gamma = np.linalg.norm(r_matrix[n_keep:, n_keep:], axis=0)
        else:
            gamma = np.zeros(r_matrix.shape[1] - n_keep)
        rho = np.sqrt(
            np.abs(w_matrix) ** 2
            + np.outer(1.0 / omega, gamma) ** 2
        )
        if np.max(rho) <= f_param:
            break
        left, right = np.unravel_index(np.argmax(rho), rho.shape)
        right += n_keep
        permutation[left], permutation[right] = (
            permutation[right],
            permutation[left],
        )
        _, r_matrix = scipy.linalg.qr(
            matrix[:, permutation],
            mode="economic",
        )
    return permutation[:n_keep]


def _layer_id(name: str) -> int:
    match = LAYER_RE.search(name)
    if match is None:
        raise ValueError(f"DRRQR: cannot parse layer from {name!r}")
    return int(match.group(1))


def load_keep_maps(
    capture_dir: Path,
    *,
    expected_layer_ids: list[int],
    tp_size: int,
    num_heads: int,
    old_dim: int,
    new_dim: int,
    captures_per_rank: int,
    source_config_sha256: str,
    calibration_sha256: str,
) -> tuple[dict[int, torch.Tensor], dict]:
    """Validate bound captures and run Strong RRQR independently per head."""

    grouped: dict[tuple[int, int], list[dict]] = {}
    capture_files = sorted(capture_dir.glob("*.pt"))
    for path in capture_files:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f"DRRQR: capture must be a mapping: {path}")
        required = {
            "schema": CAPTURE_SCHEMA,
            "target_model_id": "Qwen/Qwen3.8-27B",
            "source_config_sha256": source_config_sha256,
            "calibration_sha256": calibration_sha256,
            "stage": "post_conv_qk",
            "tp_size": tp_size,
            "old_head_k_dim": old_dim,
        }
        for name, expected in required.items():
            if payload.get(name) != expected:
                raise ValueError(
                    f"DRRQR: capture {path.name} has invalid {name}"
                )
        try:
            layer = _layer_id(payload["prefix"])
            rank = int(payload["tp_rank"])
            capture_index = int(payload["capture_index"])
            local_key_dim = int(payload["local_key_dim"])
            q, k = payload["q"], payload["k"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"DRRQR: invalid capture metadata: {path}") from error
        if rank not in range(tp_size) or capture_index < 0:
            raise ValueError(f"DRRQR: invalid rank/index in {path}")
        local_heads = num_heads // tp_size
        if local_key_dim != local_heads * old_dim:
            raise ValueError(f"DRRQR: invalid local key width in {path}")
        if (
            not isinstance(q, torch.Tensor)
            or not isinstance(k, torch.Tensor)
            or q.shape != k.shape
            or q.ndim != 2
            or q.shape[0] < 1
            or q.shape[1] != local_key_dim
            or not q.is_floating_point()
            or not k.is_floating_point()
            or not torch.isfinite(q).all()
            or not torch.isfinite(k).all()
        ):
            raise ValueError(f"DRRQR: invalid Q/K tensors in {path}")
        grouped.setdefault((layer, rank), []).append(payload)

    expected_keys = {
        (layer, rank)
        for layer in expected_layer_ids
        for rank in range(tp_size)
    }
    if set(grouped) != expected_keys:
        raise RuntimeError(
            "DRRQR: capture coverage mismatch "
            f"missing={sorted(expected_keys - set(grouped))} "
            f"extra={sorted(set(grouped) - expected_keys)}"
        )

    local_heads = num_heads // tp_size
    keep_maps: dict[int, torch.Tensor] = {}
    evidence = {
        "schema": "drrqr-selection/v1",
        "method": "Strong RRQR on concatenated post-conv Q/K activations",
        "official_commit": OFFICIAL_COMMIT,
        "official_rrqr_sha256": OFFICIAL_RRQR_SHA256,
        "f_param": 1.5,
        "max_swaps": 20,
        "subsample_threshold": 5000,
        "seed": 42,
        "capture_file_count": len(capture_files),
        "capture_manifest_sha256": hashlib.sha256(
            "".join(
                f"{path.name}:{file_sha256(path)}\n"
                for path in capture_files
            ).encode()
        ).hexdigest(),
        "layers": {},
    }
    for layer in expected_layer_ids:
        global_keep: list[int] = []
        layer_record = {"ranks": {}}
        for rank in range(tp_size):
            payloads = sorted(
                grouped[(layer, rank)],
                key=lambda item: int(item["capture_index"]),
            )
            indexes = [int(item["capture_index"]) for item in payloads]
            if indexes != list(range(captures_per_rank)):
                raise RuntimeError(
                    f"DRRQR: layer {layer} rank {rank} capture indexes {indexes}"
                )
            q = torch.cat([item["q"] for item in payloads], dim=0)
            k = torch.cat([item["k"] for item in payloads], dim=0)
            combined = torch.cat((q, k), dim=0).view(
                -1,
                local_heads,
                old_dim,
            )
            rank_keep = []
            for local_head in range(local_heads):
                selected = np.sort(
                    strong_rrqr_indices(
                        combined[:, local_head, :],
                        new_dim,
                        seed=42,
                    )
                )
                global_head = rank * local_heads + local_head
                values = (selected + global_head * old_dim).tolist()
                global_keep.extend(values)
                rank_keep.append(values)
            layer_record["ranks"][str(rank)] = {
                "captures": len(payloads),
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
                "keep_indices_by_local_head": rank_keep,
            }
        if (
            len(global_keep) != num_heads * new_dim
            or len(set(global_keep)) != len(global_keep)
        ):
            raise RuntimeError(f"DRRQR: invalid keep map for layer {layer}")
        keep_maps[layer] = torch.tensor(global_keep, dtype=torch.long)
        layer_record["global_keep_indices"] = global_keep
        evidence["layers"][str(layer)] = layer_record
    return keep_maps, evidence
