"""Canonical token-ID calibration rows shared by capture and preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def token_sha256(input_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(input_ids, separators=(",", ":")).encode()).hexdigest()


def read_calibration(path: Path, expected_sha256: str) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("DRRQR: calibration JSONL hash mismatch")
    rows = []
    seen = set()
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        row = json.loads(line)
        ids = row.get("input_ids") if isinstance(row, dict) else None
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError(f"DRRQR: calibration row {number} requires nonempty integer input_ids")
        digest = token_sha256(ids)
        if digest in seen:
            raise ValueError("DRRQR: duplicate calibration token sequence")
        seen.add(digest)
        rows.append({"input_ids": ids, "input_ids_sha256": digest, "row_index": number - 1})
    if not rows:
        raise ValueError("DRRQR: calibration JSONL is empty")
    return rows
