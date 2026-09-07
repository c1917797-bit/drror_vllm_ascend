"""Bounded runtime evidence for plugin activation and weight coverage."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from .envs import DrrqrConfig

_WRITE_LOCK = threading.Lock()


def configure_package_logging() -> None:
    logging.getLogger("drror_vllm_ascend").setLevel(logging.INFO)


def emit_evidence(
    config: DrrqrConfig,
    event: str,
    *,
    component: str,
    **fields: Any,
) -> None:
    if not config.evidence_file:
        return
    record = {
        "schema": "drror-vllm-ascend-evidence/v1",
        "event": event,
        "component": component,
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
        **fields,
    }
    encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = Path(config.evidence_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    with _WRITE_LOCK:
        descriptor = os.open(path, flags, 0o600)
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)
