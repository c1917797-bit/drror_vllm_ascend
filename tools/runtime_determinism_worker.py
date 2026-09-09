"""Diagnostic NPU worker retaining upstream model and custom operator implementations."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch
from vllm_ascend.worker.worker import NPUWorker

from diagnostic_determinism import apply_settings

WORKER_SHA = "c392da56095b255d47fe5ad7ffdfbedd269fe69851690e89a8d2ed44c073f6c3"


def emit(record):
    target = Path(os.environ["DRRQR_DIAGNOSTIC_DETERMINISM_EVIDENCE_FILE"])
    if not target.is_absolute() or Path("/drrqr-results") not in target.resolve().parents:
        raise RuntimeError("diagnostic evidence must stay within the experiment artifacts")
    payload = json.dumps({"schema": "drrqr-determinism-worker/v1", "pid": os.getpid(),
                          "epoch_s": time.time(), **record}) + "\n"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        raw = payload.encode()
        if os.write(fd, raw) != len(raw):
            raise RuntimeError("incomplete diagnostic evidence write")
    finally:
        os.close(fd)


class DeterministicNPUWorker(NPUWorker):
    def _init_worker_distributed_environment(self):
        original_file = Path(inspect.getsourcefile(NPUWorker))
        if hashlib.sha256(original_file.read_bytes()).hexdigest() != WORKER_SHA:
            raise RuntimeError("audited upstream worker source changed")
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3":
            raise RuntimeError("unexpected diagnostic device namespace")
        settings = apply_settings(torch, os.environ)
        info = {"rank": self.rank, "settings": settings,
                "upstream_worker_sha256": WORKER_SHA,
                "diagnostic_source": __file__,
                "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        emit({"event": "determinism_before_distributed_init", **info})
        result = super()._init_worker_distributed_environment()
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("deterministic option was reset during worker initialization")
        emit({"event": "determinism_after_distributed_init", **info})
        return result

