"""Diagnostic-only worker observing MC2 batch coverage; no determinism settings."""
import hashlib
import inspect
import json
import os
from pathlib import Path

from vllm_ascend.worker.worker import NPUWorker

from mc2_coverage import install_observer

WORKER_SHA = "c392da56095b255d47fe5ad7ffdfbedd269fe69851690e89a8d2ed44c073f6c3"
ADAPTER_SHA = "23a7d07e32d9d33f8d9f095dd8db7931ada1d18978d12472f116caf85e558ab4"


class CoverageNPUWorker(NPUWorker):
    def load_model(self):
        from drror_vllm_ascend.patches import prefill_mc2
        from vllm.forward_context import get_forward_context
        from vllm_ascend.ops.linear import AscendRowParallelLinear
        import mc2_coverage

        if hashlib.sha256(Path(inspect.getsourcefile(NPUWorker)).read_bytes()).hexdigest() != WORKER_SHA:
            raise RuntimeError("upstream worker source changed")
        if hashlib.sha256(Path(prefill_mc2.__file__).read_bytes()).hexdigest() != ADAPTER_SHA:
            raise RuntimeError("installed MC2 adapter source changed")
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3" or self.rank not in range(4):
            raise RuntimeError("unexpected diagnostic TP4 namespace")
        directory = Path(os.environ["DRRQR_MC2_COVERAGE_DIR"]).resolve()
        if Path("/drrqr-results") not in directory.parents:
            raise RuntimeError("coverage directory outside experiment artifacts")
        result = super().load_model()
        bindings = AscendRowParallelLinear.forward._drrqr_prefill_mc2_bindings
        if len(bindings) != 128 or not self.model_runner._drrqr_prefill_mc2_ready:
            raise RuntimeError("MC2 binding incomplete")
        target = directory / ("coverage-rank" + str(self.rank) + ".jsonl")
        stream = target.open("x", buffering=1)
        def emit(record):
            stream.write(json.dumps({"schema": "drrqr-mc2-coverage-step/v1",
                                     "rank": self.rank, "pid": os.getpid(), **record}, sort_keys=True) + "\n")
        install_observer(self.model_runner, prefill_mc2, bindings, get_forward_context,
                         active=lambda: (directory / "coverage-active").exists(), emit=emit)
        emit({"event": "coverage_ready", "target_count": len(bindings),
              "adapter_sha256": ADAPTER_SHA, "upstream_worker_sha256": WORKER_SHA,
              "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256(Path(mc2_coverage.__file__).read_bytes()).hexdigest(),
              "claim": "diagnostic route counters; no throughput or numerical equivalence claim"})
        self._drrqr_coverage_stream = stream
        return result
