#!/usr/bin/env python3
"""Run the frozen service with built-in NPU profiling and auditable artifacts.

Execute inside the existing, explicitly audited physical-NPU4-7 container.
This helper does not install packages or change any upstream source.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PLAN_SHA = "b1fcb727d3e58be66cc24148e30a5a1722d3bdc0914784f01cad02f11f741b8c"
MODULE_HASHES = {
    "selection.py": "73dda45bd1e90fe4c149f1ca8d8aa55cc72cb535bfb5a42c98a493b7341ddf24",
    "patches/gdn.py": "5252d09ae3a0802a77e435d5180e2602d92d4b3f30d2b78045dbc18b0846180c",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["dk64", "baseline"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-steps", type=int, default=30)
    args = parser.parse_args()
    if args.profile_steps < 1:
        parser.error("profile steps must be positive")
    output = args.output_dir.resolve()
    allowed = Path("/drrqr-results").resolve()
    if allowed not in output.parents or output.exists():
        parser.error("use a new directory below /drrqr-results")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3":
        parser.error("expected the audited four-device container namespace")
    installed = Path(importlib.util.find_spec("drror_vllm_ascend").origin).parent
    module_hashes = {name: digest(installed / name) for name in MODULE_HASHES}
    if module_hashes != MODULE_HASHES:
        raise RuntimeError("installed plugin differs from frozen energy preflight")
    plan = Path("/drrqr-plans/prune50-energy-kernel-v1.json")
    if digest(plan) != PLAN_SHA:
        raise RuntimeError("frozen plan hash differs")
    output.mkdir(parents=True)
    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(output / "traces"),
        "torch_profiler_with_stack": False,
        "torch_profiler_with_memory": False,
        "torch_profiler_record_shapes": False,
        "ignore_frontend": True,
        "delay_iterations": 0,
        "max_iterations": args.profile_steps,
    }
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/cache/austinov/Qwen3.8-27B",
        "--host", "0.0.0.0", "--port", "6666",
        "--served-model-name", "qwen3.8", "--trust-remote-code",
        "--dtype", "bfloat16", "--tensor-parallel-size", "4",
        "--max-model-len", "40960", "--max-num-batched-tokens", "40960",
        "--max-num-seqs", "32", "--gpu-memory-utilization", "0.9",
        "--no-enable-prefix-caching", "--async-scheduling",
        "--compilation-config", json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"}),
        "--profiler-config", json.dumps(profiler_config),
    ]
    if args.variant == "dk64":
        command += ["--hf-overrides", json.dumps({"text_config": {"linear_key_head_dim": 64}})]
    env = os.environ.copy()
    env["VLLM_ASCEND_DRRQR_ENABLE"] = "1" if args.variant == "dk64" else "0"
    env["VLLM_ASCEND_DRRQR_EVIDENCE_FILE"] = str(output / "evidence.jsonl")
    source_hashes = {}
    for package, paths in {
        "vllm": ("profiler/wrapper.py", "config/profiler.py"),
        "vllm_ascend": ("profiler/torch_npu_profiler.py", "worker/worker.py", "ops/gdn.py"),
    }.items():
        root = Path(importlib.util.find_spec(package).origin).parent
        for relative in paths:
            source_hashes[str(root / relative)] = digest(root / relative)
    manifest = {
        "schema": "drrqr-profile-service/v1", "variant": args.variant,
        "command": command, "profiler_config": profiler_config,
        "installed_plugin": str(installed), "module_sha256": module_hashes,
        "plan_sha256": PLAN_SHA if args.variant == "dk64" else None,
        "source_sha256": source_hashes,
        "wrapper_sha256": digest(Path(__file__)),
        "created_epoch_s": time.time(),
        "physical_npu_contract": [4, 5, 6, 7],
        "container_npu_namespace": [0, 1, 2, 3],
        "claim": "diagnostic profiling service; not a throughput benchmark",
    }
    (output / "service-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "service.log").open("x") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        (output / "service.pid").write_text(str(child.pid) + "\n")
        def stop(signum, frame):
            if child.poll() is None:
                child.send_signal(signum)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print(json.dumps({"service_pid": child.pid, "output_dir": str(output)}), flush=True)
        code = child.wait()
    (output / "service-exit.json").write_text(json.dumps({"exit_code": code, "epoch_s": time.time()}) + "\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
