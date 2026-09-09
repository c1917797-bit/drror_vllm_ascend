#!/usr/bin/env python3
"""Launch a provenance-locked original/packed Qwen service in the audited container."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_profile_service import PLAN_SHA, digest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "dk64", "dk32"), required=True)
    parser.add_argument("--layout", choices=("original", "packed"), required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path,
                        default=Path("/drrqr-plans/prune50-energy-kernel-v1.json"))
    parser.add_argument("--plan-sha", default=PLAN_SHA)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--plugin-version",
                        choices=("0.1.7", "0.1.8", "0.1.9", "0.1.10", "0.1.11"),
                        default="0.1.7")
    parser.add_argument("--prefill-mc2", action="store_true")
    parser.add_argument("--prefill-mc2-mixed", action="store_true")
    parser.add_argument("--diagnostic-mc2-coverage", action="store_true")
    parser.add_argument("--diagnostic-determinism", action="store_true",
                        help="Preserve GDN operators; diagnose with deterministic torch/HCCL/LCCL settings")
    args = parser.parse_args(argv)
    if args.prefill_mc2 and args.plugin_version not in ("0.1.8", "0.1.9", "0.1.10", "0.1.11"):
        parser.error("prefill MC2 requires an explicitly selected compatible local build")
    if args.prefill_mc2_mixed and (
            not args.prefill_mc2 or args.plugin_version not in ("0.1.9", "0.1.10", "0.1.11")):
        parser.error("mixed-batch MC2 requires MC2 ON and an explicit compatible local build")
    if args.diagnostic_mc2_coverage and (not args.prefill_mc2 or args.profile or args.diagnostic_determinism):
        parser.error("MC2 coverage requires MC2 ON and excludes profiler/determinism diagnostics")
    if args.diagnostic_mc2_coverage and args.plugin_version != "0.1.8":
        parser.error("historical coverage worker is pinned to the0.1.8 adapter")
    return parser, args


def main():
    parser, args = parse_args()
    output = args.output_dir.resolve()
    if Path("/drrqr-results") not in output.parents or output.exists():
        parser.error("choose a fresh directory below /drrqr-results")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3":
        parser.error("expected audited four-device container namespace")
    build = json.loads(args.build_manifest.read_text())
    if digest(Path(build["wheel"])) != build["wheel_sha256"]:
        raise RuntimeError("wheel digest differs from build manifest")
    installed = Path(importlib.util.find_spec("drror_vllm_ascend").origin).parent
    if "site-packages" not in installed.parts:
        raise RuntimeError("expected installed wheel, not implicit repository import")
    expected = {name.split("/", 1)[1]: checksum for name, checksum in build["source_sha256"].items()
                if name.startswith("drror_vllm_ascend/")}
    observed = {name: digest(installed / name) for name in expected}
    if observed != expected or importlib.metadata.version("drror-vllm-ascend-plugin") != args.plugin_version:
        raise RuntimeError("installed package differs from pinned local build")
    if {str(p.relative_to(installed)) for p in installed.rglob("*.py")} != set(expected):
        raise RuntimeError("installed package file coverage differs from the build")
    source = Path(build["source_root"]) / "drror_vllm_ascend"
    if {name: digest(source / name) for name in expected} != expected:
        raise RuntimeError("repository package changed after build; rebuild explicitly")
    if {str(p.relative_to(source)) for p in source.rglob("*.py")} != set(expected):
        raise RuntimeError("repository package file coverage differs from the build")
    reference_path = Path("/drrqr-results/profile-first-v1/baseline/service-manifest.json")
    reference = json.loads(reference_path.read_text())
    runtime = {name: digest(Path(name)) for name in reference["source_sha256"]}
    if runtime != reference["source_sha256"]:
        raise RuntimeError("upstream runtime changed from paired profile reference")
    plan = args.plan_path.resolve()
    if digest(plan) != args.plan_sha:
        raise RuntimeError("explicit DRRQR plan digest differs")
    plan_data = json.loads(plan.read_text())
    target_head_k_dim = plan_data.get("target_head_k_dim")
    if (type(target_head_k_dim) is not int or target_head_k_dim <= 0
            or target_head_k_dim >= 128 or target_head_k_dim % 16):
        raise RuntimeError("plan target head dimension is not an aligned reduction")
    if args.variant != "baseline" and args.variant != f"dk{target_head_k_dim}":
        raise RuntimeError("variant does not match the explicit plan target dimension")
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/cache/austinov/Qwen3.8-27B", "--host", "0.0.0.0", "--port", "6666",
        "--served-model-name", "qwen3.8", "--trust-remote-code",
        "--dtype", "bfloat16", "--tensor-parallel-size", "4",
        "--max-model-len", "40960", "--max-num-batched-tokens", "40960",
        "--max-num-seqs", "32", "--gpu-memory-utilization", "0.9",
        "--no-enable-prefix-caching", "--async-scheduling",
        "--compilation-config", json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"}),
    ]
    if args.variant != "baseline":
        command += ["--hf-overrides", json.dumps(
            {"text_config": {"linear_key_head_dim": target_head_k_dim}})]
    profiler = None
    if args.profile:
        profiler = {"profiler": "torch", "torch_profiler_dir": str(output / "traces"),
                    "torch_profiler_with_stack": False, "torch_profiler_with_memory": False,
                    "torch_profiler_record_shapes": False, "ignore_frontend": True,
                    "delay_iterations": 0, "max_iterations": 30}
        command += ["--profiler-config", json.dumps(profiler)]
    env = os.environ.copy()
    runtime_env = {key: env.get(key) for key in (
        "VLLM_ASCEND_ENABLE_NZ", "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1", "VLLM_BATCH_INVARIANT",
        "HCCL_OP_EXPANSION_MODE", "HCCL_DETERMINISTIC", "LCCL_DETERMINISTIC",
    )}
    if args.plugin_version in ("0.1.8", "0.1.9", "0.1.10", "0.1.11"):
        if runtime_env["VLLM_ASCEND_ENABLE_NZ"] not in (None, "1"):
            raise RuntimeError("MC2 experiment must retain original BF16 ND weight policy")
        for name in ("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "VLLM_ASCEND_ENABLE_FLASHCOMM1", "VLLM_BATCH_INVARIANT"):
            if runtime_env[name] not in (None, "0"):
                raise RuntimeError("unexpected competing runtime optimization: " + name)
    flags = {
        "VLLM_ASCEND_DRRQR_ENABLE": "1" if args.variant != "baseline" else "0",
        "VLLM_ASCEND_DRRQR_CAPTURE_ENABLE": "0",
        "VLLM_ASCEND_DRRQR_CONV_LAYOUT": "1" if args.layout == "packed" else "0",
        "VLLM_ASCEND_DRRQR_PREFILL_MC2": "1" if args.prefill_mc2 else "0",
        "VLLM_ASCEND_DRRQR_PREFILL_MC2_MIXED": "1" if args.prefill_mc2_mixed else "0",
        "VLLM_ASCEND_DRRQR_PLAN_PATH": str(plan),
        "VLLM_ASCEND_DRRQR_PLAN_SHA256": args.plan_sha,
        "VLLM_ASCEND_DRRQR_EVIDENCE_FILE": str(output / "evidence.jsonl"),
    }
    env.update(flags)
    diagnostic = None
    if args.diagnostic_determinism:
        tool_dir = Path(__file__).resolve().parent
        worker_file = tool_dir / "runtime_determinism_worker.py"
        helper_file = tool_dir / "diagnostic_determinism.py"
        command += ["--worker-cls", "runtime_determinism_worker.DeterministicNPUWorker"]
        diagnostic_flags = {
            "DRRQR_DIAGNOSTIC_DETERMINISM": "1",
            "DRRQR_DIAGNOSTIC_DETERMINISM_EVIDENCE_FILE": str(output / "determinism.jsonl"),
            "VLLM_BATCH_INVARIANT": "0",
            "HCCL_DETERMINISTIC": "strict",
            "LCCL_DETERMINISTIC": "1",
        }
        env.update(diagnostic_flags)
        env["PYTHONPATH"] = str(tool_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        diagnostic = {"flags": diagnostic_flags, "worker_sha256": digest(worker_file),
                      "helper_sha256": digest(helper_file),
                      "claim": "numerical diagnostic bundle, not a performance configuration",
                      "limits": "warn_only does not prove every custom operator is deterministic"}
    coverage = None
    if args.diagnostic_mc2_coverage:
        tool_dir = Path(__file__).resolve().parent
        command += ["--worker-cls", "mc2_coverage_worker.CoverageNPUWorker"]
        env["DRRQR_MC2_COVERAGE_DIR"] = str(output)
        env["PYTHONPATH"] = str(tool_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        coverage = {"worker_sha256": digest(tool_dir / "mc2_coverage_worker.py"),
                    "helper_sha256": digest(tool_dir / "mc2_coverage.py"),
                    "max_steps_per_rank": 8192,
                    "claim": "route counters only; diagnostic timings are not performance evidence"}
    output.mkdir(parents=True)
    manifest = {
        "schema": "drrqr-conv-layout-service/v1", "variant": args.variant, "layout": args.layout,
        "plugin_version": args.plugin_version, "prefill_mc2": args.prefill_mc2,
        "prefill_mc2_mixed": args.prefill_mc2_mixed,
        "command": command, "profiler_config": profiler, "plugin_flags": flags,
        "installed_plugin": str(installed), "module_sha256": observed,
        "build_manifest_sha256": digest(args.build_manifest), "wheel_sha256": build["wheel_sha256"],
        "source_sha256": runtime, "reference_manifest_sha256": digest(reference_path),
        "plan_sha256": args.plan_sha if args.variant != "baseline" else None,
        "target_head_k_dim": target_head_k_dim if args.variant != "baseline" else 128,
        "wrapper_sha256": digest(Path(__file__)), "created_epoch_s": time.time(),
        "physical_npu_contract": [4, 5, 6, 7], "container_npu_namespace": [0, 1, 2, 3],
        "diagnostic_determinism": diagnostic,
        "diagnostic_mc2_coverage": coverage,
        "inherited_runtime_environment": runtime_env,
        "claim": "runtime/provenance configuration; performance or quality require separate reports",
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
