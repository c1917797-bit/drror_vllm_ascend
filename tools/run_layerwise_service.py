#!/usr/bin/env python3
"""Start one bounded model-control service inside the audited NPU4-7 container.

This launcher records runtime/plan provenance only. It does not collect or
establish throughput or benchmark accuracy. The parent must verify physical
device mappings and idle devices immediately before invoking it.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import zipfile

ROOT = Path("/cache/cch/drror_vllm_ascend")
RESULT_ROOT = Path("/drrqr-results")
MODEL = "/cache/austinov/Qwen3.8-27B"
REFERENCE = RESULT_ROOT / "profile-first-v1/baseline/service-manifest.json"
HASH = re.compile(r"^[0-9a-f]{64}$")
UNIFORM_SCHEMA = "ascend-drrqr-plan/v1"
LAYERWISE_SCHEMA = "ascend-drrqr-layerwise-manifest/v1"
WORKER_NAME = "layerwise_probe_worker.py"
WORKER_CLASS = "LayerwiseProbeWorker"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_bound_json(path, checksum):
    path = Path(path)
    if not path.is_absolute() or not isinstance(checksum, str) or not HASH.fullmatch(checksum):
        raise ValueError("absolute path and explicit lowercase SHA256 required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("bound file digest differs: " + str(path))
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("bound JSON must be an object")
    return document


def write_record(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-kind", choices=("uniform", "layerwise"), required=True)
    parser.add_argument("--plan-path", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--build-manifest-sha256", required=True)
    parser.add_argument("--plugin-version", required=True)
    parser.add_argument("--reference-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-service-seconds", type=int, default=1800)
    parser.add_argument("--term-grace-seconds", type=int, default=30)
    parser.add_argument("--diagnostic-worker", type=Path)
    parser.add_argument("--diagnostic-worker-sha256")
    parser.add_argument("--diagnostic-worker-class", default=WORKER_CLASS)
    args = parser.parse_args(argv)
    if not 1 <= args.max_service_seconds <= 7200:
        parser.error("service budget must be between 1 and 7200 seconds")
    if not 1 <= args.term_grace_seconds <= 120:
        parser.error("TERM grace must be between 1 and 120 seconds")
    for name in ("plan_sha256", "build_manifest_sha256", "reference_manifest_sha256"):
        if not HASH.fullmatch(getattr(args, name)):
            parser.error(name + " must be an explicit lowercase SHA256")
    if bool(args.diagnostic_worker) != bool(args.diagnostic_worker_sha256):
        parser.error("diagnostic worker path and SHA256 must be supplied together")
    if args.diagnostic_worker_class != WORKER_CLASS:
        parser.error("only the explicitly audited layerwise probe worker is supported")
    if args.diagnostic_worker_sha256 and not HASH.fullmatch(args.diagnostic_worker_sha256):
        parser.error("diagnostic worker SHA256 is invalid")
    return parser, args


def verify_build(args):
    build = read_bound_json(args.build_manifest, args.build_manifest_sha256)
    if build.get("schema") != "drrqr-local-wheel-build/v1":
        raise ValueError("unexpected build manifest schema")
    if Path(build["source_root"]).resolve() != ROOT.resolve():
        raise ValueError("build source is not the plugin repository")
    wheel = Path(build["wheel"])
    if not wheel.is_absolute() or digest(wheel) != build["wheel_sha256"]:
        raise ValueError("wheel digest differs from the bound build")
    expected = build["source_sha256"]
    if not isinstance(expected, dict) or not expected:
        raise ValueError("build has no source hashes")
    for name, checksum in expected.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not HASH.fullmatch(checksum):
            raise ValueError("invalid relative source path/hash")
        if digest(ROOT / path) != checksum:
            raise ValueError("repository changed after build: " + name)
        if digest(args.build_manifest.parent / "source" / path) != checksum:
            raise ValueError("build snapshot differs: " + name)
    package = {name: checksum for name, checksum in expected.items()
               if name.startswith("drror_vllm_ascend/")}
    if not package:
        raise ValueError("build does not cover the plugin package")
    source_files = {str(p.relative_to(ROOT)) for p in (ROOT / "drror_vllm_ascend").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if source_files != set(package):
        raise ValueError("repository package file coverage differs from build")
    spec = importlib.util.find_spec("drror_vllm_ascend")
    if spec is None or spec.origin is None:
        raise ValueError("installed plugin is missing")
    installed = Path(spec.origin).resolve().parent
    if "site-packages" not in installed.parts:
        raise ValueError("expected installed wheel, not repository import")
    if importlib.metadata.version("drror-vllm-ascend-plugin") != args.plugin_version:
        raise ValueError("installed plugin version differs")
    installed_files = {str(p.relative_to(installed.parent)) for p in installed.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if installed_files != set(package):
        raise ValueError("installed package coverage differs from build")
    observed = {name: digest(installed.parent / name) for name in package}
    if observed != package:
        raise ValueError("installed package bytes differ from build")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("wheel integrity/duplicate member failure")
        members = {name for name in names
                   if name.startswith("drror_vllm_ascend/") and not name.endswith("/")}
        if members != set(package):
            raise ValueError("wheel package coverage differs")
        for name, checksum in package.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError("wheel package bytes differ: " + name)
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1 or ("Version: " + args.plugin_version) not in archive.read(metadata[0]).decode().splitlines():
            raise ValueError("wheel version differs")
    return {"installed_plugin": str(installed), "module_sha256": observed,
            "build_manifest_sha256": args.build_manifest_sha256,
            "build_source_sha256": expected, "wheel": str(wheel),
            "wheel_sha256": build["wheel_sha256"], "plugin_version": args.plugin_version}


def verify_runtime(reference_sha256):
    reference = read_bound_json(REFERENCE, reference_sha256)
    expected = reference.get("source_sha256")
    if not isinstance(expected, dict) or len(expected) < 5:
        raise ValueError("runtime reference does not cover frozen source files")
    required_suffixes = ("profiler/wrapper.py", "config/profiler.py",
                         "profiler/torch_npu_profiler.py", "worker/worker.py", "ops/gdn.py")
    if any(not any(str(name).endswith("/" + suffix) for name in expected)
           for suffix in required_suffixes):
        raise ValueError("runtime reference source coverage incomplete")
    observed = {}
    for name, checksum in expected.items():
        path = Path(name)
        if not path.is_absolute() or not HASH.fullmatch(checksum):
            raise ValueError("invalid runtime reference path/hash")
        observed[name] = digest(path)
    if observed != expected:
        raise ValueError("upstream runtime differs from paired original reference")
    return {"source_sha256": observed, "reference_manifest": str(REFERENCE),
            "reference_manifest_sha256": reference_sha256}


def describe_plan(plan, kind):
    if kind == "layerwise":
        reference = plan.reference
        widths = dict(plan.layer_dims)
        shared_dk = plan.max_head_k_dim
    else:
        reference = plan
        widths = {layer: plan.target_head_k_dim for layer, _ in plan.keep_indices}
        shared_dk = plan.target_head_k_dim
    if not widths or shared_dk != max(widths.values()):
        raise ValueError("shared cache envelope must equal maximum layer Dk")
    if any(type(dk) is not int or dk < 16 or dk > 128 or dk % 16 for dk in widths.values()):
        raise ValueError("plan contains an unsupported native decode width")
    if (reference.old_head_k_dim, reference.num_key_heads, reference.num_value_heads,
            reference.head_v_dim) != (128, 16, 48, 128):
        raise ValueError("not the frozen TP4 Qwen3.8 GDN geometry")
    return {"plan_kind": kind, "shared_hf_head_k_dim": shared_dk,
            "layer_head_k_dims": {str(layer): dk for layer, dk in sorted(widths.items())},
            "sum_layer_head_k_dims": sum(widths.values()),
            "num_linear_layers": len(widths),
            "shared_dimension_meaning": "maximum layer Dk for conservative cache sizing"}


def verify_plan(args):
    document = read_bound_json(args.plan_path, args.plan_sha256)
    expected_schema = LAYERWISE_SCHEMA if args.plan_kind == "layerwise" else UNIFORM_SCHEMA
    if document.get("schema") != expected_schema:
        raise ValueError("explicit plan kind differs from document schema")
    # Import only after installed package/source/wheel identity has been proved.
    from drror_vllm_ascend.envs import DrrqrConfig
    from drror_vllm_ascend.runtime_plan import load_bound_plan
    plan = load_bound_plan(DrrqrConfig(plan_path=str(args.plan_path),
                                     plan_sha256=args.plan_sha256))
    reference = plan.reference if args.plan_kind == "layerwise" else plan
    reference.validate_source(MODEL)
    detail = describe_plan(plan, args.plan_kind)
    detail.update(plan_path=str(args.plan_path), plan_sha256=args.plan_sha256,
                  source_plans=document.get("source_plans"))
    return detail


def service_command(shared_dk, executable=None):
    command = [
        executable or sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--host", "0.0.0.0", "--port", "6666",
        "--served-model-name", "qwen3.8", "--trust-remote-code",
        "--dtype", "bfloat16", "--tensor-parallel-size", "4",
        "--max-model-len", "40960", "--max-num-batched-tokens", "40960",
        "--max-num-seqs", "32", "--gpu-memory-utilization", "0.9",
        "--no-enable-prefix-caching", "--async-scheduling",
        "--compilation-config", json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"}),
        "--hf-overrides", json.dumps({"text_config": {"linear_key_head_dim": shared_dk}}),
    ]
    return command


def service_environment(args, output, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    if env.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3":
        raise ValueError("expected the audited four-device container namespace")
    for name, allowed in {
        "VLLM_ASCEND_ENABLE_NZ": (None, "1"),
        "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": (None, "0"),
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": (None, "0"),
        "VLLM_BATCH_INVARIANT": (None, "0"),
        "HCCL_DETERMINISTIC": (None, "0"),
        "LCCL_DETERMINISTIC": (None, "0"),
        "DRRQR_DIAGNOSTIC_DETERMINISM": (None, "0"),
    }.items():
        if env.get(name) not in allowed:
            raise ValueError("unadmitted inherited runtime setting: " + name)
    # Root on PYTHONPATH would bypass the installed wheel in new workers.
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        if entry and Path(entry).resolve() == ROOT.resolve():
            raise ValueError("repository root on PYTHONPATH would shadow installed wheel")
    inherited_runtime = {key: env.get(key) for key in (
        "VLLM_ASCEND_ENABLE_NZ", "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1", "VLLM_BATCH_INVARIANT",
        "HCCL_OP_EXPANSION_MODE", "HCCL_DETERMINISTIC", "LCCL_DETERMINISTIC")}
    for name in list(env):
        if name.startswith(("VLLM_ASCEND_DRRQR_", "DRRQR_DIAGNOSTIC_", "DRRQR_LAYERWISE_", "DRRQR_MC2_COVERAGE_")):
            del env[name]
    flags = {
        "VLLM_ASCEND_DRRQR_ENABLE": "1",
        "VLLM_ASCEND_DRRQR_CAPTURE_ENABLE": "0",
        "VLLM_ASCEND_DRRQR_STRICT": "1",
        "VLLM_ASCEND_DRRQR_REQUIRE_RUNTIME_HOOKS": "1",
        "VLLM_ASCEND_DRRQR_CONV_LAYOUT": "1",
        "VLLM_ASCEND_DRRQR_PREFILL_MC2": "1",
        "VLLM_ASCEND_DRRQR_PREFILL_MC2_MIXED": "1",
        "VLLM_ASCEND_DRRQR_PLAN_PATH": str(args.plan_path),
        "VLLM_ASCEND_DRRQR_PLAN_SHA256": args.plan_sha256,
        "VLLM_ASCEND_DRRQR_EVIDENCE_FILE": str(output / "evidence.jsonl"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update(flags)
    diagnostic = None
    worker_args = []
    if args.diagnostic_worker:
        worker = args.diagnostic_worker
        expected = ROOT / "tools" / WORKER_NAME
        if not worker.is_absolute() or worker.resolve() != expected.resolve() or worker.is_symlink():
            raise ValueError("only the explicit repository layerwise probe worker is allowed")
        if digest(worker) != args.diagnostic_worker_sha256:
            raise ValueError("diagnostic worker digest differs")
        env["DRRQR_LAYERWISE_PROBE_DIR"] = str(output / "worker-probe")
        env["PYTHONPATH"] = str(expected.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        worker_args = ["--worker-cls", "layerwise_probe_worker." + WORKER_CLASS]
        diagnostic = {"worker": str(worker), "worker_sha256": args.diagnostic_worker_sha256,
                      "class": WORKER_CLASS, "output_dir": env["DRRQR_LAYERWISE_PROBE_DIR"],
                      "claim": "model-control diagnostics; timings are not performance evidence"}
    return env, flags, inherited_runtime, worker_args, diagnostic


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def set_subreaper(enabled):
    """Adopt only this launcher's orphaned descendants so they can be reaped."""
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
    if libc.prctl(36, int(enabled), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    return bool(previous.value)


def reap_group_orphans(child):
    child.poll()
    if child.returncode is None:
        return
    while True:
        try:
            pid, _ = os.waitpid(-child.pid, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
        return True
    except ProcessLookupError:
        return False


def stop_group(child, grace_seconds):
    """Signal only the session/group created for this service, then reap leader."""
    pgid = child.pid
    if pgid == os.getpgrp():
        raise RuntimeError("refusing to signal launcher's own process group")
    sent = []
    if signal_group(pgid, signal.SIGTERM):
        sent.append("SIGTERM")
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        reap_group_orphans(child)
        if not group_exists(pgid):
            return {"signals": sent, "exit_code": child.wait(), "group_remaining": False}
        time.sleep(0.1)
    if signal_group(pgid, signal.SIGKILL):
        sent.append("SIGKILL")
    code = child.wait(timeout=10)
    deadline = time.monotonic() + 10
    while group_exists(pgid) and time.monotonic() < deadline:
        reap_group_orphans(child)
        time.sleep(0.1)
    return {"signals": sent, "exit_code": code, "group_remaining": group_exists(pgid)}


def run_service(command, env, output, max_seconds, term_grace):
    """One launch, with stop.request, signal forwarding and a hard time budget."""
    started = time.time()
    requested = []
    old_handlers = {}
    previous_subreaper = None
    child = None
    record = {"schema": "drrqr-layerwise-service-exit/v1", "started_epoch_s": started,
              "stop_reason": None, "service_pid": None, "process_group": None,
              "exit_code": None, "signals": [], "group_remaining": None}
    def request_stop(signum, frame):
        if not requested:
            requested.append(signum)
    try:
        previous_subreaper = set_subreaper(True)
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, request_stop)
        with (output / "service.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, env=env, cwd=str(output), stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            record.update(service_pid=child.pid, process_group=child.pid)
            write_record(output / "service-process.json",
                         {"pid": child.pid, "pgid": child.pid, "launcher_pid": os.getpid(),
                          "started_epoch_s": started, "max_service_seconds": max_seconds})
            print(json.dumps({"service_pid": child.pid, "output_dir": str(output)}), flush=True)
            deadline = time.monotonic() + max_seconds
            while True:
                if requested:
                    record["stop_reason"] = "launcher_signal"
                    record["requested_signal"] = requested[0]
                    break
                if (output / "stop.request").exists():
                    record["stop_reason"] = "stop_request"
                    break
                if child.poll() is not None:
                    record["stop_reason"] = "service_exit"
                    break
                if time.monotonic() >= deadline:
                    record["stop_reason"] = "time_budget"
                    break
                time.sleep(0.2)
            record.update(stop_group(child, term_grace))
    except BaseException as error:
        record.update(stop_reason="launcher_error", error=repr(error))
        if child is not None:
            record.update(stop_group(child, term_grace))
        raise
    finally:
        if previous_subreaper is not None:
            set_subreaper(previous_subreaper)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        record.update(ended_epoch_s=time.time(), elapsed_seconds=time.time() - started)
        write_record(output / "service-exit.json", record)
    if record["group_remaining"]:
        return 125
    if record["stop_reason"] == "time_budget":
        return 124
    if record["stop_reason"] == "launcher_signal":
        return 128 + requested[0]
    if record["stop_reason"] == "stop_request":
        return 0 if record["exit_code"] in (0, -signal.SIGTERM) else 1
    return record["exit_code"] if record["exit_code"] >= 0 else 128 - record["exit_code"]


def main(argv=None):
    parser, args = parse_args(argv)
    if Path(__file__).resolve().parent != ROOT / "tools":
        parser.error("run the wrapper from the plugin repository tools directory")
    if not args.output_dir.is_absolute():
        parser.error("output directory must be absolute")
    output = args.output_dir.resolve()
    if RESULT_ROOT.resolve() not in output.parents or output.exists():
        parser.error("choose a fresh directory below /drrqr-results")
    env, flags, inherited, worker_args, diagnostic = service_environment(args, output)
    provenance = verify_build(args)
    runtime = verify_runtime(args.reference_manifest_sha256)
    plan = verify_plan(args)
    command = service_command(plan["shared_hf_head_k_dim"]) + worker_args
    # Port collision is a failed preflight, never a reason to replace an owner.
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", 6666))
    output.mkdir(parents=True, exist_ok=False)
    if diagnostic is not None:
        (output / "worker-probe").mkdir(exist_ok=False)
    manifest = {
        "schema": "drrqr-layerwise-service/v1", **provenance, **runtime, **plan,
        "command": command, "plugin_flags": flags, "diagnostic_worker": diagnostic,
        "inherited_runtime_environment": inherited, "profiler_config": None,
        "common_optimizations": {"conv_layout": "packed", "prefill_mc2": True,
                                 "prefill_mc2_mixed": True},
        "wrapper_sha256": digest(Path(__file__)), "created_epoch_s": time.time(),
        "physical_npu_contract": [4, 5, 6, 7], "container_npu_namespace": [0, 1, 2, 3],
        "physical_mapping_verification": "external parent preflight required before launch",
        "max_service_seconds": args.max_service_seconds, "term_grace_seconds": args.term_grace_seconds,
        "stop_request_path": str(output / "stop.request"),
        "claim": "model-control service provenance; no performance or task accuracy measurement",
    }
    write_record(output / "service-manifest.json", manifest)
    return run_service(command, env, output, args.max_service_seconds, args.term_grace_seconds)


if __name__ == "__main__":
    sys.exit(main())
