#!/usr/bin/env python3
"""One frozen, profiler-disabled performance screen; not final acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

DRIVER_SHA = "cba2b1b463a1e436c818722d92147d93f8396b8d64f8da6582d7e67d2cb45822"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", required=True, type=Path)
    parser.add_argument("--benchmark-driver", required=True, type=Path)
    parser.add_argument("--experiment", choices=("conv-layout-screen-v1", "prefill-mc2-screen-v1",
                                                "prefill-mc2-mixed-v1"),
                        default="conv-layout-screen-v1")
    parser.add_argument("--concurrency", type=int, choices=(16, 32), default=16,
                        help="Frozen workload concurrency; 32 is only for the preregistered scaling screen")
    args = parser.parse_args()
    directory = args.service_dir.resolve()
    root = Path("/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight") / args.experiment
    if root not in directory.parents:
        parser.error("use a fresh service directory in the selected experiment")
    if hashlib.sha256(args.benchmark_driver.read_bytes()).hexdigest() != DRIVER_SHA:
        raise RuntimeError("frozen driver changed")
    deadline = time.monotonic() + 900
    while True:
        try:
            manifest = json.loads((directory / "service-manifest.json").read_text())
            with urlopen("http://127.0.0.1:6666/health", timeout=5) as response:
                if response.status == 200:
                    break
        except (OSError, ValueError):
            pass
        if (directory / "service-exit.json").exists():
            raise RuntimeError("service exited before readiness")
        if time.monotonic() >= deadline:
            raise TimeoutError("service not ready")
        time.sleep(5)
    if manifest["profiler_config"] is not None or manifest.get("diagnostic_determinism") is not None:
        raise RuntimeError("not the frozen profiler-disabled runtime")
    if "--profiler-config" in manifest["command"] or "--worker-cls" in manifest["command"]:
        raise RuntimeError("diagnostic runtime is not a performance candidate")
    smoke_sha = None
    if args.experiment == "prefill-mc2-screen-v1":
        smoke_path = directory / "activation-smoke.json"
        smoke = json.loads(smoke_path.read_text())
        if smoke["status"] != "activation_smoke_pass" or smoke["service_manifest_sha256"] != hashlib.sha256(
                (directory / "service-manifest.json").read_bytes()).hexdigest():
            raise RuntimeError("MC2 integration smoke is not bound to this service")
        smoke_sha = hashlib.sha256(smoke_path.read_bytes()).hexdigest()
    elif args.experiment == "prefill-mc2-mixed-v1":
        smoke_path = directory / ("mixed-sequence.validated.json"
                                  if (directory / "mixed-sequence.validated.json").exists()
                                  else "mixed-sequence.json")
        smoke = json.loads(smoke_path.read_text())
        if smoke["status"] != "sequence_probe_pass" or smoke["service_manifest_sha256"] != hashlib.sha256(
                (directory / "service-manifest.json").read_bytes()).hexdigest():
            raise RuntimeError("mixed MC2 native sequence gate is not bound to this service")
        smoke_sha = hashlib.sha256(smoke_path.read_bytes()).hexdigest()
    output = directory / "performance.run1.json"
    receipt = directory / "performance-screen.json"
    log_path = directory / "performance-client.log"
    if any(path.exists() for path in (output, receipt, log_path)):
        raise RuntimeError("preserve previous screen artifacts")
    command = [sys.executable, "-u", str(args.benchmark_driver),
               "--base-url", "http://127.0.0.1:6666", "--model", "qwen3.8",
               "--input-tokens", "32768", "--prefix-tokens", "16384",
               "--output-tokens", "1024", "--requests", "40", "--concurrency", str(args.concurrency),
               "--warmup-requests", "2", "--timeout", "1800", "--json-output", str(output)]
    record = {"schema": "drrqr-layout-performance-screen/v1", "status": "running",
              "variant": manifest["variant"], "layout": manifest["layout"],
              "experiment": args.experiment, "prefill_mc2": manifest.get("prefill_mc2", False),
              "concurrency": args.concurrency,
              "activation_smoke_sha256": smoke_sha,
              "command": command, "driver_sha256": DRIVER_SHA,
              "service_manifest_sha256": hashlib.sha256((directory / "service-manifest.json").read_bytes()).hexdigest(),
              "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "started_epoch_s": time.time(),
              "claim": "single screening run, not reproducible performance or accuracy acceptance"}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"event": "screen_started", "service_dir": str(directory)}), flush=True)
    try:
        with log_path.open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = json.loads(output.read_text())
        rows = report["results"]
        if len(rows) != 40 or sorted(r["index"] for r in rows) != list(range(40)):
            raise RuntimeError("incomplete or duplicated measured requests")
        if not all(r["ok"] and r["output_tokens"] == r["requested_output_tokens"] == 1024
                   and r["finish_reason"] == "length" for r in rows):
            raise RuntimeError("fixed work or response validity failed")
        warm = report["warmup_results"]
        if len(warm) != 2 or not all(r["ok"] and r["output_tokens"] == 1 for r in warm):
            raise RuntimeError("warmup scope changed")
        record["status"] = "screen_complete"
        record["summary"] = report["summary"]
        record["report_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    except Exception as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        raise
    finally:
        record["finished_epoch_s"] = time.time()
        receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
