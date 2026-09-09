#!/usr/bin/env python3
"""Install a verified offline plugin build in the dedicated experiment container."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import time
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifest", required=True, type=Path)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--previous-version", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    receipt = args.receipt.resolve()
    if Path("/drrqr-results") not in receipt.parents or receipt.exists():
        parser.error("fresh receipt under /drrqr-results required")
    build = json.loads(args.build_manifest.read_text())
    wheel = Path(build["wheel"])
    if digest(wheel) != args.wheel_sha256 or build["wheel_sha256"] != args.wheel_sha256:
        raise RuntimeError("wheel digest mismatch")
    expected = {k: v for k, v in build["source_sha256"].items() if k.startswith("drror_vllm_ascend/")}
    source = Path(build["source_root"])
    if {str(p.relative_to(source)) for p in (source / "drror_vllm_ascend").rglob("*.py")} != set(expected):
        raise RuntimeError("source package coverage mismatch")
    with zipfile.ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("wheel integrity failure")
        files = {n for n in archive.namelist() if n.startswith("drror_vllm_ascend/") and n.endswith(".py")}
        if files != set(expected):
            raise RuntimeError("wheel package coverage mismatch")
        for name, checksum in expected.items():
            if digest(source / name) != checksum or hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise RuntimeError("source/wheel mismatch: " + name)
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1 or ("Version: " + args.version) not in archive.read(metadata[0]).decode().splitlines():
            raise RuntimeError("unexpected wheel version")
    site = Path(sysconfig.get_path("purelib"))
    def versions():
        return [d.version for d in importlib.metadata.distributions(path=[str(site)])
                if d.metadata.get("Name") == "drror-vllm-ascend-plugin"]
    previous = versions()
    if previous != [args.previous_version]:
        raise RuntimeError("unexpected currently installed version")
    reference = Path("/drrqr-results/profile-first-v1/baseline/service-manifest.json")
    runtime = json.loads(reference.read_text())["source_sha256"]
    def check_runtime():
        if not all(digest(Path(p)) == sha for p, sha in runtime.items()):
            raise RuntimeError("upstream reference source changed")
    check_runtime()
    record = {"schema": "drrqr-local-plugin-install/v2", "status": "running",
              "previous": previous, "wheel": str(wheel), "wheel_sha256": digest(wheel),
              "build_manifest_sha256": digest(args.build_manifest),
              "reference_manifest_sha256": digest(reference), "script_sha256": digest(Path(__file__)),
              "epoch_s": time.time()}
    try:
        run = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                              "--force-reinstall", str(wheel)], capture_output=True, text=True)
        record.update(exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr)
        run.check_returncode()
        observed = {name: digest(site / name) for name in expected}
        installed_files = {str(p.relative_to(site)) for p in (site / "drror_vllm_ascend").rglob("*.py")}
        if observed != expected or installed_files != set(expected) or versions() != [args.version]:
            raise RuntimeError("installed package coverage/version/digest mismatch")
        check_runtime()
        record.update(status="installed_verified", installed=versions(), verified_module_count=len(observed),
                      source_matches_installed=True, upstream_reference_unchanged=True)
    except Exception as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
