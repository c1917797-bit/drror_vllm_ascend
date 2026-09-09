#!/usr/bin/env python3
"""Build this repository snapshot offline in the dedicated experiment container."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if Path("/drrqr-results") not in output.parents or output.exists():
        parser.error("choose a fresh build directory under /drrqr-results")
    output.mkdir(parents=True)
    stage = output / "source"
    stage.mkdir()
    shutil.copytree(root / "drror_vllm_ascend", stage / "drror_vllm_ascend",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(root / name, stage / name)
    sources = {str(p.relative_to(stage)): digest(p) for p in stage.rglob("*") if p.is_file()}
    for name, checksum in sources.items():
        if digest(root / name) != checksum:
            raise RuntimeError("source changed while taking build snapshot")
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
                    "--no-build-isolation", "--wheel-dir", str(output / "wheels"), str(stage)], check=True)
    wheels = list((output / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("expected one offline-built wheel")
    result = {"schema": "drrqr-local-wheel-build/v1", "source_root": str(root),
              "source_sha256": sources, "wheel": str(wheels[0]), "wheel_sha256": digest(wheels[0]),
              "builder_sha256": digest(Path(__file__))}
    (output / "build-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
