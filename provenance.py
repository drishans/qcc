"""Environment provenance for every benchmark result in this repo.

Every results/*.json carries the exact hardware and library versions it was
measured with — a benchmark number without its environment is a rumor.

Compiler benchmarks are CPU-bound, so the CPU model is always recorded; the
GPU block is best-effort and only matters for CUDA-Q execution checks.
"""

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

PACKAGES = ("xdsl", "openqasm3", "pyqir", "qiskit", "pytket", "cudaq", "qdk", "numpy")


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _gpu() -> dict | None:
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if smi:
            name, driver = (s.strip() for s in smi.split(",", 1))
            return {"gpu": name, "driver": driver}
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def collect() -> dict:
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": _cpu_model(),
        "os": f"{platform.system()} {platform.release()} (WSL2)"
        if "microsoft" in platform.release().lower()
        else platform.platform(),
        "python": platform.python_version(),
        "versions": {},
    }
    for pkg in PACKAGES:
        try:
            meta["versions"][pkg] = version(pkg)
        except PackageNotFoundError:
            meta["versions"][pkg] = None
    gpu = _gpu()
    if gpu:
        meta.update(gpu)
    return meta


def write(name: str, params: dict, rows: list[dict]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps({"meta": collect(), "params": params, "rows": rows}, indent=2))
    print(f"wrote {out} ({len(rows)} rows)")
    return out
