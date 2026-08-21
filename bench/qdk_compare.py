#!/usr/bin/env python3
"""qcc vs Microsoft's QDK on the QIR they emit for the same OpenQASM 3.

This is a different comparison from run_bench.py, and deliberately so: the
QDK's compiler is not an optimizing transpiler, it's a language frontend
that lowers Q#/OpenQASM to QIR and leaves gate-level optimization to
whatever consumes the QIR. So the fair question is not "who optimizes
harder" but "given identical OpenQASM 3, what QIR does each stack emit,
and what does it cost to get it".

Ground rules:
- both stacks receive the identical OpenQASM 3 text (bench suites plus
  terminal measurements, which the base profile's output recording wants);
- the metric is *QIR instructions*, counted by one shared parser over the
  emitted text, so each side pays for its own lowering choices (qcc's
  u3 → rz·ry·rz and swap → 3 cx included);
- qdk-base = QDK compile at TargetProfile.Base; qcc-O0 = parse + emit,
  no optimization (isolates lowering from optimization); qcc-O1 = the
  full pipeline;
- timings are median of 5 of the whole source→QIR call for both stacks;
- cross-execution check: the QDK's own simulator runs qcc's *optimized*
  qasm3 output for GHZ-12 and must reproduce the two-string support.
"""

from __future__ import annotations

import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import qdk.openqasm as oq
from circuits import suite
from qdk import TargetProfile
from qiskit import ClassicalRegister, QuantumCircuit, qasm3

import provenance
from qcc.backend.qasm3_out import emit_qasm3
from qcc.backend.qir import emit_qir
from qcc.frontend import parse_qasm3
from qcc.ir import extract
from qcc.passes import optimize

REPEATS = 5

_CALL = re.compile(r"call void @__quantum__qis__(\w+?)__body")
_MEASURE = {"m", "mz", "mresetz"}
_TWOQ = {"cx", "cnot", "cz", "swap"}


def qir_metrics(qir_text: str) -> dict:
    names = [m.group(1) for m in _CALL.finditer(qir_text)]
    gates = [n for n in names if n not in _MEASURE]
    return {
        "qir_gates": len(gates),
        "qir_2q": sum(1 for n in gates if n in _TWOQ),
        "by_gate": dict(sorted(Counter(gates).items())),
    }


def with_measures(qc: QuantumCircuit) -> QuantumCircuit:
    out = qc.copy()
    creg = ClassicalRegister(qc.num_qubits, "c")
    out.add_register(creg)
    out.measure(range(qc.num_qubits), range(qc.num_qubits))
    return out


def _timed(fn):
    times, result = [], None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return result, statistics.median(times)


def run_qdk(src3: str) -> dict:
    qir, t_ms = _timed(
        lambda: str(oq.compile(src3, target_profile=TargetProfile.Base))
    )
    return {**qir_metrics(qir), "t_ms": t_ms}


def run_qcc(src3: str, level: int) -> dict:
    def go():
        m = parse_qasm3(src3)
        if level:
            optimize(m)
        return emit_qir(extract(m))

    qir, t_ms = _timed(go)
    return {**qir_metrics(qir), "t_ms": t_ms}


def cross_execution_check() -> dict:
    """Microsoft's simulator executes qcc's optimized output for GHZ-12."""
    from circuits import ghz, to_basis

    src = qasm3.dumps(with_measures(to_basis(ghz(12))))
    module = parse_qasm3(src)
    optimize(module)
    optimized_src = emit_qasm3(extract(module))
    counts = Counter(oq.run(optimized_src, shots=500, as_bitstring=True))
    support = set(counts)
    ok = support <= {"0" * 12, "1" * 12} and len(support) == 2
    return {"ghz12_support": sorted(support), "passed": ok, "shots": 500}


def main() -> None:
    rows = []
    for suite_name, instance, qc in suite():
        src3 = qasm3.dumps(with_measures(qc))
        for tool, res in (
            ("qdk-base", run_qdk(src3)),
            ("qcc-O0", run_qcc(src3, 0)),
            ("qcc-O1", run_qcc(src3, 1)),
        ):
            rows.append({
                "suite": suite_name, "instance": instance,
                "n_qubits": qc.num_qubits, "tool": tool, **res,
            })
            print(
                f"[{suite_name}/{instance}] {tool:<9} "
                f"{res['qir_gates']:>4} qir gates ({res['qir_2q']} 2q) "
                f"{res['t_ms']:8.2f} ms"
            )

    xcheck = cross_execution_check()
    print(f"cross-execution (QDK runs qcc-optimized ghz12): {xcheck}")
    if not xcheck["passed"]:
        sys.exit("CROSS-EXECUTION CHECK FAILED")

    provenance.write(
        "qdk_qir_compare",
        params={
            "repeats": REPEATS,
            "timing": "median of full source->QIR compile, both stacks",
            "metric": "QIR instructions counted by shared parser; measures excluded",
            "qdk_profile": "TargetProfile.Base",
            "cross_execution": xcheck,
        },
        rows=rows,
    )


if __name__ == "__main__":
    main()
