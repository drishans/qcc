#!/usr/bin/env python3
"""qcc vs Qiskit transpiler vs pytket on the generated suites.

Ground rules (also in the README):
- every tool receives the identical basis-translated circuit; qiskit gets
  the QuantumCircuit, qcc gets qasm3.dumps of it, pytket gets qasm2.dumps;
- qiskit runs transpile() with NO backend/coupling map — logical
  optimization only, same game qcc plays;
- pytket runs FullPeepholeOptimise;
- timings are the median of 5 runs of the optimization call alone;
- "verified" (qcc rows only) = statevector equivalence up to global phase
  against the unoptimized circuit, checked with qcc's reference simulator.
  Qiskit/pytket outputs are not re-verified here.

Metrics count unitary gates only; depth likewise.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from circuits import suite
from pytket.passes import FullPeepholeOptimise
from pytket.qasm import circuit_from_qasm_str
from qiskit import QuantumCircuit, qasm2, qasm3, transpile

import provenance
from qcc.frontend import parse_qasm3
from qcc.ir import extract, metrics
from qcc.passes import optimize
from qcc.verify import equivalent

REPEATS = 5
VERIFY_MAX_QUBITS = 12


def qiskit_metrics(qc: QuantumCircuit) -> dict:
    ops = [i for i in qc.data if i.operation.name not in ("barrier", "measure")]
    return {
        "gates": len(ops),
        "gates_2q": sum(1 for i in ops if len(i.qubits) == 2),
        "depth": qc.depth(
            lambda i: i.operation.name not in ("barrier", "measure")
        ),
    }


def tket_metrics(circ) -> dict:
    from pytket.circuit import OpType

    cmds = [
        c for c in circ.get_commands()
        if c.op.type not in (OpType.Barrier, OpType.Measure)
    ]
    return {
        "gates": len(cmds),
        "gates_2q": sum(1 for c in cmds if len(c.qubits) == 2),
        "depth": circ.depth(),
    }


def _timed(fn):
    times = []
    result = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return result, statistics.median(times)


def run_qcc(src3: str, level: int) -> dict:
    before_tape = extract(parse_qasm3(src3))

    def compile_once():
        m = parse_qasm3(src3)  # fresh IR each repeat; only optimize is timed
        t0 = time.perf_counter()
        optimize(m, level)
        dt = (time.perf_counter() - t0) * 1000
        return m, dt

    times, module = [], None
    for _ in range(REPEATS):
        module, dt = compile_once()
        times.append(dt)
    after_tape = extract(module)
    m = metrics(after_tape)
    verified = None
    if after_tape.n_qubits <= VERIFY_MAX_QUBITS:
        verified = equivalent(before_tape, after_tape)
    return {
        "gates": m["gates"], "gates_2q": m["gates_2q"], "depth": m["depth"],
        "t_ms": statistics.median(times), "verified": verified,
    }


def run_qiskit(qc: QuantumCircuit, level: int) -> dict:
    out, t_ms = _timed(lambda: transpile(qc, optimization_level=level))
    return {**qiskit_metrics(out), "t_ms": t_ms, "verified": None}


def run_tket(src2: str) -> dict:
    def go():
        circ = circuit_from_qasm_str(src2)
        t0 = time.perf_counter()
        FullPeepholeOptimise().apply(circ)
        return circ, (time.perf_counter() - t0) * 1000

    times, circ = [], None
    for _ in range(REPEATS):
        circ, dt = go()
        times.append(dt)
    return {**tket_metrics(circ), "t_ms": statistics.median(times), "verified": None}


def main() -> None:
    rows = []
    for suite_name, instance, qc in suite():
        src3 = qasm3.dumps(qc)
        src2 = qasm2.dumps(qc)
        base = {
            "suite": suite_name, "instance": instance,
            "n_qubits": qc.num_qubits, **qiskit_metrics(qc),
            "tool": "input", "t_ms": 0.0, "verified": None,
        }
        rows.append(base)
        print(f"[{suite_name}/{instance}] input: {base['gates']} gates")

        for tool, res in (
            ("qcc-O1", run_qcc(src3, 1)),
            ("qcc-O2", run_qcc(src3, 2)),
            ("qiskit-O1", run_qiskit(qc, 1)),
            ("qiskit-O2", run_qiskit(qc, 2)),
            ("qiskit-O3", run_qiskit(qc, 3)),
            ("tket-full", run_tket(src2)),
        ):
            rows.append({
                "suite": suite_name, "instance": instance,
                "n_qubits": qc.num_qubits, "tool": tool, **res,
            })
            v = {True: " ✓verified", False: " ✗NOT-EQUIV", None: ""}[res["verified"]]
            print(
                f"  {tool:<10} {res['gates']:>4} gates "
                f"({res['gates_2q']} 2q, depth {res['depth']}) "
                f"{res['t_ms']:8.2f} ms{v}"
            )

    failed = [r for r in rows if r["verified"] is False]
    if failed:
        sys.exit(f"EQUIVALENCE FAILURES: {failed}")

    provenance.write(
        "compile_bench",
        params={
            "repeats": REPEATS,
            "timing": "median of optimization call only",
            "qiskit_mode": "transpile(no backend) = logical optimization only",
            "tket_pass": "FullPeepholeOptimise",
            "verify": f"qcc rows, ≤{VERIFY_MAX_QUBITS} qubits, statevector up to global phase",
        },
        rows=rows,
    )


if __name__ == "__main__":
    main()
