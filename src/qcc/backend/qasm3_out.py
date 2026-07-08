"""Tape → OpenQASM 3 text (stdgates vocabulary).

The inverse of the frontend, used to hand optimized circuits to other
toolchains and for `qcc compile --emit qasm3`. parse_qasm3(emit_qasm3(t))
must round-trip to an equivalent circuit — there's a test for that.
"""

from __future__ import annotations

from qcc.ir.tape import Tape


def _fmt(x: float) -> str:
    return repr(float(x))


def emit_qasm3(tape: Tape) -> str:
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{tape.n_qubits}] q;",
    ]
    if tape.n_cbits:
        lines.append(f"bit[{tape.n_cbits}] c;")

    for ins in tape.instrs:
        qs = ", ".join(f"q[{w}]" for w in ins.qubits)
        if ins.name == "measure":
            lines.append(f"c[{ins.cbit}] = measure q[{ins.qubits[0]}];")
        elif ins.name == "barrier":
            lines.append(f"barrier {qs};")
        elif ins.params:
            args = ", ".join(_fmt(p) for p in ins.params)
            lines.append(f"{ins.name}({args}) {qs};")
        else:
            lines.append(f"{ins.name} {qs};")

    return "\n".join(lines) + "\n"
