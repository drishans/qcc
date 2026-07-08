"""Tape → QIR (base profile) via pyqir.

The QIR base-profile QIS has no swap/sx/p/u3, so those lower here:
swap → 3 cx, sx → rx(π/2), p → rz (both up to global phase — the
compiler's standing invariant), u3(θ,φ,λ) → rz(λ)·ry(θ)·rz(φ) applied in
that order. Barriers are compile-time fences and vanish at this boundary.
"""

from __future__ import annotations

import math

import pyqir

from qcc.ir.tape import Tape


def emit_qir(tape: Tape, name: str = "qcc") -> str:
    mod = pyqir.SimpleModule(
        name, num_qubits=tape.n_qubits, num_results=max(tape.n_cbits, 1)
    )
    qis = pyqir.BasicQisBuilder(mod.builder)
    q = mod.qubits

    fixed = {
        "h": qis.h,
        "x": qis.x,
        "y": qis.y,
        "z": qis.z,
        "s": qis.s,
        "sdg": qis.s_adj,
        "t": qis.t,
        "tdg": qis.t_adj,
    }
    rot = {"rx": qis.rx, "ry": qis.ry, "rz": qis.rz}

    for ins in tape.instrs:
        w = ins.qubits
        if ins.name in fixed:
            fixed[ins.name](q[w[0]])
        elif ins.name in rot:
            rot[ins.name](ins.params[0], q[w[0]])
        elif ins.name == "p":
            qis.rz(ins.params[0], q[w[0]])
        elif ins.name == "sx":
            qis.rx(math.pi / 2, q[w[0]])
        elif ins.name == "u3":
            theta, phi, lam = ins.params
            qis.rz(lam, q[w[0]])
            qis.ry(theta, q[w[0]])
            qis.rz(phi, q[w[0]])
        elif ins.name == "cx":
            qis.cx(q[w[0]], q[w[1]])
        elif ins.name == "cz":
            qis.cz(q[w[0]], q[w[1]])
        elif ins.name == "swap":
            a, b = q[w[0]], q[w[1]]
            qis.cx(a, b)
            qis.cx(b, a)
            qis.cx(a, b)
        elif ins.name == "measure":
            assert ins.cbit is not None
            qis.mz(q[w[0]], mod.results[ins.cbit])
        elif ins.name == "barrier":
            pass
        else:  # pragma: no cover
            raise KeyError(f"no QIR lowering for {ins.name!r}")

    return mod.ir()
