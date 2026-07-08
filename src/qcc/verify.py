"""Semantic equivalence of two circuits, up to global phase.

|<ψ_a|ψ_b>| = 1 iff the pre-measurement states differ only by a global
phase. This is the invariant every pass must preserve (see qcc.gates for
why global phase is the right quotient: the IR has no controlled
modifiers, so per-wire phases factor out of the whole state).
"""

from __future__ import annotations

import numpy as np
from xdsl.dialects.builtin import ModuleOp

from qcc.backend.sim import statevector
from qcc.ir.tape import Tape, extract

TOL = 1e-8


def fidelity(a: Tape, b: Tape) -> float:
    if a.n_qubits != b.n_qubits:
        raise ValueError(f"qubit counts differ: {a.n_qubits} vs {b.n_qubits}")
    return float(abs(np.vdot(statevector(a), statevector(b))))


def equivalent(a: ModuleOp | Tape, b: ModuleOp | Tape, tol: float = TOL) -> bool:
    ta = a if isinstance(a, Tape) else extract(a)
    tb = b if isinstance(b, Tape) else extract(b)
    return abs(fidelity(ta, tb) - 1.0) < tol
