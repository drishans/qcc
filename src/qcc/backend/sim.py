"""Reference statevector simulator — plain numpy, no quantum framework.

This is the compiler's ground truth: it deliberately shares no code with
the frontends or backends being benchmarked, so an optimizer bug can't
hide behind a shared convention. State axes follow wire indices (axis k =
wire k, wire 0 = most significant bit), matching the operand-order
convention in qcc.gates.

Measures and barriers are skipped: equivalence is checked on the
pre-measurement state.
"""

from __future__ import annotations

import numpy as np

from qcc import gates
from qcc.ir.tape import Tape

MAX_QUBITS = 16


def statevector(tape: Tape) -> np.ndarray:
    """Pre-measurement state of the tape, as a flat 2^n vector."""
    n = tape.n_qubits
    if n == 0:
        raise ValueError("empty circuit")
    if n > MAX_QUBITS:
        raise ValueError(f"{n} qubits exceeds reference-simulator cap {MAX_QUBITS}")

    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0

    for ins in tape.instrs:
        if ins.name in ("barrier", "measure"):
            continue
        if len(ins.qubits) == 1:
            (w,) = ins.qubits
            u = gates.matrix_1q(ins.name, ins.params)
            state = np.tensordot(u, state, axes=[(1,), (w,)])
            state = np.moveaxis(state, 0, w)
        else:
            wa, wb = ins.qubits
            u = gates.matrix_2q(ins.name).reshape(2, 2, 2, 2)
            state = np.tensordot(u, state, axes=[(2, 3), (wa, wb)])
            state = np.moveaxis(state, (0, 1), (wa, wb))

    return state.reshape(-1)
