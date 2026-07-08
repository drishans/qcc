"""Circuit metrics off the tape. Depth counts unitary gates only —
barriers and measures are excluded, so numbers are comparable across
compilers that place measures differently."""

from __future__ import annotations

from collections import Counter

from qcc.ir.tape import Tape


def metrics(tape: Tape) -> dict:
    counts: Counter[str] = Counter()
    clock = [0] * max(tape.n_qubits, 1)
    n_1q = n_2q = 0

    for ins in tape.instrs:
        if ins.name in ("barrier", "measure"):
            continue
        counts[ins.name] += 1
        if len(ins.qubits) == 1:
            n_1q += 1
        else:
            n_2q += 1
        t = max(clock[q] for q in ins.qubits) + 1
        for q in ins.qubits:
            clock[q] = t

    return {
        "n_qubits": tape.n_qubits,
        "gates": n_1q + n_2q,
        "gates_1q": n_1q,
        "gates_2q": n_2q,
        "depth": max(clock),
        "by_gate": dict(sorted(counts.items())),
    }
