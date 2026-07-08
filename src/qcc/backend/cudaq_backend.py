"""Tape → CUDA-Q kernel (builder API), executed on the GPU when present.

CUDA-Q has no public QIR-ingestion API, so the execution path drives the
kernel builder from the optimized IR directly — the emitted QIR remains
the compiler's portable artifact. `cudaq` imports lazily: everything else
in qcc works without a GPU stack installed.
"""

from __future__ import annotations

import math
from typing import Any

from qcc.ir.tape import Tape


def build_kernel(tape: Tape) -> Any:
    import cudaq

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(tape.n_qubits)

    for ins in tape.instrs:
        w = ins.qubits
        match ins.name:
            case "h" | "x" | "y" | "z" | "s" | "sdg" | "t" | "tdg" as g:
                getattr(kernel, g)(q[w[0]])
            case "rx" | "ry" | "rz" as g:
                getattr(kernel, g)(ins.params[0], q[w[0]])
            case "p":
                kernel.r1(ins.params[0], q[w[0]])
            case "sx":
                kernel.rx(math.pi / 2, q[w[0]])
            case "u3":
                theta, phi, lam = ins.params
                kernel.u3(theta, phi, lam, q[w[0]])
            case "cx":
                kernel.cx(q[w[0]], q[w[1]])
            case "cz":
                kernel.cz(q[w[0]], q[w[1]])
            case "swap":
                kernel.swap(q[w[0]], q[w[1]])
            case "measure":
                kernel.mz(q[w[0]])
            case "barrier":
                pass
            case _:  # pragma: no cover
                raise KeyError(f"no CUDA-Q lowering for {ins.name!r}")

    return kernel


def set_best_target(prefer: str = "nvidia") -> str:
    """Pick the GPU simulator if it initializes, else CPU. Returns the name."""
    import cudaq

    for target in (prefer, "qpp-cpu"):
        try:
            cudaq.set_target(target)
            return target
        except RuntimeError:
            continue
    return cudaq.get_target().name


def sample(tape: Tape, shots: int = 1000) -> dict[str, int]:
    """Measurement counts. Bitstring character i is qubit i."""
    import cudaq

    result = cudaq.sample(build_kernel(tape), shots_count=shots)
    return {bits: result.count(bits) for bits in result}
