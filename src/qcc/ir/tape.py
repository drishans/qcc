"""IR → linear instruction tape.

The tape is the neutral, backend-facing view of a circuit: a list of
(gate, params, wires) with measures carrying their classical bit. The
simulator, the QIR emitter, the CUDA-Q backend, and the metrics all
consume this instead of walking SSA themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import SSAValue

from qcc.ir import dialect as d


@dataclass(frozen=True)
class Instr:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()
    cbit: int | None = None


@dataclass
class Tape:
    n_qubits: int
    n_cbits: int
    instrs: list[Instr] = field(default_factory=list)


def main_func(module: ModuleOp) -> FuncOp:
    for op in module.body.ops:
        if isinstance(op, FuncOp):
            return op
    raise ValueError("module has no func.func")


def extract(module: ModuleOp) -> Tape:
    """Flatten the SSA circuit into a tape, resolving values to wire indices."""
    func = main_func(module)
    wire: dict[SSAValue, int] = {}
    instrs: list[Instr] = []
    n_qubits = 0
    n_cbits = 0

    for op in func.body.ops:
        if isinstance(op, d.AllocOp):
            wire[op.result] = op.wire
            n_qubits = max(n_qubits, op.wire + 1)
        elif isinstance(op, d.Gate1Op):
            w = wire[op.qin]
            instrs.append(Instr(op.GATE_NAME, (w,), op.params))
            wire[op.qout] = w
        elif isinstance(op, d.Gate2Op):
            wa, wb = wire[op.qin0], wire[op.qin1]
            instrs.append(Instr(op.GATE_NAME, (wa, wb)))
            wire[op.qout0], wire[op.qout1] = wa, wb
        elif isinstance(op, d.BarrierOp):
            ws = tuple(wire[q] for q in op.qins)
            instrs.append(Instr("barrier", ws))
            for q, out in zip(op.qins, op.qouts):
                wire[out] = wire[q]
        elif isinstance(op, d.MeasureOp):
            w = wire[op.qin]
            instrs.append(Instr("measure", (w,), cbit=op.cbit_index))
            wire[op.qout] = w
            n_cbits = max(n_cbits, op.cbit_index + 1)
        elif isinstance(op, ReturnOp):
            break
        else:
            raise ValueError(f"tape extraction hit unexpected op {op.name}")

    return Tape(n_qubits, n_cbits, instrs)
