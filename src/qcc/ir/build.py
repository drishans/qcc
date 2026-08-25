"""Programmatic IR construction: the one place that knows how to thread
qubit SSA values through a growing circuit."""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region, SSAValue

from qcc.ir import dialect as d


class CircuitBuilder:
    """Builds a `func.func @main` full of qcc ops, tracking the frontier
    SSA value of every wire."""

    def __init__(self, n_qubits: int):
        self.block = Block()
        self.front: list[SSAValue] = []
        for i in range(n_qubits):
            alloc = d.AllocOp(i)
            self.block.add_op(alloc)
            self.front.append(alloc.result)

    def gate(self, name: str, *wires: int, params: tuple[float, ...] = ()) -> None:
        if name in d.FIXED_1Q_OPS:
            (w,) = wires
            op = d.FIXED_1Q_OPS[name](self.front[w])
            self.front[w] = op.qout
        elif name in d.ROTATION_OPS:
            (w,) = wires
            (angle,) = params
            op = d.ROTATION_OPS[name](self.front[w], angle)
            self.front[w] = op.qout
        elif name == "u3":
            (w,) = wires
            theta, phi, lam = params
            op = d.U3Op(self.front[w], theta, phi, lam)
            self.front[w] = op.qout
        elif name in d.GATE_2Q_OPS:
            wa, wb = wires
            if wa == wb:
                raise ValueError(f"{name} needs two distinct qubits, got {wires}")
            op = d.GATE_2Q_OPS[name](self.front[wa], self.front[wb])
            self.front[wa], self.front[wb] = op.qout0, op.qout1
        elif name == "id":
            return  # identity: nothing to do
        else:
            raise KeyError(f"unknown gate {name!r}")
        self.block.add_op(op)

    def barrier(self, *wires: int) -> None:
        ws = wires or tuple(range(len(self.front)))
        op = d.BarrierOp([self.front[w] for w in ws])
        self.block.add_op(op)
        for w, out in zip(ws, op.qouts):
            self.front[w] = out

    def measure(self, wire: int, cbit: int) -> None:
        op = d.MeasureOp(self.front[wire], cbit)
        self.block.add_op(op)
        self.front[wire] = op.qout

    def finish(self) -> ModuleOp:
        self.block.add_op(ReturnOp())
        func = FuncOp("main", ([], []), Region(self.block))
        return ModuleOp([func])


def build_from_tape(tape) -> ModuleOp:
    """Rebuild a tape as fresh, dominance-correct qubit SSA."""
    builder = CircuitBuilder(tape.n_qubits)
    for ins in tape.instrs:
        if ins.name == "barrier":
            builder.barrier(*ins.qubits)
        elif ins.name == "measure":
            assert ins.cbit is not None
            builder.measure(ins.qubits[0], ins.cbit)
        else:
            builder.gate(ins.name, *ins.qubits, params=ins.params)
    return builder.finish()


def replace_main_from_tape(module: ModuleOp, tape) -> None:
    """Atomically replace ``@main`` with a fresh SSA graph for ``tape``.

    KAK rewrites both wires of a region at once. Rebuilding the function makes
    every frontier and dominance relation explicit, instead of performing a
    fragile sequence of local two-result rewires.
    """
    from xdsl.rewriter import Rewriter

    from qcc.ir.tape import main_func

    old_func = main_func(module)
    replacement_module = build_from_tape(tape)
    new_func = main_func(replacement_module)
    new_func.detach()
    Rewriter.replace_op(old_func, new_func, [])
