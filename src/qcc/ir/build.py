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
