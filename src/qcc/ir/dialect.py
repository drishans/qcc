"""The `qcc` dialect: quantum circuits as SSA with value-semantics qubits.

Every gate *consumes* its qubit values and *produces* fresh ones — a qubit
wire is a chain of SSA values threaded through ops, not a mutable register.
Two things fall out of that choice:

- "next gate on this wire" is just the one use of a result value, so every
  peephole optimization is a local def-use rewrite, no DAG bookkeeping;
- no-cloning becomes a *verifier rule*: a qubit value used twice is the
  same bug as a linear type violated, and `verify_linear` catches it.

This mirrors the value-semantics half of CUDA-Q's Quake dialect and QIRO.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.builtin import (
    FloatAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    f64,
    i1,
    i64,
)
from xdsl.ir import Dialect, Operation, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)

from qcc import gates


@irdl_attr_definition
class QubitType(ParametrizedAttribute, TypeAttribute):
    """A linear qubit value. Each value is consumed by at most one op."""

    name = "qcc.qubit"


qubit = QubitType()


@irdl_op_definition
class AllocOp(IRDLOperation):
    """Materialize wire `index` of the circuit as a fresh qubit value."""

    name = "qcc.alloc"
    result = result_def(qubit)
    index = prop_def(IntegerAttr[IntegerType])

    assembly_format = "$index attr-dict"

    def __init__(self, index: int):
        super().__init__(
            result_types=[qubit], properties={"index": IntegerAttr(index, i64)}
        )

    @property
    def wire(self) -> int:
        return self.index.value.data


class Gate1Op(IRDLOperation, abc.ABC):
    """A 1q gate: consumes one qubit value, produces one."""

    GATE_NAME: ClassVar[str]  # set by subclasses; key into qcc.gates tables

    qin = operand_def(qubit)
    qout = result_def(qubit)

    assembly_format = "$qin attr-dict"

    def __init__(self, qin: Operation | SSAValue, **kwargs):
        super().__init__(operands=[qin], result_types=[qubit], **kwargs)

    @property
    def params(self) -> tuple[float, ...]:
        return ()

    def matrix(self):
        return gates.matrix_1q(self.GATE_NAME, self.params)


@irdl_op_definition
class HOp(Gate1Op):
    name = "qcc.h"
    GATE_NAME: ClassVar[str] = "h"


@irdl_op_definition
class XOp(Gate1Op):
    name = "qcc.x"
    GATE_NAME: ClassVar[str] = "x"


@irdl_op_definition
class YOp(Gate1Op):
    name = "qcc.y"
    GATE_NAME: ClassVar[str] = "y"


@irdl_op_definition
class ZOp(Gate1Op):
    name = "qcc.z"
    GATE_NAME: ClassVar[str] = "z"


@irdl_op_definition
class SOp(Gate1Op):
    name = "qcc.s"
    GATE_NAME: ClassVar[str] = "s"


@irdl_op_definition
class SdgOp(Gate1Op):
    name = "qcc.sdg"
    GATE_NAME: ClassVar[str] = "sdg"


@irdl_op_definition
class TOp(Gate1Op):
    name = "qcc.t"
    GATE_NAME: ClassVar[str] = "t"


@irdl_op_definition
class TdgOp(Gate1Op):
    name = "qcc.tdg"
    GATE_NAME: ClassVar[str] = "tdg"


@irdl_op_definition
class SxOp(Gate1Op):
    name = "qcc.sx"
    GATE_NAME: ClassVar[str] = "sx"


class Rotation1Op(Gate1Op, abc.ABC):
    """A 1q rotation with a compile-time angle. Same-axis neighbors merge."""

    angle = prop_def(FloatAttr)

    assembly_format = "`(` $angle `)` $qin attr-dict"

    def __init__(self, qin: Operation | SSAValue, angle: float | FloatAttr):
        if not isinstance(angle, FloatAttr):
            angle = FloatAttr(angle, f64)
        super().__init__(qin, properties={"angle": angle})

    @property
    def params(self) -> tuple[float, ...]:
        return (self.angle.value.data,)


@irdl_op_definition
class RxOp(Rotation1Op):
    name = "qcc.rx"
    GATE_NAME: ClassVar[str] = "rx"


@irdl_op_definition
class RyOp(Rotation1Op):
    name = "qcc.ry"
    GATE_NAME: ClassVar[str] = "ry"


@irdl_op_definition
class RzOp(Rotation1Op):
    name = "qcc.rz"
    GATE_NAME: ClassVar[str] = "rz"


@irdl_op_definition
class POp(Rotation1Op):
    """Phase gate diag(1, e^{iλ}) — rz up to global phase."""

    name = "qcc.p"
    GATE_NAME: ClassVar[str] = "p"


@irdl_op_definition
class U3Op(Gate1Op):
    """Generic 1q unitary u3(θ,φ,λ), the output of 1q fusion."""

    name = "qcc.u3"
    GATE_NAME: ClassVar[str] = "u3"

    theta = prop_def(FloatAttr)
    phi = prop_def(FloatAttr)
    lam = prop_def(FloatAttr)

    assembly_format = "`(` $theta `,` $phi `,` $lam `)` $qin attr-dict"

    def __init__(
        self, qin: Operation | SSAValue, theta: float, phi: float, lam: float
    ):
        super().__init__(
            qin,
            properties={
                "theta": FloatAttr(theta, f64),
                "phi": FloatAttr(phi, f64),
                "lam": FloatAttr(lam, f64),
            },
        )

    @property
    def params(self) -> tuple[float, ...]:
        return (
            self.theta.value.data,
            self.phi.value.data,
            self.lam.value.data,
        )


class Gate2Op(IRDLOperation, abc.ABC):
    """A 2q gate: consumes two qubit values, produces two (same order)."""

    GATE_NAME: ClassVar[str]

    qin0 = operand_def(qubit)
    qin1 = operand_def(qubit)
    qout0 = result_def(qubit)
    qout1 = result_def(qubit)

    assembly_format = "$qin0 `,` $qin1 attr-dict"

    def __init__(self, qin0: Operation | SSAValue, qin1: Operation | SSAValue):
        super().__init__(operands=[qin0, qin1], result_types=[qubit, qubit])

    def matrix(self):
        return gates.matrix_2q(self.GATE_NAME)


@irdl_op_definition
class CXOp(Gate2Op):
    """CNOT; qin0 is the control."""

    name = "qcc.cx"
    GATE_NAME: ClassVar[str] = "cx"


@irdl_op_definition
class CZOp(Gate2Op):
    name = "qcc.cz"
    GATE_NAME: ClassVar[str] = "cz"


@irdl_op_definition
class SwapOp(Gate2Op):
    name = "qcc.swap"
    GATE_NAME: ClassVar[str] = "swap"


@irdl_op_definition
class BarrierOp(IRDLOperation):
    """Optimization fence. Consumes and reproduces its qubit values, so no
    def-use chain crosses it — passes can't see through it by construction."""

    name = "qcc.barrier"
    qins = var_operand_def(qubit)
    qouts = var_result_def(qubit)

    # no assembly_format: variadic result types can't be inferred, so the
    # barrier prints in generic form

    def __init__(self, qins: Sequence[SSAValue | Operation]):
        super().__init__(operands=[qins], result_types=[[qubit] * len(qins)])


@irdl_op_definition
class MeasureOp(IRDLOperation):
    """Z-basis measurement into classical bit `cbit`."""

    name = "qcc.measure"
    qin = operand_def(qubit)
    qout = result_def(qubit)
    bit = result_def(i1)
    cbit = prop_def(IntegerAttr[IntegerType])

    assembly_format = "$qin `->` $cbit attr-dict"

    def __init__(self, qin: Operation | SSAValue, cbit: int):
        super().__init__(
            operands=[qin],
            result_types=[qubit, i1],
            properties={"cbit": IntegerAttr(cbit, i64)},
        )

    @property
    def cbit_index(self) -> int:
        return self.cbit.value.data


QCC = Dialect(
    "qcc",
    [
        AllocOp,
        HOp,
        XOp,
        YOp,
        ZOp,
        SOp,
        SdgOp,
        TOp,
        TdgOp,
        SxOp,
        RxOp,
        RyOp,
        RzOp,
        POp,
        U3Op,
        CXOp,
        CZOp,
        SwapOp,
        BarrierOp,
        MeasureOp,
    ],
    [QubitType],
)

#: fixed 1q gate name → op class (rotations and u3 are handled separately)
FIXED_1Q_OPS: dict[str, type[Gate1Op]] = {
    "h": HOp,
    "x": XOp,
    "y": YOp,
    "z": ZOp,
    "s": SOp,
    "sdg": SdgOp,
    "t": TOp,
    "tdg": TdgOp,
    "sx": SxOp,
}
ROTATION_OPS: dict[str, type[Rotation1Op]] = {
    "rx": RxOp,
    "ry": RyOp,
    "rz": RzOp,
    "p": POp,
}
GATE_2Q_OPS: dict[str, type[Gate2Op]] = {"cx": CXOp, "cz": CZOp, "swap": SwapOp}


def verify_linear(module: ModuleOp) -> None:
    """No-cloning as a verifier: every qubit SSA value is used at most once.

    (Zero uses is legal only as 'end of wire' — nothing further happens on
    that qubit.) IRDL already guarantees operand/result typing; this checks
    the linearity constraint IRDL can't express.
    """
    for op in module.walk():
        for res in op.results:
            if res.type == qubit and len(tuple(res.uses)) > 1:
                users = [use.operation.name for use in res.uses]
                raise ValueError(
                    f"qubit value {res} produced by {op.name} is used "
                    f"{len(users)} times ({', '.join(users)}); qubit values "
                    "are linear and may be consumed at most once"
                )
