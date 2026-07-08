"""OpenQASM 3 → qcc dialect.

Parses with the reference `openqasm3` package and lowers a deliberately
tight subset: qubit/bit register declarations, stdgates calls with
compile-time-constant angles, user `gate` definitions (expanded inline,
the way qiskit's exporter emits `r`, `rzz` and friends), barrier, and
register measurement. Gates the dialect doesn't carry natively (ccx,
cswap, cp, crz, u2, u) are expanded here into the core set, so the
optimizer only ever sees one gate vocabulary. Everything else is a clear
error with a line number — this is a compiler for circuits, not a general
QASM3 interpreter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import openqasm3
from openqasm3 import ast
from xdsl.dialects.builtin import ModuleOp

from qcc.ir.build import CircuitBuilder

MAX_EXPANSION_DEPTH = 32


class QasmError(Exception):
    def __init__(self, msg: str, node: ast.QASMNode | None = None):
        line = getattr(getattr(node, "span", None), "start_line", None)
        super().__init__(f"line {line}: {msg}" if line else msg)


#: aliases → canonical gate names of the dialect / expansion table
ALIASES = {
    "U": "u3",
    "u": "u3",
    "cnot": "cx",
    "CX": "cx",
    "phase": "p",
    "u1": "p",
    "cphase": "cp",
    "cu1": "cp",
    "i": "id",
}

#: gates the builder takes directly: (n_qubits, n_params)
DIRECT = {
    "id": (1, 0),
    "h": (1, 0),
    "x": (1, 0),
    "y": (1, 0),
    "z": (1, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    "sx": (1, 0),
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "p": (1, 1),
    "u3": (1, 3),
    "cx": (2, 0),
    "cz": (2, 0),
    "swap": (2, 0),
}

#: gates expanded at the frontend: (n_qubits, n_params)
EXPANDED = {
    "u2": (1, 2),
    "ccx": (3, 0),
    "cswap": (3, 0),
    "cp": (2, 1),
    "crz": (2, 1),
}

CONSTANTS = {
    "pi": math.pi,
    "π": math.pi,
    "tau": math.tau,
    "τ": math.tau,
    "euler": math.e,
    "ℇ": math.e,
}

_EMPTY: Mapping[str, float] = {}


def _const_eval(expr: ast.Expression, env: Mapping[str, float] = _EMPTY) -> float:
    """Evaluate a compile-time-constant angle expression (params in `env`)."""
    if isinstance(expr, ast.FloatLiteral | ast.IntegerLiteral):
        return float(expr.value)
    if isinstance(expr, ast.Identifier):
        if expr.name in env:
            return env[expr.name]
        if expr.name in CONSTANTS:
            return CONSTANTS[expr.name]
        raise QasmError(f"angle must be a constant, got identifier {expr.name!r}", expr)
    if isinstance(expr, ast.UnaryExpression):
        val = _const_eval(expr.expression, env)
        if expr.op.name == "-":
            return -val
        raise QasmError(f"unsupported unary operator {expr.op.name!r}", expr)
    if isinstance(expr, ast.BinaryExpression):
        lhs, rhs = _const_eval(expr.lhs, env), _const_eval(expr.rhs, env)
        match expr.op.name:
            case "+":
                return lhs + rhs
            case "-":
                return lhs - rhs
            case "*":
                return lhs * rhs
            case "/":
                return lhs / rhs
            case "**":
                return lhs**rhs
        raise QasmError(f"unsupported operator {expr.op.name!r}", expr)
    raise QasmError(f"unsupported angle expression {type(expr).__name__}", expr)


class _Lowerer:
    def __init__(self) -> None:
        self.qregs: dict[str, tuple[int, int]] = {}  # name → (offset, size)
        self.cregs: dict[str, tuple[int, int]] = {}
        self.gate_defs: dict[str, ast.QuantumGateDefinition] = {}
        self.n_qubits = 0
        self.n_cbits = 0
        self.builder: CircuitBuilder | None = None
        self._depth = 0

    # ── register / formal-qubit resolution ─────────────────────────────

    def _index_of(
        self,
        node,
        regs: dict[str, tuple[int, int]],
        what: str,
        qscope: Mapping[str, int] | None = None,
    ) -> list[int]:
        """Resolve an Identifier / IndexedIdentifier to a list of indices."""
        if isinstance(node, ast.Identifier):
            if qscope is not None and node.name in qscope:
                return [qscope[node.name]]
            if node.name not in regs:
                raise QasmError(f"unknown {what} register {node.name!r}", node)
            off, size = regs[node.name]
            return list(range(off, off + size))
        if isinstance(node, ast.IndexedIdentifier):
            name = node.name.name
            if name not in regs:
                raise QasmError(f"unknown {what} register {name!r}", node)
            off, size = regs[name]
            if len(node.indices) != 1 or len(node.indices[0]) != 1:
                raise QasmError("only single-index subscripts are supported", node)
            idx = node.indices[0][0]
            if not isinstance(idx, ast.IntegerLiteral):
                raise QasmError("subscripts must be integer literals", node)
            if not 0 <= idx.value < size:
                raise QasmError(
                    f"index {idx.value} out of range for {name}[{size}]", node
                )
            return [off + idx.value]
        raise QasmError(f"unsupported {what} reference {type(node).__name__}", node)

    def _broadcast(self, wire_lists: list[list[int]], node) -> list[tuple[int, ...]]:
        """QASM3 gate broadcasting: full registers apply elementwise."""
        n = max(len(ws) for ws in wire_lists)
        for ws in wire_lists:
            if len(ws) not in (1, n):
                raise QasmError("mismatched register sizes in gate broadcast", node)
        return [
            tuple(ws[0] if len(ws) == 1 else ws[i] for ws in wire_lists)
            for i in range(n)
        ]

    # ── gate lowering (incl. expansions) ───────────────────────────────

    def _play(self, name: str, wires: tuple[int, ...], params: tuple[float, ...]):
        b = self.builder
        assert b is not None
        if name in DIRECT:
            if name != "id":
                b.gate(name, *wires, params=params)
        elif name == "u2":
            phi, lam = params
            b.gate("u3", *wires, params=(math.pi / 2, phi, lam))
        elif name == "cp":
            (theta,) = params
            a, c = wires
            b.gate("p", a, params=(theta / 2,))
            b.gate("cx", a, c)
            b.gate("p", c, params=(-theta / 2,))
            b.gate("cx", a, c)
            b.gate("p", c, params=(theta / 2,))
        elif name == "crz":
            (theta,) = params
            a, c = wires
            b.gate("rz", c, params=(theta / 2,))
            b.gate("cx", a, c)
            b.gate("rz", c, params=(-theta / 2,))
            b.gate("cx", a, c)
        elif name == "ccx":
            a, bq, c = wires
            for g, w in (
                ("h", (c,)), ("cx", (bq, c)), ("tdg", (c,)), ("cx", (a, c)),
                ("t", (c,)), ("cx", (bq, c)), ("tdg", (c,)), ("cx", (a, c)),
                ("t", (bq,)), ("t", (c,)), ("h", (c,)), ("cx", (a, bq)),
                ("t", (a,)), ("tdg", (bq,)), ("cx", (a, bq)),
            ):
                b.gate(g, *w)
        elif name == "cswap":
            a, bq, c = wires
            b.gate("cx", c, bq)
            self._play("ccx", (a, bq, c), ())
            b.gate("cx", c, bq)
        else:  # pragma: no cover — guarded by callers
            raise KeyError(name)

    def _expand_def(
        self, gd: ast.QuantumGateDefinition, wires: tuple[int, ...],
        params: tuple[float, ...],
    ) -> None:
        """Inline a user gate definition at a call site (macro expansion)."""
        if self._depth >= MAX_EXPANSION_DEPTH:
            raise QasmError(f"gate expansion deeper than {MAX_EXPANSION_DEPTH}", gd)
        env = {arg.name: v for arg, v in zip(gd.arguments, params)}
        qscope = {q.name: w for q, w in zip(gd.qubits, wires)}
        self._depth += 1
        try:
            for st in gd.body:
                match st:
                    case ast.QuantumGate():
                        self.gate(st, qscope=qscope, penv=env)
                    case ast.QuantumPhase(modifiers=mods):
                        if mods:
                            raise QasmError("controlled gphase is unsupported", st)
                    case ast.QuantumBarrier(qubits=qs):
                        assert self.builder is not None
                        wires_b: list[int] = []
                        for q in qs:
                            wires_b += self._index_of(q, self.qregs, "qubit", qscope)
                        self.builder.barrier(*(wires_b or qscope.values()))
                    case _:
                        raise QasmError(
                            f"unsupported statement {type(st).__name__} "
                            f"in gate definition {gd.name.name!r}", st,
                        )
        finally:
            self._depth -= 1

    def gate(
        self,
        st: ast.QuantumGate,
        qscope: Mapping[str, int] | None = None,
        penv: Mapping[str, float] = _EMPTY,
    ) -> None:
        if st.modifiers:
            raise QasmError("gate modifiers (ctrl/negctrl/inv/pow) are unsupported", st)
        raw = st.name.name

        gd = self.gate_defs.get(raw)
        if gd is not None:
            nq, np_ = len(gd.qubits), len(gd.arguments)
        else:
            name = ALIASES.get(raw, raw)
            if name not in DIRECT and name not in EXPANDED:
                raise QasmError(f"unsupported gate {raw!r}", st)
            nq, np_ = DIRECT.get(name) or EXPANDED[name]

        if len(st.qubits) != nq:
            raise QasmError(
                f"{raw} expects {nq} qubit argument(s), got {len(st.qubits)}", st
            )
        if len(st.arguments) != np_:
            raise QasmError(
                f"{raw} expects {np_} parameter(s), got {len(st.arguments)}", st
            )
        params = tuple(_const_eval(a, penv) for a in st.arguments)
        wire_lists = [
            self._index_of(q, self.qregs, "qubit", qscope) for q in st.qubits
        ]
        self._ensure_builder()
        for wires in self._broadcast(wire_lists, st):
            if len(set(wires)) != len(wires):
                raise QasmError("duplicate qubit in gate operands", st)
            if gd is not None:
                self._expand_def(gd, wires, params)
            else:
                self._play(ALIASES.get(raw, raw), wires, params)

    # ── statement dispatch ─────────────────────────────────────────────

    def statement(self, st: ast.Statement) -> None:
        match st:
            case ast.Include(filename=f):
                if f not in ("stdgates.inc",):
                    raise QasmError(f"unsupported include {f!r}", st)
            case ast.QubitDeclaration(qubit=ident, size=size):
                if self.builder is not None:
                    raise QasmError("qubit declarations must precede gates", st)
                n = 1 if size is None else int(_const_eval(size))
                self.qregs[ident.name] = (self.n_qubits, n)
                self.n_qubits += n
            case ast.ClassicalDeclaration(type=typ, identifier=ident, init_expression=init):
                if not isinstance(typ, ast.BitType):
                    raise QasmError(
                        f"unsupported classical type {type(typ).__name__}", st
                    )
                if init is not None:
                    raise QasmError("initialized bit registers are unsupported", st)
                n = 1 if typ.size is None else int(_const_eval(typ.size))
                self.cregs[ident.name] = (self.n_cbits, n)
                self.n_cbits += n
            case ast.QuantumGateDefinition():
                self.gate_defs[st.name.name] = st
            case ast.QuantumGate():
                self.gate(st)
            case ast.QuantumBarrier(qubits=qs):
                self._ensure_builder()
                assert self.builder is not None
                if qs:
                    wires: list[int] = []
                    for q in qs:
                        wires += self._index_of(q, self.qregs, "qubit")
                    self.builder.barrier(*wires)
                else:
                    self.builder.barrier()
            case ast.QuantumMeasurementStatement(measure=meas, target=target):
                if target is None:
                    raise QasmError("measure without classical target", st)
                self._ensure_builder()
                assert self.builder is not None
                qs = self._index_of(meas.qubit, self.qregs, "qubit")
                cs = self._index_of(target, self.cregs, "bit")
                if len(qs) != len(cs):
                    raise QasmError("measure register size mismatch", st)
                for q, c in zip(qs, cs):
                    self.builder.measure(q, c)
            case ast.QuantumPhase(modifiers=mods):
                if mods:
                    raise QasmError("controlled gphase is unsupported", st)
                # bare gphase only shifts global phase — dropped (the whole
                # compiler works up to global phase)
            case _:
                raise QasmError(
                    f"unsupported statement {type(st).__name__}", st
                )

    def _ensure_builder(self) -> None:
        if self.builder is None:
            if self.n_qubits == 0:
                raise QasmError("no qubits declared before first gate")
            self.builder = CircuitBuilder(self.n_qubits)


def parse_qasm3(source: str) -> ModuleOp:
    """OpenQASM 3 source text → qcc IR module."""
    try:
        program = openqasm3.parse(source)
    except Exception as e:  # antlr wraps errors in its own types
        raise QasmError(f"parse error: {e}") from e

    if program.version and not str(program.version).startswith("3"):
        raise QasmError(f"OPENQASM {program.version} is not OpenQASM 3")

    lo = _Lowerer()
    for st in program.statements:
        lo.statement(st)
    lo._ensure_builder()
    assert lo.builder is not None
    return lo.builder.finish()
