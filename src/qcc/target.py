"""Hardware target descriptions and native-gate lowering.

The optimizer works on a target-independent logical gate set. A ``Target``
describes the native basis, topology, cost data, and mobility assumptions for
the next compilation layer. Lowering currently supports CX- and CZ-native
targets. Fixed nonadjacent interactions fail explicitly until routing lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from collections.abc import Mapping

from xdsl.dialects.builtin import ModuleOp

from qcc import gates
from qcc.ir import (
    Instr,
    Tape,
    extract,
    replace_main_from_tape,
    verify_linear,
)

_NONUNITARY = frozenset({"barrier", "measure"})
_KNOWN_GATES = gates.GATES_1Q | gates.GATES_2Q


class TargetError(ValueError):
    """A circuit cannot be represented by the selected target."""


@dataclass(frozen=True)
class TargetEstimate:
    gate_count: int
    two_qubit_count: int
    weighted_cost: float
    critical_path_ns: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "gate_count": self.gate_count,
            "two_qubit_count": self.two_qubit_count,
            "weighted_cost": self.weighted_cost,
            "critical_path_ns": self.critical_path_ns,
        }


@dataclass(frozen=True)
class Target:
    """Compilation target data, independent of any backend SDK.

    ``coupling=None`` means all-to-all connectivity. Undirected couplings are
    normalized, while directed couplings preserve edge orientation. Costs are
    dimensionless weights. Durations are optional nanosecond estimates and are
    reported only when every unitary gate in a circuit has one.
    """

    name: str
    native_gates: frozenset[str]
    entangler: str
    num_qubits: int | None = None
    coupling: frozenset[tuple[int, int]] | None = None
    directed_coupling: bool = False
    gate_costs: Mapping[str, float] = field(default_factory=dict)
    gate_durations_ns: Mapping[str, float] = field(default_factory=dict)
    movable_qubits: bool = False
    movable_interactions: bool = False

    def __post_init__(self) -> None:
        native = frozenset(self.native_gates)
        unknown = native - _KNOWN_GATES
        if unknown:
            raise TargetError(f"target {self.name!r} has unknown gates: {sorted(unknown)}")
        if self.entangler not in ("cx", "cz"):
            raise TargetError("current target lowering supports cx or cz entanglers")
        if self.entangler not in native:
            raise TargetError(
                f"target {self.name!r} does not include entangler {self.entangler!r}"
            )
        if self.num_qubits is not None and self.num_qubits <= 0:
            raise TargetError("num_qubits must be positive")

        coupling = self.coupling
        if coupling is not None:
            if self.num_qubits is None:
                raise TargetError("a fixed coupling requires num_qubits")
            normalized: set[tuple[int, int]] = set()
            for a, b in coupling:
                if a == b:
                    raise TargetError("coupling edges must connect distinct qubits")
                if min(a, b) < 0 or max(a, b) >= self.num_qubits:
                    raise TargetError(f"coupling edge {(a, b)} is outside the target")
                normalized.add((a, b) if self.directed_coupling else tuple(sorted((a, b))))
            coupling = frozenset(normalized)

        costs = self._validated_numbers(self.gate_costs, "cost")
        durations = self._validated_numbers(self.gate_durations_ns, "duration")
        object.__setattr__(self, "native_gates", native)
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "gate_costs", MappingProxyType(costs))
        object.__setattr__(self, "gate_durations_ns", MappingProxyType(durations))

    def _validated_numbers(
        self, values: Mapping[str, float], label: str
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        allowed = self.native_gates | {"measure"}
        for gate, value in values.items():
            if gate not in allowed:
                raise TargetError(f"{label} supplied for non-native gate {gate!r}")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise TargetError(f"{label} for {gate!r} must be finite and nonnegative")
            result[gate] = number
        return result

    def supports_pair(self, q0: int, q1: int) -> bool:
        if self.num_qubits is not None and max(q0, q1) >= self.num_qubits:
            return False
        if self.coupling is None:
            return True
        edge = (q0, q1) if self.directed_coupling else tuple(sorted((q0, q1)))
        return edge in self.coupling

    def validate(self, tape: Tape) -> None:
        if self.num_qubits is not None and tape.n_qubits > self.num_qubits:
            raise TargetError(
                f"circuit needs {tape.n_qubits} qubits, target {self.name!r} has "
                f"{self.num_qubits}"
            )
        for ins in tape.instrs:
            if ins.name in _NONUNITARY:
                continue
            if ins.name not in self.native_gates:
                raise TargetError(
                    f"gate {ins.name!r} is not native to target {self.name!r}"
                )
            expected = 1 if ins.name in gates.GATES_1Q else 2
            if len(ins.qubits) != expected:
                raise TargetError(f"gate {ins.name!r} has invalid arity")
            if expected == 2 and not self.supports_pair(*ins.qubits):
                raise TargetError(
                    f"interaction {ins.qubits} is not connected on target "
                    f"{self.name!r}; routing is not implemented yet"
                )

    def estimate(self, tape: Tape) -> TargetEstimate:
        self.validate(tape)
        unitary = [ins for ins in tape.instrs if ins.name not in _NONUNITARY]
        weighted = sum(self.gate_costs.get(ins.name, 1.0) for ins in unitary)
        has_durations = all(ins.name in self.gate_durations_ns for ins in unitary)
        critical: float | None = None
        if has_durations:
            ready = [0.0] * tape.n_qubits
            for ins in tape.instrs:
                if ins.name == "barrier":
                    boundary = max((ready[q] for q in ins.qubits), default=0.0)
                    for q in ins.qubits:
                        ready[q] = boundary
                elif ins.name != "measure":
                    end = max(ready[q] for q in ins.qubits)
                    end += self.gate_durations_ns[ins.name]
                    for q in ins.qubits:
                        ready[q] = end
            critical = max(ready, default=0.0)
        return TargetEstimate(
            gate_count=len(unitary),
            two_qubit_count=sum(len(ins.qubits) == 2 for ins in unitary),
            weighted_cost=weighted,
            critical_path_ns=critical,
        )


def _lower_1q(ins: Instr, target: Target) -> list[Instr]:
    if ins.name in target.native_gates:
        return [ins]
    matrix = gates.matrix_1q(ins.name, ins.params)
    if gates.is_identity_up_to_phase(matrix):
        return []
    theta, phi, lam = gates.zyz_angles(matrix)
    if "u3" in target.native_gates:
        return [Instr("u3", ins.qubits, (theta, phi, lam))]
    if {"rz", "ry"}.issubset(target.native_gates):
        result: list[Instr] = []
        for gate, angle in (("rz", lam), ("ry", theta), ("rz", phi)):
            if not gates.is_zero_rotation(angle):
                result.append(Instr(gate, ins.qubits, (angle,)))
        return result
    raise TargetError(
        f"target {target.name!r} cannot synthesize one-qubit gate {ins.name!r}; "
        "provide u3 or both rz and ry"
    )


def _lower_entangler(ins: Instr, target: Target) -> list[Instr]:
    if ins.name in target.native_gates:
        return [ins]
    q0, q1 = ins.qubits
    if ins.name == "swap":
        result: list[Instr] = []
        for control, target_wire in ((q0, q1), (q1, q0), (q0, q1)):
            result.extend(_lower_entangler(Instr("cx", (control, target_wire)), target))
        return result
    if ins.name not in ("cx", "cz"):
        raise TargetError(f"no lowering rule for two-qubit gate {ins.name!r}")

    other = "cz" if ins.name == "cx" else "cx"
    if other not in target.native_gates:
        raise TargetError(
            f"target {target.name!r} cannot lower {ins.name!r} to its native basis"
        )
    hadamard = Instr("h", (q1,))
    return [
        *_lower_1q(hadamard, target),
        Instr(other, (q0, q1)),
        *_lower_1q(hadamard, target),
    ]


def lower_to_target(tape: Tape, target: Target) -> Tape:
    """Lower a logical tape into ``target`` without changing wire placement."""
    lowered: list[Instr] = []
    for ins in tape.instrs:
        if ins.name in _NONUNITARY:
            lowered.append(ins)
        elif len(ins.qubits) == 1:
            lowered.extend(_lower_1q(ins, target))
        else:
            lowered.extend(_lower_entangler(ins, target))
    result = Tape(tape.n_qubits, tape.n_cbits, lowered)
    target.validate(result)
    return result


def lower_module_to_target(module: ModuleOp, target: Target) -> ModuleOp:
    """Lower ``@main`` atomically and preserve valid linear qubit SSA."""
    original = extract(module)
    lowered = lower_to_target(original, target)
    if lowered.instrs != original.instrs:
        replace_main_from_tape(module, lowered)
    module.verify()
    verify_linear(module)
    return module


_ALL_1Q = gates.GATES_1Q
BUILTIN_TARGETS: Mapping[str, Target] = MappingProxyType(
    {
        "generic": Target(
            "generic", _ALL_1Q | gates.GATES_2Q, entangler="cx"
        ),
        "cx": Target("cx", _ALL_1Q | {"cx"}, entangler="cx"),
        "cz": Target("cz", _ALL_1Q | {"cz"}, entangler="cz"),
        "qir-base": Target(
            "qir-base",
            frozenset(
                {
                    "h",
                    "x",
                    "y",
                    "z",
                    "s",
                    "sdg",
                    "t",
                    "tdg",
                    "rx",
                    "ry",
                    "rz",
                    "cx",
                    "cz",
                }
            ),
            entangler="cx",
        ),
        "cudaq": Target(
            "cudaq", _ALL_1Q | gates.GATES_2Q, entangler="cx"
        ),
    }
)


def get_target(name: str) -> Target:
    try:
        return BUILTIN_TARGETS[name]
    except KeyError as exc:
        raise TargetError(f"unknown target {name!r}") from exc
