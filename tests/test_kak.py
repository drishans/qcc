import math

import numpy as np
from qiskit.synthesis import two_qubit_cnot_decompose

from qcc import gates
from qcc.frontend import parse_qasm3
from qcc.ir import CircuitBuilder, Instr, Tape, extract, metrics, verify_linear
from qcc.passes import optimize
from qcc.passes.kak_resynthesis import collect_pair_blocks, resynthesize_tape
from qcc.synthesis import (
    KAKDecomposition,
    kak_decomposition,
    num_cx_for_kak,
    sequence_matrix,
    synthesize_two_qubit,
    unitary_fidelity,
)
from qcc.verify import fidelity


def _haar(n: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(raw)
    return q @ np.diag(np.diag(r) / np.abs(np.diag(r)))


def _interaction(x: float, y: float, z: float) -> np.ndarray:
    return KAKDecomposition(
        1,
        (np.eye(2), np.eye(2)),
        (x, y, z),
        (np.eye(2), np.eye(2)),
    ).matrix()


def _locally_dressed(
    interaction: tuple[float, float, float], rng: np.random.Generator
) -> np.ndarray:
    before = (_haar(2, rng), _haar(2, rng))
    after = (_haar(2, rng), _haar(2, rng))
    return KAKDecomposition(1, before, interaction, after).matrix()


def test_known_weyl_points_have_optimal_cx_counts():
    known = {
        0: np.eye(4),
        1: gates.matrix_2q("cx"),
        2: _interaction(0.31, 0.17, 0.0),
        3: gates.matrix_2q("swap"),
    }
    for expected, unitary in known.items():
        kak = kak_decomposition(unitary)
        result = synthesize_two_qubit(unitary)
        assert num_cx_for_kak(kak) == expected
        assert result.cx_count == expected
        assert two_qubit_cnot_decompose.num_basis_gates(unitary) == expected
        assert 1 - result.fidelity < 1e-12


def test_random_haar_u4_reconstructs_with_three_cx():
    rng = np.random.default_rng(20260825)
    for _ in range(100):
        unitary = _haar(4, rng)
        kak = kak_decomposition(unitary)
        result = synthesize_two_qubit(unitary)
        assert 0 <= abs(kak.interaction[2]) <= kak.interaction[1] + 1e-9
        assert kak.interaction[1] <= kak.interaction[0] + 1e-9
        assert kak.interaction[0] <= math.pi / 4 + 1e-9
        assert result.cx_count == 3
        assert 1 - unitary_fidelity(kak.matrix(), unitary) < 1e-12
        assert 1 - result.fidelity < 1e-12


def test_random_dressed_zero_one_two_and_three_cx_classes():
    rng = np.random.default_rng(17)
    classes = {
        0: (0.0, 0.0, 0.0),
        1: (math.pi / 4, 0.0, 0.0),
        2: (0.63, 0.21, 0.0),
        3: (0.62, 0.21, -0.08),
    }
    for expected, xyz in classes.items():
        for _ in range(20):
            unitary = _locally_dressed(xyz, rng)
            result = synthesize_two_qubit(unitary)
            assert result.cx_count == expected
            assert two_qubit_cnot_decompose.num_basis_gates(unitary) == expected
            assert 1 - result.fidelity < 1e-12


def test_weyl_boundary_and_near_degenerate_cases_are_stable():
    points = [
        (math.pi / 4, math.pi / 4, math.pi / 4),
        (math.pi / 4, math.pi / 4, -math.pi / 4),
        (math.pi / 4, 0.2, -0.2),
        (math.pi / 4 - 1e-10, 1e-10, 0.0),
        (0.4, 0.4 - 1e-11, 1e-11),
        (0.4, 0.2, 1e-7),
    ]
    for xyz in points:
        unitary = _interaction(*xyz)
        kak = kak_decomposition(unitary)
        result = synthesize_two_qubit(unitary)
        assert 1 - unitary_fidelity(kak.matrix(), unitary) < 1e-12
        assert 1 - result.fidelity < 1e-8


def test_reversed_wire_sequence_preserves_operand_order():
    unitary = gates.matrix_2q("cx")
    result = synthesize_two_qubit(unitary, 3, 1)
    assert result.cx_count == 1
    assert any(ins.name == "cx" and ins.qubits == (3, 1) for ins in result.instrs)
    actual = sequence_matrix(result.instrs, (3, 1))
    assert 1 - unitary_fidelity(actual, unitary) < 1e-12


def test_collector_is_maximal_contiguous_and_respects_fences():
    tape = Tape(
        3,
        1,
        [
            Instr("h", (0,)),
            Instr("cx", (0, 1)),
            Instr("rz", (1,), (0.2,)),
            Instr("cz", (1, 0)),
            Instr("h", (2,)),
            Instr("cx", (0, 1)),
            Instr("barrier", (0, 1, 2)),
            Instr("cx", (0, 1)),
            Instr("measure", (0,), cbit=0),
        ],
    )
    blocks = collect_pair_blocks(tape)
    assert [(b.start, b.end, b.pair) for b in blocks] == [
        (0, 4, (0, 1)),
        (5, 6, (0, 1)),
        (7, 8, (0, 1)),
    ]


def test_cost_guard_reduces_entanglers_and_never_worsens_cost():
    original = Tape(
        2,
        0,
        [
            Instr("cx", (0, 1)),
            Instr("h", (0,)),
            Instr("cx", (1, 0)),
            Instr("t", (1,)),
            Instr("cz", (0, 1)),
            Instr("rx", (0,), (0.37,)),
            Instr("cx", (0, 1)),
        ],
    )
    optimized, stats = resynthesize_tape(original)
    assert stats.considered == 1 and stats.replaced == 1
    assert stats.two_qubit_after < stats.two_qubit_before
    assert metrics(optimized)["gates_2q"] <= metrics(original)["gates_2q"]


def test_o2_rebuilds_linear_ssa_and_is_idempotent():
    src = """OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
qubit[2] q;
cx q[0], q[1];
h q[0];
cx q[1], q[0];
t q[1];
cz q[0], q[1];
rx(0.37) q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"""
    module = parse_qasm3(src)
    before = extract(module)
    optimize(module, 2)
    verify_linear(module)
    once = extract(module)
    assert fidelity(before, once) > 1 - 1e-10
    assert metrics(once)["gates_2q"] < metrics(before)["gates_2q"]
    assert [i.cbit for i in once.instrs if i.name == "measure"] == [0, 1]
    optimize(module, 2)
    twice = extract(module)
    assert metrics(twice) == metrics(once)
    assert fidelity(once, twice) > 1 - 1e-10


def test_random_whole_circuits_remain_equivalent_and_cost_monotone():
    rng = np.random.default_rng(91)
    oneq = ("h", "x", "t", "sx")
    twoq = ("cx", "cz", "swap")
    for _ in range(25):
        builder = CircuitBuilder(5)
        for _ in range(8):
            pair = tuple(int(x) for x in rng.choice(5, size=2, replace=False))
            for _ in range(5):
                builder.gate(twoq[int(rng.integers(len(twoq)))], *pair)
                wire = pair[int(rng.integers(2))]
                builder.gate(oneq[int(rng.integers(len(oneq)))], wire)
        module = builder.finish()
        before = extract(module)
        optimize(module, 2)
        verify_linear(module)
        after = extract(module)
        assert fidelity(before, after) > 1 - 1e-8
        assert metrics(after)["gates_2q"] <= metrics(before)["gates_2q"]
