"""Benchmark circuit generators.

Every suite is generated in qiskit, then basis-translated (optimization
level 0 — rewrite rules only, no optimization) to a conservative gate set
that qiskit, pytket, and qcc all ingest natively from the same text:

    h x y z s sdg t tdg rx ry rz cx cz swap

That way all three compilers see the *identical* fully-expanded circuit;
nobody gets a head start from a smarter frontend expansion. No measures,
no barriers: the comparison is pure logical-circuit optimization.
"""

from __future__ import annotations

import random

import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import efficient_su2
from qiskit.synthesis import synth_qft_full

BASIS = ["h", "x", "y", "z", "s", "sdg", "t", "tdg",
         "rx", "ry", "rz", "cx", "cz", "swap"]


def to_basis(qc: QuantumCircuit) -> QuantumCircuit:
    out = transpile(qc, basis_gates=BASIS, optimization_level=0)
    names = {i.operation.name for i in out.data}
    assert names <= set(BASIS), f"basis translation leaked {names - set(BASIS)}"
    return out


def ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def qft(n: int) -> QuantumCircuit:
    return synth_qft_full(n, do_swaps=True)


def qaoa_maxcut(n: int, p: int, seed: int) -> QuantumCircuit:
    g = nx.random_regular_graph(3, n, seed=seed)
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(p):
        gamma, beta = rng.uniform(0, 3.14), rng.uniform(0, 3.14)
        for a, b in g.edges:
            qc.rzz(gamma, a, b)
        qc.rx(2 * beta, range(n))
    return qc


def random_cliffordt(n: int, n_gates: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    one_q = ["h", "x", "z", "s", "sdg", "t", "tdg"]
    qc = QuantumCircuit(n)
    for _ in range(n_gates):
        if rng.random() < 0.7:
            getattr(qc, rng.choice(one_q))(rng.randrange(n))
        else:
            a, b = rng.sample(range(n), 2)
            getattr(qc, rng.choice(["cx", "cz"]))(a, b)
    return qc


def vqe_su2(n: int, reps: int, seed: int) -> QuantumCircuit:
    qc = efficient_su2(n, reps=reps)
    rng = random.Random(seed)
    return qc.assign_parameters(
        [rng.uniform(-3.14, 3.14) for _ in qc.parameters]
    ).decompose()


def adder(bits: int) -> QuantumCircuit:
    """Ripple-carry adder (Cuccaro) — Toffoli-heavy arithmetic."""
    from qiskit.synthesis import adder_ripple_c04

    return adder_ripple_c04(bits)


def suite() -> list[tuple[str, str, QuantumCircuit]]:
    """(suite, instance, circuit) triples, already basis-translated."""
    out: list[tuple[str, str, QuantumCircuit]] = []
    out.append(("ghz", "n12", ghz(12)))
    out.append(("qft", "n6", qft(6)))
    out.append(("qft", "n10", qft(10)))
    for seed in (11, 12):
        out.append(("qaoa", f"n12p2s{seed}", qaoa_maxcut(12, 2, seed)))
    for seed in (1, 2, 3):
        out.append(("cliffordt", f"n8g400s{seed}", random_cliffordt(8, 400, seed)))
    out.append(("vqe", "n10r3", vqe_su2(10, 3, seed=5)))
    out.append(("adder", "b4", adder(4)))
    return [(s, i, to_basis(qc)) for s, i, qc in out]
