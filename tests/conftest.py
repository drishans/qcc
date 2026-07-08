import random

import pytest

from qcc.ir.build import CircuitBuilder

FIXED = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"]
ROT = ["rx", "ry", "rz", "p"]
TWOQ = ["cx", "cz", "swap"]


def random_circuit(rng: random.Random, n_qubits: int, n_gates: int,
                   barriers: bool = False) -> CircuitBuilder:
    b = CircuitBuilder(n_qubits)
    for i in range(n_gates):
        r = rng.random()
        if barriers and r > 0.97:
            b.barrier()
        elif r < 0.35:
            b.gate(rng.choice(FIXED), rng.randrange(n_qubits))
        elif r < 0.55:
            b.gate(
                rng.choice(ROT), rng.randrange(n_qubits),
                params=(rng.uniform(-3.2, 3.2),),
            )
        elif n_qubits >= 2:
            wa, wb = rng.sample(range(n_qubits), 2)
            b.gate(rng.choice(TWOQ), wa, wb)
    return b


@pytest.fixture
def rng():
    return random.Random(20260707)
