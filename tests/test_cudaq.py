import random

import numpy as np
import pytest

from conftest import random_circuit

from qcc.backend.sim import statevector
from qcc.frontend import parse_qasm3
from qcc.ir import extract
from qcc.passes import optimize

cudaq = pytest.importorskip("cudaq")

HDR = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'


@pytest.fixture(scope="module")
def cpu_target():
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def _bit_reversal_perm(n: int) -> list[int]:
    return [int(format(i, f"0{n}b")[::-1], 2) for i in range(2**n)]


def test_cudaq_state_matches_reference(cpu_target, rng: random.Random):
    from qcc.backend.cudaq_backend import build_kernel

    for _ in range(3):
        module = random_circuit(rng, 4, 50).finish()
        optimize(module)
        tape = extract(module)
        state = np.array(cudaq.get_state(build_kernel(tape)))
        reference = statevector(tape)
        fidelity = abs(np.vdot(state[_bit_reversal_perm(tape.n_qubits)], reference))
        assert abs(fidelity - 1) < 1e-6


def test_cudaq_bell_counts(cpu_target):
    from qcc.backend.cudaq_backend import sample

    source = (
        HDR
        + "bit[2] c;\nh q[0];\ncx q[0], q[1];\n"
        + "c[0] = measure q[0];\nc[1] = measure q[1];\n"
    )
    counts = sample(extract(parse_qasm3(source)), shots=1000)
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 1000
