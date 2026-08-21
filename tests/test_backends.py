import random

import numpy as np
import pytest

from conftest import random_circuit

from qcc.backend.qasm3_out import emit_qasm3
from qcc.backend.qir import emit_qir
from qcc.backend.sim import statevector
from qcc.frontend import parse_qasm3
from qcc.ir import extract
from qcc.passes import optimize
from qcc.verify import equivalent

HDR = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'


def test_qir_bell_structure():
    src = HDR + "bit[2] c;\nh q[0];\ncx q[0], q[1];\nc[0] = measure q[0];\nc[1] = measure q[1];\n"
    ir = emit_qir(extract(parse_qasm3(src)), name="bell")
    assert "__quantum__qis__h__body" in ir
    assert "__quantum__qis__cnot__body" in ir
    assert ir.count("__quantum__qis__mz__body") >= 2  # decl + calls
    assert '"required_num_qubits"="2"' in ir
    assert "entry_point" in ir


def test_qir_lowers_whole_gate_set():
    src = HDR + (
        "sx q[0];\np(0.3) q[0];\nu3(0.1,0.2,0.3) q[1];\nswap q[0], q[1];\n"
        "cz q[0], q[1];\nrz(0.5) q[0];\n"
    )
    m = parse_qasm3(src)
    ir = emit_qir(extract(m))
    # everything must resolve to base-profile intrinsics
    assert "u3" not in ir and "swap" not in ir.lower()
    assert "__quantum__qis__rx__body" in ir  # sx lowering
    assert ir.count("call void @__quantum__qis__cnot__body") == 3  # swap = 3 cx


def test_qir_u3_elides_zero_angles():
    # a u3 that is really one rotation must cost one QIR instruction
    src = HDR + "u3(0.5, 0.0, 0.0) q[0];\nu3(0.0, 0.3, 0.4) q[1];\n"
    ir = emit_qir(extract(parse_qasm3(src)))
    assert ir.count("call void @__quantum__qis__ry__body") == 1
    assert ir.count("call void @__quantum__qis__rz__body") == 2  # 0.3 and 0.4
    src = HDR + "u3(0.1, 0.2, 0.3) q[0];\n"
    ir = emit_qir(extract(parse_qasm3(src)))
    assert ir.count("call void @__quantum__qis__r") == 3  # full case unchanged


def test_qasm3_roundtrip_random(rng: random.Random):
    for _ in range(10):
        m = random_circuit(rng, 4, 60).finish()
        optimize(m)
        t = extract(m)
        rt = extract(parse_qasm3(emit_qasm3(t)))
        assert equivalent(t, rt)


# ── CUDA-Q (skipped automatically where cudaq isn't installed) ─────────

cudaq = pytest.importorskip("cudaq")


@pytest.fixture(scope="module")
def cpu_target():
    cudaq.set_target("qpp-cpu")  # deterministic, no GPU needed in CI
    yield
    cudaq.reset_target()


def _bit_reversal_perm(n: int) -> list[int]:
    return [int(format(i, f"0{n}b")[::-1], 2) for i in range(2**n)]


def test_cudaq_state_matches_reference(cpu_target, rng: random.Random):
    from qcc.backend.cudaq_backend import build_kernel

    for _ in range(3):
        m = random_circuit(rng, 4, 50).finish()
        optimize(m)
        t = extract(m)
        state = np.array(cudaq.get_state(build_kernel(t)))
        ref = statevector(t)
        # cudaq orders amplitudes with qubit 0 least significant; we use
        # qubit 0 most significant — compare through the bit reversal
        f = abs(np.vdot(state[_bit_reversal_perm(t.n_qubits)], ref))
        assert abs(f - 1) < 1e-6


def test_cudaq_bell_counts(cpu_target):
    from qcc.backend.cudaq_backend import sample

    src = HDR + "bit[2] c;\nh q[0];\ncx q[0], q[1];\nc[0] = measure q[0];\nc[1] = measure q[1];\n"
    counts = sample(extract(parse_qasm3(src)), shots=1000)
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 1000
