import math

import numpy as np
import pytest

from qcc.backend.sim import statevector
from qcc.frontend import QasmError, parse_qasm3
from qcc.ir import extract, metrics

HDR2 = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'
HDR3 = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\n'


def test_bell_amplitudes():
    src = HDR2 + "bit[2] c;\nh q[0];\ncx q[0], q[1];\nc[0] = measure q[0];\nc[1] = measure q[1];\n"
    sv = statevector(extract(parse_qasm3(src)))
    assert np.allclose(sv, [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])


def test_register_broadcast():
    src = HDR3 + "bit[3] c;\nh q;\nc = measure q;\n"
    t = extract(parse_qasm3(src))
    assert metrics(t)["by_gate"] == {"h": 3}
    assert sum(1 for i in t.instrs if i.name == "measure") == 3


def test_angle_constant_folding():
    src = HDR2 + "rz(pi/2) q[0];\nrz(2*pi - pi/2) q[1];\nrz(-pi**2/pi) q[0];\n"
    t = extract(parse_qasm3(src))
    assert abs(t.instrs[0].params[0] - math.pi / 2) < 1e-12
    assert abs(t.instrs[1].params[0] - (2 * math.pi - math.pi / 2)) < 1e-12
    assert abs(t.instrs[2].params[0] + math.pi) < 1e-12


def test_ccx_truth_table():
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                src = HDR3
                for bit, wire in ((a, 0), (b, 1), (c, 2)):
                    if bit:
                        src += f"x q[{wire}];\n"
                src += "ccx q[0], q[1], q[2];\n"
                sv = statevector(extract(parse_qasm3(src)))
                expect = (a << 2) | (b << 1) | (c ^ (a & b))
                assert int(np.argmax(np.abs(sv))) == expect
                assert abs(abs(sv[expect]) - 1) < 1e-9


def _controlled_matrix(gate: str, theta: float) -> np.ndarray:
    cols = []
    for k in range(4):
        src = HDR2
        if k & 2:
            src += "x q[0];\n"
        if k & 1:
            src += "x q[1];\n"
        src += f"{gate}({theta}) q[0], q[1];\n"
        cols.append(statevector(extract(parse_qasm3(src))))
    return np.column_stack(cols)


def test_cp_and_crz_expansions():
    th = 0.7345
    assert np.allclose(
        _controlled_matrix("cp", th), np.diag([1, 1, 1, np.exp(1j * th)]), atol=1e-9
    )
    assert np.allclose(
        _controlled_matrix("crz", th),
        np.diag([1, 1, np.exp(-1j * th / 2), np.exp(1j * th / 2)]),
        atol=1e-9,
    )


def test_u_aliases_and_u2():
    src = HDR2 + "U(0.1, 0.2, 0.3) q[0];\nu2(0.2, 0.3) q[1];\n"
    t = extract(parse_qasm3(src))
    assert t.instrs[0].name == "u3"
    assert t.instrs[1].name == "u3"
    assert abs(t.instrs[1].params[0] - math.pi / 2) < 1e-12


@pytest.mark.parametrize(
    "src,msg",
    [
        (HDR2 + "ch q[0], q[1];\n", "unsupported gate"),
        (HDR2 + "ctrl @ x q[0], q[1];\n", "modifiers"),
        (HDR2 + "h q[5];\n", "out of range"),
        (HDR2 + "cx q[0], q[0];\n", "duplicate qubit"),
        (HDR2 + "rz(x) q[0];\n", "constant"),
        ('OPENQASM 3.0;\ninclude "other.inc";\n', "unsupported include"),
        (HDR2 + "bit[2] c;\nc[0] = measure q;\n", "size mismatch"),
    ],
)
def test_clear_errors(src, msg):
    with pytest.raises(QasmError, match=msg):
        parse_qasm3(src)


def test_gate_definition_expansion():
    # the shape qiskit's exporter emits for rzz
    th = 0.6182
    src = (
        'OPENQASM 3.0;\ninclude "stdgates.inc";\n'
        "gate rzz(p0) _gate_q_0, _gate_q_1 {\n"
        "  cx _gate_q_0, _gate_q_1;\n  rz(p0) _gate_q_1;\n  cx _gate_q_0, _gate_q_1;\n"
        "}\nqubit[2] q;\n"
    )
    cols = []
    for k in range(4):
        s = src
        if k & 2:
            s += "x q[0];\n"
        if k & 1:
            s += "x q[1];\n"
        s += f"rzz({th}) q[0], q[1];\n"
        cols.append(statevector(extract(parse_qasm3(s))))
    u = np.column_stack(cols)
    e = np.exp
    expect = np.diag([e(-1j * th / 2), e(1j * th / 2), e(1j * th / 2), e(-1j * th / 2)])
    assert np.allclose(u, expect, atol=1e-9)


def test_qiskit_qasm3_import():
    qiskit = pytest.importorskip("qiskit")
    from qiskit import qasm3
    from qiskit.circuit.library import efficient_su2

    qc = efficient_su2(3, reps=1).assign_parameters(
        [0.1 * i for i in range(12)]
    )
    m = parse_qasm3(qasm3.dumps(qc.decompose()))
    assert metrics(extract(m))["gates"] > 0
