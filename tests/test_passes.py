from qcc.frontend import parse_qasm3
from qcc.ir import extract, metrics
from qcc.passes import optimize
from qcc.verify import equivalent

HDR = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'


def _opt_counts(body: str):
    src = HDR + body
    before = extract(parse_qasm3(src))
    m = optimize(parse_qasm3(src))
    after = extract(m)
    assert equivalent(before, after), body
    return metrics(after)


def test_self_inverse_pairs_cancel():
    for g in ("h", "x", "y", "z"):
        assert _opt_counts(f"{g} q[0];\n{g} q[0];\n")["gates"] == 0


def test_adjoint_pairs_cancel_both_orders():
    for a, b in (("s", "sdg"), ("sdg", "s"), ("t", "tdg"), ("tdg", "t")):
        assert _opt_counts(f"{a} q[0];\n{b} q[0];\n")["gates"] == 0


def test_cancellation_cascades():
    # x h h x  — inner pair exposes the outer pair
    assert _opt_counts("x q[0];\nh q[0];\nh q[0];\nx q[0];\n")["gates"] == 0


def test_2q_pairs():
    assert _opt_counts("cx q[0],q[1];\ncx q[0],q[1];\n")["gates"] == 0
    assert _opt_counts("cz q[0],q[1];\ncz q[1],q[0];\n")["gates"] == 0
    assert _opt_counts("swap q[0],q[1];\nswap q[1],q[0];\n")["gates"] == 0
    # reversed cx is NOT an inverse pair
    assert _opt_counts("cx q[0],q[1];\ncx q[1],q[0];\n")["gates"] == 2


def test_rotation_merge_and_zero_drop():
    c = _opt_counts("rz(0.3) q[0];\nrz(0.4) q[0];\n")
    assert c["by_gate"] == {"rz": 1}
    assert _opt_counts("rx(0.3) q[0];\nrx(-0.3) q[0];\n")["gates"] == 0
    assert _opt_counts("p(0) q[0];\n")["gates"] == 0


def test_fusion_collapses_1q_runs():
    c = _opt_counts("h q[0];\ns q[0];\nh q[0];\n")
    assert c["gates"] == 1
    # a long alternating run still fuses to one gate
    c = _opt_counts("h q[0];\nt q[0];\nh q[0];\nt q[0];\nh q[0];\nt q[0];\n")
    assert c["gates"] == 1


def test_fusion_recognizes_hidden_identity():
    # sx·sx = x, then x cancels x
    assert _opt_counts("sx q[0];\nsx q[0];\nx q[0];\n")["gates"] == 0


def test_barrier_blocks_optimization():
    c = _opt_counts("h q[0];\nbarrier q;\nh q[0];\n")
    assert c["gates"] == 2


def test_measure_blocks_fusion():
    src = "bit[2] c;\nh q[0];\nc[0] = measure q[0];\nh q[0];\n"
    m = optimize(parse_qasm3(HDR + src))
    assert metrics(extract(m))["gates"] == 2


def test_optimize_is_idempotent():
    src = HDR + "h q[0];\nt q[0];\ncx q[0],q[1];\nrz(0.5) q[1];\ncx q[0],q[1];\n"
    m = optimize(parse_qasm3(src))
    once = metrics(extract(m))
    m = optimize(m)
    assert metrics(extract(m)) == once
