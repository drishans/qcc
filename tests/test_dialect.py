import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.parser import Parser

from qcc.ir import QCC, CircuitBuilder, extract, metrics, verify_linear
from qcc.ir import dialect as d


def _ctx() -> Context:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(QCC)
    return ctx


def build_bell():
    b = CircuitBuilder(2)
    b.gate("h", 0)
    b.gate("cx", 0, 1)
    b.measure(0, 0)
    b.measure(1, 1)
    return b.finish()


def test_print_parse_round_trip():
    m = build_bell()
    m.verify()
    m2 = Parser(_ctx(), str(m)).parse_module()
    m2.verify()
    assert str(m) == str(m2)
    assert metrics(extract(m)) == metrics(extract(m2))


def test_linearity_verifier_catches_cloning():
    b = CircuitBuilder(1)
    b.gate("h", 0)
    h = list(b.block.ops)[-1]
    # use the same qubit value twice — forbidden
    b.block.add_op(d.XOp(h.qout))
    b.block.add_op(d.YOp(h.qout))
    m = b.finish()
    with pytest.raises(ValueError, match="linear"):
        verify_linear(m)


def test_tape_and_metrics():
    m = build_bell()
    t = extract(m)
    assert t.n_qubits == 2 and t.n_cbits == 2
    names = [i.name for i in t.instrs]
    assert names == ["h", "cx", "measure", "measure"]
    mm = metrics(t)
    assert mm["gates"] == 2 and mm["gates_2q"] == 1 and mm["depth"] == 2


def test_two_qubit_gate_needs_distinct_wires():
    b = CircuitBuilder(2)
    with pytest.raises(ValueError, match="distinct"):
        b.gate("cx", 1, 1)
