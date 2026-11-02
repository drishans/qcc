import pytest

from qcc.cli import main
from qcc.frontend import parse_qasm3
from qcc.ir import Instr, Tape, extract
from qcc.target import (
    BUILTIN_TARGETS,
    Target,
    TargetError,
    lower_to_target,
)
from qcc.verify import equivalent


def _tape(body: str) -> Tape:
    source = (
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\n' + body
    )
    return extract(parse_qasm3(source))


@pytest.mark.parametrize("name", ["cx", "cz", "qir-base", "generic", "cudaq"])
def test_builtin_targets_are_self_consistent(name: str):
    target = BUILTIN_TARGETS[name]
    assert target.entangler in target.native_gates


def test_cx_and_cz_lowering_are_equivalent_and_native():
    original = _tape(
        "h q[0];\ncx q[0], q[1];\ncz q[1], q[2];\nswap q[0], q[2];\n"
    )
    for name in ("cx", "cz"):
        target = BUILTIN_TARGETS[name]
        lowered = lower_to_target(original, target)
        assert equivalent(original, lowered)
        assert {ins.name for ins in lowered.instrs} <= target.native_gates
        assert all(
            ins.name == target.entangler
            for ins in lowered.instrs
            if len(ins.qubits) == 2
        )


def test_qir_target_lowers_non_base_gates():
    original = _tape(
        "sx q[0];\np(0.2) q[1];\nu3(0.1,0.2,0.3) q[2];\n"
        "swap q[0], q[2];\n"
    )
    target = BUILTIN_TARGETS["qir-base"]
    lowered = lower_to_target(original, target)
    assert equivalent(original, lowered)
    assert {ins.name for ins in lowered.instrs} <= target.native_gates


def test_fixed_connectivity_rejects_nonadjacent_interaction():
    target = Target(
        "line-3",
        BUILTIN_TARGETS["cx"].native_gates,
        entangler="cx",
        num_qubits=3,
        coupling=frozenset({(0, 1), (1, 2)}),
    )
    target.validate(_tape("cx q[0], q[1];\n"))
    with pytest.raises(TargetError, match="routing is not implemented"):
        target.validate(_tape("cx q[0], q[2];\n"))


def test_target_estimate_uses_cost_and_parallel_duration():
    target = Target(
        "costed",
        frozenset({"h", "cx"}),
        entangler="cx",
        gate_costs={"h": 1, "cx": 10},
        gate_durations_ns={"h": 5, "cx": 200},
    )
    tape = Tape(
        2,
        0,
        [Instr("h", (0,)), Instr("h", (1,)), Instr("cx", (0, 1))],
    )
    estimate = target.estimate(tape)
    assert estimate.gate_count == 3
    assert estimate.two_qubit_count == 1
    assert estimate.weighted_cost == 12
    assert estimate.critical_path_ns == 205


def test_cli_lowers_to_named_target(tmp_path, capsys):
    source = tmp_path / "bell.qasm"
    source.write_text(
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'
        "h q[0];\ncx q[0], q[1];\n",
        encoding="utf-8",
    )
    assert main(["compile", str(source), "-O0", "--target", "cz", "--emit", "qasm3"]) == 0
    output = capsys.readouterr().out
    assert "cz q[0], q[1]" in output
    assert "cx q[0], q[1]" not in output


def test_cli_lists_targets(capsys):
    assert main(["targets"]) == 0
    output = capsys.readouterr().out
    assert "generic:" in output
    assert "qir-base:" in output
