"""Collect maximal contiguous pair blocks and resynthesize them with KAK.

A block contains only unitary operations on one logical-qubit pair and at
least one 2q gate. Measurement, barriers, disjoint work, or a gate involving
a third qubit ends the block. Replacement is atomic: the optimized tape is
rebuilt as fresh SSA, then the old ``@main`` function is swapped out once.

The acceptance order is lexicographic ``(2q gates, total gates, depth)``.
KAK therefore cannot make the expensive metric worse merely to canonicalize
a block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass

from qcc.ir import Instr, Tape, extract, replace_main_from_tape, verify_linear
from qcc.synthesis import sequence_matrix, synthesize_two_qubit


@dataclass(frozen=True)
class PairBlock:
    start: int
    end: int
    pair: tuple[int, int]


@dataclass(frozen=True)
class KAKStats:
    considered: int = 0
    replaced: int = 0
    two_qubit_before: int = 0
    two_qubit_after: int = 0


def _is_unitary_1q(ins: Instr, pair: tuple[int, int]) -> bool:
    return (
        ins.name not in ("barrier", "measure")
        and len(ins.qubits) == 1
        and ins.qubits[0] in pair
    )


def _is_pair_2q(ins: Instr, pair: tuple[int, int]) -> bool:
    return len(ins.qubits) == 2 and tuple(sorted(ins.qubits)) == pair


def collect_pair_blocks(tape: Tape) -> tuple[PairBlock, ...]:
    """Find non-overlapping maximal contiguous blocks on each wire pair."""
    instrs = tape.instrs
    blocks: list[PairBlock] = []
    floor = 0
    i = 0
    while i < len(instrs):
        ins = instrs[i]
        if len(ins.qubits) != 2 or ins.name == "barrier":
            i += 1
            continue
        pair = tuple(sorted(ins.qubits))
        start = i
        while start > floor and _is_unitary_1q(instrs[start - 1], pair):
            start -= 1
        end = i + 1
        while end < len(instrs):
            nxt = instrs[end]
            if _is_unitary_1q(nxt, pair) or _is_pair_2q(nxt, pair):
                end += 1
            else:
                break
        blocks.append(PairBlock(start, end, pair))
        floor = end
        i = end
    return tuple(blocks)


def _cost(instrs: list[Instr] | tuple[Instr, ...], pair: tuple[int, int]) -> tuple[int, int, int]:
    depth = {pair[0]: 0, pair[1]: 0}
    twoq = 0
    for ins in instrs:
        if len(ins.qubits) == 2:
            twoq += 1
        layer = max(depth[q] for q in ins.qubits) + 1
        for q in ins.qubits:
            depth[q] = layer
    return twoq, len(instrs), max(depth.values(), default=0)


def resynthesize_tape(tape: Tape) -> tuple[Tape, KAKStats]:
    """Return a cost-monotone KAK-resynthesized tape and its statistics."""
    blocks = collect_pair_blocks(tape)
    if not blocks:
        return tape, KAKStats()
    out: list[Instr] = []
    cursor = 0
    replaced = 0
    before_2q = 0
    after_2q = 0
    for block in blocks:
        out.extend(tape.instrs[cursor : block.start])
        original = tape.instrs[block.start : block.end]
        original_cost = _cost(original, block.pair)
        before_2q += original_cost[0]
        try:
            candidate = synthesize_two_qubit(
                sequence_matrix(original, block.pair), *block.pair
            ).instrs
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            candidate = tuple(original)
        candidate_cost = _cost(candidate, block.pair)
        if candidate_cost < original_cost:
            out.extend(candidate)
            after_2q += candidate_cost[0]
            replaced += 1
        else:
            out.extend(original)
            after_2q += original_cost[0]
        cursor = block.end
    out.extend(tape.instrs[cursor:])
    return (
        Tape(tape.n_qubits, tape.n_cbits, out),
        KAKStats(len(blocks), replaced, before_2q, after_2q),
    )


class KAKResynthesisPass(ModulePass):
    name = "qcc-kak-resynthesis"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        tape, stats = resynthesize_tape(extract(op))
        if stats.replaced:
            replace_main_from_tape(op, tape)
            op.verify()
            verify_linear(op)
