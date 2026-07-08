"""Cancel adjacent adjoint pairs.

Value semantics makes "adjacent" precise: gate B is adjacent to gate A on
a wire iff B consumes the value A produced. No DAG scan, no commutation
table — just one def-use hop.
"""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from qcc import gates
from qcc.ir import dialect as d


def sole_user(value):
    uses = tuple(value.uses)
    return uses[0].operation if len(uses) == 1 else None


class CancelAdjoint1Q(RewritePattern):
    """h·h, x·x, …, s·sdg, sdg·s, t·tdg, tdg·t → nothing."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: d.Gate1Op, rewriter: PatternRewriter, /):
        partner = gates.ADJOINT_1Q.get(op.GATE_NAME)
        if partner is None:
            return
        nxt = sole_user(op.qout)
        if not isinstance(nxt, d.Gate1Op) or nxt.GATE_NAME != partner:
            return
        rewriter.replace_all_uses_with(nxt.qout, op.qin)
        rewriter.erase_op(nxt)
        rewriter.erase_op(op)


class CancelAdjoint2Q(RewritePattern):
    """cx·cx, cz·cz, swap·swap on the same wire pair → nothing.

    cx must repeat with the same orientation; cz and swap are symmetric,
    so the crossed pairing cancels too.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: d.Gate2Op, rewriter: PatternRewriter, /):
        n0, n1 = sole_user(op.qout0), sole_user(op.qout1)
        if n0 is None or n0 is not n1 or type(n0) is not type(op):
            return
        nxt = n0
        straight = nxt.qin0 is op.qout0 and nxt.qin1 is op.qout1
        crossed = nxt.qin0 is op.qout1 and nxt.qin1 is op.qout0
        if straight or (crossed and op.GATE_NAME in gates.SYMMETRIC_2Q):
            if straight:
                rewriter.replace_all_uses_with(nxt.qout0, op.qin0)
                rewriter.replace_all_uses_with(nxt.qout1, op.qin1)
            else:
                rewriter.replace_all_uses_with(nxt.qout0, op.qin1)
                rewriter.replace_all_uses_with(nxt.qout1, op.qin0)
            rewriter.erase_op(nxt)
            rewriter.erase_op(op)


class CancelInversesPass(ModulePass):
    name = "qcc-cancel-inverses"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([CancelAdjoint1Q(), CancelAdjoint2Q()])
        ).rewrite_module(op)
