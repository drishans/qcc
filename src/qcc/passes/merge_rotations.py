"""Merge same-axis rotation neighbors and drop no-op rotations.

rz(a)·rz(b) → rz(a+b) exactly (same for rx, ry, p). A merged angle of
~0 mod 2π erases the gate: R(2π) = -I is a global phase, and the whole
compiler works up to global phase.
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
from qcc.passes.cancel_inverses import sole_user


class MergeRotationPair(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: d.Rotation1Op, rewriter: PatternRewriter, /):
        nxt = sole_user(op.qout)
        if type(nxt) is not type(op):
            return
        assert isinstance(nxt, d.Rotation1Op)
        total = op.params[0] + nxt.params[0]
        if gates.is_zero_rotation(total):
            rewriter.replace_all_uses_with(nxt.qout, op.qin)
            rewriter.erase_op(nxt)
            rewriter.erase_op(op)
        else:
            merged = type(op)(op.qin, gates.normalize_angle(total))
            rewriter.replace_op(nxt, merged)
            rewriter.erase_op(op)


class DropZeroRotation(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: d.Rotation1Op, rewriter: PatternRewriter, /):
        if gates.is_zero_rotation(op.params[0]):
            rewriter.replace_all_uses_with(op.qout, op.qin)
            rewriter.erase_op(op)


class DropIdentityU3(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: d.U3Op, rewriter: PatternRewriter, /):
        if gates.is_identity_up_to_phase(op.matrix()):
            rewriter.replace_all_uses_with(op.qout, op.qin)
            rewriter.erase_op(op)


class MergeRotationsPass(ModulePass):
    name = "qcc-merge-rotations"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [DropZeroRotation(), DropIdentityU3(), MergeRotationPair()]
            )
        ).rewrite_module(op)
