"""Commutation-aware canonicalization: push commuting 1q gates rightward
through 2q gates so the cancellation and fusion passes see new neighbors.

Two facts do all the work:
- diagonal 1q gates (z, s, sdg, t, tdg, rz, p) commute with a CX at its
  *control*, and with a CZ at *either* wire (CZ is diagonal);
- X-axis 1q gates (x, rx, sx) commute with a CX at its *target*
  (they're polynomials in X, and CX conjugation fixes X on the target).

A push happens only when it provably pays: walking forward along the
wire through every 2q gate the pushed gate commutes with, there must be
a 1q gate to merge into (fusion picks it up) — or the 2q gate's twin
sits directly before, so the pair cancels once the 1q gate is out of
the way:

    cx a,b ; t a ; cx a,b   →   cx ; cx ; t   →   t

Unconditional pushing is a real trap — it strands rotations away from
their fusion partners and *regressed* the VQE benchmark 67 → 91 gates
before the guard was added. This is the lite version of Qiskit's
CommutativeCancellation.
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
from xdsl.rewriter import InsertPoint

from qcc.ir import dialect as d
from qcc.passes.cancel_inverses import sole_user

#: 1q gates diagonal in the Z basis
DIAGONAL = {"z", "s", "sdg", "t", "tdg", "rz", "p"}
#: 1q gates that are polynomials in X
X_AXIS = {"x", "rx", "sx"}


def _clone_gate(op: d.Gate1Op, qin) -> d.Gate1Op:
    """Rebuild the same 1q gate on a different input value."""
    if isinstance(op, d.Rotation1Op):
        return type(op)(qin, op.params[0])
    return type(op)(qin)


def _commutes_entering(diag: bool, xax: bool, twoq: d.Gate2Op, value) -> bool:
    """Does a diag/x-axis 1q gate commute with `twoq` entering at `value`?"""
    if isinstance(twoq, d.CZOp):
        return diag
    if isinstance(twoq, d.CXOp):
        return (diag and twoq.qin0 is value) or (xax and twoq.qin1 is value)
    return False


MAX_LOOKAHEAD = 32


def _push_pays(diag: bool, xax: bool, start: d.Gate2Op, value) -> bool:
    """Walk the wire forward through commuting 2q gates; a push pays iff
    the chain ends at a 1q gate the pushed gate can merge with."""
    twoq, val = start, value
    for _ in range(MAX_LOOKAHEAD):
        val = twoq.qout0 if twoq.qin0 is val else twoq.qout1
        user = sole_user(val)
        if isinstance(user, d.Gate1Op):
            return True
        if not isinstance(user, d.Gate2Op) or not _commutes_entering(
            diag, xax, user, val
        ):
            return False
        twoq = user
    return False


class PushThrough2Q(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: d.Gate1Op, rewriter: PatternRewriter, /):
        name = op.GATE_NAME
        diag, xax = name in DIAGONAL, name in X_AXIS
        if not (diag or xax):
            return
        nxt = sole_user(op.qout)
        if not isinstance(nxt, d.Gate2Op):
            return

        commutes = (
            isinstance(nxt, d.CZOp) and diag
            or isinstance(nxt, d.CXOp) and diag and nxt.qin0 is op.qout
            or isinstance(nxt, d.CXOp) and xax and nxt.qin1 is op.qout
        )
        if not commutes:
            return

        # ── guard: only push when it pays ──────────────────────────────
        merges = _push_pays(diag, xax, nxt, op.qout)

        prev = op.qin.owner
        cancels = False
        if isinstance(prev, d.Gate2Op) and type(prev) is type(nxt):
            if nxt.qin0 is op.qout:
                same_wire = prev.qout0 is op.qin
                other_adjacent = prev.qout1 is nxt.qin1
            else:
                same_wire = prev.qout1 is op.qin
                other_adjacent = prev.qout0 is nxt.qin0
            cancels = same_wire and other_adjacent

        if not (merges or cancels):
            return

        if nxt.qin0 is op.qout:
            new_2q = type(nxt)(op.qin, nxt.qin1)
            moved = _clone_gate(op, new_2q.qout0)
            other_out, other_new = nxt.qout1, new_2q.qout1
            this_out = nxt.qout0
        else:
            new_2q = type(nxt)(nxt.qin0, op.qin)
            moved = _clone_gate(op, new_2q.qout1)
            other_out, other_new = nxt.qout0, new_2q.qout0
            this_out = nxt.qout1

        # insert right before the 2q op: the other wire's operand may be
        # defined between `op` and `nxt`, so earlier would break dominance
        rewriter.insert_op(new_2q, InsertPoint.before(nxt))
        rewriter.insert_op(moved, InsertPoint.before(nxt))
        rewriter.replace_all_uses_with(this_out, moved.qout)
        rewriter.replace_all_uses_with(other_out, other_new)
        rewriter.erase_op(nxt)
        rewriter.erase_op(op)


class CommuteDiagonalsPass(ModulePass):
    name = "qcc-commute-push"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([PushThrough2Q()])
        ).rewrite_module(op)
