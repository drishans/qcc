"""The optimization pipeline.

-O0 is the identity. -O1 iterates cheap local rewrites to a fixpoint. -O2
runs O1, then repeats cost-monotone KAK resynthesis plus O1 cleanup until the
tape stops changing. Strict block-cost improvement prevents oscillation.
"""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp

from qcc.ir import extract
from qcc.ir.dialect import verify_linear
from qcc.passes.cancel_inverses import CancelInversesPass
from qcc.passes.commute_diagonals import CommuteDiagonalsPass
from qcc.passes.fuse_1q import FuseSingleQubitPass
from qcc.passes.kak_resynthesis import KAKResynthesisPass
from qcc.passes.merge_rotations import MergeRotationsPass

MAX_ITERATIONS = 25
MAX_KAK_ITERATIONS = 8


def _n_ops(module: ModuleOp) -> int:
    return sum(1 for _ in module.walk())


def _run_o1(ctx: Context, module: ModuleOp) -> None:
    passes = [
        CancelInversesPass(),
        MergeRotationsPass(),
        CommuteDiagonalsPass(),
        CancelInversesPass(),
        MergeRotationsPass(),
        FuseSingleQubitPass(),
    ]
    before = _n_ops(module)
    for _ in range(MAX_ITERATIONS):
        for p in passes:
            p.apply(ctx, module)
        after = _n_ops(module)
        if after == before:
            break
        before = after


def optimize(module: ModuleOp, level: int = 1) -> ModuleOp:
    """Run the -O<level> pipeline in place (and return the module)."""
    if level <= 0:
        return module
    ctx = Context()
    _run_o1(ctx, module)
    if level >= 2:
        for _ in range(MAX_KAK_ITERATIONS):
            before = tuple(extract(module).instrs)
            KAKResynthesisPass().apply(ctx, module)
            _run_o1(ctx, module)
            if tuple(extract(module).instrs) == before:
                break
    module.verify()
    verify_linear(module)
    return module
