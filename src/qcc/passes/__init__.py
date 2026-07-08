"""The optimization pipeline.

-O0 is the identity. -O1 iterates cancellation → rotation merging → 1q
fusion to a fixpoint: each pass only ever removes or coarsens gates, so
the op count is monotone non-increasing and termination is by counting.
"""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp

from qcc.ir.dialect import verify_linear
from qcc.passes.cancel_inverses import CancelInversesPass
from qcc.passes.fuse_1q import FuseSingleQubitPass
from qcc.passes.merge_rotations import MergeRotationsPass

MAX_ITERATIONS = 25


def _n_ops(module: ModuleOp) -> int:
    return sum(1 for _ in module.walk())


def optimize(module: ModuleOp, level: int = 1) -> ModuleOp:
    """Run the -O<level> pipeline in place (and return the module)."""
    if level <= 0:
        return module
    ctx = Context()
    passes = [CancelInversesPass(), MergeRotationsPass(), FuseSingleQubitPass()]
    before = _n_ops(module)
    for _ in range(MAX_ITERATIONS):
        for p in passes:
            p.apply(ctx, module)
        after = _n_ops(module)
        if after == before:
            break
        before = after
    module.verify()
    verify_linear(module)
    return module
