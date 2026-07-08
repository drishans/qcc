"""Fuse maximal runs of single-qubit gates into one u3 (or rz, or nothing).

Multiply the run's 2x2 matrices, then resynthesize: identity-up-to-phase
runs vanish, diagonal products become one rz, everything else becomes one
u3 via ZYZ Euler angles. Runs never cross a 2q gate, a barrier, or a
measure — those ops aren't Gate1Op, so the def-use chain ends there.

This is where Clifford strings like h·s·h stop being "gates from the nice
set" and become one float-angled gate; total gate count is what this
compiler minimizes, not T-count.
"""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter

from qcc import gates
from qcc.ir import dialect as d
from qcc.passes.cancel_inverses import sole_user


def _runs(module: ModuleOp):
    """Maximal 1q runs, each as a list of ops in program order."""
    seen: set = set()
    runs: list[list[d.Gate1Op]] = []
    for op in module.walk():
        if not isinstance(op, d.Gate1Op) or op in seen:
            continue
        # only start from the head of a chain
        prev = op.qin.owner if hasattr(op.qin, "owner") else None
        if isinstance(prev, d.Gate1Op):
            continue
        run: list[d.Gate1Op] = []
        cur: d.Gate1Op | None = op
        while isinstance(cur, d.Gate1Op):
            run.append(cur)
            seen.add(cur)
            cur = sole_user(cur.qout)  # type: ignore[assignment]
        if len(run) >= 2:
            runs.append(run)
    return runs


class FuseSingleQubitPass(ModulePass):
    name = "qcc-fuse-1q"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for run in _runs(op):
            u = run[0].matrix()
            for g in run[1:]:
                u = g.matrix() @ u  # program order: later gates multiply left

            first, last = run[0], run[-1]
            if gates.is_identity_up_to_phase(u):
                last.qout.replace_all_uses_with(first.qin)
            else:
                theta, phi, lam = gates.zyz_angles(u)
                if abs(theta) < gates.TOL:
                    new_op: d.Gate1Op = d.RzOp(
                        first.qin, gates.normalize_angle(phi + lam)
                    )
                else:
                    new_op = d.U3Op(first.qin, theta, phi, lam)
                Rewriter.insert_op(new_op, InsertPoint.before(first))
                last.qout.replace_all_uses_with(new_op.qout)
            for g in reversed(run):
                Rewriter.erase_op(g)
