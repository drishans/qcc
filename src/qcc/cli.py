"""The qcc command line.

    qcc compile bell.qasm                 # optimize, print MLIR
    qcc compile bell.qasm --emit qir      # ... emit QIR instead
    qcc compile bell.qasm -O0 --stats     # no optimization, show metrics
    qcc compile bell.qasm --verify        # check optimized ≡ original
    qcc run bell.qasm --shots 2000        # execute on CUDA-Q
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qcc.frontend import parse_qasm3
from qcc.ir import extract, metrics, verify_linear
from qcc.passes import optimize


def _load(path: str):
    module = parse_qasm3(Path(path).read_text(encoding="utf-8"))
    module.verify()
    verify_linear(module)
    return module


def cmd_compile(args: argparse.Namespace) -> int:
    module = _load(args.input)
    before = extract(module)
    optimize(module, args.opt_level)
    after = extract(module)

    if args.verify:
        from qcc.verify import equivalent, fidelity

        if not equivalent(before, after):
            print(
                f"VERIFICATION FAILED: fidelity {fidelity(before, after)}",
                file=sys.stderr,
            )
            return 1
        print("verified: optimized ≡ original (up to global phase)", file=sys.stderr)

    if args.stats:
        print(
            json.dumps({"before": metrics(before), "after": metrics(after)}, indent=2),
            file=sys.stderr,
        )

    if args.emit == "mlir":
        out = str(module)
    elif args.emit == "qir":
        from qcc.backend.qir import emit_qir

        out = emit_qir(after, name=Path(args.input).stem)
    else:
        from qcc.backend.qasm3_out import emit_qasm3

        out = emit_qasm3(after)

    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
    else:
        print(out)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from qcc.backend.cudaq_backend import sample, set_best_target

    module = _load(args.input)
    optimize(module, args.opt_level)
    tape = extract(module)
    target = set_best_target()
    counts = sample(tape, shots=args.shots)
    print(f"target: {target}", file=sys.stderr)
    for bits in sorted(counts, key=counts.get, reverse=True):  # type: ignore[arg-type]
        print(f"{bits} {counts[bits]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qcc", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="compile OpenQASM 3 to mlir/qir/qasm3")
    c.add_argument("input")
    c.add_argument("-O", dest="opt_level", type=int, default=1, choices=(0, 1))
    c.add_argument("--emit", choices=("mlir", "qir", "qasm3"), default="mlir")
    c.add_argument("-o", "--output")
    c.add_argument("--stats", action="store_true", help="print gate metrics to stderr")
    c.add_argument("--verify", action="store_true", help="simulate and check equivalence")
    c.set_defaults(fn=cmd_compile)

    r = sub.add_parser("run", help="execute on CUDA-Q")
    r.add_argument("input")
    r.add_argument("-O", dest="opt_level", type=int, default=1, choices=(0, 1))
    r.add_argument("--shots", type=int, default=1000)
    r.set_defaults(fn=cmd_run)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
