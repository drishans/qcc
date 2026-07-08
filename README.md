# qcc — a small optimizing quantum compiler

The compiler behind the drishan.com field-guide series on compilers ×
quantum. OpenQASM 3 in, optimized QIR out, executed with CUDA-Q, and
benchmarked honestly against Qiskit's transpiler and pytket.

The interesting part is the middle: circuits live in an MLIR-style SSA
dialect (built with [xDSL](https://github.com/xdslproject/xdsl)) where
qubits are *values*, not registers — every gate consumes its qubit values
and produces new ones. Def-use chains then give "next gate on this wire"
for free, and every optimization is a local rewrite:

```
OpenQASM 3 ──parse──▶ quantum dialect ──rewrites──▶ QIR (pyqir)
              (openqasm3 AST)   │                     │
                                │ cancel adjoint pairs │ CUDA-Q kernel
                                │ merge rotations      ▼
                                │ fuse 1q runs → U3   GPU execution
                                ▼
                          verified equivalent (numpy reference sim)
```

## Setup

Runs on Linux or WSL2. The compiler itself is pure Python; CUDA-Q
execution wants an NVIDIA GPU but falls back to CPU simulation.

```bash
# venv outside the repo if the repo lives on /mnt/c (NTFS venvs are slow)
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/qcc
uv sync
uv run qcc compile examples/bell.qasm --emit mlir   # see the IR
```

## Layout

| Path | What it is |
| --- | --- |
| `src/qcc/ir/` | the dialect: qubit type, gate ops, linearity verifier, metrics |
| `src/qcc/frontend/` | OpenQASM 3 → dialect (subset; clear errors otherwise) |
| `src/qcc/passes/` | the optimizer: cancellation, rotation merging, 1q fusion |
| `src/qcc/backend/` | QIR emission (pyqir), CUDA-Q execution, reference simulator |
| `tests/` | unit tests + differential testing (random circuits, state equivalence) |
| `bench/` | the harness: qcc vs Qiskit O0–O3 vs pytket on generated suites |
| `results/` | committed benchmark JSONs, each stamped by `provenance.py` |

## Reproducing the numbers

Every number in the series comes from a `results/*.json` here, stamped
with the environment it was measured on:

```bash
uv run python bench/run_bench.py
uv run python figures.py --out /path/to/figures
```

## License

MIT. The prose of the tutorial series is CC BY 4.0 on the site.
