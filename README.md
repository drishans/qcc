# qcc — a small optimizing quantum compiler

The compiler behind the drishan.com field-guide series on compilers ×
quantum. OpenQASM 3 in, optimized QIR out, executed with CUDA-Q, and
benchmarked honestly against Qiskit's transpiler and pytket.

The interesting part is the middle: circuits live in an MLIR-style SSA
dialect (built with [xDSL](https://github.com/xdslproject/xdsl)) where
qubits are *values*, not registers — every gate consumes its qubit values
and produces new ones. Def-use chains then give "next gate on this wire"
for free, no-cloning becomes a linearity rule the verifier checks, and
every optimization is a local rewrite:

```
OpenQASM 3 ──parse──▶ quantum dialect ──rewrites──▶ QIR (pyqir)
              (openqasm3 AST)   │                     │
                                │ cancel adjoint pairs │ CUDA-Q kernel
                                │ merge rotations      ▼
                                │ commute-and-cancel  GPU execution
                                │ fuse 1q runs → u3
                                ▼
                          verified equivalent (numpy reference sim)
```

Every optimized circuit is checked for statevector equivalence (up to
global phase) against the unoptimized one, with a reference simulator
that shares no code with anything being benchmarked. `tests/` also runs
differential testing over random circuits — that suite caught every real
bug the passes ever had.

## Setup

Runs on Linux or WSL2. The compiler itself is pure Python; CUDA-Q
execution wants an NVIDIA GPU but falls back to CPU simulation.

```bash
# venv outside the repo if the repo lives on /mnt/c (NTFS venvs are slow)
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/qcc
uv sync
uv run qcc compile examples/redundant.qasm --stats --verify   # watch it shrink
uv run qcc compile examples/bell.qasm --emit qir              # the artifact
uv run qcc run examples/bell.qasm --shots 2000                # CUDA-Q, GPU if present
```

## Layout

| Path | What it is |
| --- | --- |
| `src/qcc/ir/` | the dialect: qubit type, gate ops, linearity verifier, metrics |
| `src/qcc/frontend/` | OpenQASM 3 → dialect (subset + gate-def expansion; clear errors otherwise) |
| `src/qcc/passes/` | the optimizer: cancellation, rotation merging, commutation pushing, 1q fusion |
| `src/qcc/backend/` | QIR emission (pyqir), CUDA-Q execution, qasm3 output, reference simulator |
| `tests/` | unit tests + differential testing (random circuits, state equivalence) |
| `bench/` | the harness: qcc vs Qiskit O0–O3 vs pytket on generated suites |
| `results/` | committed benchmark JSONs, each stamped by `provenance.py` |

## The benchmark, and what it does and doesn't claim

All three compilers receive the *identical* basis-translated circuit
(h/x/y/z/s/sdg/t/tdg/rx/ry/rz/cx/cz/swap). Qiskit runs `transpile()`
with no backend — logical optimization only, the same game qcc plays.
pytket runs `FullPeepholeOptimise`. Timings are the median of 5 runs of
the optimization call alone. Total gates surviving, from
`results/compile_bench.json`:

| suite | input | qcc -O1 | qiskit O1 | qiskit O2/O3 | tket |
| --- | ---: | ---: | ---: | ---: | ---: |
| ghz n12 | 12 | 12 | 12 | 12 | 12 |
| qft n6 | 84 | **68** | 74 | 61 | 61 |
| qft n10 | 240 | **194** | 222 | 181 | 181 |
| qaoa n12 p2 (s11) | 144 | **135** | 144 | 144 | 138 |
| qaoa n12 p2 (s12) | 144 | **134** | 144 | 144 | 140 |
| clifford+T n8 (s1) | 400 | **201** | 251 | 216 | 210 |
| clifford+T n8 (s2) | 400 | **178** | 218 | 198 | 191 |
| clifford+T n8 (s3) | 400 | **205** | 250 | 220 | 219 |
| vqe su2 n10 r3 | 267 | **67** | 67 | 67 | 91 |
| adder b4 | 137 | **125** | 129 | 129 | 132 |

Four local rewrite families match or beat qiskit O2/O3 on total gates
everywhere except QFT, at a fraction of tket's compile time (roughly
1–40 ms per circuit here vs 90–1100 ms). The honest flip side: this
pipeline never touches 2q structure. tket's Clifford resynthesis removes
~20% of 2q gates on the random Clifford+T suite and qiskit O2+ wins QFT
via 2q-block collection + resynthesis. That gap — `KAK`/block
resynthesis — is precisely the next pass this compiler doesn't have.

## And the Q# stack?

Microsoft's QDK is a different kind of tool and gets its own comparison
(`bench/qdk_compare.py`, results in `results/qdk_qir_compare.json`): it is a
language frontend that lowers Q#/OpenQASM 3 to QIR and by design leaves
gate-level optimization to whatever consumes the QIR. Measured, that's
exactly what it does: on 8 of 10 suites the QDK's base-profile QIR is
instruction-for-instruction the size of *unoptimized* qcc (`-O0`); the two
QFT rows differ only because the QDK keeps a native `swap` intrinsic while
qcc's pyqir emission expands swap into 3 cx.

QIR instructions emitted for the same OpenQASM 3 (measures excluded;
timings are the median source→QIR compile):

| suite | QDK (base) | qcc -O0 | qcc -O1 |
| --- | ---: | ---: | ---: |
| ghz n12 | 12 | 12 | 12 |
| qft n6 | 84 | 90 | **80** |
| qft n10 | 240 | 250 | **214** |
| qaoa n12 p2 (a/b) | 144 | 144 | 153 / 154 |
| clifford+T n8 (a/b/c) | 320 / 305 / 307 | same | **226 / 218 / 244** |
| vqe su2 n10 r3 | 267 | 267 | **130** |
| adder b4 | 113 | 113 | **109** |

Median compile time: QDK 84 ms, qcc 33 ms end-to-end. The QAOA rows show a
metric artifact worth understanding rather than papering over: fusion
coarsens pairs of cheap 1q gates into u3, and base-profile QIR has no
generic-1q intrinsic, so a 3-angle u3 re-expands to three rotations. Every
optimizing compiler makes this trade (qiskit's `u`, tket's TK1); the
2q count, which dominates hardware cost, is unchanged, and near-zero ZYZ
angles are elided at emission so a u3 that is really one rotation costs
one instruction.

The cross-stack check also passes: the QDK's own simulator executes qcc's
*optimized* qasm3 output for GHZ-12 and reproduces exactly the two-string
support (recorded in the results JSON).

Reproduce with:

```bash
uv run python bench/run_bench.py
uv run python bench/qdk_compare.py
uv run python figures.py --out /path/to/figures
```

## License

MIT. The prose of the tutorial series is CC BY 4.0 on the site.
