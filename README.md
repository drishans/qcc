# qcc: an optimizing quantum compiler

[![CI](https://github.com/drishans/qcc/actions/workflows/ci.yml/badge.svg)](https://github.com/drishans/qcc/actions/workflows/ci.yml)

**Small enough to read, complete enough to benchmark.**

qcc takes an OpenQASM 3 circuit, represents qubit state as linear SSA values,
applies verified optimizations including exact KAK two-qubit resynthesis,
lowers to a target-native gate set, and emits QIR, OpenQASM 3, or an executable
CUDA-Q kernel.

The project is an exploration of the compiler problems underneath a quantum
SDK: IR design, no-cloning verification, rewrite profitability, unitary
synthesis, target modeling, lowering, correctness oracles, and fair compiler
benchmarks. The frontend mapping, dialect, verifier, pass pipeline, synthesis,
target lowering, simulator, and benchmark harness all live in this repository.

## Why qcc is interesting

- **Qubits are SSA values.** Gates consume one logical qubit state and produce
  the next. Def-use chains expose circuit order, and a linear-use verifier
  turns no-cloning into a compiler invariant.
- **Optimization is more than peepholes.** `-O1` performs cancellation,
  rotation merging, profitable commutation, and single-qubit fusion. `-O2`
  adds KAK/Weyl decomposition with exact 0-3 CX resynthesis.
- **The compiler is target-aware.** A target describes native gates,
  connectivity, gate costs, durations, qubit capacity, and mobility
  assumptions without depending on a vendor SDK.
- **Correctness is executable.** Optimized circuits are checked against an
  independent NumPy statevector simulator, up to global phase, and random
  differential tests attack the pass pipeline.
- **The comparisons are reproducible.** Committed result files include tool
  versions, hardware, parameters, and raw rows for qcc, Qiskit, and pytket.

## Quick start

Prerequisites: Linux or WSL2, Python 3.12+, and
[`uv`](https://docs.astral.sh/uv/). CUDA-Q is optional.

```bash
git clone https://github.com/drishans/qcc.git
cd qcc

# Keeping the environment outside an NTFS-mounted repo is much faster on WSL2.
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/qcc
uv sync --group dev

# Optimize, lower to the QIR basis, verify equivalence, and emit LLVM IR.
uv run qcc compile examples/redundant.qasm \
  -O2 --target qir-base --emit qir --stats --verify \
  -o /tmp/redundant.ll
```

The example begins with 12 gates and finishes with a Bell-state circuit of one
Hadamard and one entangler. `--verify` independently checks that the rewrite
preserved the circuit unitary.

Explore the target model or inspect target-native OpenQASM:

```bash
uv run qcc targets
uv run qcc compile examples/bell.qasm -O2 --target cz --emit qasm3
```

To execute through CUDA-Q:

```bash
uv sync --extra cudaq --group dev
uv run qcc run examples/bell.qasm --shots 2000
```

CUDA-Q uses an NVIDIA GPU when a supported target is available and otherwise
can run with its CPU simulator.

## Compiler architecture

```text
OpenQASM 3
    |
    v
openqasm3 AST
    |
    v
qcc SSA dialect  -->  linearity verifier
    |
    +--> -O1 local rewrites
    +--> -O2 KAK two-qubit resynthesis
    |
    v
target-native lowering  -->  gate set, topology, costs, durations
    |
    +--> QIR via pyqir
    +--> OpenQASM 3
    +--> CUDA-Q execution
```

### Qubits as linear SSA values

A Bell circuit in qcc's textual IR looks like this:

```mlir
%0 = qcc.alloc 0 : i64
%1 = qcc.alloc 1 : i64
%2 = qcc.h %0
%3, %4 = qcc.cx %2, %1
```

`%0` and `%2` are successive logical states of the same qubit, not two
physical qubits. Once `%0` is consumed by `qcc.h`, using it again is invalid.
This representation makes next-on-wire relationships ordinary SSA use-def
relationships and lets the verifier reject accidental fan-out.

The dialect is built with
[`xDSL`](https://github.com/xdslproject/xdsl), which provides MLIR-style
operations, regions, verification, and rewrite infrastructure in Python. The
choice keeps the compiler inspectable and fast to evolve. The roadmap records
measured triggers for moving to native LLVM MLIR if interoperability or pass
performance eventually requires it.

### Optimization levels

| Level | Pipeline |
| --- | --- |
| `-O0` | Parse and verify with no logical optimization |
| `-O1` | Cancel inverses, merge rotations, commute diagonal gates when it pays, clean up, and fuse maximal single-qubit runs into `u3` |
| `-O2` | Run `-O1`, resynthesize profitable same-pair two-qubit blocks with KAK, then repeat cleanup to a fixpoint |

The KAK pass collects a maximal block acting on one qubit pair, computes its
unitary, maps it into the Weyl chamber, and synthesizes an exact circuit using
the minimal 0-3 CX class for that interaction. A candidate is accepted only
when it strictly improves the lexicographic cost `(two-qubit gates, total
gates)`, so canonicalization cannot silently make the expensive metric worse.

### Targets and lowering

Built-in targets cover the full logical gate set, CX-native and CZ-native
bases, the QIR base profile, and CUDA-Q. A custom `Target` can specify:

- native one- and two-qubit gates;
- all-to-all, directed, or undirected connectivity;
- qubit capacity;
- weighted gate costs and durations;
- movable-qubit and movable-interaction assumptions.

Target lowering happens after logical optimization. A fixed topology currently
rejects nonadjacent interactions with a clear routing error. Placement and
SWAP routing are intentionally the next compiler layer, not behavior hidden in
the target model.

## Measured against Qiskit and pytket

The benchmark asks a narrow, useful question: how well does each compiler
optimize the same logical circuit before hardware placement and routing?

Each tool receives an identical basis-translated input. Qiskit runs
`transpile()` without a backend or coupling map, pytket runs
`FullPeepholeOptimise`, and qcc runs its own pass pipeline. Timings are the
median of five optimization calls. qcc outputs up to 12 qubits are checked for
statevector equivalence against the unoptimized input.

Selected total unitary gate counts from the committed baseline:

| Circuit | Input | qcc `-O2` | Qiskit O2 | pytket |
| --- | ---: | ---: | ---: | ---: |
| GHZ, 12 qubits | 12 | 12 | 12 | 12 |
| QFT, 10 qubits | 240 | 194 | **181** | **181** |
| Clifford+T, seed 1 | 400 | **204** | 216 | 210 |
| Clifford+T, seed 2 | 400 | **181** | 198 | 191 |
| Clifford+T, seed 3 | 400 | **203** | 220 | 219 |
| VQE SU2, 10 qubits | 267 | **67** | **67** | 91 |
| Adder, 4 bits | 137 | **125** | 129 | 132 |

Across all ten suites, qcc `-O2` leaves 1,323 total gates versus 1,372 for
Qiskit O2, while its two-qubit count is 704 versus Qiskit's 703. On all three
random Clifford+T instances, qcc produces fewer two-qubit gates than Qiskit O2.
Median optimization time is 28.5 ms for qcc `-O2`, 12.4 ms for Qiskit O2, and
344.7 ms for pytket in this baseline.

The misses matter too. QFT exposes qcc's current lack of broader circuit
restructuring: final swaps and isolated controlled phases do not form
profitable same-pair KAK blocks, so Qiskit and pytket win those rows. This is a
compiler-pass comparison, not a claim that qcc yet replaces hardware-aware
production transpilation.

The full data and provenance are in
[`results/compile_bench.json`](results/compile_bench.json). Reproduce them with:

```bash
uv sync --extra bench --group dev
uv run python bench/run_bench.py
uv run python figures.py --out /tmp/qcc-figures
```

## Correctness and interoperability

qcc uses several independent checks rather than trusting a pass because its
output looks smaller:

- xDSL structural verification and qcc's linear qubit-use verifier run after
  compilation;
- a separate NumPy statevector simulator checks equivalence up to global
  phase;
- randomized circuits exercise combinations that hand-written examples miss;
- CI tests Python 3.12 and 3.13 and installs the built wheel in an isolated
  environment;
- optional CUDA-Q tests compare execution against the reference simulator.

For interoperability, qcc emits QIR through `pyqir`, round-trippable OpenQASM
3, and CUDA-Q kernels. The repository also retains a compact Microsoft QDK/Q#
comparison as a cross-stack QIR check, not as an optimizer contest.
[`bench/qdk_compare.py`](bench/qdk_compare.py) records source-to-QIR size and
timing, and verifies that the QDK simulator can execute qcc's optimized output.
Its raw results live in
[`results/qdk_qir_compare.json`](results/qdk_qir_compare.json).

## Scope and roadmap

qcc is a working research compiler, not a drop-in production transpiler. It
currently focuses on a deliberately scoped OpenQASM 3 gate-level frontend,
logical optimization, native-basis lowering, QIR emission, and local
execution. Unsupported language features fail explicitly.

The next milestones are physical placement and routing, structured pass
diagnostics, and a measured decision on native MLIR. Neutral-atom movement and
scheduling, where NAKO-style metrics become useful, follow after qcc can emit a
physical placement witness. See [`ROADMAP.md`](ROADMAP.md) for the concrete
criteria.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/qcc/frontend/` | OpenQASM 3 AST to qcc dialect, including gate-definition expansion |
| `src/qcc/ir/` | Quantum operations, qubit type, builders, tape view, metrics, and linearity verification |
| `src/qcc/passes/` | Cancellation, rotation merging, commutation, single-qubit fusion, and KAK resynthesis |
| `src/qcc/synthesis/` | KAK/Weyl decomposition and exact 0-3 CX synthesis |
| `src/qcc/target.py` | Native gates, topology, resource metadata, and basis lowering |
| `src/qcc/backend/` | QIR, OpenQASM 3, CUDA-Q, and the independent statevector simulator |
| `tests/` | Unit, differential, backend, target, and synthesis tests |
| `bench/` | Qiskit, pytket, and QDK comparison harnesses |
| `results/` | Versioned benchmark outputs with provenance |

The companion
[*Building a Quantum Compiler*](https://drishan.com/series/building-a-quantum-compiler/)
field guide explains the IR, verifier, passes, backends, and benchmark design
step by step.

## License

Original qcc code is MIT. The simultaneous-diagonalization routine used by the
KAK decomposition is adapted from Cirq 1.6.1 under Apache-2.0. See `NOTICE` and
`LICENSES/Apache-2.0.txt`.
