# qcc roadmap

The next milestones extend qcc from logical optimization into physical
compilation without weakening its equivalence and provenance guarantees.

## 4. Static-topology placement and routing

- Add an explicit logical-to-physical mapping state.
- Choose an initial placement for a target coupling graph.
- Insert SWAPs for nonadjacent interactions and update the mapping after each
  movement.
- Verify equivalence modulo the final output permutation.
- Benchmark SWAP count and routed depth against Qiskit and pytket on identical
  targets, with QKNOB providing known near-optimal references.
- Treat neutral-atom scheduling as a later target-specific extension. NAKO
  becomes relevant once qcc emits physical placement and movement schedules.

## 5. Structured pass diagnostics

- Give each pass an optional result record with candidates considered,
  accepted rewrites, rejection reasons, metric deltas, and elapsed time.
- Add a CLI JSON trace that preserves pass order and before/after metrics.
- Feed the same trace format into benchmark provenance and regression tests.
- Keep diagnostics observational so enabling them cannot change pass behavior.

## 6. Native MLIR migration decision

Stay on xDSL while compiler capability is the limiting factor. Reconsider a
native LLVM MLIR implementation only when one of these measured triggers is
present:

- Python pass execution is a demonstrated bottleneck on representative IRs.
- qcc needs direct interoperability with upstream MLIR dialects or tools.
- a C++ pass/plugin ABI is required by a backend or collaborator.
- circuit sizes expose unacceptable memory or serialization overhead.

Any migration starts with a textual-IR compatibility fixture and benchmark so
the decision is based on measured interoperability or performance, not assumed
speed.
