"""Independent KAK/Weyl decomposition and exact CX synthesis for U(4).

The public surface is deliberately NumPy + qcc data types only. Qiskit and
pytket remain external baselines; neither is called by the production pass.

The magic-basis canonicalization and special-orthogonal bidiagonalization are
adapted from Cirq 1.6.1 (Apache-2.0), itself implementing the Cartan/KAK
construction described by Tucci and the optimal CNOT results of Shende,
Bullock, and Markov. This file is substantially modified for qcc's big-endian
wire convention, U3 output, and CX-only entangling basis. See NOTICE.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from qcc import gates
from qcc.ir.tape import Instr

ATOL = 1e-9
RTOL = 1e-7

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)

MAGIC = np.array(
    [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
    dtype=complex,
) * math.sqrt(0.5)
MAGIC_DAG = MAGIC.conj().T
GAMMA = np.array(
    [[1, 1, 1, 1], [1, 1, -1, -1], [-1, 1, -1, 1], [1, -1, -1, 1]],
    dtype=float,
) * 0.25


@dataclass(frozen=True)
class KAKDecomposition:
    """U = phase (A0⊗A1) exp(i[xXX+yYY+zZZ]) (B0⊗B1)."""

    global_phase: complex
    before: tuple[np.ndarray, np.ndarray]
    interaction: tuple[float, float, float]
    after: tuple[np.ndarray, np.ndarray]

    def matrix(self) -> np.ndarray:
        x, y, z = self.interaction
        nonlocal_u = _pauli_interaction(Z, z) @ _pauli_interaction(Y, y)
        nonlocal_u = nonlocal_u @ _pauli_interaction(X, x)
        return (
            self.global_phase
            * np.kron(*self.after)
            @ nonlocal_u
            @ np.kron(*self.before)
        )


@dataclass(frozen=True)
class SynthesisResult:
    instrs: tuple[Instr, ...]
    cx_count: int
    fidelity: float
    kak: KAKDecomposition


def unitary_fidelity(actual: np.ndarray, expected: np.ndarray) -> float:
    """Phase-blind normalized trace overlap for two 4x4 unitaries."""
    return float(abs(np.trace(actual.conj().T @ expected)) / 4)


def _pauli_interaction(pauli: np.ndarray, angle: float) -> np.ndarray:
    pp = np.kron(pauli, pauli)
    return math.cos(angle) * I4 + 1j * math.sin(angle) * pp


def _block_diag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + b.shape[0], a.shape[1] + b.shape[1]))
    out[: a.shape[0], : a.shape[1]] = a
    out[a.shape[0] :, a.shape[1] :] = b
    return out


def _groups(length: int, equal) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = start + 1
        while end < length and equal(start, end):
            end += 1
        result.append((start, end))
        start = end
    return result


def _simultaneous_diagonalizer(
    symmetric: np.ndarray, diagonal: np.ndarray
) -> np.ndarray:
    """Orthogonal P preserving diagonal while diagonalizing symmetric."""
    ranges = _groups(
        diagonal.shape[0],
        lambda i, j: np.isclose(
            diagonal[i, i], diagonal[j, j], rtol=RTOL, atol=ATOL
        ),
    )
    p = np.zeros(symmetric.shape, dtype=float)
    for start, end in ranges:
        _, vecs = np.linalg.eigh(symmetric[start:end, start:end])
        p[start:end, start:end] = vecs
    return p


def _bidiagonalize_real_pair(
    real: np.ndarray, imag: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """L,R with L real R and L imag R both diagonal."""
    base_left, singular, base_right = np.linalg.svd(real)
    base_diag = np.diag(singular)
    rank = len(singular)
    while rank and abs(base_diag[rank - 1, rank - 1]) < ATOL:
        rank -= 1
    nonzero_diag = base_diag[:rank, :rank]

    semi = base_left.T @ imag @ base_right.T
    overlap = semi[:rank, :rank]
    overlap_adjust = _simultaneous_diagonalizer(overlap, nonzero_diag)

    extra = semi[rank:, rank:]
    if extra.size:
        extra_left, _, extra_right = np.linalg.svd(extra)
    else:
        extra_left = np.zeros((0, 0))
        extra_right = np.zeros((0, 0))

    left_adjust = _block_diag(overlap_adjust, extra_left)
    right_adjust = _block_diag(overlap_adjust.T, extra_right)
    left = left_adjust.T @ base_left.T
    right = base_right.T @ right_adjust.T
    return left, right


def _bidiagonalize_unitary(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = _bidiagonalize_real_pair(np.real(mat), np.imag(mat))
    if np.linalg.det(left) < 0:
        left[0, :] *= -1
    if np.linalg.det(right) < 0:
        right[:, 0] *= -1
    diagonal = np.diag(left @ mat @ right)
    return left, diagonal, right


def _kron_factor(matrix: np.ndarray) -> tuple[complex, np.ndarray, np.ndarray]:
    """Factor a known 4x4 Kronecker product into two SU(2) matrices."""
    a, b = max(
        ((i, j) for i in range(4) for j in range(4)),
        key=lambda ij: abs(matrix[ij]),
    )
    f0 = np.zeros((2, 2), dtype=complex)
    f1 = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            f0[(a >> 1) ^ i, (b >> 1) ^ j] = matrix[a ^ (i << 1), b ^ (j << 1)]
            f1[(a & 1) ^ i, (b & 1) ^ j] = matrix[a ^ i, b ^ j]
    for factor in (f0, f1):
        root = np.sqrt(np.linalg.det(factor))
        if abs(root) > ATOL:
            factor /= root
    phase = matrix[a, b] / (f0[a >> 1, b >> 1] * f1[a & 1, b & 1])
    if phase.real < 0:
        f0 *= -1
        phase *= -1
    if not np.allclose(matrix, phase * np.kron(f0, f1), rtol=RTOL, atol=ATOL):
        raise ArithmeticError("failed to factor SO(4) element into SU(2) products")
    return phase, f0, f1


def _so4_to_magic_su2s(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformed = MAGIC @ mat @ MAGIC_DAG
    _, a, b = _kron_factor(transformed)
    return a, b


def _canonicalize(x: float, y: float, z: float) -> KAKDecomposition:
    phase = complex(1)
    left = [I2.copy(), I2.copy()]
    right = [I2.copy(), I2.copy()]
    v = [x, y, z]
    flippers = [1j * X, 1j * Y, 1j * Z]
    swappers = [
        1j * np.array([[1, -1j], [1j, -1]]) / math.sqrt(2),
        1j * np.array([[1, 1], [1, -1]]) / math.sqrt(2),
        1j * np.array([[0, 1 - 1j], [1 + 1j, 0]]) / math.sqrt(2),
    ]

    def shift(k: int, step: int) -> None:
        nonlocal phase
        v[k] += step * math.pi / 2
        phase *= 1j**step
        power = np.linalg.matrix_power(flippers[k], step % 4)
        right[0] = power @ right[0]
        right[1] = power @ right[1]

    def negate(k0: int, k1: int) -> None:
        nonlocal phase
        v[k0] *= -1
        v[k1] *= -1
        phase *= -1
        s = flippers[3 - k0 - k1]
        left[1] = left[1] @ s
        right[1] = s @ right[1]

    def swap(k0: int, k1: int) -> None:
        v[k0], v[k1] = v[k1], v[k0]
        s = swappers[3 - k0 - k1]
        left[0] = left[0] @ s
        left[1] = left[1] @ s
        right[0] = s @ right[0]
        right[1] = s @ right[1]

    def canonical_shift(k: int) -> None:
        while v[k] <= -math.pi / 4:
            shift(k, 1)
        while v[k] > math.pi / 4:
            shift(k, -1)

    for k in range(3):
        canonical_shift(k)
    if abs(v[0]) < abs(v[1]):
        swap(0, 1)
    if abs(v[1]) < abs(v[2]):
        swap(1, 2)
    if abs(v[0]) < abs(v[1]):
        swap(0, 1)
    if v[0] < 0:
        negate(0, 2)
    if v[1] < 0:
        negate(1, 2)
    canonical_shift(2)
    if v[0] > math.pi / 4 - ATOL and v[2] < 0:
        shift(0, -1)
        negate(0, 2)

    return KAKDecomposition(
        phase,
        (right[1], right[0]),
        (float(v[0]), float(v[1]), float(v[2])),
        (left[1], left[0]),
    )


def kak_decomposition(unitary: np.ndarray) -> KAKDecomposition:
    """Canonical KAK decomposition in the Weyl chamber."""
    unitary = np.asarray(unitary, dtype=complex)
    if unitary.shape != (4, 4):
        raise ValueError(f"expected a 4x4 unitary, got {unitary.shape}")
    if not np.allclose(unitary.conj().T @ unitary, I4, rtol=RTOL, atol=ATOL):
        raise ValueError("input is not unitary")

    left, diagonal, right = _bidiagonalize_unitary(MAGIC_DAG @ unitary @ MAGIC)
    a0, a1 = _so4_to_magic_su2s(left.T)
    b0, b1 = _so4_to_magic_su2s(right.T)
    w, x, y, z = (GAMMA @ np.angle(diagonal).reshape(-1, 1)).ravel()
    canonical = _canonicalize(float(x), float(y), float(z))

    before0 = canonical.before[0] @ b0
    before1 = canonical.before[1] @ b1
    after0 = a0 @ canonical.after[0]
    after1 = a1 @ canonical.after[1]
    result = KAKDecomposition(
        np.exp(1j * w) * canonical.global_phase,
        (before0, before1),
        canonical.interaction,
        (after0, after1),
    )
    if 1 - unitary_fidelity(result.matrix(), unitary) > 1e-8:
        raise ArithmeticError("KAK reconstruction failed")
    return result


def num_cx_for_kak(kak: KAKDecomposition) -> int:
    """Exact minimal CX count from canonical Weyl coordinates."""
    x, y, z = kak.interaction
    if max(abs(x), abs(y), abs(z)) < ATOL:
        return 0
    if abs(x - math.pi / 4) < ATOL and abs(y) < ATOL and abs(z) < ATOL:
        return 1
    if abs(z) < ATOL:
        return 2
    return 3


def _append_matrix(raw: list[Instr], wire: int, matrix: np.ndarray) -> None:
    if gates.is_identity_up_to_phase(matrix):
        return
    theta, phi, lam = gates.zyz_angles(matrix)
    if abs(theta) < gates.TOL:
        angle = gates.normalize_angle(phi + lam)
        if not gates.is_zero_rotation(angle):
            raw.append(Instr("rz", (wire,), (angle,)))
    else:
        raw.append(Instr("u3", (wire,), (theta, phi, lam)))


def _append_cz_as_cx(raw: list[Instr], q0: int, q1: int) -> None:
    raw.append(Instr("h", (q1,)))
    raw.append(Instr("cx", (q0, q1)))
    raw.append(Instr("h", (q1,)))


def _append_nonlocal(
    raw: list[Instr], q0: int, q1: int, interaction: tuple[float, float, float]
) -> None:
    x, y, z = interaction
    if max(abs(x), abs(y), abs(z)) < ATOL:
        return
    if abs(x - math.pi / 4) < ATOL and abs(y) < ATOL and abs(z) < ATOL:
        raw.extend(
            [
                Instr("ry", (q0,), (-math.pi / 2,)),
                Instr("ry", (q1,), (-math.pi / 2,)),
            ]
        )
        _append_cz_as_cx(raw, q0, q1)
        raw.extend(
            [
                Instr("rz", (q0,), (-math.pi / 2,)),
                Instr("rz", (q1,), (-math.pi / 2,)),
                Instr("ry", (q0,), (math.pi / 2,)),
                Instr("ry", (q1,), (math.pi / 2,)),
            ]
        )
        return
    if abs(z) < ATOL and abs(y) < ATOL:
        raw.append(Instr("h", (q1,)))
        _append_cz_as_cx(raw, q0, q1)
        raw.append(Instr("rx", (q0,), (-2 * x,)))
        _append_cz_as_cx(raw, q0, q1)
        raw.append(Instr("h", (q1,)))
        return
    if abs(z) < ATOL:
        raw.extend([Instr("rx", (q0,), (math.pi / 2,)), Instr("h", (q1,))])
        _append_cz_as_cx(raw, q0, q1)
        raw.extend(
            [
                Instr("h", (q1,)),
                Instr("rx", (q0,), (-2 * x,)),
                Instr("ry", (q1,), (-2 * y,)),
                Instr("h", (q1,)),
            ]
        )
        _append_cz_as_cx(raw, q0, q1)
        raw.extend([Instr("h", (q1,)), Instr("rx", (q0,), (-math.pi / 2,))])
        return

    raw.extend([Instr("rx", (q0,), (math.pi / 2,)), Instr("h", (q1,))])
    _append_cz_as_cx(raw, q0, q1)
    raw.extend(
        [
            Instr("h", (q1,)),
            Instr("rx", (q0,), (-2 * x + math.pi / 2,)),
            Instr("ry", (q1,), (-2 * y + math.pi / 2,)),
            Instr("h", (q0,)),
        ]
    )
    _append_cz_as_cx(raw, q0, q1)
    raw.extend(
        [
            Instr("h", (q0,)),
            Instr("rx", (q1,), (-math.pi / 2,)),
            Instr("rz", (q1,), (-2 * z + math.pi / 2,)),
            Instr("h", (q1,)),
        ]
    )
    _append_cz_as_cx(raw, q0, q1)
    raw.append(Instr("h", (q1,)))


def _fuse_local_layers(raw: list[Instr], q0: int, q1: int) -> tuple[Instr, ...]:
    pending = {q0: I2.copy(), q1: I2.copy()}
    out: list[Instr] = []

    def flush() -> None:
        for wire in (q0, q1):
            _append_matrix(out, wire, pending[wire])
            pending[wire] = I2.copy()

    for ins in raw:
        if len(ins.qubits) == 1:
            (wire,) = ins.qubits
            pending[wire] = gates.matrix_1q(ins.name, ins.params) @ pending[wire]
        else:
            flush()
            out.append(ins)
    flush()
    return tuple(out)


def sequence_matrix(
    instrs: tuple[Instr, ...] | list[Instr], pair: tuple[int, int]
) -> np.ndarray:
    """Matrix of a pair-local instruction sequence in |pair[0] pair[1]> order."""
    q0, q1 = pair
    result = I4.copy()
    swap = gates.matrix_2q("swap")
    for ins in instrs:
        if len(ins.qubits) == 1:
            local = gates.matrix_1q(ins.name, ins.params)
            op = np.kron(local, I2) if ins.qubits[0] == q0 else np.kron(I2, local)
        else:
            op = gates.matrix_2q(ins.name)
            if ins.qubits == (q1, q0):
                op = swap @ op @ swap
            elif ins.qubits != (q0, q1):
                raise ValueError(f"instruction {ins} is outside pair {pair}")
        result = op @ result
    return result


def synthesize_two_qubit(
    unitary: np.ndarray, q0: int = 0, q1: int = 1
) -> SynthesisResult:
    """Synthesize U(4) exactly into local U3/Rz gates and 0–3 CX gates."""
    if q0 == q1:
        raise ValueError("two-qubit synthesis needs distinct wires")
    kak = kak_decomposition(unitary)
    raw: list[Instr] = []
    _append_matrix(raw, q0, kak.before[0])
    _append_matrix(raw, q1, kak.before[1])
    _append_nonlocal(raw, q0, q1, kak.interaction)
    _append_matrix(raw, q0, kak.after[0])
    _append_matrix(raw, q1, kak.after[1])
    instrs = _fuse_local_layers(raw, q0, q1)
    actual = sequence_matrix(instrs, (q0, q1))
    fidelity = unitary_fidelity(actual, unitary)
    if 1 - fidelity > 1e-8:
        raise ArithmeticError(f"CX synthesis failed: fidelity={fidelity:.16g}")
    cx_count = sum(ins.name == "cx" for ins in instrs)
    expected = num_cx_for_kak(kak)
    if cx_count != expected:
        raise ArithmeticError(f"expected {expected} CX gates, emitted {cx_count}")
    return SynthesisResult(instrs, cx_count, fidelity, kak)
