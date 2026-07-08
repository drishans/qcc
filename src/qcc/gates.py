"""Gate metadata and the small amount of linear algebra the compiler needs.

Everything here is plain numpy — no IR imports — so the dialect, the passes,
the simulator, and the tests all draw from one table of truth.

Conventions:
- 1q matrices are 2x2 in the computational basis.
- 2q matrices are 4x4 in basis |q0 q1> where q0 is the FIRST operand
  (the control for cx) and the most significant bit.
- Equivalence throughout the compiler is *up to global phase*: an
  uncontrolled gate's global phase tensor-factors out of the whole state,
  so replacing a run U with V where U = e^{iγ}V is safe. This invariant
  would break if gates could later acquire control modifiers — the IR has
  none after the frontend.
"""

from __future__ import annotations

import cmath
import math

import numpy as np

TOL = 1e-9

# ─── 1q fixed gates ──────────────────────────────────────────────────────

_SQ2 = 1 / math.sqrt(2)

_FIXED_1Q: dict[str, np.ndarray] = {
    "id": np.eye(2, dtype=complex),
    "h": np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, cmath.exp(-1j * math.pi / 4)]], dtype=complex),
    "sx": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
}

#: gates that are their own inverse
SELF_ADJOINT_1Q = frozenset({"id", "h", "x", "y", "z"})
SELF_ADJOINT_2Q = frozenset({"cx", "cz", "swap"})

#: name → name of its adjoint (self-adjoint gates map to themselves)
ADJOINT_1Q: dict[str, str] = {
    **{g: g for g in SELF_ADJOINT_1Q},
    "s": "sdg",
    "sdg": "s",
    "t": "tdg",
    "tdg": "t",
}

#: rotation gates: name → axis. rz/rx/ry compose additively; p is the phase
#: gate diag(1, e^{iλ}), which also composes additively and equals rz up to
#: global phase.
ROTATIONS = ("rx", "ry", "rz", "p")

#: 2q gates that are symmetric under operand exchange
SYMMETRIC_2Q = frozenset({"cz", "swap"})

_2Q: dict[str, np.ndarray] = {
    "cx": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    ),
    "cz": np.diag([1, 1, 1, -1]).astype(complex),
    "swap": np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    ),
}

#: every 1q gate name the IR knows, fixed and parametric
GATES_1Q = frozenset(_FIXED_1Q) | frozenset(ROTATIONS) | {"u3"}
GATES_2Q = frozenset(_2Q)


def matrix_1q(name: str, params: tuple[float, ...] = ()) -> np.ndarray:
    """2x2 unitary for a 1q gate."""
    if name in _FIXED_1Q:
        return _FIXED_1Q[name]
    if name == "rx":
        (theta,) = params
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "ry":
        (theta,) = params
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz":
        (theta,) = params
        return np.array(
            [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]],
            dtype=complex,
        )
    if name == "p":
        (lam,) = params
        return np.array([[1, 0], [0, cmath.exp(1j * lam)]], dtype=complex)
    if name == "u3":
        theta, phi, lam = params
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array(
            [
                [c, -cmath.exp(1j * lam) * s],
                [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c],
            ],
            dtype=complex,
        )
    raise KeyError(f"unknown 1q gate {name!r}")


def matrix_2q(name: str) -> np.ndarray:
    """4x4 unitary in |q0 q1> basis, first operand = most significant bit."""
    return _2Q[name]


# ─── the linear algebra behind 1q fusion ─────────────────────────────────


def is_identity_up_to_phase(u: np.ndarray, tol: float = TOL) -> bool:
    """Is U = e^{iγ}·I? (Safe to erase under the global-phase invariant.)"""
    if abs(abs(u[0, 0]) - 1) > tol:
        return False
    return bool(np.allclose(u, u[0, 0] * np.eye(len(u)), atol=tol))


def zyz_angles(u: np.ndarray, tol: float = TOL) -> tuple[float, float, float]:
    """Angles (theta, phi, lam) with u3(theta,phi,lam) = e^{iγ}·U.

    Standard ZYZ Euler decomposition: strip the global phase to get an
    SU(2) matrix V, then V = Rz(phi)·Ry(theta)·Rz(lam), and
    u3(θ,φ,λ) = e^{i(φ+λ)/2}·Rz(φ)Ry(θ)Rz(λ).
    """
    det = u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]
    v = u * cmath.exp(-1j * cmath.phase(det) / 2)  # det(v) = 1

    theta = 2 * math.atan2(abs(v[1, 0]), abs(v[0, 0]))
    if abs(v[1, 0]) < tol:  # diagonal: only phi+lam is defined
        return 0.0, 0.0, 2 * cmath.phase(v[1, 1])
    if abs(v[0, 0]) < tol:  # anti-diagonal: only phi-lam is defined
        return math.pi, 2 * cmath.phase(v[1, 0]), 0.0
    phi = cmath.phase(v[1, 1]) + cmath.phase(v[1, 0])
    lam = cmath.phase(v[1, 1]) - cmath.phase(v[1, 0])
    return theta, phi, lam


def normalize_angle(theta: float) -> float:
    """Map an angle to (-pi, pi]."""
    theta = math.fmod(theta, 2 * math.pi)
    if theta > math.pi:
        theta -= 2 * math.pi
    elif theta <= -math.pi:
        theta += 2 * math.pi
    return theta


def is_zero_rotation(theta: float, tol: float = TOL) -> bool:
    """Rotation by ~0 mod 2π. R(2π) = -I for rx/ry/rz — identity up to
    global phase; p(2π) = I exactly. Both erasable under the invariant."""
    return abs(normalize_angle(theta)) < tol
