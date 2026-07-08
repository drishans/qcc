import math
import random

import numpy as np
import pytest

from qcc import gates


def test_fixed_gates_are_unitary():
    for name in ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"):
        u = gates.matrix_1q(name)
        assert np.allclose(u @ u.conj().T, np.eye(2), atol=1e-12), name


def test_adjoint_pairs_multiply_to_identity():
    for a, b in gates.ADJOINT_1Q.items():
        u = gates.matrix_1q(a) @ gates.matrix_1q(b)
        assert gates.is_identity_up_to_phase(u), (a, b)


def test_rotations_compose_additively():
    for name in gates.ROTATIONS:
        u = gates.matrix_1q(name, (0.4,)) @ gates.matrix_1q(name, (0.3,))
        v = gates.matrix_1q(name, (0.7,))
        assert np.allclose(u, v, atol=1e-12), name


def test_full_rotation_is_identity_up_to_phase():
    for name in gates.ROTATIONS:
        u = gates.matrix_1q(name, (2 * math.pi,))
        assert gates.is_identity_up_to_phase(u), name
    assert gates.is_zero_rotation(2 * math.pi)
    assert gates.is_zero_rotation(-2 * math.pi)
    assert not gates.is_zero_rotation(math.pi)


def test_zyz_reconstructs_random_products():
    rng = random.Random(7)
    names = ["h", "x", "s", "t", "sx", "rx", "ry", "rz", "p"]
    for _ in range(200):
        u = np.eye(2, dtype=complex)
        for _ in range(rng.randint(1, 8)):
            name = rng.choice(names)
            params = (rng.uniform(-6, 6),) if name in gates.ROTATIONS else ()
            u = gates.matrix_1q(name, params) @ u
        theta, phi, lam = gates.zyz_angles(u)
        v = gates.matrix_1q("u3", (theta, phi, lam))
        # u3 result must equal u up to global phase
        w = v @ u.conj().T
        assert gates.is_identity_up_to_phase(w, tol=1e-8)


def test_zyz_diagonal_and_antidiagonal_edge_cases():
    for u in (
        gates.matrix_1q("rz", (0.9,)),
        gates.matrix_1q("p", (2.2,)),
        gates.matrix_1q("x"),
        gates.matrix_1q("y"),
    ):
        theta, phi, lam = gates.zyz_angles(u)
        v = gates.matrix_1q("u3", (theta, phi, lam))
        assert gates.is_identity_up_to_phase(v @ u.conj().T, tol=1e-9)


@pytest.mark.parametrize("angle,expect", [(0, 0), (math.pi, math.pi),
                                          (3 * math.pi, math.pi),
                                          (-3 * math.pi, math.pi),
                                          (2 * math.pi, 0)])
def test_normalize_angle(angle, expect):
    assert abs(gates.normalize_angle(angle) - expect) < 1e-12 or (
        abs(abs(gates.normalize_angle(angle)) - math.pi) < 1e-12
        and abs(expect) == math.pi
    )
