"""Differential testing: the optimizer must preserve semantics on circuits
it has never seen. This is the test that catches real compiler bugs."""

import random

from conftest import random_circuit

from qcc.ir import extract, metrics, verify_linear
from qcc.passes import optimize
from qcc.verify import fidelity


def test_random_circuits_stay_equivalent(rng: random.Random):
    for trial in range(30):
        n = rng.randint(1, 6)
        b = random_circuit(rng, n, 80, barriers=True)
        m = b.finish()
        before = extract(m)
        optimize(m)
        verify_linear(m)
        after = extract(m)
        f = fidelity(before, after)
        assert abs(f - 1) < 1e-9, (trial, n, f)
        assert metrics(after)["gates"] <= metrics(before)["gates"]


def test_optimizer_actually_optimizes(rng: random.Random):
    """Aggregate sanity: across many random circuits the pipeline should
    remove a nontrivial fraction of gates (they're dense in 1q runs)."""
    total_before = total_after = 0
    for _ in range(10):
        b = random_circuit(rng, 4, 100)
        m = b.finish()
        total_before += metrics(extract(m))["gates"]
        optimize(m)
        total_after += metrics(extract(m))["gates"]
    assert total_after < 0.85 * total_before, (total_before, total_after)
