"""
tests/test_crossover.py -- Tests for crossover operators.
"""

import numpy as np
import pytest

from ea_toolkit.crossover import SBXCrossover, UniformCrossover


@pytest.fixture
def param_spec():
    return {f'x{i}': (-5.0, 5.0) for i in range(5)}


@pytest.fixture
def parent_pair(param_spec):
    rng = np.random.default_rng(42)
    names = sorted(param_spec.keys())
    p1 = {n: float(rng.uniform(-5, 5)) for n in names}
    p2 = {n: float(rng.uniform(-5, 5)) for n in names}
    return p1, p2


class TestSBXCrossover:
    def test_produces_two_children(self, param_spec, parent_pair):
        sbx = SBXCrossover(eta=20.0)
        rng = np.random.default_rng(42)
        c1, c2 = sbx.crossover(parent_pair[0], parent_pair[1], param_spec, rng)
        assert set(c1.keys()) == set(param_spec.keys())
        assert set(c2.keys()) == set(param_spec.keys())

    def test_children_within_bounds(self, param_spec, parent_pair):
        sbx = SBXCrossover(eta=1.0)  # Wide spread
        rng = np.random.default_rng(42)
        for _ in range(50):
            c1, c2 = sbx.crossover(parent_pair[0], parent_pair[1], param_spec, rng)
            for name in param_spec:
                lo, hi = param_spec[name]
                assert lo <= c1[name] <= hi
                assert lo <= c2[name] <= hi

    def test_high_eta_tight(self, param_spec, parent_pair):
        """High eta should produce children close to parents."""
        sbx = SBXCrossover(eta=100.0, probability=1.0)
        rng = np.random.default_rng(42)
        p1, p2 = parent_pair
        diffs = []
        for _ in range(100):
            c1, _ = sbx.crossover(p1, p2, param_spec, rng)
            for name in param_spec:
                mid = (p1[name] + p2[name]) / 2
                diffs.append(abs(c1[name] - mid))
        mean_diff = np.mean(diffs)
        assert mean_diff < 3.0, f"High eta should keep children near parents, got {mean_diff}"

    def test_identical_parents(self, param_spec):
        """Identical parents should produce identical children."""
        p = {f'x{i}': 1.0 for i in range(5)}
        sbx = SBXCrossover(eta=20.0)
        rng = np.random.default_rng(42)
        c1, c2 = sbx.crossover(p, p, param_spec, rng)
        for name in param_spec:
            assert c1[name] == p[name]
            assert c2[name] == p[name]

    def test_reproducibility(self, param_spec, parent_pair):
        sbx = SBXCrossover(eta=20.0)
        c1a, c2a = sbx.crossover(*parent_pair, param_spec, np.random.default_rng(99))
        c1b, c2b = sbx.crossover(*parent_pair, param_spec, np.random.default_rng(99))
        assert c1a == c1b
        assert c2a == c2b


class TestUniformCrossover:
    def test_produces_two_children(self, param_spec, parent_pair):
        ux = UniformCrossover(swap_probability=0.5)
        rng = np.random.default_rng(42)
        c1, c2 = ux.crossover(parent_pair[0], parent_pair[1], param_spec, rng)
        assert set(c1.keys()) == set(param_spec.keys())
        assert set(c2.keys()) == set(param_spec.keys())

    def test_children_from_parents(self, param_spec, parent_pair):
        """Each child value should come from one of the parents."""
        ux = UniformCrossover(swap_probability=0.5)
        rng = np.random.default_rng(42)
        p1, p2 = parent_pair
        c1, c2 = ux.crossover(p1, p2, param_spec, rng)
        for name in param_spec:
            assert c1[name] in (p1[name], p2[name])
            assert c2[name] in (p1[name], p2[name])

    def test_complementary_children(self, param_spec, parent_pair):
        """If c1 gets p2's value, c2 should get p1's value."""
        ux = UniformCrossover(swap_probability=0.5)
        rng = np.random.default_rng(42)
        p1, p2 = parent_pair
        c1, c2 = ux.crossover(p1, p2, param_spec, rng)
        for name in param_spec:
            if c1[name] == p2[name]:
                assert c2[name] == p1[name]
            else:
                assert c2[name] == p2[name]

    def test_swap_zero_copies_parents(self, param_spec, parent_pair):
        """swap_probability=0 should copy parents exactly."""
        ux = UniformCrossover(swap_probability=0.0)
        rng = np.random.default_rng(42)
        p1, p2 = parent_pair
        c1, c2 = ux.crossover(p1, p2, param_spec, rng)
        assert c1 == p1
        assert c2 == p2

    def test_swap_one_swaps_all(self, param_spec, parent_pair):
        """swap_probability=1 should swap all values."""
        ux = UniformCrossover(swap_probability=1.0)
        rng = np.random.default_rng(42)
        p1, p2 = parent_pair
        c1, c2 = ux.crossover(p1, p2, param_spec, rng)
        assert c1 == p2
        assert c2 == p1
