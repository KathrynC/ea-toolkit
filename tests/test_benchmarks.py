"""
tests/test_benchmarks.py -- Tests for benchmark fitness functions.
"""

import numpy as np
import pytest

from ea_toolkit.benchmarks import (
    SphereFitness,
    RosenbrockFitness,
    RastriginFitness,
    AckleyFitness,
    ZDT1Fitness,
)


class TestSphereFitness:
    def test_optimum_at_origin(self):
        sf = SphereFitness(n_dims=3)
        result = sf.evaluate({'x0': 0, 'x1': 0, 'x2': 0})
        assert abs(result['fitness']) < 1e-10

    def test_negative_away_from_origin(self):
        sf = SphereFitness(n_dims=3)
        result = sf.evaluate({'x0': 1, 'x1': 1, 'x2': 1})
        assert result['fitness'] == -3.0

    def test_param_spec(self):
        sf = SphereFitness(n_dims=5, bounds=(-10, 10))
        spec = sf.param_spec()
        assert len(spec) == 5
        assert spec['x0'] == (-10, 10)


class TestRosenbrockFitness:
    def test_optimum_at_ones(self):
        rf = RosenbrockFitness(n_dims=3)
        result = rf.evaluate({'x0': 1, 'x1': 1, 'x2': 1})
        assert abs(result['fitness']) < 1e-10

    def test_negative_away_from_optimum(self):
        rf = RosenbrockFitness(n_dims=3)
        result = rf.evaluate({'x0': 0, 'x1': 0, 'x2': 0})
        assert result['fitness'] < 0

    def test_valley_structure(self):
        """Point on the valley floor should be better than off it."""
        rf = RosenbrockFitness(n_dims=2)
        on_valley = rf.evaluate({'x0': 0.5, 'x1': 0.25})  # x1 = x0^2
        off_valley = rf.evaluate({'x0': 0.5, 'x1': 2.0})
        assert on_valley['fitness'] > off_valley['fitness']


class TestRastriginFitness:
    def test_optimum_at_origin(self):
        rf = RastriginFitness(n_dims=3)
        result = rf.evaluate({'x0': 0, 'x1': 0, 'x2': 0})
        assert abs(result['fitness']) < 1e-10

    def test_local_optima_exist(self):
        """Near-integer points should be local optima (worse than origin)."""
        rf = RastriginFitness(n_dims=1)
        at_origin = rf.evaluate({'x0': 0})
        at_one = rf.evaluate({'x0': 1.0})
        assert at_origin['fitness'] > at_one['fitness']

    def test_raw_rastrigin_included(self):
        rf = RastriginFitness(n_dims=2)
        result = rf.evaluate({'x0': 1, 'x1': 1})
        assert 'raw_rastrigin' in result
        assert result['raw_rastrigin'] > 0


class TestAckleyFitness:
    def test_optimum_at_origin(self):
        af = AckleyFitness(n_dims=3)
        result = af.evaluate({'x0': 0, 'x1': 0, 'x2': 0})
        assert abs(result['fitness']) < 1e-10

    def test_negative_away_from_origin(self):
        af = AckleyFitness(n_dims=3)
        result = af.evaluate({'x0': 1, 'x1': 1, 'x2': 1})
        assert result['fitness'] < 0

    def test_raw_ackley_included(self):
        af = AckleyFitness(n_dims=2)
        result = af.evaluate({'x0': 1, 'x1': 0})
        assert 'raw_ackley' in result


class TestZDT1Fitness:
    def test_param_spec(self):
        zdt = ZDT1Fitness(n_dims=10)
        spec = zdt.param_spec()
        assert len(spec) == 10
        for name in spec:
            assert spec[name] == (0.0, 1.0)

    def test_pareto_optimal_point(self):
        """On the Pareto front, x_1..x_{n-1} = 0."""
        zdt = ZDT1Fitness(n_dims=5)
        result = zdt.evaluate({'x0': 0.5, 'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0})
        assert abs(result['f1'] - 0.5) < 1e-10
        # f2 = 1 - sqrt(f1/g) = 1 - sqrt(0.5) ≈ 0.293
        expected_f2 = 1 - np.sqrt(0.5)
        assert abs(result['f2'] - expected_f2) < 1e-6

    def test_dominated_point(self):
        """Non-zero x_1..x_{n-1} should give worse f2."""
        zdt = ZDT1Fitness(n_dims=5)
        optimal = zdt.evaluate({'x0': 0.5, 'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0})
        dominated = zdt.evaluate({'x0': 0.5, 'x1': 0.5, 'x2': 0.5, 'x3': 0.5, 'x4': 0.5})
        assert optimal['f2'] < dominated['f2']  # Lower f2 is better


class TestAllBenchmarks:
    def test_all_have_param_spec(self):
        """All benchmarks should implement param_spec correctly."""
        benchmarks = [
            SphereFitness(3), RosenbrockFitness(3),
            RastriginFitness(3), AckleyFitness(3), ZDT1Fitness(5),
        ]
        for bf in benchmarks:
            spec = bf.param_spec()
            assert isinstance(spec, dict)
            assert len(spec) > 0
            for name, (lo, hi) in spec.items():
                assert lo < hi

    def test_all_return_fitness(self):
        """All benchmarks should return dicts with 'fitness' key."""
        benchmarks = [
            SphereFitness(3), RosenbrockFitness(3),
            RastriginFitness(3), AckleyFitness(3), ZDT1Fitness(5),
        ]
        for bf in benchmarks:
            spec = bf.param_spec()
            params = {name: (lo + hi) / 2 for name, (lo, hi) in spec.items()}
            result = bf.evaluate(params)
            assert 'fitness' in result
            assert isinstance(result['fitness'], float)
