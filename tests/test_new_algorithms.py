"""
tests/test_new_algorithms.py -- Tests for DifferentialEvolution and CMAES.
"""

import numpy as np
import pytest

from ea_toolkit.algorithms import DifferentialEvolution, CMAES
from ea_toolkit.benchmarks import SphereFitness, RosenbrockFitness, RastriginFitness
from ea_toolkit.callbacks import HistoryRecorder


# ── Differential Evolution tests ────────────────────────────────────────────

class TestDifferentialEvolution:
    def test_converges_on_sphere(self):
        sf = SphereFitness(n_dims=5)
        de = DifferentialEvolution(sf, pop_size=30, F=0.8, CR=0.9, seed=42)
        history = de.run(budget=3000)
        best = de.best()
        assert best is not None
        assert best['fitness'] > -0.1, (
            f"DE should converge on sphere, got {best['fitness']}")

    def test_budget_respected(self):
        sf = SphereFitness(n_dims=3)
        de = DifferentialEvolution(sf, pop_size=10, seed=42)
        history = de.run(budget=50)
        assert len(history) == 50

    def test_history_records(self):
        sf = SphereFitness(n_dims=3)
        de = DifferentialEvolution(sf, pop_size=10, seed=42)
        de.run(budget=30)
        for entry in de.history:
            assert 'params' in entry
            assert 'fitness' in entry

    def test_ask_tell_interface(self):
        sf = SphereFitness(n_dims=3)
        de = DifferentialEvolution(sf, pop_size=10, seed=42)

        # First ask returns initial population
        candidates = de.ask()
        assert len(candidates) == 10
        assert all(isinstance(c, dict) for c in candidates)

        # Evaluate and tell
        evaluations = [(c, sf.evaluate(c)) for c in candidates]
        de.tell(evaluations)

        # Second ask returns trial vectors
        trials = de.ask()
        assert len(trials) == 10

    def test_ask_tell_matches_run(self):
        """Ask-tell should produce comparable results to run()."""
        sf = SphereFitness(n_dims=3)

        # Run mode
        de_run = DifferentialEvolution(sf, pop_size=10, seed=42)
        de_run.run(budget=100)

        # Ask-tell mode with same seed
        de_at = DifferentialEvolution(sf, pop_size=10, seed=42)
        while len(de_at.history) < 100:
            candidates = de_at.ask()
            evals = []
            for c in candidates:
                if len(de_at.history) + len(evals) >= 100:
                    break
                evals.append((c, sf.evaluate(c)))
            de_at.tell(evals)

        # Both should find reasonable solutions
        assert de_run.best()['fitness'] > -5.0
        assert de_at.best()['fitness'] > -5.0

    def test_on_rastrigin(self):
        """DE should handle multimodal functions."""
        rf = RastriginFitness(n_dims=3)
        de = DifferentialEvolution(rf, pop_size=30, seed=42)
        de.run(budget=3000)
        best = de.best()
        assert best is not None
        assert best['fitness'] > -10.0


# ── CMA-ES tests ───────────────────────────────────────────────────────────

class TestCMAES:
    def test_converges_on_sphere(self):
        sf = SphereFitness(n_dims=5)
        cma = CMAES(sf, sigma0=2.0, seed=42)
        history = cma.run(budget=2000)
        best = cma.best()
        assert best is not None
        assert best['fitness'] > -0.01, (
            f"CMA-ES should converge on sphere, got {best['fitness']}")

    def test_budget_respected(self):
        sf = SphereFitness(n_dims=3)
        cma = CMAES(sf, sigma0=1.0, seed=42)
        history = cma.run(budget=50)
        assert len(history) <= 50

    def test_history_records(self):
        sf = SphereFitness(n_dims=3)
        cma = CMAES(sf, sigma0=1.0, seed=42)
        cma.run(budget=30)
        for entry in cma.history:
            assert 'params' in entry
            assert 'fitness' in entry

    def test_ask_tell_interface(self):
        sf = SphereFitness(n_dims=3)
        cma = CMAES(sf, sigma0=1.0, seed=42)

        candidates = cma.ask()
        assert len(candidates) > 0
        assert all(isinstance(c, dict) for c in candidates)

        evaluations = [(c, sf.evaluate(c)) for c in candidates]
        cma.tell(evaluations)

        # Second generation
        candidates2 = cma.ask()
        assert len(candidates2) == len(candidates)

    def test_sigma_adapts(self):
        """Sigma should change over the course of optimization."""
        sf = SphereFitness(n_dims=5)
        cma = CMAES(sf, sigma0=2.0, seed=42)
        initial_sigma = cma.sigma
        cma.run(budget=500)
        final_sigma = cma.sigma
        assert final_sigma != initial_sigma, "Sigma should adapt"

    def test_properties(self):
        sf = SphereFitness(n_dims=3)
        cma = CMAES(sf, sigma0=1.0, seed=42)

        # Before initialization
        assert cma.mean is None
        assert cma.covariance is None

        cma.run(budget=30)

        # After running
        assert cma.mean is not None
        assert len(cma.mean) == 3
        assert cma.covariance is not None
        assert cma.covariance.shape == (3, 3)

    def test_custom_pop_size(self):
        sf = SphereFitness(n_dims=3)
        cma = CMAES(sf, sigma0=1.0, pop_size=20, seed=42)
        candidates = cma.ask()
        assert len(candidates) == 20

    def test_rosenbrock(self):
        """CMA-ES should make progress on Rosenbrock."""
        rf = RosenbrockFitness(n_dims=3, bounds=(-5.0, 10.0))
        cma = CMAES(rf, sigma0=3.0, seed=42)
        cma.run(budget=5000)
        best = cma.best()
        assert best is not None
        # Should at least find something better than a random point
        assert best['fitness'] > -1000.0

    def test_callbacks(self):
        sf = SphereFitness(n_dims=3)
        hr = HistoryRecorder()
        cma = CMAES(sf, sigma0=1.0, seed=42)
        cma.callbacks = [hr]
        cma.run(budget=100)
        assert len(hr.generations) > 0


# ── Cross-algorithm comparison ──────────────────────────────────────────────

class TestNewAlgorithmComparison:
    def test_de_vs_cmaes_sphere(self):
        """Both DE and CMA-ES should converge on sphere."""
        sf = SphereFitness(n_dims=5)
        budget = 3000

        de = DifferentialEvolution(sf, pop_size=30, seed=42)
        de.run(budget=budget)

        cma = CMAES(sf, sigma0=2.0, seed=42)
        cma.run(budget=budget)

        assert de.best()['fitness'] > -1.0
        assert cma.best()['fitness'] > -1.0

    def test_all_eight_algorithms_sphere(self):
        """All 8 algorithms should work on sphere."""
        from ea_toolkit.algorithms import (
            HillClimber, OnePlusLambdaES, RidgeWalker,
            CliffMapper, NoveltySeeker, EnsembleExplorer,
        )
        sf = SphereFitness(n_dims=3)
        algorithms = [
            HillClimber(sf, seed=42),
            OnePlusLambdaES(sf, lam=5, seed=42),
            RidgeWalker(sf, n_candidates=3, seed=42),
            CliffMapper(sf, n_probes=5, seed=42),
            NoveltySeeker(sf, n_candidates=3, k_nearest=3, seed=42),
            EnsembleExplorer(sf, n_walkers=5, seed=42),
            DifferentialEvolution(sf, pop_size=10, seed=42),
            CMAES(sf, sigma0=2.0, seed=42),
        ]
        for algo in algorithms:
            history = algo.run(budget=50)
            assert len(history) <= 50, (
                f"{algo.__class__.__name__} exceeded budget")
            best = algo.best()
            assert best is not None
            assert 'fitness' in best
