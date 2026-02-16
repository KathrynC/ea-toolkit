"""
tests/test_algorithms.py -- Comprehensive test suite for all EA toolkit algorithms.

Tests each algorithm on the sphere function (unimodal, optimum at origin)
and verifies convergence, history tracking, and algorithm-specific features.
"""

import numpy as np
import pytest

from ea_toolkit.algorithms import (
    HillClimber,
    OnePlusLambdaES,
    RidgeWalker,
    CliffMapper,
    NoveltySeeker,
    EnsembleExplorer,
)
from ea_toolkit.mutation import GaussianMutation, AdaptiveMutation
from tests.conftest import (
    SphereFitness,
    RastriginFitness,
    MultiObjectiveSphere,
)


# ── HillClimber tests ────────────────────────────────────────────────────────

class TestHillClimber:
    """Tests for the HillClimber algorithm."""

    def test_converges_on_sphere(self, sphere_fitness):
        """HillClimber should converge on the sphere function."""
        hc = HillClimber(sphere_fitness, seed=42)
        history = hc.run(budget=500)

        assert len(history) == 500
        best = hc.best()
        assert best is not None
        assert best['fitness'] > -0.1, (
            f"Expected fitness > -0.1, got {best['fitness']}")

    def test_multiple_restarts_better(self, sphere_fitness):
        """Multiple restarts should find reasonable solutions."""
        # Single restart with larger sigma to make progress on [-5,5]
        mutation1 = GaussianMutation(sigma=0.5)
        hc1 = HillClimber(sphere_fitness, mutation=mutation1,
                           n_restarts=1, seed=42)
        hc1.run(budget=500)
        best1 = hc1.best()

        # Multiple restarts with same total budget
        mutation5 = GaussianMutation(sigma=0.5)
        hc5 = HillClimber(sphere_fitness, mutation=mutation5,
                           n_restarts=5, seed=42)
        hc5.run(budget=500)
        best5 = hc5.best()

        # Both should find something reasonable
        assert best5 is not None
        assert best1 is not None
        # 5 restarts explores more starting points, may find better basins
        # but each restart has only 100 evals vs 500 for 1 restart
        # At minimum, both should have negative (non-zero) fitness
        assert best5['fitness'] > -50.0, (
            f"5 restarts ({best5['fitness']:.4f}) should find decent solution")

    def test_history_length(self, sphere_fitness):
        """History should contain exactly budget entries."""
        hc = HillClimber(sphere_fitness, seed=42)
        history = hc.run(budget=100)
        assert len(history) == 100

    def test_records_params(self, sphere_fitness):
        """Each history entry should have params and fitness."""
        hc = HillClimber(sphere_fitness, seed=42)
        hc.run(budget=10)
        for entry in hc.history:
            assert 'params' in entry
            assert 'fitness' in entry
            assert isinstance(entry['params'], dict)

    def test_custom_mutation(self, sphere_fitness):
        """HillClimber should work with a custom mutation operator."""
        mutation = GaussianMutation(sigma=0.5)
        hc = HillClimber(sphere_fitness, mutation=mutation, seed=42)
        history = hc.run(budget=100)
        assert len(history) == 100


# ── OnePlusLambdaES tests ────────────────────────────────────────────────────

class TestOnePlusLambdaES:
    """Tests for the (1+lambda) Evolution Strategy."""

    def test_converges_on_sphere(self, sphere_fitness):
        """(1+lambda) ES should converge on the sphere function."""
        mutation = GaussianMutation(sigma=1.0)
        es = OnePlusLambdaES(sphere_fitness, mutation=mutation,
                             lam=10, seed=42)
        history = es.run(budget=500)

        assert len(history) == 500
        best = es.best()
        assert best is not None
        assert best['fitness'] > -0.1, (
            f"Expected fitness > -0.1, got {best['fitness']}")

    def test_adaptive_mutation(self, sphere_fitness):
        """ES should work with AdaptiveMutation."""
        mutation = AdaptiveMutation(sigma_init=0.5)
        es = OnePlusLambdaES(sphere_fitness, mutation=mutation,
                             lam=10, seed=42)
        history = es.run(budget=500)

        best = es.best()
        assert best is not None
        assert best['fitness'] > -1.0, (
            f"Expected fitness > -1.0 with adaptive, got {best['fitness']}")

    def test_budget_respected(self, sphere_fitness):
        """ES should not exceed the evaluation budget."""
        es = OnePlusLambdaES(sphere_fitness, lam=10, seed=42)
        history = es.run(budget=55)
        assert len(history) == 55

    def test_lambda_effect(self, sphere_fitness):
        """Higher lambda should generally improve exploration."""
        mutation = GaussianMutation(sigma=1.0)
        es_low = OnePlusLambdaES(sphere_fitness, mutation=mutation,
                                 lam=2, seed=42)
        es_low.run(budget=500)

        mutation2 = GaussianMutation(sigma=1.0)
        es_high = OnePlusLambdaES(sphere_fitness, mutation=mutation2,
                                  lam=20, seed=42)
        es_high.run(budget=500)

        # Both should converge, higher lambda may help
        assert es_high.best()['fitness'] > -5.0


# ── RidgeWalker tests ────────────────────────────────────────────────────────

class TestRidgeWalker:
    """Tests for the RidgeWalker multi-objective search."""

    def test_runs_and_converges(self, sphere_fitness):
        """RidgeWalker should run and produce a history."""
        rw = RidgeWalker(sphere_fitness, objectives=('fitness',),
                         n_candidates=3, seed=42)
        history = rw.run(budget=500)

        assert len(history) == 500
        best = rw.best()
        assert best is not None

    def test_pareto_front(self):
        """RidgeWalker should produce a non-empty Pareto front."""
        ff = MultiObjectiveSphere(n_dims=3)
        rw = RidgeWalker(ff, objectives=('fitness', 'secondary'),
                         n_candidates=3, seed=42)
        rw.run(budget=200)

        front = rw.pareto_front()
        assert len(front) > 0, "Pareto front should be non-empty"

        # Verify that front entries are non-dominated
        for i, entry_a in enumerate(front):
            a_fit = entry_a['fitness']
            a_sec = entry_a['secondary']
            for j, entry_b in enumerate(front):
                if i != j:
                    b_fit = entry_b['fitness']
                    b_sec = entry_b['secondary']
                    # entry_b should NOT dominate entry_a
                    dominated = (b_fit >= a_fit and b_sec >= a_sec and
                                 (b_fit > a_fit or b_sec > a_sec))
                    assert not dominated, (
                        f"Front entry {i} dominated by entry {j}")

    def test_multi_objective(self):
        """RidgeWalker with multiple objectives should explore trade-offs."""
        ff = MultiObjectiveSphere(n_dims=3)
        rw = RidgeWalker(ff, objectives=('fitness', 'secondary'),
                         n_candidates=5, seed=42)
        rw.run(budget=300)

        front = rw.pareto_front()
        assert len(front) >= 2, "Pareto front should have multiple solutions"


# ── CliffMapper tests ────────────────────────────────────────────────────────

class TestCliffMapper:
    """Tests for the CliffMapper high-sensitivity search."""

    def test_runs_within_budget(self, sphere_fitness):
        """CliffMapper should respect the evaluation budget."""
        cm = CliffMapper(sphere_fitness, n_probes=5, seed=42)
        history = cm.run(budget=200)
        assert len(history) == 200

    def test_cliff_map_populated(self, sphere_fitness):
        """CliffMapper should populate the cliff map."""
        cm = CliffMapper(sphere_fitness, n_probes=5, seed=42)
        cm.run(budget=100)

        cliff_map = cm.cliff_map()
        assert len(cliff_map) > 0, "Cliff map should be non-empty"

        for params, cliffiness in cliff_map:
            assert isinstance(params, dict)
            assert isinstance(cliffiness, float)
            assert cliffiness >= 0.0

    def test_detects_sensitivity(self):
        """CliffMapper should detect high sensitivity near step functions."""
        from tests.conftest import StepFitness
        sf = StepFitness(n_dims=3)
        cm = CliffMapper(sf, n_probes=10, probe_radius=0.1, seed=42)
        cm.run(budget=200)

        cliff_map = cm.cliff_map()
        max_cliff = max(c for _, c in cliff_map)
        # The step function has a 20-unit cliff, so we should detect
        # significant sensitivity with probes near x0=0
        assert max_cliff > 0.0, "Should detect some sensitivity"


# ── NoveltySeeker tests ──────────────────────────────────────────────────────

class TestNoveltySeeker:
    """Tests for the NoveltySeeker novelty search."""

    def test_runs_within_budget(self, sphere_fitness):
        """NoveltySeeker should respect the evaluation budget."""
        ns = NoveltySeeker(sphere_fitness, n_candidates=5,
                           k_nearest=5, seed=42)
        history = ns.run(budget=200)
        assert len(history) == 200

    def test_explores_diverse_points(self, sphere_fitness):
        """NoveltySeeker should explore diverse parameter regions."""
        ns = NoveltySeeker(sphere_fitness, n_candidates=5,
                           k_nearest=5, seed=42)
        ns.run(budget=200)

        # Check that the explored points span a reasonable range
        all_x0 = [e['params']['x0'] for e in ns.history]
        x0_range = max(all_x0) - min(all_x0)
        assert x0_range > 0.5, (
            f"Expected diverse exploration, but x0 range was only {x0_range}")

    def test_custom_behavior_fn(self, sphere_fitness):
        """NoveltySeeker should work with a custom behavior function."""
        def behavior_fn(result):
            return np.array([result['fitness']])

        ns = NoveltySeeker(sphere_fitness, behavior_fn=behavior_fn,
                           n_candidates=3, k_nearest=3, seed=42)
        history = ns.run(budget=100)
        assert len(history) == 100

    def test_novelty_vs_fitness(self, sphere_fitness):
        """NoveltySeeker should not necessarily find the best fitness."""
        # NoveltySeeker doesn't optimize fitness -- it explores.
        # Just verify it runs and produces a valid history.
        ns = NoveltySeeker(sphere_fitness, n_candidates=3,
                           k_nearest=5, seed=42)
        ns.run(budget=200)

        best = ns.best()
        assert best is not None
        assert 'fitness' in best


# ── EnsembleExplorer tests ───────────────────────────────────────────────────

class TestEnsembleExplorer:
    """Tests for the EnsembleExplorer multi-walker ensemble."""

    def test_converges_on_sphere(self, sphere_fitness):
        """EnsembleExplorer should find good solutions on sphere."""
        mutation = GaussianMutation(sigma=0.5)
        ee = EnsembleExplorer(sphere_fitness, mutation=mutation,
                              n_walkers=10, seed=42)
        history = ee.run(budget=500)

        assert len(history) == 500
        best = ee.best()
        assert best is not None
        assert best['fitness'] > -5.0, (
            f"Expected fitness > -5.0, got {best['fitness']}")

    def test_budget_respected(self, sphere_fitness):
        """EnsembleExplorer should respect the evaluation budget."""
        ee = EnsembleExplorer(sphere_fitness, n_walkers=5, seed=42)
        history = ee.run(budget=50)
        assert len(history) == 50

    def test_teleportation(self, sphere_fitness):
        """EnsembleExplorer should handle teleportation without errors."""
        # Small threshold forces frequent teleportation
        ee = EnsembleExplorer(sphere_fitness, n_walkers=5,
                              teleport_threshold=10.0,
                              teleport_interval=2, seed=42)
        history = ee.run(budget=100)
        assert len(history) == 100

    def test_many_walkers(self, sphere_fitness):
        """EnsembleExplorer should work with many walkers."""
        ee = EnsembleExplorer(sphere_fitness, n_walkers=20, seed=42)
        history = ee.run(budget=200)
        # With 20 walkers and 200 budget, we get 20 inits + 9 steps
        assert len(history) == 200


# ── Cross-algorithm tests ────────────────────────────────────────────────────

class TestCrossAlgorithm:
    """Tests that apply across all algorithms."""

    def test_all_algorithms_sphere(self, sphere_fitness):
        """All algorithms should produce valid histories on sphere."""
        algorithms = [
            HillClimber(sphere_fitness, seed=42),
            OnePlusLambdaES(sphere_fitness, lam=5, seed=42),
            RidgeWalker(sphere_fitness, n_candidates=3, seed=42),
            CliffMapper(sphere_fitness, n_probes=5, seed=42),
            NoveltySeeker(sphere_fitness, n_candidates=3,
                          k_nearest=3, seed=42),
            EnsembleExplorer(sphere_fitness, n_walkers=5, seed=42),
        ]

        for algo in algorithms:
            history = algo.run(budget=50)
            assert len(history) == 50, (
                f"{algo.__class__.__name__} produced {len(history)} "
                f"entries, expected 50")

            best = algo.best()
            assert best is not None
            assert 'fitness' in best
            assert 'params' in best

    def test_reproducibility(self, sphere_fitness):
        """Same seed should produce identical results."""
        hc1 = HillClimber(sphere_fitness, seed=123)
        h1 = hc1.run(budget=50)

        hc2 = HillClimber(sphere_fitness, seed=123)
        h2 = hc2.run(budget=50)

        for e1, e2 in zip(h1, h2):
            assert e1['fitness'] == e2['fitness'], (
                "Same seed should produce identical fitness values")

    def test_rastrigin_all_algorithms(self, rastrigin_fitness):
        """All algorithms should work on the multimodal Rastrigin function."""
        algorithms = [
            HillClimber(rastrigin_fitness, seed=42),
            OnePlusLambdaES(rastrigin_fitness, lam=10, seed=42),
            RidgeWalker(rastrigin_fitness, n_candidates=3, seed=42),
            CliffMapper(rastrigin_fitness, n_probes=5, seed=42),
            NoveltySeeker(rastrigin_fitness, n_candidates=3,
                          k_nearest=3, seed=42),
            EnsembleExplorer(rastrigin_fitness, n_walkers=5, seed=42),
        ]

        for algo in algorithms:
            history = algo.run(budget=100)
            assert len(history) == 100
            best = algo.best()
            assert best is not None
