"""
tests/test_callbacks.py -- Tests for the callback system.
"""

import numpy as np
import pytest

from ea_toolkit.base import Callback
from ea_toolkit.callbacks import (
    ConvergenceChecker,
    ProgressPrinter,
    HistoryRecorder,
)
from ea_toolkit.algorithms import HillClimber, OnePlusLambdaES, DifferentialEvolution
from ea_toolkit.mutation import GaussianMutation
from tests.conftest import SphereFitness


class TestCallbackBase:
    def test_callback_has_all_methods(self):
        cb = Callback()
        cb.on_start(None)
        cb.on_generation(None, 0, 0.0)
        cb.on_improvement(None, 0.0, 1.0)
        cb.on_finish(None)


class TestConvergenceChecker:
    def test_stops_when_stale(self):
        cc = ConvergenceChecker(patience=5, min_delta=1e-8)
        # Simulate 10 generations with no improvement
        for gen in range(10):
            result = cc.on_generation(None, gen, 1.0)
            if gen < 5:
                assert result is None  # Not yet stale
            else:
                assert result is False  # Should request stop

    def test_resets_on_improvement(self):
        cc = ConvergenceChecker(patience=5, min_delta=0.01)
        for gen in range(4):
            cc.on_generation(None, gen, 1.0)
        # Improvement resets counter
        cc.on_generation(None, 4, 2.0)
        for gen in range(5, 9):
            result = cc.on_generation(None, gen, 2.0)
            assert result is None  # Not yet stale again

    def test_on_start_resets(self):
        cc = ConvergenceChecker(patience=3)
        for gen in range(5):
            cc.on_generation(None, gen, 1.0)
        cc.on_start(None)
        result = cc.on_generation(None, 0, 1.0)
        assert result is None  # Reset


class TestHistoryRecorder:
    def test_records_generations(self):
        hr = HistoryRecorder()
        hr.on_start(None)
        hr.on_generation(None, 0, 1.0)
        hr.on_generation(None, 1, 2.0)
        hr.on_generation(None, 2, 2.0)
        assert len(hr.generations) == 3
        assert hr.generations[0]['improved'] is True
        assert hr.generations[1]['improved'] is True
        assert hr.generations[2]['improved'] is False


class TestCallbackIntegration:
    def test_callbacks_on_algorithm(self):
        """Callbacks set on algorithm.callbacks should work with DE."""
        sf = SphereFitness(n_dims=3)
        hr = HistoryRecorder()
        de = DifferentialEvolution(sf, pop_size=10, seed=42)
        de.callbacks = [hr]
        de.run(budget=100)

        assert len(hr.generations) > 0
        assert hr.generations[0]['best_fitness'] is not None

    def test_convergence_early_stop(self):
        """ConvergenceChecker should cause early stopping in DE."""
        sf = SphereFitness(n_dims=3)
        cc = ConvergenceChecker(patience=3, min_delta=100.0)  # Impossible threshold
        de = DifferentialEvolution(sf, pop_size=10, seed=42)
        de.callbacks = [cc]
        history = de.run(budget=10000)
        # Should stop well before 10000 evals due to early stopping
        assert len(history) < 10000
