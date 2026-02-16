"""
tests/test_zimmerman_bridge.py -- Tests for the Zimmerman-toolkit bridge.

Adapter tests (FitnessAsSimulator, SimulatorAsFitness) always run.
Integration tests that require the zimmerman package are skipped if
zimmerman is not importable.
"""

import numpy as np
import pytest

from ea_toolkit.benchmarks import SphereFitness, RastriginFitness
from ea_toolkit.zimmerman_bridge import (
    FitnessAsSimulator,
    SimulatorAsFitness,
)


# ── Adapter Tests (no zimmerman dependency) ──────────────────────────────────


class TestFitnessAsSimulator:
    def test_run_delegates_to_evaluate(self):
        sf = SphereFitness(n_dims=3)
        sim = FitnessAsSimulator(sf)
        result = sim.run({'x0': 1.0, 'x1': 0.0, 'x2': 0.0})
        assert 'fitness' in result
        assert result['fitness'] == -1.0

    def test_param_spec_passthrough(self):
        sf = SphereFitness(n_dims=5, bounds=(-10, 10))
        sim = FitnessAsSimulator(sf)
        spec = sim.param_spec()
        assert len(spec) == 5
        assert spec['x0'] == (-10, 10)

    def test_all_keys_preserved(self):
        """Extra keys from evaluate() should be visible to Zimmerman tools."""
        rf = RastriginFitness(n_dims=2)
        sim = FitnessAsSimulator(rf)
        result = sim.run({'x0': 1.0, 'x1': 0.0})
        assert 'fitness' in result
        assert 'raw_rastrigin' in result

    def test_multiple_calls(self):
        sf = SphereFitness(n_dims=2)
        sim = FitnessAsSimulator(sf)
        r1 = sim.run({'x0': 0.0, 'x1': 0.0})
        r2 = sim.run({'x0': 1.0, 'x1': 1.0})
        assert r1['fitness'] > r2['fitness']


class TestSimulatorAsFitness:
    def _make_dummy_simulator(self):
        """Create a duck-typed Simulator without importing zimmerman."""
        class DummySim:
            def run(self, params):
                x = params.get('x', 0)
                y = params.get('y', 0)
                return {'cost': x**2 + y**2, 'dist': abs(x) + abs(y)}

            def param_spec(self):
                return {'x': (-5.0, 5.0), 'y': (-5.0, 5.0)}

        return DummySim()

    def test_evaluate_wraps_run(self):
        sim = self._make_dummy_simulator()
        fitness = SimulatorAsFitness(sim, fitness_key='cost')
        result = fitness.evaluate({'x': 3.0, 'y': 4.0})
        assert result['fitness'] == 25.0
        assert result['cost'] == 25.0  # Original key preserved

    def test_negate(self):
        sim = self._make_dummy_simulator()
        fitness = SimulatorAsFitness(sim, fitness_key='cost', negate=True)
        result = fitness.evaluate({'x': 3.0, 'y': 4.0})
        assert result['fitness'] == -25.0

    def test_param_spec_passthrough(self):
        sim = self._make_dummy_simulator()
        fitness = SimulatorAsFitness(sim, fitness_key='cost')
        spec = fitness.param_spec()
        assert spec == {'x': (-5.0, 5.0), 'y': (-5.0, 5.0)}

    def test_works_with_ea_algorithm(self):
        """SimulatorAsFitness should work with ea-toolkit algorithms."""
        from ea_toolkit import HillClimber

        sim = self._make_dummy_simulator()
        fitness = SimulatorAsFitness(sim, fitness_key='cost', negate=True)
        hc = HillClimber(fitness, seed=42)
        hc.run(budget=50)
        best = hc.best()
        assert best is not None
        assert 'fitness' in best

    def test_default_fitness_key(self):
        """If the simulator already returns 'fitness', no key needed."""
        class FitnessSim:
            def run(self, params):
                return {'fitness': -params['x']**2}
            def param_spec(self):
                return {'x': (-5.0, 5.0)}

        fitness = SimulatorAsFitness(FitnessSim())
        result = fitness.evaluate({'x': 2.0})
        assert result['fitness'] == -4.0


class TestRoundTrip:
    def test_fitness_to_sim_and_back(self):
        """FitnessFunction -> Simulator -> FitnessFunction should preserve behavior."""
        sf = SphereFitness(n_dims=3)
        sim = FitnessAsSimulator(sf)
        fitness_back = SimulatorAsFitness(sim, fitness_key='fitness')

        params = {'x0': 1.0, 'x1': 2.0, 'x2': 3.0}
        original = sf.evaluate(params)
        roundtrip = fitness_back.evaluate(params)
        assert original['fitness'] == roundtrip['fitness']


# ── Integration Tests (require zimmerman-toolkit) ───────────────────────────

try:
    import zimmerman
    HAS_ZIMMERMAN = True
except ImportError:
    HAS_ZIMMERMAN = False

zimmerman_required = pytest.mark.skipif(
    not HAS_ZIMMERMAN,
    reason="zimmerman-toolkit not importable"
)


@zimmerman_required
class TestSobolOnFitness:
    def test_sobol_on_sphere(self):
        from ea_toolkit.zimmerman_bridge import sobol_on_fitness

        sf = SphereFitness(n_dims=3)
        result = sobol_on_fitness(sf, n_base=32, seed=42)
        assert 'fitness' in result
        assert 'S1' in result['fitness']
        assert 'ST' in result['fitness']
        # All 3 params should have similar sensitivity on sphere
        s1 = result['fitness']['S1']
        assert len(s1) == 3


@zimmerman_required
class TestFalsifyFitness:
    def test_sphere_is_clean(self):
        from ea_toolkit.zimmerman_bridge import falsify_fitness

        sf = SphereFitness(n_dims=3)
        report = falsify_fitness(sf, n_random=20, n_boundary=10,
                                 n_adversarial=10, seed=42)
        assert report['summary']['violations_found'] == 0


@zimmerman_required
class TestContrastiveAroundBest:
    def test_contrastive_after_optimization(self):
        from ea_toolkit import DifferentialEvolution
        from ea_toolkit.zimmerman_bridge import contrastive_around_best

        sf = SphereFitness(n_dims=3)
        de = DifferentialEvolution(sf, pop_size=10, seed=42)
        de.run(budget=100)

        result = contrastive_around_best(
            de, sf,
            outcome_fn=lambda r: 'good' if r.get('fitness', 0) > -1.0 else 'bad',
            n_attempts=20, seed=42,
        )
        assert 'found' in result


@zimmerman_required
class TestOptimizeAndInterrogate:
    def test_full_pipeline(self):
        from ea_toolkit import CMAES
        from ea_toolkit.zimmerman_bridge import optimize_and_interrogate

        sf = SphereFitness(n_dims=3)
        report = optimize_and_interrogate(
            sf, CMAES,
            algorithm_kwargs={'sigma0': 2.0},
            budget=100,
            sobol_n_base=32,
            seed=42,
        )
        assert 'best' in report
        assert report['best'] is not None
        assert 'sobol' in report
        assert 'falsification' in report
        assert 'contrastive' in report
        assert report['falsification']['summary']['violations_found'] == 0


@zimmerman_required
class TestZimmermanProtocolCompliance:
    def test_adapter_satisfies_simulator_protocol(self):
        """FitnessAsSimulator should satisfy zimmerman's Simulator protocol."""
        from zimmerman.base import Simulator

        sf = SphereFitness(n_dims=3)
        sim = FitnessAsSimulator(sf)
        assert isinstance(sim, Simulator)
