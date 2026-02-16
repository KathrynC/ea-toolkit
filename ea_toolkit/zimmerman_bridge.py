"""
ea_toolkit.zimmerman_bridge -- Bidirectional bridge to the Zimmerman Toolkit.

The ea-toolkit and zimmerman-toolkit share nearly identical simulator
protocols but name the methods differently:

    ea-toolkit FitnessFunction:   evaluate(params) -> dict, param_spec() -> dict
    zimmerman Simulator:          run(params) -> dict,      param_spec() -> dict

This module provides adapters to cross-connect the two toolkits, plus
convenience functions that combine evolutionary optimization with
Zimmerman's black-box interrogation tools (Sobol sensitivity, falsification,
contrastive generation, POSIWID auditing).

The zimmerman-toolkit is an optional dependency. The bridge module
imports it lazily so the core ea-toolkit works without it installed.
"""

from __future__ import annotations

from ea_toolkit.base import FitnessFunction


# ── Adapters ─────────────────────────────────────────────────────────────────


class FitnessAsSimulator:
    """Wrap an ea-toolkit FitnessFunction as a Zimmerman Simulator.

    Delegates run() to evaluate(). The returned dict is passed through
    unchanged, so all keys (including 'fitness') are visible to Zimmerman
    analysis tools.

    Example:
        from ea_toolkit.benchmarks import SphereFitness
        from ea_toolkit.zimmerman_bridge import FitnessAsSimulator
        from zimmerman import sobol_sensitivity

        sf = SphereFitness(n_dims=5)
        sim = FitnessAsSimulator(sf)
        result = sobol_sensitivity(sim, n_base=128)
    """

    def __init__(self, fitness_fn: FitnessFunction):
        self._fitness_fn = fitness_fn

    def run(self, params: dict) -> dict:
        """Execute the fitness function (Zimmerman Simulator protocol)."""
        return self._fitness_fn.evaluate(params)

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds (shared protocol)."""
        return self._fitness_fn.param_spec()


class SimulatorAsFitness(FitnessFunction):
    """Wrap a Zimmerman Simulator as an ea-toolkit FitnessFunction.

    Delegates evaluate() to run(). A fitness_key specifies which output
    key becomes the 'fitness' value required by ea-toolkit algorithms.

    Example:
        from zimmerman.base import SimulatorWrapper
        from ea_toolkit.zimmerman_bridge import SimulatorAsFitness
        from ea_toolkit import CMAES

        def my_model(params):
            return {'cost': params['x']**2 + params['y']**2}

        sim = SimulatorWrapper(my_model, {'x': (-5, 5), 'y': (-5, 5)})
        fitness = SimulatorAsFitness(sim, fitness_key='cost', negate=True)
        cma = CMAES(fitness, sigma0=2.0, seed=42)
        cma.run(budget=1000)
    """

    def __init__(self, simulator, fitness_key: str = 'fitness',
                 negate: bool = False):
        """
        Args:
            simulator: Object with run(params) and param_spec() methods.
            fitness_key: Which key in the simulator's output dict to use
                as the fitness value. Default 'fitness'.
            negate: If True, negate the value (for converting minimization
                outputs to ea-toolkit's maximization convention).
        """
        self._simulator = simulator
        self._fitness_key = fitness_key
        self._negate = negate

    def evaluate(self, params: dict) -> dict:
        result = self._simulator.run(params)
        raw = result.get(self._fitness_key, 0.0)
        fitness = -float(raw) if self._negate else float(raw)
        return {**result, 'fitness': fitness}

    def param_spec(self) -> dict[str, tuple[float, float]]:
        return self._simulator.param_spec()


# ── Convenience Functions ────────────────────────────────────────────────────


def sobol_on_fitness(fitness_fn: FitnessFunction,
                     n_base: int = 256,
                     seed: int = 42,
                     output_keys: list[str] | None = None) -> dict:
    """Run Zimmerman Sobol sensitivity analysis on a FitnessFunction.

    Identifies which parameters drive the most variance in each output.

    Args:
        fitness_fn: ea-toolkit fitness function to analyze.
        n_base: Saltelli base sample count. Total sims = n_base * (D + 2).
        seed: Random seed.
        output_keys: Specific outputs to analyze. Default: all numeric.

    Returns:
        Sobol result dict with S1, ST, interaction indices per output,
        plus rankings. See zimmerman.sobol.sobol_sensitivity docs.
    """
    from zimmerman import sobol_sensitivity
    sim = FitnessAsSimulator(fitness_fn)
    return sobol_sensitivity(sim, n_base=n_base, seed=seed,
                             output_keys=output_keys)


def falsify_fitness(fitness_fn: FitnessFunction,
                    assertions: list | None = None,
                    n_random: int = 100,
                    n_boundary: int = 50,
                    n_adversarial: int = 50,
                    seed: int = 42) -> dict:
    """Run Zimmerman falsification on a FitnessFunction.

    Systematically probes for NaN, Inf, assertion violations, and
    exceptions using random, boundary, and adversarial strategies.

    Args:
        fitness_fn: ea-toolkit fitness function to test.
        assertions: List of callables (result_dict -> bool). Default checks
            for NaN, Inf, and all-finite values.
        n_random: Random sample count.
        n_boundary: Boundary test count (corners, edges, faces).
        n_adversarial: Perturbation count near found violations.
        seed: Random seed.

    Returns:
        Falsification report with violations list and summary stats.
        See zimmerman.falsifier.Falsifier docs.
    """
    from zimmerman import Falsifier
    sim = FitnessAsSimulator(fitness_fn)
    f = Falsifier(sim, assertions=assertions)
    return f.falsify(n_random=n_random, n_boundary=n_boundary,
                     n_adversarial=n_adversarial, seed=seed)


def contrastive_around_best(algorithm, fitness_fn: FitnessFunction,
                            outcome_fn=None,
                            n_attempts: int = 100,
                            max_delta_frac: float = 0.1,
                            seed: int = 42) -> dict:
    """Find the smallest parameter change that flips the outcome near
    the algorithm's best solution.

    This reveals fragility: if a tiny perturbation flips the outcome,
    the algorithm converged to a knife-edge region.

    Args:
        algorithm: An ea-toolkit Algorithm that has been run (has a best()).
        fitness_fn: The fitness function used.
        outcome_fn: Categorizes results (default: sign of fitness).
        n_attempts: Number of random directions to try.
        max_delta_frac: Search within this fraction of parameter ranges.
        seed: Random seed.

    Returns:
        Contrastive result dict with found, delta, perturbation_magnitude.
        See zimmerman.contrastive.ContrastiveGenerator docs.
    """
    from zimmerman import ContrastiveGenerator
    sim = FitnessAsSimulator(fitness_fn)
    gen = ContrastiveGenerator(sim, outcome_fn=outcome_fn)
    best = algorithm.best()
    if best is None:
        return {'found': False, 'error': 'Algorithm has no best solution'}
    return gen.find_contrastive(best['params'], n_attempts=n_attempts,
                                max_delta_frac=max_delta_frac, seed=seed)


def optimize_and_interrogate(fitness_fn: FitnessFunction,
                             algorithm_cls,
                             algorithm_kwargs: dict | None = None,
                             budget: int = 1000,
                             sobol_n_base: int = 128,
                             falsify: bool = True,
                             contrastive: bool = True,
                             seed: int = 42) -> dict:
    """Full pipeline: optimize, then interrogate with Zimmerman tools.

    1. Run an evolutionary algorithm to find the best solution.
    2. Run Sobol sensitivity to identify which parameters matter.
    3. Optionally falsify the fitness function for numerical stability.
    4. Optionally find contrastive pairs near the optimum.

    Args:
        fitness_fn: Problem to solve and interrogate.
        algorithm_cls: Algorithm class (e.g., CMAES, DifferentialEvolution).
        algorithm_kwargs: Extra keyword args for the algorithm constructor.
        budget: Evaluation budget for the optimizer.
        sobol_n_base: Saltelli base sample count for sensitivity analysis.
        falsify: Whether to run falsification.
        contrastive: Whether to find contrastive pairs near optimum.
        seed: Random seed (shared by optimizer and analysis).

    Returns:
        Dict with keys:
            'best': best solution found by the algorithm
            'history_length': number of evaluations used
            'sobol': Sobol sensitivity result dict
            'falsification': falsification report (if falsify=True)
            'contrastive': contrastive result (if contrastive=True)
    """
    kwargs = dict(algorithm_kwargs or {})
    kwargs.setdefault('seed', seed)
    algo = algorithm_cls(fitness_fn, **kwargs)
    algo.run(budget=budget)

    report = {
        'best': algo.best(),
        'history_length': len(algo.history),
        'sobol': sobol_on_fitness(fitness_fn, n_base=sobol_n_base, seed=seed),
    }

    if falsify:
        report['falsification'] = falsify_fitness(fitness_fn, seed=seed)

    if contrastive:
        report['contrastive'] = contrastive_around_best(
            algo, fitness_fn, seed=seed)

    return report
