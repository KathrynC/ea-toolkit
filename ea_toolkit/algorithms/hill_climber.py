"""
ea_toolkit.algorithms.hill_climber -- Parallel hill climber with restarts.

Extracted from walker_competition.py run_hill_climber(): greedy local search
that accepts mutations only when they strictly improve fitness. Supports
multiple restarts to escape local optima.
"""

import numpy as np

from ea_toolkit.base import Algorithm, FitnessFunction, MutationOperator
from ea_toolkit.mutation import GaussianMutation


class HillClimber(Algorithm):
    """Parallel hill climber with configurable restarts.

    Each restart performs greedy local search: mutate the current best,
    accept only if fitness strictly improves. The overall best across
    all restarts is tracked.

    Extracted from walker_competition.py run_hill_climber().

    Args:
        fitness_fn: fitness function to optimize.
        mutation: mutation operator. Defaults to GaussianMutation(sigma=0.1).
        n_restarts: number of independent restarts. Default 1.
        seed: random seed for reproducibility.
    """

    def __init__(self, fitness_fn: FitnessFunction,
                 mutation: MutationOperator | None = None,
                 n_restarts: int = 1,
                 seed: int | None = None):
        if mutation is None:
            mutation = GaussianMutation(sigma=0.1)
        super().__init__(fitness_fn, mutation=mutation, seed=seed)
        self.n_restarts = n_restarts

    def run(self, budget: int) -> list[dict]:
        """Run the hill climber with the given evaluation budget.

        Divides the budget evenly among restarts. Each restart:
        1. Samples a random starting point.
        2. Iteratively mutates and accepts strict improvements.

        Args:
            budget: total number of fitness evaluations to perform.

        Returns:
            list of all evaluation history entries.
        """
        param_spec = self.fitness_fn.param_spec()
        names = sorted(param_spec.keys())

        # Divide budget among restarts
        budget_per_restart = max(1, budget // self.n_restarts)

        for restart in range(self.n_restarts):
            remaining = min(budget_per_restart,
                            budget - len(self.history))
            if remaining <= 0:
                break

            # Random starting point
            current_params = {}
            for name in names:
                lo, hi = param_spec[name]
                current_params[name] = float(self.rng.uniform(lo, hi))

            current_result = self.fitness_fn.evaluate(current_params)
            self._record(current_params, current_result)
            remaining -= 1

            current_fitness = current_result.get('fitness', float('-inf'))

            # Greedy local search
            for _ in range(remaining):
                if len(self.history) >= budget:
                    break

                # Mutate
                new_params = self.mutation.mutate(current_params, param_spec,
                                                  self.rng)
                new_result = self.fitness_fn.evaluate(new_params)
                self._record(new_params, new_result)

                new_fitness = new_result.get('fitness', float('-inf'))

                # Accept only strict improvement
                if new_fitness > current_fitness:
                    current_params = new_params
                    current_fitness = new_fitness

        return self.history
