"""
ea_toolkit.algorithms.es -- (1+lambda) Evolution Strategy.

Extracted from temporal_optimizer.py evolve(): each generation produces
lambda children via mutation, keeps the best of parent-or-children.
Supports adaptive sigma via the 1/5th success rule.
"""

import numpy as np

from ea_toolkit.base import Algorithm, FitnessFunction, MutationOperator
from ea_toolkit.mutation import GaussianMutation, AdaptiveMutation


class OnePlusLambdaES(Algorithm):
    """(1+lambda) Evolution Strategy.

    Each generation:
    1. Generates lambda children by mutating the parent.
    2. Evaluates all children.
    3. Keeps the best of parent and all children.

    If the mutation operator is an AdaptiveMutation, the 1/5th success
    rule is automatically applied to adapt sigma.

    Extracted from temporal_optimizer.py evolve().

    Args:
        fitness_fn: fitness function to optimize.
        mutation: mutation operator. Defaults to GaussianMutation(sigma=0.1).
        lam: number of children per generation (lambda). Default 10.
        seed: random seed for reproducibility.
    """

    def __init__(self, fitness_fn: FitnessFunction,
                 mutation: MutationOperator | None = None,
                 lam: int = 10,
                 seed: int | None = None):
        if mutation is None:
            mutation = GaussianMutation(sigma=0.1)
        super().__init__(fitness_fn, mutation=mutation, seed=seed)
        self.lam = lam

    def run(self, budget: int) -> list[dict]:
        """Run the (1+lambda) ES with the given evaluation budget.

        Args:
            budget: total number of fitness evaluations to perform.

        Returns:
            list of all evaluation history entries.
        """
        param_spec = self.fitness_fn.param_spec()
        names = sorted(param_spec.keys())

        # Initialize parent randomly
        parent_params = {}
        for name in names:
            lo, hi = param_spec[name]
            parent_params[name] = float(self.rng.uniform(lo, hi))

        parent_result = self.fitness_fn.evaluate(parent_params)
        self._record(parent_params, parent_result)
        parent_fitness = parent_result.get('fitness', float('-inf'))

        # Run generations
        while len(self.history) < budget:
            best_child_params = None
            best_child_fitness = float('-inf')
            generation_improved = False

            # Generate lambda children
            for _ in range(self.lam):
                if len(self.history) >= budget:
                    break

                child_params = self.mutation.mutate(parent_params, param_spec,
                                                    self.rng)
                child_result = self.fitness_fn.evaluate(child_params)
                self._record(child_params, child_result)

                child_fitness = child_result.get('fitness', float('-inf'))
                if child_fitness > best_child_fitness:
                    best_child_fitness = child_fitness
                    best_child_params = child_params

            # (1+lambda): keep better of parent and best child
            if best_child_params is not None and best_child_fitness > parent_fitness:
                parent_params = best_child_params
                parent_fitness = best_child_fitness
                generation_improved = True

            # Report to adaptive mutation if applicable
            if isinstance(self.mutation, AdaptiveMutation):
                self.mutation.report_success(generation_improved)

        return self.history
