"""
ea_toolkit.algorithms.de -- Differential Evolution (DE/rand/1/bin).

A population-based optimizer where new candidates are created by combining
differences between existing population members. Extremely competitive for
real-valued optimization, especially on multimodal landscapes.

Supports the ask-tell interface natively.
"""

import numpy as np

from ea_toolkit.base import Algorithm, FitnessFunction


class DifferentialEvolution(Algorithm):
    """Differential Evolution with DE/rand/1/bin strategy.

    Each generation:
    1. For each target vector x_i in the population:
       a. Select 3 distinct random vectors r1, r2, r3 (not i)
       b. Mutant: v = r1 + F * (r2 - r3)
       c. Trial: binomial crossover of v with x_i (probability CR)
       d. If trial is better than x_i, replace it
    2. Repeat until budget exhausted.

    Supports the ask-tell interface: ask() returns NP trial vectors,
    tell() performs selection (replace if better).

    Args:
        fitness_fn: fitness function to optimize.
        pop_size: population size (NP). Default 50.
        F: mutation scale factor (differential weight). Default 0.8.
        CR: crossover rate. Default 0.9.
        seed: random seed for reproducibility.
    """

    def __init__(self, fitness_fn: FitnessFunction,
                 pop_size: int = 50,
                 F: float = 0.8,
                 CR: float = 0.9,
                 seed: int | None = None):
        super().__init__(fitness_fn, seed=seed)
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self._initialized = False
        self._population: list[dict] = []
        self._pop_fitness: list[float] = []
        self._trials: list[dict] = []
        self._trial_targets: list[int] = []
        self._generation = 0

    def _initialize(self):
        """Initialize the population."""
        param_spec = self.fitness_fn.param_spec()
        names = sorted(param_spec.keys())

        self._population = []
        self._pop_fitness = []

        for _ in range(self.pop_size):
            params = {}
            for name in names:
                lo, hi = param_spec[name]
                params[name] = float(self.rng.uniform(lo, hi))
            self._population.append(params)
            self._pop_fitness.append(float('-inf'))

        self._initialized = True

    def ask(self) -> list[dict]:
        """Return trial vectors for the current generation.

        On the first call, returns the initial population for evaluation.
        On subsequent calls, returns NP trial vectors generated via
        DE/rand/1/bin mutation and crossover.

        Returns:
            list of parameter dicts to evaluate.
        """
        if not self._initialized:
            self._initialize()
            # First ask: return initial population for evaluation
            self._trials = list(self._population)
            self._trial_targets = list(range(self.pop_size))
            return list(self._trials)

        param_spec = self.fitness_fn.param_spec()
        names = sorted(param_spec.keys())
        n_dim = len(names)
        NP = len(self._population)

        self._trials = []
        self._trial_targets = []

        for i in range(NP):
            # Select 3 distinct random indices, all different from i
            candidates = [j for j in range(NP) if j != i]
            chosen = self.rng.choice(candidates, size=3, replace=False)
            r1, r2, r3 = int(chosen[0]), int(chosen[1]), int(chosen[2])

            # Mutation: v = r1 + F * (r2 - r3)
            target = self._population[i]
            donor1 = self._population[r1]
            donor2 = self._population[r2]
            donor3 = self._population[r3]

            mutant = {}
            for name in names:
                mutant[name] = donor1[name] + self.F * (donor2[name] - donor3[name])

            # Binomial crossover
            j_rand = self.rng.integers(0, n_dim)  # Ensure at least one from mutant
            trial = {}
            for idx, name in enumerate(names):
                lo, hi = param_spec[name]
                if self.rng.random() < self.CR or idx == j_rand:
                    trial[name] = float(np.clip(mutant[name], lo, hi))
                else:
                    trial[name] = target[name]

            self._trials.append(trial)
            self._trial_targets.append(i)

        return list(self._trials)

    def tell(self, evaluations: list[tuple[dict, dict]]) -> None:
        """Report evaluation results and perform selection.

        For each trial vector, if it's better than the corresponding
        target vector, replace the target.

        Args:
            evaluations: list of (params_dict, result_dict) tuples.
        """
        for idx, (params, result) in enumerate(evaluations):
            self._record(params, result)

            trial_fitness = result.get('fitness', float('-inf'))

            if idx < len(self._trial_targets):
                target_idx = self._trial_targets[idx]
                if trial_fitness > self._pop_fitness[target_idx]:
                    self._population[target_idx] = dict(params)
                    self._pop_fitness[target_idx] = trial_fitness

        self._generation += 1

        # Notify callbacks
        best_fitness = max(self._pop_fitness) if self._pop_fitness else float('-inf')
        self._notify('on_generation',
                     generation=self._generation,
                     best_fitness=best_fitness)

    def run(self, budget: int) -> list[dict]:
        """Run DE with the given evaluation budget.

        Args:
            budget: total number of fitness evaluations to perform.

        Returns:
            list of all evaluation history entries.
        """
        self._notify('on_start')

        # Initialize and evaluate initial population
        candidates = self.ask()
        evaluations = []
        for params in candidates:
            if len(self.history) + len(evaluations) >= budget:
                break
            result = self.fitness_fn.evaluate(params)
            evaluations.append((params, result))
        if evaluations:
            self.tell(evaluations)

        # Main loop
        while len(self.history) < budget:
            candidates = self.ask()
            evaluations = []
            for params in candidates:
                if len(self.history) + len(evaluations) >= budget:
                    break
                result = self.fitness_fn.evaluate(params)
                evaluations.append((params, result))
            if evaluations:
                self.tell(evaluations)
            else:
                break

            # Check for early stopping via callbacks
            best_f = max(self._pop_fitness) if self._pop_fitness else float('-inf')
            if not self._notify('on_generation',
                                generation=self._generation,
                                best_fitness=best_f):
                break

        self._notify('on_finish')
        return self.history
