"""
ea_toolkit.algorithms.ensemble -- Multi-algorithm ensemble explorer.

Extracted from walker_competition.py run_ensemble_explorer(): parallel
hill climbers with periodic teleportation of converged walkers to
maintain population diversity.
"""

import numpy as np

from ea_toolkit.base import Algorithm, FitnessFunction, MutationOperator
from ea_toolkit.mutation import GaussianMutation


class EnsembleExplorer(Algorithm):
    """Multi-walker ensemble with behavioral convergence detection.

    Runs n_walkers parallel hill climbers. Periodically checks for
    behavioral convergence: if two walkers are within teleport_threshold
    in normalized parameter space, the worse one is teleported to a
    random location to maintain diversity.

    Extracted from walker_competition.py run_ensemble_explorer().

    Args:
        fitness_fn: fitness function to optimize.
        mutation: mutation operator. Defaults to GaussianMutation(sigma=0.1).
        n_walkers: number of parallel hill climbers. Default 20.
        teleport_threshold: normalized distance below which walkers are
            considered converged. Default 0.3.
        teleport_interval: check for convergence every this many steps.
            Default 10.
        seed: random seed for reproducibility.
    """

    def __init__(self, fitness_fn: FitnessFunction,
                 mutation: MutationOperator | None = None,
                 n_walkers: int = 20,
                 teleport_threshold: float = 0.3,
                 teleport_interval: int = 10,
                 seed: int | None = None):
        if mutation is None:
            mutation = GaussianMutation(sigma=0.1)
        super().__init__(fitness_fn, mutation=mutation, seed=seed)
        self.n_walkers = n_walkers
        self.teleport_threshold = teleport_threshold
        self.teleport_interval = teleport_interval

    def _normalize_params(self, params_list: list[dict],
                          param_spec: dict) -> np.ndarray:
        """Normalize parameter vectors to [0, 1] for convergence detection.

        Args:
            params_list: list of parameter dicts.
            param_spec: parameter bounds.

        Returns:
            (n_walkers, n_dims) normalized array.
        """
        names = sorted(param_spec.keys())
        lows = np.array([param_spec[n][0] for n in names])
        highs = np.array([param_spec[n][1] for n in names])
        ranges = highs - lows
        ranges[ranges < 1e-12] = 1.0

        vecs = []
        for params in params_list:
            vec = np.array([params[n] for n in names])
            vecs.append((vec - lows) / ranges)

        return np.array(vecs)

    def run(self, budget: int) -> list[dict]:
        """Run the ensemble explorer with the given evaluation budget.

        Args:
            budget: total number of fitness evaluations to perform.

        Returns:
            list of all evaluation history entries.
        """
        param_spec = self.fitness_fn.param_spec()
        names = sorted(param_spec.keys())

        # Clamp n_walkers to budget
        actual_walkers = min(self.n_walkers, budget)

        # Initialize walkers
        walker_params = []
        walker_fitness = []

        for _ in range(actual_walkers):
            if len(self.history) >= budget:
                break

            params = {}
            for name in names:
                lo, hi = param_spec[name]
                params[name] = float(self.rng.uniform(lo, hi))

            result = self.fitness_fn.evaluate(params)
            self._record(params, result)

            walker_params.append(params)
            walker_fitness.append(result.get('fitness', float('-inf')))

        # Main loop: parallel hill climbing with teleportation
        step = 0
        while len(self.history) < budget:
            # Each walker takes one step
            for wi in range(len(walker_params)):
                if len(self.history) >= budget:
                    break

                new_params = self.mutation.mutate(walker_params[wi],
                                                  param_spec, self.rng)
                new_result = self.fitness_fn.evaluate(new_params)
                self._record(new_params, new_result)

                new_fitness = new_result.get('fitness', float('-inf'))

                # Accept only strict improvement
                if new_fitness > walker_fitness[wi]:
                    walker_params[wi] = new_params
                    walker_fitness[wi] = new_fitness

            step += 1

            # Periodic teleportation to prevent convergence
            if step % self.teleport_interval == 0 and len(walker_params) > 1:
                norm_vecs = self._normalize_params(walker_params, param_spec)
                n_w = len(walker_params)

                for wi in range(n_w):
                    for wj in range(wi + 1, n_w):
                        dist = float(np.linalg.norm(norm_vecs[wi] - norm_vecs[wj]))
                        if dist < self.teleport_threshold:
                            # Teleport the worse walker
                            if walker_fitness[wi] < walker_fitness[wj]:
                                worse = wi
                            else:
                                worse = wj

                            # Reset to random position
                            new_params = {}
                            for name in names:
                                lo, hi = param_spec[name]
                                new_params[name] = float(
                                    self.rng.uniform(lo, hi))

                            walker_params[worse] = new_params
                            walker_fitness[worse] = float('-inf')

        return self.history
