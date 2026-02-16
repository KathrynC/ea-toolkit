"""
ea_toolkit.algorithms.cliff_mapper -- High-sensitivity region search.

Extracted from walker_competition.py run_cliff_mapper(): probes multiple
directions at each step and walks toward the largest absolute fitness
change, deliberately seeking "cliffs" in the fitness landscape.
"""

import numpy as np

from ea_toolkit.base import Algorithm, FitnessFunction, MutationOperator
from ea_toolkit.mutation import GaussianMutation


class CliffMapper(Algorithm):
    """Cliff mapper: seeks high-sensitivity regions in the fitness landscape.

    At each step, probes n_probes random directions at a small radius,
    measures the absolute fitness change in each direction, and moves
    toward the direction with the largest |delta fitness|. This
    deliberately walks toward "cliffs" -- regions where small parameter
    changes produce large fitness changes.

    Extracted from walker_competition.py run_cliff_mapper().

    Args:
        fitness_fn: fitness function to optimize.
        mutation: mutation operator (used for generating probe directions).
            Defaults to GaussianMutation(sigma=probe_radius).
        n_probes: number of probe directions per step. Default 10.
        probe_radius: perturbation radius for probing. Default 0.05.
        seed: random seed for reproducibility.
    """

    def __init__(self, fitness_fn: FitnessFunction,
                 mutation: MutationOperator | None = None,
                 n_probes: int = 10,
                 probe_radius: float = 0.05,
                 seed: int | None = None):
        if mutation is None:
            mutation = GaussianMutation(sigma=probe_radius)
        super().__init__(fitness_fn, mutation=mutation, seed=seed)
        self.n_probes = n_probes
        self.probe_radius = probe_radius
        self._cliff_map: list[tuple[dict, float]] = []

    def run(self, budget: int) -> list[dict]:
        """Run the cliff mapper with the given evaluation budget.

        Args:
            budget: total number of fitness evaluations to perform.

        Returns:
            list of all evaluation history entries.
        """
        param_spec = self.fitness_fn.param_spec()
        names = sorted(param_spec.keys())
        n_dim = len(names)

        # Random starting point
        current_params = {}
        for name in names:
            lo, hi = param_spec[name]
            current_params[name] = float(self.rng.uniform(lo, hi))

        current_result = self.fitness_fn.evaluate(current_params)
        self._record(current_params, current_result)
        current_fitness = current_result.get('fitness', float('-inf'))

        # Record initial cliffiness (zero for the first point)
        self._cliff_map.append((dict(current_params), 0.0))

        # Main loop: probe and walk toward cliffs
        while len(self.history) < budget:
            probes = []

            for _ in range(self.n_probes):
                if len(self.history) >= budget:
                    break

                # Generate a random probe direction
                direction = self.rng.standard_normal(n_dim)
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    direction = np.ones(n_dim)
                    norm = np.linalg.norm(direction)
                direction = direction / norm

                # Perturb current params along this direction
                current_vec = np.array([current_params[n] for n in names])
                probe_vec = current_vec + self.probe_radius * direction

                # Clamp to bounds
                for i, name in enumerate(names):
                    lo, hi = param_spec[name]
                    probe_vec[i] = np.clip(probe_vec[i], lo, hi)

                probe_params = {n: float(probe_vec[i])
                                for i, n in enumerate(names)}

                probe_result = self.fitness_fn.evaluate(probe_params)
                self._record(probe_params, probe_result)

                probe_fitness = probe_result.get('fitness', float('-inf'))
                delta = abs(probe_fitness - current_fitness)
                probes.append((probe_params, probe_result, delta))

            if not probes:
                break

            # Move toward the probe with largest |delta fitness|
            best_probe = max(probes, key=lambda x: x[2])
            current_params = best_probe[0]
            current_result = best_probe[1]
            current_fitness = current_result.get('fitness', float('-inf'))
            cliffiness = best_probe[2]

            self._cliff_map.append((dict(current_params), cliffiness))

        return self.history

    def cliff_map(self) -> list[tuple[dict, float]]:
        """Return the cliff map: list of (params, cliffiness) from history.

        Returns:
            list of (params_dict, cliffiness_float) tuples, one per step.
            Cliffiness is the maximum |delta fitness| observed at that
            step across all probe directions.
        """
        return list(self._cliff_map)
