"""
ea_toolkit.algorithms.ridge_walker -- Multi-objective Pareto search.

Extracted from walker_competition.py run_ridge_walker(): each step generates
n_candidates, filters to non-dominated solutions, and picks the one farthest
in objective space to encourage Pareto front exploration.
"""

import numpy as np

from ea_toolkit.base import Algorithm, FitnessFunction, MutationOperator
from ea_toolkit.mutation import GaussianMutation


def _is_dominated(a_vals: np.ndarray, b_vals: np.ndarray) -> bool:
    """Test if b Pareto-dominates a (all objectives higher-is-better).

    Dominance: b >= a on all objectives and b > a on at least one.

    Args:
        a_vals: objective values for candidate a.
        b_vals: objective values for candidate b.

    Returns:
        True if b dominates a.
    """
    return bool(np.all(b_vals >= a_vals) and np.any(b_vals > a_vals))


class RidgeWalker(Algorithm):
    """Multi-objective Pareto search along fitness ridges.

    Each step generates n_candidates mutations, filters to those not
    dominated by the current point, and moves to the one farthest
    in objective space. This encourages exploration along the Pareto
    front rather than stagnation.

    Extracted from walker_competition.py run_ridge_walker().

    Args:
        fitness_fn: fitness function. Must return dict with keys matching
            the objectives tuple.
        mutation: mutation operator. Defaults to GaussianMutation(sigma=0.1).
        objectives: tuple of objective names to optimize (all maximized).
            Default ('fitness',).
        n_candidates: number of candidates per step. Default 3.
        seed: random seed for reproducibility.
    """

    def __init__(self, fitness_fn: FitnessFunction,
                 mutation: MutationOperator | None = None,
                 objectives: tuple[str, ...] = ('fitness',),
                 n_candidates: int = 3,
                 seed: int | None = None):
        if mutation is None:
            mutation = GaussianMutation(sigma=0.1)
        super().__init__(fitness_fn, mutation=mutation, seed=seed)
        self.objectives = objectives
        self.n_candidates = n_candidates

    def _obj_values(self, result: dict) -> np.ndarray:
        """Extract objective values from a result dict."""
        return np.array([result.get(obj, 0.0) for obj in self.objectives])

    def run(self, budget: int) -> list[dict]:
        """Run the ridge walker with the given evaluation budget.

        Args:
            budget: total number of fitness evaluations to perform.

        Returns:
            list of all evaluation history entries.
        """
        param_spec = self.fitness_fn.param_spec()
        names = sorted(param_spec.keys())

        # Random starting point
        current_params = {}
        for name in names:
            lo, hi = param_spec[name]
            current_params[name] = float(self.rng.uniform(lo, hi))

        current_result = self.fitness_fn.evaluate(current_params)
        self._record(current_params, current_result)
        current_obj = self._obj_values(current_result)

        # Main loop
        while len(self.history) < budget:
            candidates = []

            # Generate n_candidates
            for _ in range(self.n_candidates):
                if len(self.history) >= budget:
                    break

                cand_params = self.mutation.mutate(current_params, param_spec,
                                                   self.rng)
                cand_result = self.fitness_fn.evaluate(cand_params)
                self._record(cand_params, cand_result)

                cand_obj = self._obj_values(cand_result)
                candidates.append((cand_params, cand_result, cand_obj))

            if not candidates:
                break

            # Filter to non-dominated candidates (not dominated by current)
            non_dominated = [
                (p, r, o) for p, r, o in candidates
                if not _is_dominated(o, current_obj)
            ]

            if non_dominated:
                # Pick the one farthest from current in objective space
                best_dist = -1.0
                best_p, best_r, best_o = non_dominated[0]

                for p, r, o in non_dominated:
                    dist = float(np.linalg.norm(o - current_obj))
                    if dist > best_dist:
                        best_dist = dist
                        best_p, best_r, best_o = p, r, o

                current_params = best_p
                current_result = best_r
                current_obj = best_o

        return self.history

    def pareto_front(self) -> list[dict]:
        """Extract the Pareto front from the history.

        Returns:
            list of non-dominated history entries.
        """
        if not self.history:
            return []

        front = []
        for i, entry_a in enumerate(self.history):
            a_obj = self._obj_values(entry_a)
            dominated = False
            for j, entry_b in enumerate(self.history):
                if i != j:
                    b_obj = self._obj_values(entry_b)
                    if _is_dominated(a_obj, b_obj):
                        dominated = True
                        break
            if not dominated:
                front.append(entry_a)
        return front
