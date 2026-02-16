"""
ea_toolkit.algorithms.novelty_seeker -- Novelty search algorithm.

Extracted from walker_competition.py run_novelty_seeker(): at each step,
generates multiple candidates and moves to the one with the highest
k-nearest-neighbor novelty score, regardless of fitness.
"""

import numpy as np

from ea_toolkit.base import Algorithm, FitnessFunction, MutationOperator
from ea_toolkit.mutation import GaussianMutation


def _default_behavior_fn(result: dict) -> np.ndarray:
    """Default behavior function: extract all numeric values from result.

    Args:
        result: evaluation result dict.

    Returns:
        numpy array of all float/int values in the dict.
    """
    values = []
    for key in sorted(result.keys()):
        val = result[key]
        if isinstance(val, (int, float, np.integer, np.floating)):
            values.append(float(val))
    if not values:
        values = [0.0]
    return np.array(values)


def _normalize_behavioral_vecs(vecs: list[np.ndarray]) -> np.ndarray:
    """Min-max normalize behavioral vectors to [0, 1] per dimension.

    Handles constant dimensions by setting their range to 1.0.

    Args:
        vecs: list of behavioral vectors.

    Returns:
        numpy array of shape (len(vecs), n_dims) with values in [0, 1].
    """
    if not vecs:
        return np.array([])
    arr = np.array(vecs)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-12] = 1.0
    return (arr - mins) / ranges


def _knn_novelty(bvec: np.ndarray, archive: np.ndarray,
                 k: int = 15) -> float:
    """Compute k-nearest-neighbor novelty score.

    Args:
        bvec: behavioral vector to score.
        archive: (n, d) array of archived behavioral vectors.
        k: number of nearest neighbors.

    Returns:
        Mean Euclidean distance to k nearest neighbors.
        Returns inf if archive is empty.
    """
    if len(archive) == 0:
        return float('inf')

    dists = np.linalg.norm(archive - bvec, axis=1)
    k_actual = min(k, len(dists))
    if k_actual >= len(dists):
        # Use all distances
        return float(np.mean(dists))
    indices = np.argpartition(dists, k_actual)[:k_actual]
    return float(np.mean(dists[indices]))


class NoveltySeeker(Algorithm):
    """Novelty search: maximize behavioral diversity, not fitness.

    At each step, generates n_candidates mutations, extracts behavioral
    descriptors, normalizes them together with the archive, and moves
    to the candidate with the highest k-NN novelty score -- regardless
    of its fitness.

    Extracted from walker_competition.py run_novelty_seeker().

    Args:
        fitness_fn: fitness function to evaluate.
        mutation: mutation operator. Defaults to GaussianMutation(sigma=0.2).
        behavior_fn: callable that extracts a behavioral descriptor from
            a result dict. Default: use all numeric values.
        n_candidates: number of candidates per step. Default 5.
        k_nearest: number of nearest neighbors for novelty computation.
            Default 15.
        seed: random seed for reproducibility.
    """

    def __init__(self, fitness_fn: FitnessFunction,
                 mutation: MutationOperator | None = None,
                 behavior_fn=None,
                 n_candidates: int = 5,
                 k_nearest: int = 15,
                 seed: int | None = None):
        if mutation is None:
            mutation = GaussianMutation(sigma=0.2)
        super().__init__(fitness_fn, mutation=mutation, seed=seed)
        self.behavior_fn = behavior_fn or _default_behavior_fn
        self.n_candidates = n_candidates
        self.k_nearest = k_nearest
        self._behavior_archive: list[np.ndarray] = []

    def run(self, budget: int) -> list[dict]:
        """Run novelty search with the given evaluation budget.

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

        bvec = self.behavior_fn(current_result)
        self._behavior_archive.append(bvec)

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

                cand_bvec = self.behavior_fn(cand_result)
                candidates.append((cand_params, cand_result, cand_bvec))

            if not candidates:
                break

            # Normalize archive + candidates together
            all_bvecs = self._behavior_archive + [c[2] for c in candidates]
            normalized = _normalize_behavioral_vecs(all_bvecs)
            n_archive = len(self._behavior_archive)
            norm_archive = normalized[:n_archive]

            # Pick the most novel candidate
            best_novelty = -1.0
            best_idx = 0

            for ci, (_, _, _) in enumerate(candidates):
                norm_bvec = normalized[n_archive + ci]
                nov = _knn_novelty(norm_bvec, norm_archive, k=self.k_nearest)
                if nov > best_novelty:
                    best_novelty = nov
                    best_idx = ci

            # Move to the most novel candidate
            current_params = candidates[best_idx][0]

            # Add all candidates to behavior archive
            for _, _, bvec in candidates:
                self._behavior_archive.append(bvec)

        return self.history
