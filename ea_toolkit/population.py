"""
ea_toolkit.population -- Population management utilities.

Provides:
- PopulationManager: tracks individuals with params + fitness.
- random_population(): generate n random individuals within bounds.
- elitism(): keep the top n_elite individuals.
- diversity_metric(): mean pairwise normalized distance.
- behavioral_diversity(): diversity based on behavioral descriptors.
"""

import numpy as np


class PopulationManager:
    """Manages a population of individuals, each with params and fitness.

    Individuals are stored as dicts with at least 'params' and 'fitness'
    keys. The manager provides utilities for adding, sorting, and
    analyzing the population.
    """

    def __init__(self):
        self.individuals: list[dict] = []

    def add(self, params: dict, fitness: float, **extra) -> dict:
        """Add an individual to the population.

        Args:
            params: parameter configuration dict.
            fitness: fitness value.
            **extra: additional metadata to store with the individual.

        Returns:
            The created individual dict.
        """
        individual = {'params': dict(params), 'fitness': fitness, **extra}
        self.individuals.append(individual)
        return individual

    def best(self, n: int = 1) -> list[dict]:
        """Return the top n individuals by fitness.

        Args:
            n: number of top individuals to return. Default 1.

        Returns:
            list of the top n individuals, sorted by fitness descending.
        """
        sorted_pop = sorted(self.individuals,
                            key=lambda x: x.get('fitness', float('-inf')),
                            reverse=True)
        return sorted_pop[:n]

    def size(self) -> int:
        """Return the current population size."""
        return len(self.individuals)

    def clear(self) -> None:
        """Remove all individuals from the population."""
        self.individuals = []

    def replace(self, new_individuals: list[dict]) -> None:
        """Replace the entire population with new individuals.

        Args:
            new_individuals: list of individual dicts to set as the population.
        """
        self.individuals = list(new_individuals)


def random_population(n: int, param_spec: dict,
                      rng: np.random.Generator) -> list[dict]:
    """Generate n random individuals within the parameter bounds.

    Each individual is a dict with a 'params' key mapping parameter
    names to uniformly sampled values within [low, high].

    Args:
        n: number of individuals to generate.
        param_spec: dict mapping parameter names to (low, high) bounds.
        rng: numpy random generator for reproducibility.

    Returns:
        list of n dicts, each with a 'params' key.
    """
    names = sorted(param_spec.keys())
    individuals = []

    for _ in range(n):
        params = {}
        for name in names:
            lo, hi = param_spec[name]
            params[name] = float(rng.uniform(lo, hi))
        individuals.append({'params': params})

    return individuals


def elitism(population: list[dict], n_elite: int) -> list[dict]:
    """Keep the top n_elite individuals from a population.

    Args:
        population: list of individual dicts with 'fitness' keys.
        n_elite: number of top individuals to retain.

    Returns:
        list of the n_elite best individuals, sorted by fitness descending.
    """
    sorted_pop = sorted(population,
                        key=lambda x: x.get('fitness', float('-inf')),
                        reverse=True)
    return sorted_pop[:n_elite]


def diversity_metric(population: list[dict], param_spec: dict) -> float:
    """Compute the mean pairwise normalized Euclidean distance.

    Parameters are normalized to [0, 1] using their bounds before
    computing distances, so all dimensions contribute equally.

    Args:
        population: list of individual dicts with 'params' keys.
        param_spec: dict mapping parameter names to (low, high) bounds.

    Returns:
        float: mean pairwise normalized distance. Returns 0.0 if the
        population has fewer than 2 individuals.
    """
    if len(population) < 2:
        return 0.0

    names = sorted(param_spec.keys())
    n_dim = len(names)

    # Build normalized parameter matrix
    ranges = np.array([param_spec[n][1] - param_spec[n][0]
                       for n in names], dtype=float)
    # Avoid division by zero for zero-range parameters
    ranges[ranges < 1e-12] = 1.0

    lows = np.array([param_spec[n][0] for n in names], dtype=float)

    vecs = []
    for ind in population:
        p = ind.get('params', ind)
        vec = np.array([p[n] for n in names], dtype=float)
        # Normalize to [0, 1]
        vec_norm = (vec - lows) / ranges
        vecs.append(vec_norm)

    vecs = np.array(vecs)
    n = len(vecs)

    # Compute mean pairwise distance
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += np.linalg.norm(vecs[i] - vecs[j])
            count += 1

    return total_dist / count if count > 0 else 0.0


def behavioral_diversity(population: list[dict],
                         behavior_key: str = 'behavior') -> float:
    """Compute diversity based on behavioral descriptors.

    If individuals have behavioral descriptors (e.g., from a behavior_fn),
    this computes the mean pairwise Euclidean distance between those
    descriptors after min-max normalization.

    Args:
        population: list of individual dicts.
        behavior_key: key in each individual dict that maps to a
            behavioral descriptor (list or array of floats). Default 'behavior'.

    Returns:
        float: mean pairwise distance between normalized behavioral
        descriptors. Returns 0.0 if fewer than 2 individuals have
        behavioral descriptors.
    """
    # Extract behavioral vectors from individuals that have them
    bvecs = []
    for ind in population:
        if behavior_key in ind:
            bv = ind[behavior_key]
            if bv is not None:
                bvecs.append(np.asarray(bv, dtype=float))

    if len(bvecs) < 2:
        return 0.0

    arr = np.array(bvecs)

    # Min-max normalize per dimension
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-12] = 1.0
    normalized = (arr - mins) / ranges

    # Mean pairwise distance
    n = len(normalized)
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += np.linalg.norm(normalized[i] - normalized[j])
            count += 1

    return total_dist / count if count > 0 else 0.0
