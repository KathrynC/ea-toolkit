"""
ea_toolkit.selection -- Selection strategies for evolutionary algorithms.

Three strategies for choosing individuals from a population:

- TournamentSelection: k-tournament selection with configurable tournament size.
- TruncationSelection: keep the top fraction of the population.
- EpsilonGreedy: with probability epsilon pick random, else pick best.
"""

import numpy as np

from ea_toolkit.base import SelectionStrategy


class TournamentSelection(SelectionStrategy):
    """Tournament selection: randomly sample k individuals, keep the best.

    Repeat n times (with replacement) to produce n selected individuals.
    Higher tournament sizes increase selection pressure.

    Args:
        tournament_size: number of individuals in each tournament. Default 3.
        seed: optional random seed for reproducibility.
    """

    def __init__(self, tournament_size: int = 3, seed: int | None = None):
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng(seed)

    def select(self, population: list, n: int) -> list:
        """Select n individuals via tournament selection.

        Args:
            population: list of dicts, each with at least a 'fitness' key.
            n: number of individuals to select.

        Returns:
            list of n selected individuals.
        """
        if not population:
            return []

        selected = []
        pop_size = len(population)
        k = min(self.tournament_size, pop_size)

        for _ in range(n):
            # Sample k individuals (indices) without replacement
            indices = self.rng.choice(pop_size, size=k, replace=False)
            # Pick the best from the tournament
            best_idx = max(indices,
                           key=lambda i: population[i].get('fitness',
                                                           float('-inf')))
            selected.append(population[best_idx])

        return selected


class TruncationSelection(SelectionStrategy):
    """Truncation selection: keep the top fraction of the population.

    Sorts the population by fitness (descending) and returns the top
    fraction. If n exceeds the truncated pool, individuals are recycled
    from the pool.

    Args:
        fraction: fraction of the population to keep (0.0 to 1.0). Default 0.5.
    """

    def __init__(self, fraction: float = 0.5):
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.fraction = fraction

    def select(self, population: list, n: int) -> list:
        """Select n individuals via truncation.

        Args:
            population: list of dicts, each with at least a 'fitness' key.
            n: number of individuals to select.

        Returns:
            list of n selected individuals from the top fraction.
        """
        if not population:
            return []

        # Sort by fitness descending
        sorted_pop = sorted(population,
                            key=lambda x: x.get('fitness', float('-inf')),
                            reverse=True)

        # Keep top fraction
        keep = max(1, int(len(sorted_pop) * self.fraction))
        pool = sorted_pop[:keep]

        # If n > pool size, cycle through the pool
        selected = []
        for i in range(n):
            selected.append(pool[i % len(pool)])

        return selected


class EpsilonGreedy(SelectionStrategy):
    """Epsilon-greedy selection: explore with probability epsilon, else exploit.

    With probability epsilon, pick a random individual from the population.
    Otherwise, pick the individual with the highest fitness.

    Args:
        epsilon: exploration probability (0.0 to 1.0). Default 0.1.
        seed: optional random seed for reproducibility.
    """

    def __init__(self, epsilon: float = 0.1, seed: int | None = None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select(self, population: list, n: int) -> list:
        """Select n individuals via epsilon-greedy.

        Args:
            population: list of dicts, each with at least a 'fitness' key.
            n: number of individuals to select.

        Returns:
            list of n selected individuals.
        """
        if not population:
            return []

        # Find the best individual
        best = max(population,
                   key=lambda x: x.get('fitness', float('-inf')))

        selected = []
        for _ in range(n):
            if self.rng.random() < self.epsilon:
                # Explore: pick random
                idx = self.rng.integers(0, len(population))
                selected.append(population[idx])
            else:
                # Exploit: pick best
                selected.append(best)

        return selected
