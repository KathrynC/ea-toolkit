"""
ea_toolkit.crossover -- Crossover operators for evolutionary algorithms.

Two operators for recombining parameter vectors:

- SBXCrossover: Simulated Binary Crossover for continuous parameters.
  The standard crossover for real-valued GAs (Deb & Agrawal 1995).

- UniformCrossover: Swap each parameter independently with probability p.
  Simple and effective when parameters are weakly coupled.
"""

import numpy as np

from ea_toolkit.base import CrossoverOperator


class SBXCrossover(CrossoverOperator):
    """Simulated Binary Crossover (SBX) for continuous parameters.

    Simulates the behavior of single-point crossover on binary strings
    but operates directly on real values. The distribution index eta
    controls how far offspring can be from parents: low eta produces
    diverse offspring, high eta produces offspring close to parents.

    The standard crossover operator for real-valued genetic algorithms.
    Used in NSGA-II, pymoo, and most modern multi-objective EAs.

    Args:
        eta: distribution index (>= 0). Default 20.0 (tight around parents).
            eta=1: wide spread. eta=20: tight. eta=100: very tight.
        probability: probability of crossover per parameter. Default 0.9.
    """

    def __init__(self, eta: float = 20.0, probability: float = 0.9):
        self.eta = eta
        self.probability = probability

    def crossover(self, parent1: dict, parent2: dict, param_spec: dict,
                  rng: np.random.Generator) -> tuple[dict, dict]:
        """Produce two offspring via SBX.

        For each parameter, with probability self.probability, apply the
        SBX spread factor. Otherwise, copy parent values unchanged.

        Args:
            parent1: first parent parameter dict.
            parent2: second parent parameter dict.
            param_spec: parameter bounds.
            rng: random generator.

        Returns:
            Tuple of (child1, child2) parameter dicts.
        """
        names = sorted(param_spec.keys())
        child1 = {}
        child2 = {}

        for name in names:
            lo, hi = param_spec[name]
            p1 = parent1[name]
            p2 = parent2[name]

            if rng.random() < self.probability and abs(p1 - p2) > 1e-14:
                # Apply SBX
                u = rng.random()

                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (self.eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1.0))

                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

                child1[name] = float(np.clip(c1, lo, hi))
                child2[name] = float(np.clip(c2, lo, hi))
            else:
                # No crossover — copy parents
                child1[name] = p1
                child2[name] = p2

        return child1, child2


class UniformCrossover(CrossoverOperator):
    """Uniform crossover: swap each parameter independently with probability p.

    For each parameter, flip a coin. If heads, child1 gets parent2's value
    and child2 gets parent1's value. Otherwise, keep parent values.

    Simple and effective when parameters are weakly coupled or when
    the problem has no strong epistasis between parameters.

    Args:
        swap_probability: probability of swapping each parameter. Default 0.5.
    """

    def __init__(self, swap_probability: float = 0.5):
        self.swap_probability = swap_probability

    def crossover(self, parent1: dict, parent2: dict, param_spec: dict,
                  rng: np.random.Generator) -> tuple[dict, dict]:
        """Produce two offspring via uniform crossover.

        Args:
            parent1: first parent parameter dict.
            parent2: second parent parameter dict.
            param_spec: parameter bounds.
            rng: random generator.

        Returns:
            Tuple of (child1, child2) parameter dicts.
        """
        names = sorted(param_spec.keys())
        child1 = {}
        child2 = {}

        for name in names:
            if rng.random() < self.swap_probability:
                child1[name] = parent2[name]
                child2[name] = parent1[name]
            else:
                child1[name] = parent1[name]
                child2[name] = parent2[name]

        return child1, child2
