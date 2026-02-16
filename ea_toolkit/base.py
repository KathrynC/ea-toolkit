"""
ea_toolkit.base -- Abstract base classes for evolutionary algorithm components.

Provides the foundational interfaces that all algorithms, mutation operators,
selection strategies, and fitness functions must implement. Extracted and
generalized from walker_competition.py and temporal_optimizer.py.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class FitnessFunction(ABC):
    """Abstract fitness function that users must implement.

    Subclasses must provide:
        evaluate(params) -> dict with at least a 'fitness' key.
        param_spec() -> dict mapping param names to (low, high) bounds.
    """

    @abstractmethod
    def evaluate(self, params: dict) -> dict:
        """Evaluate a parameter configuration.

        Args:
            params: dict mapping parameter names to float values.

        Returns:
            dict with at least a 'fitness' key (float). May include
            additional metrics (e.g., behavioral descriptors).
        """
        ...

    @abstractmethod
    def param_spec(self) -> dict:
        """Return the parameter specification.

        Returns:
            dict mapping parameter names to (low, high) tuples
            defining the valid bounds for each parameter.
        """
        ...


class MutationOperator(ABC):
    """Abstract mutation operator.

    Subclasses must implement mutate() to perturb a parameter dict,
    respecting the bounds defined in param_spec.
    """

    @abstractmethod
    def mutate(self, params: dict, param_spec: dict,
               rng: np.random.Generator) -> dict:
        """Mutate a parameter configuration.

        Args:
            params: dict mapping parameter names to current values.
            param_spec: dict mapping parameter names to (low, high) bounds.
            rng: numpy random Generator for reproducibility.

        Returns:
            New dict with mutated parameter values, clamped to bounds.
        """
        ...


class SelectionStrategy(ABC):
    """Abstract selection strategy for choosing individuals from a population."""

    @abstractmethod
    def select(self, population: list, n: int) -> list:
        """Select n individuals from a population.

        Args:
            population: list of dicts, each with at least a 'fitness' key.
            n: number of individuals to select.

        Returns:
            list of n selected individuals (dicts).
        """
        ...


class Algorithm(ABC):
    """Base class for all evolutionary algorithms.

    All algorithms share: run(budget) -> history.
    Provides common infrastructure for fitness tracking, history recording,
    and best-individual retrieval.
    """

    def __init__(self, fitness_fn: FitnessFunction,
                 mutation: 'MutationOperator | None' = None,
                 selection: 'SelectionStrategy | None' = None,
                 seed: int | None = None):
        """Initialize the algorithm.

        Args:
            fitness_fn: FitnessFunction instance to optimize.
            mutation: optional MutationOperator for generating variants.
            selection: optional SelectionStrategy for population-based methods.
            seed: random seed for reproducibility.
        """
        self.fitness_fn = fitness_fn
        self.mutation = mutation
        self.selection = selection
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []
        self._best: dict | None = None

    @abstractmethod
    def run(self, budget: int) -> list[dict]:
        """Run the algorithm for a given evaluation budget.

        Args:
            budget: maximum number of fitness evaluations to perform.

        Returns:
            list of history entries, each a dict with 'params' key
            plus all keys from the fitness evaluation result.
        """
        ...

    def best(self) -> dict | None:
        """Return the best individual found so far.

        Returns:
            dict with 'params' and 'fitness' keys, or None if no
            evaluations have been performed.
        """
        if not self.history:
            return None
        return max(self.history,
                   key=lambda x: x.get('fitness', float('-inf')))

    def _record(self, params: dict, result: dict) -> dict:
        """Record an evaluation in the history and update the best.

        Args:
            params: the parameter configuration that was evaluated.
            result: the evaluation result dict (must contain 'fitness').

        Returns:
            The recorded history entry (params merged with result).
        """
        entry = {'params': dict(params), **result}
        self.history.append(entry)
        if (self._best is None or
                result.get('fitness', float('-inf')) >
                self._best.get('fitness', float('-inf'))):
            self._best = entry
        return entry
