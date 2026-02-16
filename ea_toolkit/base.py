"""
ea_toolkit.base -- Abstract base classes for evolutionary algorithm components.

Provides the foundational interfaces that all algorithms, mutation operators,
selection strategies, crossover operators, and fitness functions must implement.
Extracted and generalized from walker_competition.py and temporal_optimizer.py.
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


class CrossoverOperator(ABC):
    """Abstract crossover operator for recombining two parent solutions.

    Subclasses must implement crossover() to produce two offspring
    from two parents, respecting parameter bounds.
    """

    @abstractmethod
    def crossover(self, parent1: dict, parent2: dict, param_spec: dict,
                  rng: np.random.Generator) -> tuple[dict, dict]:
        """Produce two offspring from two parents.

        Args:
            parent1: first parent parameter dict.
            parent2: second parent parameter dict.
            param_spec: dict mapping parameter names to (low, high) bounds.
            rng: numpy random Generator for reproducibility.

        Returns:
            Tuple of (child1, child2) parameter dicts.
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


class Callback:
    """Base callback class for hooking into algorithm events.

    Override any method to receive notifications. Return False from
    on_generation() to request early stopping.
    """

    def on_start(self, algorithm: 'Algorithm') -> None:
        """Called when the algorithm starts running."""
        pass

    def on_generation(self, algorithm: 'Algorithm', generation: int,
                      best_fitness: float) -> bool | None:
        """Called after each generation.

        Args:
            algorithm: the running algorithm instance.
            generation: current generation number (0-indexed).
            best_fitness: best fitness found so far.

        Returns:
            False to request early stopping, or None/True to continue.
        """
        pass

    def on_improvement(self, algorithm: 'Algorithm',
                       old_fitness: float, new_fitness: float) -> None:
        """Called when a new best fitness is found."""
        pass

    def on_finish(self, algorithm: 'Algorithm') -> None:
        """Called when the algorithm finishes."""
        pass


class Algorithm(ABC):
    """Base class for all evolutionary algorithms.

    All algorithms share: run(budget) -> history.
    Provides common infrastructure for fitness tracking, history recording,
    best-individual retrieval, callbacks, and optional ask-tell interface.
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
        self.callbacks: list[Callback] = []

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

    def ask(self) -> list[dict]:
        """Return the next batch of candidates to evaluate.

        Part of the ask-tell interface. Not all algorithms support this.
        Algorithms that do will override this method.

        Returns:
            list of parameter dicts to evaluate.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support ask-tell. Use run().")

    def tell(self, evaluations: list[tuple[dict, dict]]) -> None:
        """Report evaluation results for candidates from ask().

        Part of the ask-tell interface. Not all algorithms support this.

        Args:
            evaluations: list of (params_dict, result_dict) tuples where
                each result_dict must contain a 'fitness' key.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support ask-tell. Use run().")

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
        old_best_fitness = (self._best.get('fitness', float('-inf'))
                            if self._best else float('-inf'))

        entry = {'params': dict(params), **result}
        self.history.append(entry)

        new_fitness = result.get('fitness', float('-inf'))
        if new_fitness > old_best_fitness:
            self._best = entry
            self._notify('on_improvement',
                         old_fitness=old_best_fitness,
                         new_fitness=new_fitness)
        return entry

    def _notify(self, event: str, **kwargs) -> bool:
        """Notify all callbacks of an event.

        Args:
            event: method name on Callback to call.
            **kwargs: arguments to pass to the callback method.

        Returns:
            False if any callback explicitly returns False, True otherwise.
        """
        for cb in self.callbacks:
            method = getattr(cb, event, None)
            if method:
                result = method(self, **kwargs)
                if result is False:
                    return False
        return True
