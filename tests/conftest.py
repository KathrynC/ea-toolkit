"""
tests/conftest.py -- Shared fixtures for the ea_toolkit test suite.

Provides:
- SphereFitness: f(x) = -sum(x_i^2), optimum at origin.
- RastriginFitness: multimodal test function.
- sphere_param_spec: 5D parameter specification for the sphere function.
- default_mutation: default GaussianMutation operator.
"""

import numpy as np
import pytest

from ea_toolkit.base import FitnessFunction
from ea_toolkit.mutation import GaussianMutation


class SphereFitness(FitnessFunction):
    """Sphere function: f(x) = -sum(x_i^2).

    Optimum at the origin (all zeros) with fitness = 0.0.
    A simple unimodal function for testing convergence.

    Args:
        n_dims: number of dimensions. Default 5.
        bounds: (low, high) bounds for each parameter. Default (-5, 5).
    """

    def __init__(self, n_dims: int = 5,
                 bounds: tuple[float, float] = (-5.0, 5.0)):
        self.n_dims = n_dims
        self.bounds = bounds

    def evaluate(self, params: dict) -> dict:
        """Evaluate the negative sphere function.

        Args:
            params: dict mapping 'x0', 'x1', ... to float values.

        Returns:
            dict with 'fitness' = -sum(x_i^2).
        """
        values = np.array([params[f'x{i}'] for i in range(self.n_dims)])
        fitness = -float(np.sum(values ** 2))
        return {'fitness': fitness}

    def param_spec(self) -> dict:
        """Return parameter spec for n_dims dimensions."""
        return {f'x{i}': self.bounds for i in range(self.n_dims)}


class RastriginFitness(FitnessFunction):
    """Rastrigin function (negated for maximization).

    f(x) = -(A*n + sum(x_i^2 - A*cos(2*pi*x_i)))

    A highly multimodal function with many local optima.
    Global optimum at origin with fitness = 0.0.

    Args:
        n_dims: number of dimensions. Default 5.
        bounds: (low, high) bounds for each parameter. Default (-5.12, 5.12).
        A: Rastrigin parameter controlling multimodality. Default 10.
    """

    def __init__(self, n_dims: int = 5,
                 bounds: tuple[float, float] = (-5.12, 5.12),
                 A: float = 10.0):
        self.n_dims = n_dims
        self.bounds = bounds
        self.A = A

    def evaluate(self, params: dict) -> dict:
        """Evaluate the negated Rastrigin function.

        Args:
            params: dict mapping 'x0', 'x1', ... to float values.

        Returns:
            dict with 'fitness' = -rastrigin(x), plus 'raw_rastrigin'
            for the un-negated value.
        """
        values = np.array([params[f'x{i}'] for i in range(self.n_dims)])
        raw = self.A * self.n_dims + np.sum(
            values ** 2 - self.A * np.cos(2 * np.pi * values))
        fitness = -float(raw)
        return {'fitness': fitness, 'raw_rastrigin': float(raw)}

    def param_spec(self) -> dict:
        """Return parameter spec for n_dims dimensions."""
        return {f'x{i}': self.bounds for i in range(self.n_dims)}


class MultiObjectiveSphere(FitnessFunction):
    """Multi-objective sphere: returns fitness and a secondary objective.

    fitness = -sum(x_i^2)
    secondary = -sum((x_i - 1)^2)

    Pareto front exists because the two optima (origin vs all-ones)
    conflict.

    Args:
        n_dims: number of dimensions. Default 5.
    """

    def __init__(self, n_dims: int = 5):
        self.n_dims = n_dims

    def evaluate(self, params: dict) -> dict:
        values = np.array([params[f'x{i}'] for i in range(self.n_dims)])
        fitness = -float(np.sum(values ** 2))
        secondary = -float(np.sum((values - 1.0) ** 2))
        return {'fitness': fitness, 'secondary': secondary}

    def param_spec(self) -> dict:
        return {f'x{i}': (-5.0, 5.0) for i in range(self.n_dims)}


class StepFitness(FitnessFunction):
    """Step function with a cliff at x0 = 0.

    fitness = 10 if x0 > 0 else -10 (plus small contribution from other dims).

    Used to test cliff detection.

    Args:
        n_dims: number of dimensions. Default 5.
    """

    def __init__(self, n_dims: int = 5):
        self.n_dims = n_dims

    def evaluate(self, params: dict) -> dict:
        x0 = params['x0']
        base = 10.0 if x0 > 0 else -10.0
        # Small contribution from other dims to break ties
        other = sum(params.get(f'x{i}', 0.0) * 0.001
                    for i in range(1, self.n_dims))
        return {'fitness': base + other}

    def param_spec(self) -> dict:
        return {f'x{i}': (-5.0, 5.0) for i in range(self.n_dims)}


class LinearFitness(FitnessFunction):
    """Linear function: f(x) = sum(c_i * x_i).

    Used to test gradient estimation (analytical gradient = coefficients).

    Args:
        coefficients: list of coefficients. Default [1, 2, 3, 4, 5].
    """

    def __init__(self, coefficients: list[float] | None = None):
        if coefficients is None:
            coefficients = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.coefficients = coefficients
        self.n_dims = len(coefficients)

    def evaluate(self, params: dict) -> dict:
        values = np.array([params[f'x{i}'] for i in range(self.n_dims)])
        coeffs = np.array(self.coefficients)
        fitness = float(np.dot(coeffs, values))
        return {'fitness': fitness}

    def param_spec(self) -> dict:
        return {f'x{i}': (-5.0, 5.0) for i in range(self.n_dims)}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sphere_fitness():
    """5D sphere fitness function."""
    return SphereFitness(n_dims=5)


@pytest.fixture
def rastrigin_fitness():
    """5D Rastrigin fitness function."""
    return RastriginFitness(n_dims=5)


@pytest.fixture
def multi_objective_sphere():
    """5D multi-objective sphere fitness function."""
    return MultiObjectiveSphere(n_dims=5)


@pytest.fixture
def step_fitness():
    """5D step function with cliff at x0=0."""
    return StepFitness(n_dims=5)


@pytest.fixture
def linear_fitness():
    """5D linear fitness function."""
    return LinearFitness()


@pytest.fixture
def sphere_param_spec():
    """Parameter specification for 5D sphere."""
    return {f'x{i}': (-5.0, 5.0) for i in range(5)}


@pytest.fixture
def default_mutation():
    """Default Gaussian mutation operator."""
    return GaussianMutation(sigma=0.1)
