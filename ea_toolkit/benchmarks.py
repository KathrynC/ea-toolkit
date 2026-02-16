"""
ea_toolkit.benchmarks -- Standard benchmark fitness functions.

Provides well-known test functions for validating and comparing algorithms:

- SphereFitness: f = -sum(x^2). Unimodal, separable. Optimum at origin.
- RosenbrockFitness: f = -sum(100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2).
  Unimodal but with a narrow curved valley. Optimum at (1,1,...,1).
- RastriginFitness: f = -(An + sum(x_i^2 - A*cos(2*pi*x_i))).
  Highly multimodal (many local optima). Optimum at origin.
- AckleyFitness: Multimodal with a nearly flat outer region.
  Optimum at origin.
- ZDT1Fitness: Bi-objective test problem with convex Pareto front.

All functions are negated for the maximization convention (fitness=0 at optimum).
"""

import numpy as np

from ea_toolkit.base import FitnessFunction


class SphereFitness(FitnessFunction):
    """Sphere function: f(x) = -sum(x_i^2).

    The simplest test function. Unimodal, separable, smooth.
    Optimum at the origin with fitness = 0.0.

    Args:
        n_dims: number of dimensions. Default 5.
        bounds: (low, high) bounds for each parameter. Default (-5, 5).
    """

    def __init__(self, n_dims: int = 5,
                 bounds: tuple[float, float] = (-5.0, 5.0)):
        self.n_dims = n_dims
        self.bounds = bounds

    def evaluate(self, params: dict) -> dict:
        values = np.array([params[f'x{i}'] for i in range(self.n_dims)])
        return {'fitness': -float(np.sum(values ** 2))}

    def param_spec(self) -> dict:
        return {f'x{i}': self.bounds for i in range(self.n_dims)}


class RosenbrockFitness(FitnessFunction):
    """Rosenbrock function (negated for maximization).

    f(x) = -sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)

    A narrow curved valley makes this deceptively difficult despite
    being unimodal. Tests an algorithm's ability to follow ridges.
    Optimum at (1, 1, ..., 1) with fitness = 0.0.

    Args:
        n_dims: number of dimensions. Default 5.
        bounds: (low, high) bounds for each parameter. Default (-5, 10).
    """

    def __init__(self, n_dims: int = 5,
                 bounds: tuple[float, float] = (-5.0, 10.0)):
        self.n_dims = n_dims
        self.bounds = bounds

    def evaluate(self, params: dict) -> dict:
        values = np.array([params[f'x{i}'] for i in range(self.n_dims)])
        total = 0.0
        for i in range(self.n_dims - 1):
            total += 100 * (values[i + 1] - values[i] ** 2) ** 2 + \
                     (1 - values[i]) ** 2
        return {'fitness': -float(total)}

    def param_spec(self) -> dict:
        return {f'x{i}': self.bounds for i in range(self.n_dims)}


class RastriginFitness(FitnessFunction):
    """Rastrigin function (negated for maximization).

    f(x) = -(A*n + sum(x_i^2 - A*cos(2*pi*x_i)))

    A highly multimodal function with regularly distributed local optima.
    The global optimum is at the origin with fitness = 0.0.
    Approximately 10^n local optima in n dimensions.

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
        values = np.array([params[f'x{i}'] for i in range(self.n_dims)])
        raw = self.A * self.n_dims + np.sum(
            values ** 2 - self.A * np.cos(2 * np.pi * values))
        return {'fitness': -float(raw), 'raw_rastrigin': float(raw)}

    def param_spec(self) -> dict:
        return {f'x{i}': self.bounds for i in range(self.n_dims)}


class AckleyFitness(FitnessFunction):
    """Ackley function (negated for maximization).

    f(x) = -(a*exp(-b*sqrt(mean(x_i^2)))
             - exp(mean(cos(c*x_i))) + a + e)

    Multimodal with a nearly flat outer region that gives gradient-based
    methods trouble. A large hole at the origin. Optimum at origin
    with fitness = 0.0.

    Args:
        n_dims: number of dimensions. Default 5.
        bounds: (low, high) bounds for each parameter. Default (-5, 5).
        a: depth parameter. Default 20.
        b: width parameter. Default 0.2.
        c: cosine frequency. Default 2*pi.
    """

    def __init__(self, n_dims: int = 5,
                 bounds: tuple[float, float] = (-5.0, 5.0),
                 a: float = 20.0, b: float = 0.2,
                 c: float = 2 * np.pi):
        self.n_dims = n_dims
        self.bounds = bounds
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, params: dict) -> dict:
        values = np.array([params[f'x{i}'] for i in range(self.n_dims)])
        n = len(values)

        sum_sq = np.sum(values ** 2)
        sum_cos = np.sum(np.cos(self.c * values))

        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        raw = term1 + term2 + self.a + np.e

        return {'fitness': -float(raw), 'raw_ackley': float(raw)}

    def param_spec(self) -> dict:
        return {f'x{i}': self.bounds for i in range(self.n_dims)}


class ZDT1Fitness(FitnessFunction):
    """ZDT1 bi-objective test problem.

    f1(x) = x0
    f2(x) = g * (1 - sqrt(x0/g))
    g(x) = 1 + 9 * sum(x_1..x_{n-1}) / (n-1)

    Both objectives are to be minimized (negated here for maximization).
    The Pareto front is where g=1, i.e., x_1=...=x_{n-1}=0, and
    f2 = 1 - sqrt(f1), a convex curve from (0,1) to (1,0).

    Args:
        n_dims: number of dimensions. Default 30.
    """

    def __init__(self, n_dims: int = 30):
        self.n_dims = n_dims

    def evaluate(self, params: dict) -> dict:
        values = np.array([params[f'x{i}'] for i in range(self.n_dims)])
        x0 = values[0]

        if self.n_dims > 1:
            g = 1 + 9 * np.sum(values[1:]) / (self.n_dims - 1)
        else:
            g = 1.0

        f1 = x0
        f2 = g * (1 - np.sqrt(max(x0 / g, 0.0)))

        # Negate for maximization; use f1 as primary fitness
        return {
            'fitness': -float(f1),
            'f1': float(f1),
            'f2': float(f2),
            'neg_f2': -float(f2),
        }

    def param_spec(self) -> dict:
        return {f'x{i}': (0.0, 1.0) for i in range(self.n_dims)}
