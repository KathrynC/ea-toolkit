"""
ea_toolkit.mutation -- Mutation operators for evolutionary algorithms.

Three operators extracted from walker_competition.py and temporal_optimizer.py:

- GaussianMutation: Perturb along a random unit vector with configurable sigma.
  From walker_competition.py perturb() pattern (radius-based perturbation along
  a random direction on the unit sphere).

- CauchyMutation: Heavy-tailed Cauchy distribution for occasional large jumps.
  Useful for escaping local optima in rugged landscapes.

- AdaptiveMutation: Sigma adapts based on success rate using the 1/5th rule
  from evolution strategy theory. From temporal_optimizer.py adaptive patterns.
"""

import numpy as np

from ea_toolkit.base import MutationOperator


def _params_to_vec(params: dict, param_spec: dict) -> tuple[np.ndarray, list[str]]:
    """Convert a params dict to a numpy array with consistent ordering.

    Args:
        params: parameter name -> value mapping.
        param_spec: parameter name -> (low, high) bounds.

    Returns:
        (vector, names) where vector is a numpy array and names is the
        ordered list of parameter names.
    """
    names = sorted(param_spec.keys())
    vec = np.array([params[n] for n in names], dtype=float)
    return vec, names


def _vec_to_params(vec: np.ndarray, names: list[str]) -> dict:
    """Convert a numpy array back to a params dict.

    Args:
        vec: numpy array of parameter values.
        names: ordered list of parameter names matching the array.

    Returns:
        dict mapping parameter names to float values.
    """
    return {n: float(vec[i]) for i, n in enumerate(names)}


def _clamp(vec: np.ndarray, names: list[str], param_spec: dict) -> np.ndarray:
    """Clamp a parameter vector to the bounds defined in param_spec.

    Args:
        vec: numpy array of parameter values.
        names: ordered list of parameter names.
        param_spec: parameter name -> (low, high) bounds.

    Returns:
        Clamped numpy array.
    """
    result = vec.copy()
    for i, n in enumerate(names):
        lo, hi = param_spec[n]
        result[i] = np.clip(result[i], lo, hi)
    return result


class GaussianMutation(MutationOperator):
    """Perturb parameters along a random unit vector with Gaussian step size.

    Extracted from walker_competition.py perturb() pattern: generates a
    uniformly random direction on the N-dimensional unit sphere and moves
    the parameter vector by a step drawn from N(0, sigma) in that direction.

    This produces isotropic perturbations whose magnitude is controlled
    by sigma, while the direction is uniformly distributed on the sphere.

    Args:
        sigma: standard deviation of the step size. Default 0.1.
    """

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def mutate(self, params: dict, param_spec: dict,
               rng: np.random.Generator) -> dict:
        """Apply Gaussian mutation along a random unit direction.

        Args:
            params: current parameter values.
            param_spec: parameter bounds.
            rng: numpy random generator.

        Returns:
            New parameter dict with mutation applied and values clamped.
        """
        vec, names = _params_to_vec(params, param_spec)
        n_dim = len(vec)

        # Random direction on unit sphere (from walker_competition.py perturb())
        direction = rng.standard_normal(n_dim)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            direction = np.ones(n_dim)
            norm = np.linalg.norm(direction)
        direction = direction / norm

        # Step size drawn from Gaussian
        step = rng.normal(0, self.sigma)
        new_vec = vec + step * direction

        new_vec = _clamp(new_vec, names, param_spec)
        return _vec_to_params(new_vec, names)


class CauchyMutation(MutationOperator):
    """Heavy-tailed Cauchy mutation for occasional large jumps.

    Uses the Cauchy distribution (Student-t with 1 degree of freedom)
    which has much heavier tails than the Gaussian. This means most
    mutations are small, but occasionally a very large jump occurs,
    helping to escape local optima in rugged fitness landscapes.

    The perturbation is applied independently to each parameter dimension,
    scaled by the parameter range and a configurable scale factor.

    Args:
        scale: scale parameter for the Cauchy distribution, expressed
            as a fraction of each parameter's range. Default 0.05.
    """

    def __init__(self, scale: float = 0.05):
        self.scale = scale

    def mutate(self, params: dict, param_spec: dict,
               rng: np.random.Generator) -> dict:
        """Apply Cauchy mutation independently per dimension.

        Args:
            params: current parameter values.
            param_spec: parameter bounds.
            rng: numpy random generator.

        Returns:
            New parameter dict with mutation applied and values clamped.
        """
        vec, names = _params_to_vec(params, param_spec)
        n_dim = len(vec)

        # Compute per-dimension ranges for scaling
        ranges = np.array([param_spec[n][1] - param_spec[n][0] for n in names],
                          dtype=float)

        # Cauchy samples via inverse CDF: tan(pi * (U - 0.5))
        u = rng.uniform(0, 1, size=n_dim)
        cauchy_samples = np.tan(np.pi * (u - 0.5))

        new_vec = vec + self.scale * ranges * cauchy_samples
        new_vec = _clamp(new_vec, names, param_spec)
        return _vec_to_params(new_vec, names)


class AdaptiveMutation(MutationOperator):
    """Adaptive Gaussian mutation using the 1/5th success rule.

    From evolution strategy (ES) theory: if the fraction of successful
    mutations exceeds 1/5, increase sigma to explore more broadly;
    if it falls below 1/5, decrease sigma to exploit locally.

    The adaptation is performed over a sliding window of recent mutations.
    This operator tracks its own state (sigma, success history) across
    calls to mutate().

    Extracted from temporal_optimizer.py adaptive mutation patterns.

    Args:
        sigma_init: initial mutation step size. Default 0.1.
        sigma_min: minimum allowed sigma. Default 0.001.
        sigma_max: maximum allowed sigma. Default 1.0.
        window: number of recent mutations to track for success rate. Default 20.
        adaptation_rate: multiplicative factor for sigma adjustment. Default 1.2.
    """

    def __init__(self, sigma_init: float = 0.1, sigma_min: float = 0.001,
                 sigma_max: float = 1.0, window: int = 20,
                 adaptation_rate: float = 1.2):
        self.sigma = sigma_init
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.window = window
        self.adaptation_rate = adaptation_rate
        self._successes: list[bool] = []
        self._last_fitness: float | None = None

    def report_success(self, improved: bool) -> None:
        """Report whether the last mutation was successful (improved fitness).

        This must be called after each mutate() + evaluate cycle to keep
        the success rate estimate current.

        Args:
            improved: True if the mutated individual had higher fitness
                than its parent.
        """
        self._successes.append(improved)
        if len(self._successes) > self.window:
            self._successes = self._successes[-self.window:]

        # Adapt sigma using 1/5th rule
        if len(self._successes) >= self.window:
            success_rate = sum(self._successes) / len(self._successes)
            if success_rate > 0.2:
                # Too many successes -> increase sigma to explore more
                self.sigma = min(self.sigma * self.adaptation_rate,
                                 self.sigma_max)
            elif success_rate < 0.2:
                # Too few successes -> decrease sigma to exploit locally
                self.sigma = max(self.sigma / self.adaptation_rate,
                                 self.sigma_min)

    def mutate(self, params: dict, param_spec: dict,
               rng: np.random.Generator) -> dict:
        """Apply adaptive Gaussian mutation along a random unit direction.

        Uses the current (adapted) sigma value. Call report_success()
        after evaluating the mutant to update the adaptation.

        Args:
            params: current parameter values.
            param_spec: parameter bounds.
            rng: numpy random generator.

        Returns:
            New parameter dict with mutation applied and values clamped.
        """
        vec, names = _params_to_vec(params, param_spec)
        n_dim = len(vec)

        # Random direction on unit sphere
        direction = rng.standard_normal(n_dim)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            direction = np.ones(n_dim)
            norm = np.linalg.norm(direction)
        direction = direction / norm

        step = rng.normal(0, self.sigma)
        new_vec = vec + step * direction

        new_vec = _clamp(new_vec, names, param_spec)
        return _vec_to_params(new_vec, names)
