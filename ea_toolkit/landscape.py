"""
ea_toolkit.landscape -- Fitness landscape analysis tools.

Extracted from atlas_cliffiness.py and related landscape analysis scripts.

Provides:
- probe_cliffiness(): probe around a point to measure sensitivity.
- roughness_ratio(): ratio of local variation to global range.
- sign_flip_rate(): fraction of adjacent samples with sign change.
- gradient_estimate(): finite-difference gradient approximation.
- LandscapeAnalyzer: comprehensive landscape analysis class.
"""

import numpy as np

from ea_toolkit.base import FitnessFunction


def _params_to_vec(params: dict, param_spec: dict) -> tuple[np.ndarray, list[str]]:
    """Convert params dict to array with consistent ordering."""
    names = sorted(param_spec.keys())
    vec = np.array([params[n] for n in names], dtype=float)
    return vec, names


def _vec_to_params(vec: np.ndarray, names: list[str]) -> dict:
    """Convert array back to params dict."""
    return {n: float(vec[i]) for i, n in enumerate(names)}


def _clamp_vec(vec: np.ndarray, names: list[str],
               param_spec: dict) -> np.ndarray:
    """Clamp vector to bounds."""
    result = vec.copy()
    for i, n in enumerate(names):
        lo, hi = param_spec[n]
        result[i] = np.clip(result[i], lo, hi)
    return result


def probe_cliffiness(fitness_fn: FitnessFunction, params: dict,
                     param_spec: dict, radius: float = 0.1,
                     n_directions: int = 6,
                     rng: np.random.Generator | None = None) -> float:
    """Probe around a point to measure the maximum sensitivity (cliffiness).

    Generates n_directions random unit vectors, perturbs the parameter
    point by radius in each direction, evaluates fitness, and returns
    the maximum absolute fitness change. Higher values indicate the
    point is near a "cliff" in the fitness landscape.

    Extracted from atlas_cliffiness.py probe_gradient() pattern.

    Args:
        fitness_fn: fitness function to evaluate.
        params: center point parameter dict.
        param_spec: parameter bounds dict.
        radius: perturbation radius in parameter space. Default 0.1.
        n_directions: number of random directions to probe. Default 6.
        rng: numpy random generator. If None, creates a default one.

    Returns:
        float: maximum absolute fitness change across all probe
        directions (the "cliffiness" score).
    """
    if rng is None:
        rng = np.random.default_rng()

    vec, names = _params_to_vec(params, param_spec)
    n_dim = len(vec)

    # Evaluate base fitness
    base_result = fitness_fn.evaluate(params)
    base_fitness = base_result.get('fitness', 0.0)

    max_delta = 0.0

    for _ in range(n_directions):
        # Random direction on unit sphere
        direction = rng.standard_normal(n_dim)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            direction = np.ones(n_dim)
            norm = np.linalg.norm(direction)
        direction = direction / norm

        # Perturb and evaluate
        perturbed_vec = vec + radius * direction
        perturbed_vec = _clamp_vec(perturbed_vec, names, param_spec)
        perturbed_params = _vec_to_params(perturbed_vec, names)

        perturbed_result = fitness_fn.evaluate(perturbed_params)
        perturbed_fitness = perturbed_result.get('fitness', 0.0)

        delta = abs(perturbed_fitness - base_fitness)
        if delta > max_delta:
            max_delta = delta

    return max_delta


def roughness_ratio(fitness_values: list | np.ndarray) -> float:
    """Compute the roughness ratio: local variation / global range.

    The roughness ratio measures how "noisy" a fitness landscape is
    relative to its overall range. High values (near 1.0) indicate
    a very rugged landscape; low values (near 0.0) indicate smoothness.

    Local variation is measured as the mean absolute difference between
    consecutive values. Global range is max - min.

    Args:
        fitness_values: sequence of fitness values (e.g., sampled along
            a 1D transect through the landscape).

    Returns:
        float: roughness ratio in [0, inf). Returns 0.0 if the global
        range is zero or there are fewer than 2 values.
    """
    values = np.asarray(fitness_values, dtype=float)
    if len(values) < 2:
        return 0.0

    global_range = values.max() - values.min()
    if global_range < 1e-12:
        return 0.0

    # Mean absolute consecutive difference
    local_variation = np.mean(np.abs(np.diff(values)))

    return float(local_variation / global_range)


def sign_flip_rate(gradient_samples: list | np.ndarray) -> float:
    """Compute the fraction of adjacent gradient samples that change sign.

    A high sign flip rate indicates a highly non-monotonic (rugged)
    fitness landscape. Used to detect regions where the gradient
    direction is unstable.

    Args:
        gradient_samples: sequence of gradient values (can be scalar
            gradients or individual components). Sign changes are
            detected between consecutive elements.

    Returns:
        float: fraction of adjacent pairs with a sign change, in [0, 1].
        Returns 0.0 if there are fewer than 2 samples.
    """
    samples = np.asarray(gradient_samples, dtype=float)
    if len(samples) < 2:
        return 0.0

    signs = np.sign(samples)
    # Count sign changes (where sign[i] != sign[i+1] and neither is zero)
    n_flips = 0
    n_pairs = 0
    for i in range(len(signs) - 1):
        if signs[i] != 0 and signs[i + 1] != 0:
            n_pairs += 1
            if signs[i] != signs[i + 1]:
                n_flips += 1
        elif signs[i] != 0 or signs[i + 1] != 0:
            # One is zero, the other is not -- count as a pair but not a flip
            n_pairs += 1

    return float(n_flips / n_pairs) if n_pairs > 0 else 0.0


def gradient_estimate(fitness_fn: FitnessFunction, params: dict,
                      param_spec: dict,
                      epsilon: float = 0.01) -> dict[str, float]:
    """Estimate the fitness gradient via central finite differences.

    For each parameter dimension, evaluates fitness at +epsilon and
    -epsilon from the current point and computes the central difference
    approximation: df/dx_i = (f(x+e_i) - f(x-e_i)) / (2*epsilon).

    Args:
        fitness_fn: fitness function to evaluate.
        params: center point parameter dict.
        param_spec: parameter bounds dict.
        epsilon: step size for finite differences. Default 0.01.

    Returns:
        dict mapping parameter names to estimated partial derivatives.
    """
    names = sorted(param_spec.keys())
    gradient = {}

    for name in names:
        lo, hi = param_spec[name]

        # Forward point
        params_plus = dict(params)
        params_plus[name] = min(params[name] + epsilon, hi)

        # Backward point
        params_minus = dict(params)
        params_minus[name] = max(params[name] - epsilon, lo)

        # Actual step size (may differ from epsilon at boundaries)
        actual_step = params_plus[name] - params_minus[name]
        if actual_step < 1e-12:
            gradient[name] = 0.0
            continue

        # Evaluate both points
        f_plus = fitness_fn.evaluate(params_plus).get('fitness', 0.0)
        f_minus = fitness_fn.evaluate(params_minus).get('fitness', 0.0)

        gradient[name] = (f_plus - f_minus) / actual_step

    return gradient


class LandscapeAnalyzer:
    """Comprehensive fitness landscape analysis.

    Samples the landscape at random points and computes statistics
    about smoothness, sensitivity, and gradient structure.

    Args:
        fitness_fn: fitness function to analyze.
        seed: random seed for reproducibility.
    """

    def __init__(self, fitness_fn: FitnessFunction,
                 seed: int | None = None):
        self.fitness_fn = fitness_fn
        self.param_spec = fitness_fn.param_spec()
        self.rng = np.random.default_rng(seed)

    def run_analysis(self, n_samples: int = 50,
                     budget: int = 500) -> dict:
        """Run a landscape analysis within the given evaluation budget.

        Samples n_samples random points, evaluates fitness and gradients,
        probes cliffiness, and computes aggregate statistics.

        The total number of fitness evaluations is bounded by budget.
        Each sample requires: 1 (base) + 2*n_dim (gradient) + n_directions
        (cliffiness) evaluations. If the budget is insufficient for
        all n_samples, fewer samples are taken.

        Args:
            n_samples: desired number of sample points.
            budget: maximum total fitness evaluations to perform.

        Returns:
            dict with landscape statistics:
                - n_samples: actual number of samples taken.
                - fitness_mean, fitness_std: fitness statistics.
                - fitness_min, fitness_max: fitness range.
                - roughness: roughness ratio of sampled fitness values.
                - mean_cliffiness: average probe cliffiness.
                - max_cliffiness: maximum observed cliffiness.
                - mean_gradient_magnitude: average gradient L2 norm.
                - gradient_sign_flip_rate: sign instability per dimension.
                - evals_used: total fitness evaluations performed.
        """
        names = sorted(self.param_spec.keys())
        n_dim = len(names)

        # Estimate evaluations per sample:
        # 1 base + 2*n_dim gradient + 6 cliffiness probes
        n_cliff_probes = min(6, n_dim)
        evals_per_sample = 1 + 2 * n_dim + n_cliff_probes
        actual_n_samples = min(n_samples, max(1, budget // evals_per_sample))

        fitness_values = []
        cliffiness_values = []
        gradient_magnitudes = []
        gradient_components = {n: [] for n in names}
        evals_used = 0

        for _ in range(actual_n_samples):
            if evals_used >= budget:
                break

            # Random sample point
            params = {}
            for name in names:
                lo, hi = self.param_spec[name]
                params[name] = float(self.rng.uniform(lo, hi))

            # Evaluate fitness
            result = self.fitness_fn.evaluate(params)
            fitness = result.get('fitness', 0.0)
            fitness_values.append(fitness)
            evals_used += 1

            # Gradient estimate (2 * n_dim evaluations)
            if evals_used + 2 * n_dim <= budget:
                grad = gradient_estimate(self.fitness_fn, params,
                                         self.param_spec, epsilon=0.01)
                grad_vec = np.array([grad[n] for n in names])
                gradient_magnitudes.append(float(np.linalg.norm(grad_vec)))
                for n in names:
                    gradient_components[n].append(grad[n])
                evals_used += 2 * n_dim

            # Cliffiness probe
            if evals_used + n_cliff_probes <= budget:
                cliff = probe_cliffiness(self.fitness_fn, params,
                                         self.param_spec,
                                         radius=0.1,
                                         n_directions=n_cliff_probes,
                                         rng=self.rng)
                cliffiness_values.append(cliff)
                evals_used += n_cliff_probes

        fitness_arr = np.array(fitness_values)

        # Compute per-dimension sign flip rates
        dim_flip_rates = {}
        for n in names:
            if len(gradient_components[n]) >= 2:
                dim_flip_rates[n] = sign_flip_rate(gradient_components[n])
            else:
                dim_flip_rates[n] = 0.0
        avg_flip_rate = np.mean(list(dim_flip_rates.values())) if dim_flip_rates else 0.0

        stats = {
            'n_samples': len(fitness_values),
            'fitness_mean': float(np.mean(fitness_arr)) if len(fitness_arr) > 0 else 0.0,
            'fitness_std': float(np.std(fitness_arr)) if len(fitness_arr) > 0 else 0.0,
            'fitness_min': float(np.min(fitness_arr)) if len(fitness_arr) > 0 else 0.0,
            'fitness_max': float(np.max(fitness_arr)) if len(fitness_arr) > 0 else 0.0,
            'roughness': roughness_ratio(fitness_arr) if len(fitness_arr) >= 2 else 0.0,
            'mean_cliffiness': float(np.mean(cliffiness_values)) if cliffiness_values else 0.0,
            'max_cliffiness': float(np.max(cliffiness_values)) if cliffiness_values else 0.0,
            'mean_gradient_magnitude': float(np.mean(gradient_magnitudes)) if gradient_magnitudes else 0.0,
            'gradient_sign_flip_rate': float(avg_flip_rate),
            'evals_used': evals_used,
        }

        return stats
