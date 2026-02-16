"""
ea_toolkit.algorithms.cma_es -- Covariance Matrix Adaptation Evolution Strategy.

The gold standard for continuous black-box optimization in moderate dimensions.
Adapts a full covariance matrix to learn the local geometry of the fitness
landscape, with step-size control via cumulative path length.

Implements the (mu/mu_w, lambda)-CMA-ES with:
- Weighted recombination of mu best individuals
- Cumulative step-size adaptation (CSA)
- Rank-one and rank-mu covariance matrix updates
- Eigendecomposition for efficient sampling

Reference: Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial."
arXiv:1604.00772.

Supports the ask-tell interface natively.
"""

import numpy as np

from ea_toolkit.base import Algorithm, FitnessFunction


class CMAES(Algorithm):
    """Covariance Matrix Adaptation Evolution Strategy.

    CMA-ES maintains a multivariate normal distribution over the search
    space and adapts both the covariance matrix (search direction/shape)
    and step size (search magnitude) based on the success of recent
    generations.

    Args:
        fitness_fn: fitness function to optimize.
        sigma0: initial step size. Default 0.5.
        pop_size: offspring population size (lambda). If None, uses
            4 + floor(3 * ln(n)) which is the standard default.
        seed: random seed for reproducibility.
    """

    def __init__(self, fitness_fn: FitnessFunction,
                 sigma0: float = 0.5,
                 pop_size: int | None = None,
                 seed: int | None = None):
        super().__init__(fitness_fn, seed=seed)
        self.sigma0 = sigma0
        self._pop_size_override = pop_size
        self._state: dict | None = None

    def _init_state(self):
        """Initialize all CMA-ES state variables."""
        spec = self.fitness_fn.param_spec()
        names = sorted(spec.keys())
        n = len(names)

        # Population size
        lam = self._pop_size_override or (4 + int(3 * np.log(n)))
        mu = lam // 2

        # Recombination weights (log-linear, normalized)
        raw_weights = np.array(
            [np.log((lam + 1) / 2) - np.log(i + 1) for i in range(mu)])
        weights = raw_weights / raw_weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step-size control parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)
                   + c_sigma)

        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1,
                   2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Expected length of N(0,I) vector
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initialize mean at random point within bounds
        mean = np.array([
            self.rng.uniform(spec[name][0], spec[name][1])
            for name in names
        ])

        self._state = {
            'names': names,
            'spec': spec,
            'n': n,
            'lam': lam,
            'mu': mu,
            'weights': weights,
            'mu_eff': mu_eff,
            'c_sigma': c_sigma,
            'd_sigma': d_sigma,
            'c_c': c_c,
            'c_1': c_1,
            'c_mu': c_mu,
            'chi_n': chi_n,
            'mean': mean,
            'sigma': self.sigma0,
            'C': np.eye(n),
            'p_sigma': np.zeros(n),
            'p_c': np.zeros(n),
            'B': np.eye(n),
            'D': np.ones(n),
            'invsqrtC': np.eye(n),
            'gen': 0,
            'eigeneval': 0,
        }

    def _update_eigensystem(self):
        """Update eigendecomposition of C."""
        s = self._state
        n = s['n']

        # Only update every n/10 generations for efficiency
        if s['gen'] - s['eigeneval'] > s['lam'] / (s['c_1'] + s['c_mu']) / n / 10:
            s['eigeneval'] = s['gen']
            C = s['C']
            # Enforce symmetry
            C = np.triu(C) + np.triu(C, 1).T
            eigenvalues, B = np.linalg.eigh(C)
            # Ensure positive definiteness
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            D = np.sqrt(eigenvalues)
            s['B'] = B
            s['D'] = D
            s['invsqrtC'] = B @ np.diag(1.0 / D) @ B.T
            s['C'] = C

    def ask(self) -> list[dict]:
        """Sample lambda candidates from the current distribution.

        Returns:
            list of parameter dicts to evaluate.
        """
        if self._state is None:
            self._init_state()

        s = self._state
        self._update_eigensystem()

        candidates = []
        for _ in range(s['lam']):
            # Sample: x = mean + sigma * B * D * z, where z ~ N(0, I)
            z = self.rng.standard_normal(s['n'])
            y = s['B'] @ (s['D'] * z)
            x = s['mean'] + s['sigma'] * y

            # Clamp to bounds
            params = {}
            for i, name in enumerate(s['names']):
                lo, hi = s['spec'][name]
                params[name] = float(np.clip(x[i], lo, hi))
            candidates.append(params)

        return candidates

    def tell(self, evaluations: list[tuple[dict, dict]]) -> None:
        """Update the distribution based on evaluation results.

        Performs weighted recombination, evolution path updates,
        covariance matrix adaptation, and step-size control.

        Args:
            evaluations: list of (params_dict, result_dict) tuples.
        """
        s = self._state
        n = s['n']
        names = s['names']

        # Record all evaluations
        for params, result in evaluations:
            self._record(params, result)

        # Need at least mu evaluations for a proper update
        if len(evaluations) < s['mu']:
            # Partial generation at budget end — record but skip update
            best_f = (self._best.get('fitness', float('-inf'))
                      if self._best else float('-inf'))
            self._notify('on_generation',
                         generation=s.get('gen', 0),
                         best_fitness=best_f)
            return

        # Sort by fitness (descending — we maximize)
        sorted_evals = sorted(
            evaluations,
            key=lambda e: e[1].get('fitness', float('-inf')),
            reverse=True
        )

        # Old mean
        mean_old = s['mean'].copy()

        # New mean: weighted recombination of mu best
        mean_new = np.zeros(n)
        for i in range(s['mu']):
            params = sorted_evals[i][0]
            vec = np.array([params[name] for name in names])
            mean_new += s['weights'][i] * vec

        # Step in mean-space (in sigma units)
        y_w = (mean_new - mean_old) / s['sigma']

        # ── Cumulative step-size adaptation (CSA) ──
        s['p_sigma'] = ((1 - s['c_sigma']) * s['p_sigma'] +
                        np.sqrt(s['c_sigma'] * (2 - s['c_sigma']) *
                                s['mu_eff']) *
                        s['invsqrtC'] @ y_w)

        # ── Heaviside function for stalling detection ──
        s['gen'] += 1
        norm_p_sigma = np.linalg.norm(s['p_sigma'])
        expected_under_random = (
            np.sqrt(1 - (1 - s['c_sigma']) ** (2 * s['gen'])) *
            s['chi_n']
        )
        h_sigma = (1.0 if norm_p_sigma / expected_under_random
                   < (1.4 + 2 / (n + 1)) else 0.0)

        # ── Evolution path for covariance ──
        s['p_c'] = ((1 - s['c_c']) * s['p_c'] +
                    h_sigma *
                    np.sqrt(s['c_c'] * (2 - s['c_c']) * s['mu_eff']) *
                    y_w)

        # ── Covariance matrix update ──
        # Rank-one update
        rank_one = np.outer(s['p_c'], s['p_c'])

        # Rank-mu update
        rank_mu = np.zeros((n, n))
        for i in range(s['mu']):
            params = sorted_evals[i][0]
            vec = np.array([params[name] for name in names])
            yi = (vec - mean_old) / s['sigma']
            rank_mu += s['weights'][i] * np.outer(yi, yi)

        # Combine
        old_factor = (1 - s['c_1'] - s['c_mu'] +
                      (1 - h_sigma) * s['c_1'] * s['c_c'] * (2 - s['c_c']))
        s['C'] = (old_factor * s['C'] +
                  s['c_1'] * rank_one +
                  s['c_mu'] * rank_mu)

        # ── Step-size update ──
        s['sigma'] *= np.exp(
            (s['c_sigma'] / s['d_sigma']) *
            (norm_p_sigma / s['chi_n'] - 1)
        )

        # ── Update mean ──
        s['mean'] = mean_new

        # Notify callbacks
        best_f = (self._best.get('fitness', float('-inf'))
                  if self._best else float('-inf'))
        self._notify('on_generation',
                     generation=s['gen'],
                     best_fitness=best_f)

    def run(self, budget: int) -> list[dict]:
        """Run CMA-ES with the given evaluation budget.

        Args:
            budget: total number of fitness evaluations to perform.

        Returns:
            list of all evaluation history entries.
        """
        self._notify('on_start')

        while len(self.history) < budget:
            candidates = self.ask()
            evaluations = []
            for params in candidates:
                if len(self.history) + len(evaluations) >= budget:
                    break
                result = self.fitness_fn.evaluate(params)
                evaluations.append((params, result))

            if evaluations:
                self.tell(evaluations)
            else:
                break

        self._notify('on_finish')
        return self.history

    @property
    def sigma(self) -> float:
        """Current step size."""
        return self._state['sigma'] if self._state else self.sigma0

    @property
    def mean(self) -> np.ndarray | None:
        """Current distribution mean."""
        return self._state['mean'].copy() if self._state else None

    @property
    def covariance(self) -> np.ndarray | None:
        """Current covariance matrix."""
        return self._state['C'].copy() if self._state else None
