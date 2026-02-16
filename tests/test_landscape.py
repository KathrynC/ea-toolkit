"""
tests/test_landscape.py -- Tests for landscape analysis tools.

Tests:
- probe_cliffiness on a step function (should detect cliff).
- roughness_ratio on smooth vs noisy data.
- gradient_estimate on a linear function (should match analytical gradient).
- sign_flip_rate on various sequences.
- LandscapeAnalyzer on the sphere function.
"""

import numpy as np
import pytest

from ea_toolkit.landscape import (
    probe_cliffiness,
    roughness_ratio,
    sign_flip_rate,
    gradient_estimate,
    LandscapeAnalyzer,
)
from tests.conftest import (
    SphereFitness,
    StepFitness,
    LinearFitness,
)


class TestProbeCliffiness:
    """Tests for the probe_cliffiness function."""

    def test_detects_cliff_on_step_function(self):
        """probe_cliffiness should detect the cliff in a step function."""
        sf = StepFitness(n_dims=3)
        # Point near the cliff (x0 close to 0)
        params = {'x0': 0.01, 'x1': 0.0, 'x2': 0.0}
        spec = sf.param_spec()

        cliffiness = probe_cliffiness(sf, params, spec, radius=0.1,
                                      n_directions=20,
                                      rng=np.random.default_rng(42))

        # The step function has a 20-unit cliff at x0=0
        # With radius 0.1 and enough directions, some probes should cross it
        assert cliffiness > 1.0, (
            f"Expected high cliffiness near step, got {cliffiness}")

    def test_low_cliffiness_far_from_cliff(self):
        """probe_cliffiness should be low far from any cliff."""
        sf = StepFitness(n_dims=3)
        # Point far from the cliff
        params = {'x0': 3.0, 'x1': 0.0, 'x2': 0.0}
        spec = sf.param_spec()

        cliffiness = probe_cliffiness(sf, params, spec, radius=0.1,
                                      n_directions=10,
                                      rng=np.random.default_rng(42))

        # Far from x0=0, the step function is flat (fitness ~10 everywhere)
        assert cliffiness < 1.0, (
            f"Expected low cliffiness far from step, got {cliffiness}")

    def test_sphere_cliffiness(self):
        """probe_cliffiness on sphere should be proportional to distance."""
        sf = SphereFitness(n_dims=3)
        spec = sf.param_spec()

        # Near origin: low gradient -> low cliffiness
        cliff_near = probe_cliffiness(
            sf, {'x0': 0.1, 'x1': 0.0, 'x2': 0.0}, spec,
            radius=0.05, n_directions=10,
            rng=np.random.default_rng(42))

        # Far from origin: high gradient -> higher cliffiness
        cliff_far = probe_cliffiness(
            sf, {'x0': 3.0, 'x1': 3.0, 'x2': 3.0}, spec,
            radius=0.05, n_directions=10,
            rng=np.random.default_rng(42))

        assert cliff_far > cliff_near, (
            f"Expected higher cliffiness far from origin "
            f"({cliff_far}) vs near ({cliff_near})")


class TestRoughnessRatio:
    """Tests for the roughness_ratio function."""

    def test_smooth_data(self):
        """Smooth data should have a low roughness ratio."""
        # Linearly increasing data
        values = np.linspace(0, 10, 100)
        r = roughness_ratio(values)
        # For linear data, local variation / global range = step / range
        # step = 10/99, range = 10, ratio ~ 0.01
        assert r < 0.05, f"Expected low roughness for smooth data, got {r}"

    def test_noisy_data(self):
        """Noisy data should have a high roughness ratio."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal(100)
        r = roughness_ratio(values)
        assert r > 0.1, f"Expected high roughness for noisy data, got {r}"

    def test_constant_data(self):
        """Constant data should have roughness ratio 0."""
        values = np.ones(50)
        r = roughness_ratio(values)
        assert r == 0.0

    def test_single_value(self):
        """Single value should return 0."""
        assert roughness_ratio([5.0]) == 0.0

    def test_two_values(self):
        """Two values should work correctly."""
        r = roughness_ratio([0.0, 10.0])
        # local variation = 10, global range = 10, ratio = 1.0
        assert abs(r - 1.0) < 1e-10

    def test_alternating(self):
        """Alternating values should have high roughness."""
        values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        r = roughness_ratio(values)
        assert r == 1.0


class TestSignFlipRate:
    """Tests for the sign_flip_rate function."""

    def test_all_positive(self):
        """All positive values should have 0 sign flips."""
        values = [1.0, 2.0, 3.0, 4.0]
        rate = sign_flip_rate(values)
        assert rate == 0.0

    def test_all_negative(self):
        """All negative values should have 0 sign flips."""
        values = [-1.0, -2.0, -3.0]
        rate = sign_flip_rate(values)
        assert rate == 0.0

    def test_alternating_signs(self):
        """Alternating signs should have flip rate near 1.0."""
        values = [1.0, -1.0, 1.0, -1.0, 1.0]
        rate = sign_flip_rate(values)
        assert rate == 1.0

    def test_single_flip(self):
        """One sign change in the middle."""
        values = [1.0, 1.0, -1.0, -1.0]
        rate = sign_flip_rate(values)
        # 3 pairs: (+,+), (+,-), (-,-) -> 1 flip out of 3
        assert abs(rate - 1.0 / 3.0) < 1e-10

    def test_empty_or_single(self):
        """Edge cases: empty or single value."""
        assert sign_flip_rate([]) == 0.0
        assert sign_flip_rate([5.0]) == 0.0

    def test_with_zeros(self):
        """Zeros should be handled gracefully."""
        values = [1.0, 0.0, -1.0]
        rate = sign_flip_rate(values)
        # (1, 0): pair counted but no flip. (0, -1): pair counted but no flip
        # Actually: sign(1)=1, sign(0)=0, sign(-1)=-1
        # Pair (1,0): sign[i]=1, sign[i+1]=0 -> n_pairs+=1, no flip
        # Pair (0,-1): sign[i]=0, sign[i+1]=-1 -> n_pairs+=1, no flip
        assert rate == 0.0


class TestGradientEstimate:
    """Tests for the gradient_estimate function."""

    def test_linear_function(self):
        """Gradient of a linear function should match coefficients."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
        lf = LinearFitness(coefficients=coeffs)
        spec = lf.param_spec()

        params = {'x0': 0.0, 'x1': 0.0, 'x2': 0.0, 'x3': 0.0, 'x4': 0.0}
        grad = gradient_estimate(lf, params, spec, epsilon=0.01)

        for i, c in enumerate(coeffs):
            name = f'x{i}'
            assert abs(grad[name] - c) < 0.1, (
                f"Gradient for {name}: expected {c}, got {grad[name]}")

    def test_sphere_gradient_direction(self):
        """Sphere gradient should point toward origin (negative of position)."""
        sf = SphereFitness(n_dims=3)
        spec = sf.param_spec()

        params = {'x0': 2.0, 'x1': 1.0, 'x2': -1.0}
        grad = gradient_estimate(sf, params, spec, epsilon=0.01)

        # Gradient of -sum(x^2) is -2*x, so at x0=2 -> grad=-4
        assert grad['x0'] < 0, "Gradient should be negative at x0=2"
        assert grad['x1'] < 0, "Gradient should be negative at x1=1"
        assert grad['x2'] > 0, "Gradient should be positive at x2=-1"

    def test_gradient_at_origin(self):
        """Sphere gradient at origin should be near zero."""
        sf = SphereFitness(n_dims=3)
        spec = sf.param_spec()

        params = {'x0': 0.0, 'x1': 0.0, 'x2': 0.0}
        grad = gradient_estimate(sf, params, spec, epsilon=0.01)

        for name in ['x0', 'x1', 'x2']:
            assert abs(grad[name]) < 0.1, (
                f"Gradient at origin should be ~0, got {grad[name]} for {name}")


class TestLandscapeAnalyzer:
    """Tests for the LandscapeAnalyzer class."""

    def test_basic_analysis(self):
        """LandscapeAnalyzer should produce valid statistics."""
        sf = SphereFitness(n_dims=3)
        analyzer = LandscapeAnalyzer(sf, seed=42)

        stats = analyzer.run_analysis(n_samples=10, budget=500)

        assert 'n_samples' in stats
        assert stats['n_samples'] > 0
        assert 'fitness_mean' in stats
        assert 'fitness_std' in stats
        assert 'roughness' in stats
        assert 'mean_cliffiness' in stats
        assert 'mean_gradient_magnitude' in stats
        assert 'evals_used' in stats
        assert stats['evals_used'] <= 500

    def test_sphere_statistics(self):
        """Sphere landscape should have negative mean fitness."""
        sf = SphereFitness(n_dims=3)
        analyzer = LandscapeAnalyzer(sf, seed=42)

        stats = analyzer.run_analysis(n_samples=20, budget=1000)

        # Sphere function is always non-positive
        assert stats['fitness_max'] <= 0.01
        assert stats['fitness_mean'] < 0.0
        assert stats['fitness_std'] > 0.0

    def test_budget_respected(self):
        """Analyzer should not exceed the evaluation budget."""
        sf = SphereFitness(n_dims=3)
        analyzer = LandscapeAnalyzer(sf, seed=42)

        stats = analyzer.run_analysis(n_samples=100, budget=50)
        assert stats['evals_used'] <= 50
