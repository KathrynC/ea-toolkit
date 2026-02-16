"""
tests/test_atlas.py -- Tests for the AtlasBuilder.

Tests:
- Phase 1 probe structure and gradient magnitudes.
- Phase 2 slice pairs and grid shapes.
- Phase 3 anatomy radii and roughness.
- Budget respect, save/load roundtrip, reproducibility.
- Visualization file creation.
- Graceful degradation under low budget.
"""

import json
import os

import numpy as np
import pytest

from ea_toolkit.atlas import AtlasBuilder
from tests.conftest import SphereFitness, RastriginFitness


class TestAtlasBuild:
    """Tests for AtlasBuilder.build()."""

    def test_build_returns_dict(self):
        """Atlas should have expected top-level keys."""
        sf = SphereFitness(n_dims=3)
        ab = AtlasBuilder(sf, seed=42)
        atlas = ab.build(n_probes=10, budget=500)

        assert isinstance(atlas, dict)
        for key in ('meta', 'probes', 'stats', 'slices', 'anatomy'):
            assert key in atlas, f"Missing key: {key}"

    def test_phase1_probes(self):
        """Phase 1 should produce correct number of probes with required fields."""
        sf = SphereFitness(n_dims=3)
        ab = AtlasBuilder(sf, seed=42)
        atlas = ab.build(n_probes=10, budget=1000)

        probes = atlas['probes']
        assert len(probes) == 10

        for probe in probes:
            assert 'params' in probe
            assert 'fitness' in probe
            assert 'gradient' in probe
            assert 'gradient_magnitude' in probe
            assert 'cliffiness' in probe
            assert isinstance(probe['params'], dict)
            assert isinstance(probe['fitness'], float)

    def test_phase1_gradient_magnitudes(self):
        """Gradients should be computed at each probe point."""
        sf = SphereFitness(n_dims=3)
        ab = AtlasBuilder(sf, seed=42)
        atlas = ab.build(n_probes=10, budget=1000)

        probes = atlas['probes']
        # At least some probes should have non-zero gradient magnitudes
        # (sphere function has gradient everywhere except origin)
        grad_mags = [p['gradient_magnitude'] for p in probes]
        assert any(g > 0 for g in grad_mags), \
            "Expected at least some non-zero gradient magnitudes"

    def test_phase2_slice_pairs(self):
        """Phase 2 should generate correct number of slices."""
        sf = SphereFitness(n_dims=3)
        ab = AtlasBuilder(sf, seed=42)
        atlas = ab.build(n_probes=10, n_slice_pairs=2,
                         slice_resolution=10, budget=2000)

        slices = atlas['slices']
        assert len(slices) == 2

        for s in slices:
            assert 'param_x' in s
            assert 'param_y' in s
            assert 'x_vals' in s
            assert 'y_vals' in s
            assert 'fitness_grid' in s
            assert 'cliffiness_grid' in s
            assert 'center_pos' in s

    def test_phase2_grid_shape(self):
        """Each slice grid should have correct resolution."""
        sf = SphereFitness(n_dims=3)
        ab = AtlasBuilder(sf, seed=42)
        res = 15
        atlas = ab.build(n_probes=5, n_slice_pairs=1,
                         slice_resolution=res, budget=2000)

        for s in atlas['slices']:
            grid = np.array(s['fitness_grid'])
            assert grid.shape == (res, res), \
                f"Expected ({res}, {res}), got {grid.shape}"
            assert len(s['x_vals']) == res
            assert len(s['y_vals']) == res

    def test_phase3_anatomy_radii(self):
        """Phase 3 should sample correct number of radii."""
        sf = SphereFitness(n_dims=3)
        ab = AtlasBuilder(sf, seed=42)
        n_radii = 15
        atlas = ab.build(n_probes=10, n_anatomy_radii=n_radii,
                         budget=2000)

        anatomy = atlas['anatomy']
        assert len(anatomy['radii']) == n_radii
        assert len(anatomy['fitness_profile']) == n_radii

    def test_phase3_roughness(self):
        """Roughness should be computed on the anatomy transect."""
        sf = SphereFitness(n_dims=3)
        ab = AtlasBuilder(sf, seed=42)
        atlas = ab.build(n_probes=10, n_anatomy_radii=20,
                         budget=2000)

        anatomy = atlas['anatomy']
        assert 'roughness' in anatomy
        assert 'sign_flip_rate' in anatomy
        assert isinstance(anatomy['roughness'], float)
        assert anatomy['roughness'] >= 0.0

    def test_budget_respected(self):
        """Total evaluations should not exceed budget."""
        sf = SphereFitness(n_dims=3)
        ab = AtlasBuilder(sf, seed=42)
        budget = 500
        atlas = ab.build(n_probes=50, budget=budget)

        assert atlas['meta']['budget_used'] <= budget, \
            f"Used {atlas['meta']['budget_used']} > budget {budget}"

    def test_save_load_roundtrip(self, tmp_path):
        """Save to JSON, load, compare."""
        sf = SphereFitness(n_dims=3)
        ab = AtlasBuilder(sf, seed=42)
        atlas = ab.build(n_probes=5, budget=500)

        path = str(tmp_path / "test_atlas.json")
        ab.save(path)
        loaded = AtlasBuilder.load(path)

        assert loaded['meta'] == atlas['meta']
        assert len(loaded['probes']) == len(atlas['probes'])
        assert loaded['stats']['fitness_mean'] == \
            pytest.approx(atlas['stats']['fitness_mean'])

    def test_reproducibility(self):
        """Same seed should produce identical atlas."""
        sf = SphereFitness(n_dims=3)

        ab1 = AtlasBuilder(sf, seed=42)
        atlas1 = ab1.build(n_probes=10, budget=500)

        ab2 = AtlasBuilder(sf, seed=42)
        atlas2 = ab2.build(n_probes=10, budget=500)

        for p1, p2 in zip(atlas1['probes'], atlas2['probes']):
            assert p1['fitness'] == pytest.approx(p2['fitness'])
            assert p1['cliffiness'] == pytest.approx(p2['cliffiness'])

    def test_visualize_creates_files(self, tmp_path):
        """5 PNG files should be created."""
        sf = SphereFitness(n_dims=3)
        ab = AtlasBuilder(sf, seed=42)
        ab.build(n_probes=5, n_slice_pairs=1,
                 slice_resolution=5, budget=500)

        ab.visualize(output_dir=str(tmp_path))

        expected_files = [
            'atlas_fitness_distribution.png',
            'atlas_cliffiness_vs_fitness.png',
            'atlas_slice_heatmaps.png',
            'atlas_cliff_anatomy.png',
            'atlas_parameter_sensitivity.png',
        ]
        for fname in expected_files:
            fpath = tmp_path / fname
            assert fpath.exists(), f"Missing: {fname}"
            assert fpath.stat().st_size > 0, f"Empty: {fname}"

    def test_low_budget(self):
        """AtlasBuilder should degrade gracefully with very small budget."""
        sf = SphereFitness(n_dims=3)
        ab = AtlasBuilder(sf, seed=42)
        atlas = ab.build(n_probes=100, budget=10)

        # Should still produce valid structure, just fewer probes
        assert 'meta' in atlas
        assert 'probes' in atlas
        assert atlas['meta']['budget_used'] <= 10
        # At least 1 probe should be possible with budget=10
        assert len(atlas['probes']) >= 1


class TestAtlasRastrigin:
    """Tests using Rastrigin to verify multimodal landscape detection."""

    def test_rastrigin_cliffiness(self):
        """Rastrigin landscape should show non-trivial cliffiness."""
        rf = RastriginFitness(n_dims=3)
        ab = AtlasBuilder(rf, seed=42)
        atlas = ab.build(n_probes=10, budget=1000)

        # Rastrigin is multimodal, should have some cliffiness
        cliff_vals = [p['cliffiness'] for p in atlas['probes']]
        assert any(c > 0 for c in cliff_vals), \
            "Expected some cliffiness in Rastrigin landscape"
