"""
ea_toolkit.atlas -- Multi-phase fitness landscape atlas builder.

Composes landscape.py tools (probe_cliffiness, gradient_estimate,
roughness_ratio, sign_flip_rate) into a structured, 3-phase atlas
that maps the full parameter space:

Phase 1: Random Probing — sample random points, compute fitness +
         gradient + cliffiness at each.
Phase 2: 2D Slices — generate heatmap grids for the most sensitive
         parameter pairs.
Phase 3: Cliff Anatomy — profile the fitness along the steepest
         gradient transect through the cliffiest point.

Output is a JSON-serializable dict compatible with viz-tools.
"""

import json

import numpy as np

from ea_toolkit.base import FitnessFunction
from ea_toolkit.landscape import (
    _params_to_vec,
    _vec_to_params,
    _clamp_vec,
    probe_cliffiness,
    gradient_estimate,
    roughness_ratio,
    sign_flip_rate,
)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class AtlasBuilder:
    """Multi-phase fitness landscape atlas builder.

    Composes landscape analysis tools into a structured atlas with
    three phases: random probing, 2D slice heatmaps, and cliff
    anatomy profiling.

    Args:
        fitness_fn: fitness function to analyze (must implement
            FitnessFunction protocol).
        seed: random seed for reproducibility.
    """

    def __init__(self, fitness_fn: FitnessFunction,
                 seed: int | None = None):
        self.fitness_fn = fitness_fn
        self.param_spec = fitness_fn.param_spec()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._atlas = None
        self._evals_used = 0

    def _eval(self, params: dict) -> float:
        """Evaluate fitness and increment counter."""
        result = self.fitness_fn.evaluate(params)
        self._evals_used += 1
        return result.get('fitness', 0.0)

    # ── Phase 1: Random Probing ──────────────────────────────────────────

    def _phase1(self, n_probes: int, budget: int) -> list[dict]:
        """Sample random points, compute fitness + gradient + cliffiness."""
        names = sorted(self.param_spec.keys())
        n_dim = len(names)
        probes = []

        for _ in range(n_probes):
            if self._evals_used >= budget:
                break

            # Random sample point
            params = {}
            for name in names:
                lo, hi = self.param_spec[name]
                params[name] = float(self.rng.uniform(lo, hi))

            # Fitness (1 eval)
            fitness = self._eval(params)

            # Gradient (2 * n_dim evals)
            if self._evals_used + 2 * n_dim > budget:
                probes.append({
                    'params': params,
                    'fitness': fitness,
                    'gradient': {n: 0.0 for n in names},
                    'gradient_magnitude': 0.0,
                    'cliffiness': 0.0,
                })
                continue

            grad = gradient_estimate(self.fitness_fn, params,
                                     self.param_spec, epsilon=0.01)
            self._evals_used += 2 * n_dim
            grad_vec = np.array([grad[n] for n in names])
            grad_mag = float(np.linalg.norm(grad_vec))

            # Cliffiness (n_directions evals)
            n_directions = min(6, n_dim)
            if self._evals_used + n_directions > budget:
                probes.append({
                    'params': params,
                    'fitness': fitness,
                    'gradient': grad,
                    'gradient_magnitude': grad_mag,
                    'cliffiness': 0.0,
                })
                continue

            cliff = probe_cliffiness(
                self.fitness_fn, params, self.param_spec,
                radius=0.1, n_directions=n_directions, rng=self.rng)
            self._evals_used += n_directions

            probes.append({
                'params': params,
                'fitness': fitness,
                'gradient': grad,
                'gradient_magnitude': grad_mag,
                'cliffiness': cliff,
            })

        return probes

    # ── Phase 2: 2D Slices ───────────────────────────────────────────────

    def _phase2(self, probes: list[dict], n_slice_pairs: int,
                slice_resolution: int, budget: int) -> list[dict]:
        """Generate 2D fitness/cliffiness grids for top parameter pairs."""
        names = sorted(self.param_spec.keys())
        n_dim = len(names)

        if n_dim < 2 or not probes:
            return []

        # Find best point (highest fitness)
        best_probe = max(probes, key=lambda p: p['fitness'])
        center_params = best_probe['params']

        # Rank parameter pairs by combined gradient magnitude
        pair_scores = {}
        for probe in probes:
            grad = probe['gradient']
            for i in range(n_dim):
                for j in range(i + 1, n_dim):
                    key = (names[i], names[j])
                    score = abs(grad[names[i]]) + abs(grad[names[j]])
                    pair_scores[key] = pair_scores.get(key, 0.0) + score

        # Sort pairs by total gradient contribution
        sorted_pairs = sorted(pair_scores.keys(),
                              key=lambda k: pair_scores[k], reverse=True)
        top_pairs = sorted_pairs[:n_slice_pairs]

        slices = []
        for param_x, param_y in top_pairs:
            if self._evals_used >= budget:
                break

            lo_x, hi_x = self.param_spec[param_x]
            lo_y, hi_y = self.param_spec[param_y]
            x_vals = np.linspace(lo_x, hi_x, slice_resolution).tolist()
            y_vals = np.linspace(lo_y, hi_y, slice_resolution).tolist()

            # Evaluate fitness at each grid cell
            fitness_grid = []
            for yi in range(slice_resolution):
                row = []
                for xi in range(slice_resolution):
                    if self._evals_used >= budget:
                        row.append(0.0)
                        continue
                    # Hold other params at best point
                    p = dict(center_params)
                    p[param_x] = x_vals[xi]
                    p[param_y] = y_vals[yi]
                    f = self._eval(p)
                    row.append(f)
                fitness_grid.append(row)

            # Compute cliffiness grid from adjacent cells (no extra evals)
            fitness_arr = np.array(fitness_grid)
            cliffiness_grid = np.zeros_like(fitness_arr)
            for yi in range(slice_resolution):
                for xi in range(slice_resolution):
                    neighbors = []
                    if yi > 0:
                        neighbors.append(fitness_arr[yi - 1, xi])
                    if yi < slice_resolution - 1:
                        neighbors.append(fitness_arr[yi + 1, xi])
                    if xi > 0:
                        neighbors.append(fitness_arr[yi, xi - 1])
                    if xi < slice_resolution - 1:
                        neighbors.append(fitness_arr[yi, xi + 1])
                    if neighbors:
                        neighbor_mean = np.mean(neighbors)
                        cliffiness_grid[yi, xi] = abs(
                            fitness_arr[yi, xi] - neighbor_mean)

            # Find center position in grid coordinates
            cx = float(center_params[param_x])
            cy = float(center_params[param_y])

            slices.append({
                'param_x': param_x,
                'param_y': param_y,
                'x_vals': x_vals,
                'y_vals': y_vals,
                'fitness_grid': fitness_arr.tolist(),
                'cliffiness_grid': cliffiness_grid.tolist(),
                'center_pos': [cx, cy],
            })

        return slices

    # ── Phase 3: Cliff Anatomy ───────────────────────────────────────────

    def _phase3(self, probes: list[dict], n_anatomy_radii: int,
                budget: int) -> dict:
        """Profile fitness along steepest gradient through cliffiest point."""
        names = sorted(self.param_spec.keys())

        if not probes:
            return {
                'center': {},
                'direction': {},
                'radii': [],
                'fitness_profile': [],
                'roughness': 0.0,
                'sign_flip_rate': 0.0,
            }

        # Find highest-cliffiness point
        cliff_probe = max(probes, key=lambda p: p['cliffiness'])
        center_params = cliff_probe['params']
        grad = cliff_probe['gradient']

        # Normalized gradient direction
        grad_vec = np.array([grad[n] for n in names])
        grad_norm = np.linalg.norm(grad_vec)
        if grad_norm < 1e-12:
            direction_vec = np.ones(len(names)) / np.sqrt(len(names))
        else:
            direction_vec = grad_vec / grad_norm
        direction = {n: float(direction_vec[i]) for i, n in enumerate(names)}

        # Sample fitness at equally-spaced radii along transect
        center_vec, _ = _params_to_vec(center_params, self.param_spec)
        radii = np.linspace(-0.3, 0.3, n_anatomy_radii).tolist()
        fitness_profile = []

        for r in radii:
            if self._evals_used >= budget:
                fitness_profile.append(0.0)
                continue
            sample_vec = center_vec + r * direction_vec
            sample_vec = _clamp_vec(sample_vec, names, self.param_spec)
            sample_params = _vec_to_params(sample_vec, names)
            f = self._eval(sample_params)
            fitness_profile.append(f)

        # Compute roughness and sign flip rate along transect
        profile_arr = np.array(fitness_profile)
        transect_roughness = roughness_ratio(profile_arr)
        diffs = np.diff(profile_arr)
        transect_sfr = sign_flip_rate(diffs.tolist())

        return {
            'center': center_params,
            'direction': direction,
            'radii': radii,
            'fitness_profile': fitness_profile,
            'roughness': float(transect_roughness),
            'sign_flip_rate': float(transect_sfr),
        }

    # ── Public API ───────────────────────────────────────────────────────

    def build(self, n_probes: int = 100, n_slice_pairs: int = 3,
              slice_resolution: int = 30, n_anatomy_radii: int = 20,
              budget: int = 5000) -> dict:
        """Build the atlas: probe, slice, and profile.

        Budget allocation: 40% Phase 1, 45% Phase 2, 15% Phase 3.
        Unspent budget rolls forward to the next phase.

        Args:
            n_probes: number of random probe points (Phase 1).
            n_slice_pairs: number of 2D parameter pair slices (Phase 2).
            slice_resolution: grid resolution per slice axis (Phase 2).
            n_anatomy_radii: number of radii along cliff transect (Phase 3).
            budget: maximum total fitness evaluations.

        Returns:
            dict with keys: meta, probes, stats, slices, anatomy.
        """
        self._evals_used = 0
        self.rng = np.random.default_rng(self.seed)
        names = sorted(self.param_spec.keys())

        # Budget allocation
        budget_p1 = int(budget * 0.40)
        budget_p2 = int(budget * 0.45)
        # Phase 3 gets the remainder

        # Phase 1: Random Probing
        probes = self._phase1(n_probes, budget_p1)

        # Phase 2: 2D Slices (budget rolls over from Phase 1)
        budget_p2_effective = budget_p1 + budget_p2
        slices = self._phase2(probes, n_slice_pairs, slice_resolution,
                              budget_p2_effective)

        # Phase 3: Cliff Anatomy (remainder)
        anatomy = self._phase3(probes, n_anatomy_radii, budget)

        # Compute aggregate stats
        fitness_values = [p['fitness'] for p in probes]
        cliff_values = [p['cliffiness'] for p in probes]
        grad_mags = [p['gradient_magnitude'] for p in probes]

        if fitness_values:
            fitness_arr = np.array(fitness_values)
            stats = {
                'fitness_mean': float(np.mean(fitness_arr)),
                'fitness_std': float(np.std(fitness_arr)),
                'fitness_min': float(np.min(fitness_arr)),
                'fitness_max': float(np.max(fitness_arr)),
                'roughness': roughness_ratio(fitness_arr),
                'mean_cliffiness': float(np.mean(cliff_values)),
                'max_cliffiness': float(np.max(cliff_values)),
                'mean_gradient_magnitude': float(np.mean(grad_mags)),
            }
        else:
            stats = {
                'fitness_mean': 0.0, 'fitness_std': 0.0,
                'fitness_min': 0.0, 'fitness_max': 0.0,
                'roughness': 0.0, 'mean_cliffiness': 0.0,
                'max_cliffiness': 0.0, 'mean_gradient_magnitude': 0.0,
            }

        self._atlas = {
            'meta': {
                'param_names': names,
                'n_dims': len(names),
                'budget_used': self._evals_used,
                'seed': self.seed,
            },
            'probes': probes,
            'stats': stats,
            'slices': slices,
            'anatomy': anatomy,
        }
        return self._atlas

    def save(self, path: str):
        """Save atlas to JSON file.

        Args:
            path: output file path (should end in .json).
        """
        if self._atlas is None:
            raise RuntimeError("No atlas built yet. Call build() first.")
        with open(path, 'w') as f:
            json.dump(self._atlas, f, cls=_NumpyEncoder, indent=2)

    @staticmethod
    def load(path: str) -> dict:
        """Load atlas from JSON file.

        Args:
            path: path to atlas JSON file.

        Returns:
            dict with keys: meta, probes, stats, slices, anatomy.
        """
        with open(path) as f:
            return json.load(f)

    def visualize(self, output_dir: str = '.'):
        """Generate 5 matplotlib PNG plots from the atlas.

        Plots:
        1. atlas_fitness_distribution.png — histogram of Phase 1 fitness
        2. atlas_cliffiness_vs_fitness.png — scatter plot
        3. atlas_slice_heatmaps.png — one subplot per 2D slice
        4. atlas_cliff_anatomy.png — fitness vs radius along gradient
        5. atlas_parameter_sensitivity.png — mean |gradient| per param

        Args:
            output_dir: directory for output PNGs. Default '.'.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if self._atlas is None:
            raise RuntimeError("No atlas built yet. Call build() first.")

        atlas = self._atlas
        probes = atlas['probes']
        slices = atlas['slices']
        anatomy = atlas['anatomy']
        names = atlas['meta']['param_names']

        # 1. Fitness distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        fitness_vals = [p['fitness'] for p in probes]
        ax.hist(fitness_vals, bins=30, color='steelblue', edgecolor='white')
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Count')
        ax.set_title('Phase 1: Fitness Distribution')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/atlas_fitness_distribution.png', dpi=150)
        plt.close(fig)

        # 2. Cliffiness vs fitness
        fig, ax = plt.subplots(figsize=(8, 5))
        cliff_vals = [p['cliffiness'] for p in probes]
        ax.scatter(cliff_vals, fitness_vals, alpha=0.6, c='steelblue',
                   edgecolors='white', linewidth=0.5)
        ax.set_xlabel('Cliffiness')
        ax.set_ylabel('Fitness')
        ax.set_title('Phase 1: Cliffiness vs Fitness')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/atlas_cliffiness_vs_fitness.png', dpi=150)
        plt.close(fig)

        # 3. 2D slice heatmaps
        n_slices = len(slices)
        if n_slices > 0:
            fig, axes = plt.subplots(1, n_slices,
                                     figsize=(6 * n_slices, 5))
            if n_slices == 1:
                axes = [axes]
            for idx, s in enumerate(slices):
                ax = axes[idx]
                grid = np.array(s['fitness_grid'])
                im = ax.pcolormesh(s['x_vals'], s['y_vals'], grid,
                                   cmap='viridis', shading='auto')
                fig.colorbar(im, ax=ax, label='Fitness')
                ax.plot(s['center_pos'][0], s['center_pos'][1],
                        'ro', markersize=8)
                ax.set_xlabel(s['param_x'])
                ax.set_ylabel(s['param_y'])
                ax.set_title(f"{s['param_x']} vs {s['param_y']}")
            fig.suptitle('Phase 2: 2D Slice Heatmaps')
            fig.tight_layout()
            fig.savefig(f'{output_dir}/atlas_slice_heatmaps.png', dpi=150)
            plt.close(fig)
        else:
            # Create placeholder
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, 'No slices generated',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Phase 2: 2D Slice Heatmaps')
            fig.savefig(f'{output_dir}/atlas_slice_heatmaps.png', dpi=150)
            plt.close(fig)

        # 4. Cliff anatomy profile
        fig, ax = plt.subplots(figsize=(8, 5))
        if anatomy['radii']:
            ax.plot(anatomy['radii'], anatomy['fitness_profile'],
                    'o-', color='steelblue', linewidth=2, markersize=4)
            ax.set_xlabel('Radius along gradient direction')
            ax.set_ylabel('Fitness')
            ax.text(0.02, 0.98,
                    f"roughness={anatomy['roughness']:.3f}\n"
                    f"sign_flip_rate={anatomy['sign_flip_rate']:.3f}",
                    transform=ax.transAxes, va='top', fontsize=9,
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat',
                              alpha=0.5))
        ax.set_title('Phase 3: Cliff Anatomy Profile')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/atlas_cliff_anatomy.png', dpi=150)
        plt.close(fig)

        # 5. Parameter sensitivity
        fig, ax = plt.subplots(figsize=(8, 5))
        mean_grads = {}
        for name in names:
            grads = [abs(p['gradient'].get(name, 0.0)) for p in probes]
            mean_grads[name] = float(np.mean(grads)) if grads else 0.0
        sorted_names = sorted(mean_grads, key=mean_grads.get, reverse=True)
        bars = [mean_grads[n] for n in sorted_names]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_names)))
        ax.barh(sorted_names, bars, color=colors)
        ax.set_xlabel('Mean |gradient|')
        ax.set_title('Parameter Sensitivity')
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(f'{output_dir}/atlas_parameter_sensitivity.png', dpi=150)
        plt.close(fig)
