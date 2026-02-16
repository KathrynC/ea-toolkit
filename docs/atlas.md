# atlas

Multi-phase fitness landscape atlas builder.

---

## Overview

The `AtlasBuilder` composes the low-level tools from `landscape.py` (`probe_cliffiness`, `gradient_estimate`, `roughness_ratio`, `sign_flip_rate`) into a structured, 3-phase atlas that maps the full parameter space. Extracted from the `atlas_cliffiness.py` pattern in the Evolutionary-Robotics project, but generalized to work with any `FitnessFunction`.

Output is a JSON-serializable dict compatible with viz-tools (D3.js Atlas Explorer, P5.js Atlas Terrain).

---

## Class

### `AtlasBuilder(fitness_fn, seed=None)`

| Argument | Type | Description |
|----------|------|-------------|
| `fitness_fn` | `FitnessFunction` | Fitness function to analyze |
| `seed` | `int \| None` | Random seed for reproducibility |

---

## Methods

### `build(n_probes=100, n_slice_pairs=3, slice_resolution=30, n_anatomy_radii=20, budget=5000) -> dict`

Build the atlas in three phases.

| Argument | Default | Description |
|----------|---------|-------------|
| `n_probes` | 100 | Number of random probe points (Phase 1) |
| `n_slice_pairs` | 3 | Number of 2D parameter pair slices (Phase 2) |
| `slice_resolution` | 30 | Grid resolution per slice axis (Phase 2) |
| `n_anatomy_radii` | 20 | Radii along cliff transect (Phase 3) |
| `budget` | 5000 | Maximum total fitness evaluations |

**Budget allocation:** 40% Phase 1, 45% Phase 2, 15% Phase 3. Unspent budget rolls forward.

**Returns:** dict with keys `meta`, `probes`, `stats`, `slices`, `anatomy`.

---

### `save(path)`

Save atlas to JSON file. Handles numpy type serialization.

### `AtlasBuilder.load(path)` (static)

Load atlas dict from JSON file.

### `visualize(output_dir='.')`

Generate 5 matplotlib PNGs (Agg backend):

| File | Content |
|------|---------|
| `atlas_fitness_distribution.png` | Histogram of Phase 1 fitness values |
| `atlas_cliffiness_vs_fitness.png` | Scatter: cliffiness (x) vs fitness (y) |
| `atlas_slice_heatmaps.png` | Pcolormesh for each 2D slice |
| `atlas_cliff_anatomy.png` | Line plot: fitness vs radius along gradient |
| `atlas_parameter_sensitivity.png` | Horizontal bars: mean |gradient| per param |

---

## Phases

### Phase 1: Random Probing

Sample `n_probes` random points. At each point, compute:
- Fitness (1 eval)
- Gradient via central finite differences (2 × n_dim evals)
- Cliffiness via random perturbation (min(6, n_dim) evals)

**Output:** list of `{params, fitness, gradient, gradient_magnitude, cliffiness}`.

### Phase 2: 2D Slices

Rank parameter pairs by combined gradient magnitude from Phase 1. For the top `n_slice_pairs` pairs, generate a `slice_resolution × slice_resolution` grid holding other parameters at the best point from Phase 1. Compute a cliffiness grid from adjacent-cell differences (no extra evaluations).

**Output:** list of `{param_x, param_y, x_vals, y_vals, fitness_grid, cliffiness_grid, center_pos}`.

### Phase 3: Cliff Anatomy

Take the highest-cliffiness point from Phase 1. Sample fitness at `n_anatomy_radii` equally-spaced radii along the normalized gradient direction. Compute roughness and sign-flip rate along the transect.

**Output:** `{center, direction, radii, fitness_profile, roughness, sign_flip_rate}`.

---

## JSON Output Format

```json
{
  "meta": {"param_names": [...], "n_dims": 5, "budget_used": 4800, "seed": 42},
  "probes": [{"params": {...}, "fitness": -2.3, "gradient": {...},
              "gradient_magnitude": 1.4, "cliffiness": 0.8}, ...],
  "stats": {"fitness_mean": ..., "fitness_std": ..., "roughness": ...,
            "mean_cliffiness": ..., ...},
  "slices": [{"param_x": "x0", "param_y": "x1", "x_vals": [...],
              "y_vals": [...], "fitness_grid": [[...]], "cliffiness_grid": [[...]],
              "center_pos": [cx, cy]}, ...],
  "anatomy": {"center": {...}, "direction": {...}, "radii": [...],
              "fitness_profile": [...], "roughness": 0.3, "sign_flip_rate": 0.15}
}
```

---

## Example

```python
from ea_toolkit import AtlasBuilder, RastriginFitness

ab = AtlasBuilder(RastriginFitness(n_dims=5), seed=42)
atlas = ab.build(n_probes=50, budget=3000)

# Save for viz-tools
ab.save("atlas.json")

# Generate plots
ab.visualize(output_dir="output/")

# Load later
atlas = AtlasBuilder.load("atlas.json")
print(f"Budget used: {atlas['meta']['budget_used']}")
print(f"Mean cliffiness: {atlas['stats']['mean_cliffiness']:.3f}")
```

---

## Dependencies

- `landscape.py` — `probe_cliffiness`, `gradient_estimate`, `roughness_ratio`, `sign_flip_rate`, `_params_to_vec`, `_vec_to_params`, `_clamp_vec`
- `base.py` — `FitnessFunction` protocol
- `json`, `numpy` — serialization and numerics
- `matplotlib` (optional, for `visualize()`)
