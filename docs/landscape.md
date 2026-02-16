# landscape

Fitness landscape analysis tools.

---

## Overview

Tools for characterizing the structure of a fitness landscape — how smooth, rugged, or cliff-ridden it is. Extracted from `atlas_cliffiness.py` and related landscape analysis scripts in the Evolutionary-Robotics project. The `LandscapeAnalyzer` class provides a comprehensive one-call analysis; the standalone functions are building blocks.

---

## Functions

### `probe_cliffiness(fitness_fn, params, param_spec, radius=0.1, n_directions=6, rng=None) -> float`

Probe around a point to measure maximum sensitivity. Generates `n_directions` random unit vectors, perturbs by `radius`, evaluates fitness, and returns the maximum absolute fitness change.

| Argument | Default | Description |
|----------|---------|-------------|
| `fitness_fn` | required | Fitness function to probe |
| `params` | required | Center point |
| `param_spec` | required | Parameter bounds |
| `radius` | 0.1 | Perturbation radius |
| `n_directions` | 6 | Number of probe directions |
| `rng` | None | Random generator |

**Cost:** `n_directions + 1` evaluations (1 base + n probes).

**Interpretation:** High values indicate the point is near a "cliff" — a region where small parameter changes produce large fitness changes.

---

### `roughness_ratio(fitness_values) -> float`

Ratio of local variation to global range. Local variation = mean absolute consecutive difference. Global range = max - min.

| Value | Interpretation |
|-------|---------------|
| ~0.0 | Smooth landscape (e.g., linear) |
| ~0.5 | Moderately rugged |
| ~1.0 | Extremely rugged (e.g., alternating) |

Returns 0.0 for constant data or fewer than 2 values.

---

### `sign_flip_rate(gradient_samples) -> float`

Fraction of adjacent gradient samples that change sign. High values indicate a highly non-monotonic (rugged) landscape where the gradient direction is unstable.

Returns 0.0 for fewer than 2 samples. Zeros are handled gracefully (counted as pairs but not as flips).

---

### `gradient_estimate(fitness_fn, params, param_spec, epsilon=0.01) -> dict[str, float]`

Central finite difference gradient: `df/dx_i = (f(x+e_i) - f(x-e_i)) / (2*epsilon)`.

**Cost:** `2 * n_dims` evaluations.

Handles boundary effects by using the actual achieved step size when clamped.

---

## Class

### `LandscapeAnalyzer`

Comprehensive landscape analysis that samples random points and computes aggregate statistics.

```python
analyzer = LandscapeAnalyzer(fitness_fn, seed=42)
stats = analyzer.run_analysis(n_samples=50, budget=500)
```

| Argument | Default | Description |
|----------|---------|-------------|
| `n_samples` | 50 | Desired number of sample points |
| `budget` | 500 | Maximum total fitness evaluations |

**Per-sample cost:** 1 (base) + 2*n_dim (gradient) + min(6, n_dim) (cliffiness). Samples are reduced if budget is insufficient.

**Returns dict with:**

| Key | Description |
|-----|-------------|
| `n_samples` | Actual samples taken |
| `fitness_mean`, `fitness_std` | Fitness distribution |
| `fitness_min`, `fitness_max` | Fitness range |
| `roughness` | Roughness ratio of sampled values |
| `mean_cliffiness`, `max_cliffiness` | Sensitivity statistics |
| `mean_gradient_magnitude` | Average gradient L2 norm |
| `gradient_sign_flip_rate` | Per-dimension sign instability, averaged |
| `evals_used` | Total fitness evaluations consumed |

---

## Source

`ea_toolkit/landscape.py` — 338 lines.
