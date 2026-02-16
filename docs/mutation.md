# mutation

Three mutation operators for perturbing parameter vectors.

---

## Overview

Implements 3 mutation strategies extracted from `walker_competition.py` (Gaussian, Cauchy) and `temporal_optimizer.py` (Adaptive). All operators implement the `MutationOperator` protocol: `mutate(params, param_spec, rng) -> dict`.

Internal helpers convert between `dict` and `np.ndarray` with consistent sorted-key ordering and handle bounds clamping.

---

## Operators

### `GaussianMutation`

Perturb along a random unit vector on the N-dimensional unit sphere with step size drawn from N(0, sigma). Produces isotropic perturbations — direction is uniform, magnitude is controlled by sigma.

Extracted from `walker_competition.py` `perturb()` pattern.

```python
GaussianMutation(sigma: float = 0.1)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma` | 0.1 | Standard deviation of the step size |

**Algorithm:**
1. Generate random direction on unit sphere via normalized Gaussian samples
2. Draw step from N(0, sigma)
3. Move: `new = current + step * direction`
4. Clamp to bounds

---

### `CauchyMutation`

Heavy-tailed Cauchy distribution applied independently per dimension, scaled by parameter range. Most mutations are small, but occasional large jumps help escape local optima in rugged landscapes.

```python
CauchyMutation(scale: float = 0.05)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale` | 0.05 | Scale factor as fraction of each parameter's range |

**Algorithm:**
1. Compute per-dimension ranges from `param_spec`
2. Generate Cauchy samples via inverse CDF: `tan(pi * (U - 0.5))`
3. Move: `new = current + scale * ranges * cauchy_samples`
4. Clamp to bounds

**Comparison with Gaussian:** Cauchy's heavy tails mean ~25% of mutations are larger than 1 sigma-equivalent, vs ~5% for Gaussian. Better for multimodal landscapes (e.g., Rastrigin).

---

### `AdaptiveMutation`

Gaussian mutation with sigma adapted by the 1/5th success rule from evolution strategy theory. If success rate > 1/5, increase sigma (explore more broadly). If < 1/5, decrease sigma (exploit locally). Tracks success history over a sliding window.

Extracted from `temporal_optimizer.py` adaptive mutation patterns.

```python
AdaptiveMutation(sigma_init: float = 0.1,
                 sigma_min: float = 0.001,
                 sigma_max: float = 1.0,
                 window: int = 20,
                 adaptation_rate: float = 1.2)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_init` | 0.1 | Initial step size |
| `sigma_min` | 0.001 | Minimum allowed sigma |
| `sigma_max` | 1.0 | Maximum allowed sigma |
| `window` | 20 | Sliding window for success rate |
| `adaptation_rate` | 1.2 | Multiplicative adjustment factor |

**Required callback:**

```python
mutation.report_success(improved: bool)
```

Must be called after each mutate + evaluate cycle. `OnePlusLambdaES` calls this automatically.

**Adaptation rule:**
- Success rate > 0.2: `sigma *= adaptation_rate` (up to sigma_max)
- Success rate < 0.2: `sigma /= adaptation_rate` (down to sigma_min)
- Adaptation triggers only after `window` observations accumulated

---

## Internal Helpers

| Function | Description |
|----------|-------------|
| `_params_to_vec(params, param_spec)` | Dict → numpy array with sorted key ordering |
| `_vec_to_params(vec, names)` | Numpy array → dict |
| `_clamp(vec, names, param_spec)` | Clamp vector to bounds |

---

## Source

`ea_toolkit/mutation.py` — 252 lines.
