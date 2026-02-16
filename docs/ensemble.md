# EnsembleExplorer

Multi-walker ensemble with behavioral convergence detection and teleportation.

---

## Overview

Runs `n_walkers` parallel hill climbers. Periodically checks for convergence: if two walkers are within `teleport_threshold` in normalized parameter space, the worse one is teleported to a random location. This maintains population diversity even as individual walkers converge to local optima.

Extracted from `walker_competition.py` `run_ensemble_explorer()`.

---

## Constructor

```python
EnsembleExplorer(fitness_fn: FitnessFunction,
                 mutation: MutationOperator | None = None,
                 n_walkers: int = 20,
                 teleport_threshold: float = 0.3,
                 teleport_interval: int = 10,
                 seed: int | None = None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fitness_fn` | required | Fitness function to optimize |
| `mutation` | `GaussianMutation(sigma=0.1)` | Mutation operator |
| `n_walkers` | 20 | Number of parallel hill climbers |
| `teleport_threshold` | 0.3 | Normalized distance below which walkers are converged |
| `teleport_interval` | 10 | Check for convergence every N steps |
| `seed` | None | Random seed |

---

## Algorithm

1. Initialize `n_walkers` at random positions, evaluate all
2. Each step: each walker mutates and accepts strict improvements
3. Every `teleport_interval` steps:
   a. Normalize all walker positions to [0, 1] per dimension
   b. For each pair: if normalized distance < `teleport_threshold`, teleport the worse walker to a random position (fitness reset to -inf)
4. Repeat until budget exhausted

**Budget allocation:** First `n_walkers` evaluations initialize the walkers. Remaining budget gives `(budget - n_walkers) / n_walkers` steps per walker approximately.

---

## Convergence Detection

Walker positions are normalized using parameter bounds so all dimensions contribute equally to the distance metric. Two walkers at normalized distance < 0.3 (default) are considered converged — they're exploring the same basin. The worse walker is teleported to maintain diversity.

---

## Usage

```python
from ea_toolkit import EnsembleExplorer, GaussianMutation

# Default configuration
ee = EnsembleExplorer(my_fitness, seed=42)
history = ee.run(budget=2000)

# Aggressive diversity maintenance
ee = EnsembleExplorer(my_fitness,
                       mutation=GaussianMutation(sigma=0.3),
                       n_walkers=30,
                       teleport_threshold=0.5,
                       teleport_interval=5, seed=42)
history = ee.run(budget=5000)
best = ee.best()
```

---

## Walker Count Trade-offs

| Walkers | Init Cost | Steps (budget=1000) | Exploration | Description |
|---------|-----------|---------------------|-------------|-------------|
| 5 | 5 evals | ~199 steps per walker | Low | Few independent basins |
| 20 | 20 evals | ~49 steps per walker | High | Good basin coverage (default) |
| 50 | 50 evals | ~19 steps per walker | Highest | Many basins, minimal per-basin search |

For multimodal landscapes, more walkers with teleportation outperforms fewer walkers without.

---

## Source

`ea_toolkit/algorithms/ensemble.py` — 154 lines.
