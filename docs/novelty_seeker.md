# NoveltySeeker

Novelty-driven search — maximize behavioral diversity, not fitness.

---

## Overview

At each step, generates `n_candidates` mutations, extracts behavioral descriptors, normalizes them together with the archive, and moves to the candidate with the highest k-nearest-neighbor novelty score — regardless of its fitness. The entire behavioral archive grows over time, providing an increasingly refined novelty landscape.

This implements the novelty search paradigm (Lehman & Stanley, 2011): the only selection pressure is to do something *different* from what's been done before. Fitness is recorded but not optimized.

Extracted from `walker_competition.py` `run_novelty_seeker()`.

---

## Constructor

```python
NoveltySeeker(fitness_fn: FitnessFunction,
              mutation: MutationOperator | None = None,
              behavior_fn: callable | None = None,
              n_candidates: int = 5,
              k_nearest: int = 15,
              seed: int | None = None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fitness_fn` | required | Fitness function (evaluates, but fitness not used for selection) |
| `mutation` | `GaussianMutation(sigma=0.2)` | Mutation operator (larger default sigma for exploration) |
| `behavior_fn` | All numeric result values | Extracts behavioral descriptor from result dict |
| `n_candidates` | 5 | Candidates per step |
| `k_nearest` | 15 | k for k-NN novelty score |
| `seed` | None | Random seed |

---

## Algorithm

1. Initialize at random point, evaluate, add behavior to archive
2. Each step:
   a. Generate `n_candidates` mutations
   b. Extract behavioral descriptors for all candidates
   c. Min-max normalize archive + candidates together
   d. Compute k-NN novelty score for each candidate against the normalized archive
   e. Move to the most novel candidate
   f. Add all candidate behaviors to archive
3. Repeat until budget exhausted

**Novelty score:** Mean Euclidean distance to k nearest neighbors in the normalized behavioral archive. Higher = more novel.

---

## Behavior Functions

The `behavior_fn` maps an evaluation result dict to a numpy array. The default extracts all numeric values:

```python
# Default: all numeric values from result
def default_behavior_fn(result):
    return np.array([v for v in sorted_values if isinstance(v, (int, float))])

# Custom: use specific metrics
def my_behavior_fn(result):
    return np.array([result['gait_speed'], result['stability']])
```

---

## Usage

```python
from ea_toolkit import NoveltySeeker

# Basic novelty search
ns = NoveltySeeker(my_fitness, n_candidates=5, k_nearest=10, seed=42)
ns.run(budget=1000)

# Novelty search with custom behavior descriptor
ns = NoveltySeeker(my_fitness,
                    behavior_fn=lambda r: np.array([r['speed'], r['height']]),
                    n_candidates=10, k_nearest=15, seed=42)
ns.run(budget=2000)

# Fitness is recorded but not optimized
best_fitness = ns.best()  # Best fitness found (by coincidence, not selection)
```

---

## Internal Helpers

| Function | Description |
|----------|-------------|
| `_default_behavior_fn(result)` | Extract all numeric values from result dict |
| `_normalize_behavioral_vecs(vecs)` | Min-max normalize to [0,1] per dimension |
| `_knn_novelty(bvec, archive, k)` | Mean distance to k nearest neighbors |

---

## Source

`ea_toolkit/algorithms/novelty_seeker.py` — 185 lines.
