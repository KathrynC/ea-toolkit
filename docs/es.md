# OnePlusLambdaES

(1+lambda) Evolution Strategy.

---

## Overview

Each generation produces lambda children via mutation from the single parent, evaluates all children, and keeps the best of parent-or-children. The "(1+lambda)" naming convention means 1 parent + lambda offspring, with plus-selection (parent competes with children).

If the mutation operator is an `AdaptiveMutation`, the 1/5th success rule is automatically applied — `report_success()` is called after each generation.

Extracted from `temporal_optimizer.py` `evolve()`.

---

## Constructor

```python
OnePlusLambdaES(fitness_fn: FitnessFunction,
                mutation: MutationOperator | None = None,
                lam: int = 10,
                seed: int | None = None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fitness_fn` | required | Fitness function to optimize |
| `mutation` | `GaussianMutation(sigma=0.1)` | Mutation operator |
| `lam` | 10 | Children per generation (lambda) |
| `seed` | None | Random seed |

---

## Algorithm

1. Initialize parent randomly within bounds
2. Each generation:
   a. Generate `lam` children by mutating the parent
   b. Evaluate all children
   c. If best child > parent fitness: replace parent
   d. If mutation is `AdaptiveMutation`: call `report_success(improved)`
3. Repeat until budget exhausted

**Generations = budget / (lam + 1)** approximately (first parent costs 1 eval).

---

## Usage

```python
from ea_toolkit import OnePlusLambdaES, AdaptiveMutation

# Basic usage
es = OnePlusLambdaES(my_fitness, lam=10, seed=42)
history = es.run(budget=1000)  # ~100 generations

# With adaptive sigma
es = OnePlusLambdaES(my_fitness,
                      mutation=AdaptiveMutation(sigma_init=0.5),
                      lam=20, seed=42)
history = es.run(budget=2000)  # ~100 generations, sigma auto-tunes
```

---

## Lambda Trade-offs

| Lambda | Generations (budget=1000) | Description |
|--------|--------------------------|-------------|
| 2 | ~333 | Fast iterations, noisy selection |
| 10 | ~91 | Balanced (default) |
| 50 | ~20 | Strong per-generation selection, few iterations |

Higher lambda gives more reliable per-generation improvement but fewer total generations.

---

## Source

`ea_toolkit/algorithms/es.py` — 97 lines.
