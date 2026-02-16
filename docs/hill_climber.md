# HillClimber

Parallel hill climber with configurable restarts.

---

## Overview

Greedy local search that accepts mutations only when they strictly improve fitness. Supports multiple restarts to escape local optima — the evaluation budget is divided evenly among restarts, and the overall best across all restarts is tracked.

Extracted from `walker_competition.py` `run_hill_climber()`.

---

## Constructor

```python
HillClimber(fitness_fn: FitnessFunction,
            mutation: MutationOperator | None = None,
            n_restarts: int = 1,
            seed: int | None = None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fitness_fn` | required | Fitness function to optimize |
| `mutation` | `GaussianMutation(sigma=0.1)` | Mutation operator |
| `n_restarts` | 1 | Number of independent restarts |
| `seed` | None | Random seed |

---

## Algorithm

For each restart:
1. Sample a random starting point within bounds
2. Evaluate fitness
3. Loop: mutate → evaluate → accept only if strictly better

Budget is divided as `budget // n_restarts` per restart.

---

## Usage

```python
from ea_toolkit import HillClimber, GaussianMutation

# Single run
hc = HillClimber(my_fitness, seed=42)
history = hc.run(budget=1000)

# With restarts and larger step size
hc = HillClimber(my_fitness,
                  mutation=GaussianMutation(sigma=0.5),
                  n_restarts=5, seed=42)
history = hc.run(budget=1000)  # 200 evals per restart
best = hc.best()
```

---

## Trade-offs

| Restarts | Evals/restart | Exploration | Exploitation |
|----------|---------------|-------------|-------------|
| 1 | Full budget | Low (one basin) | High |
| 5 | Budget/5 | Higher (5 basins) | Lower per basin |
| 20 | Budget/20 | Highest | Minimal per basin |

For smooth unimodal landscapes, 1 restart suffices. For multimodal landscapes (Rastrigin), more restarts with larger sigma is better.

---

## Source

`ea_toolkit/algorithms/hill_climber.py` — 95 lines.
