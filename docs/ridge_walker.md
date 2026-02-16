# RidgeWalker

Multi-objective Pareto search along fitness ridges.

---

## Overview

At each step, generates `n_candidates` mutations, filters to those not dominated by the current point, and moves to the one farthest in objective space. This encourages exploration along the Pareto front rather than stagnation at a single non-dominated point.

Provides a `pareto_front()` method to extract the non-dominated set from the full history.

Extracted from `walker_competition.py` `run_ridge_walker()`.

---

## Constructor

```python
RidgeWalker(fitness_fn: FitnessFunction,
            mutation: MutationOperator | None = None,
            objectives: tuple[str, ...] = ('fitness',),
            n_candidates: int = 3,
            seed: int | None = None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fitness_fn` | required | Must return dict with keys matching `objectives` |
| `mutation` | `GaussianMutation(sigma=0.1)` | Mutation operator |
| `objectives` | `('fitness',)` | Objective names (all maximized) |
| `n_candidates` | 3 | Candidates per step |
| `seed` | None | Random seed |

---

## Algorithm

1. Initialize at random point, evaluate
2. Each step:
   a. Generate `n_candidates` mutations
   b. Filter to non-dominated candidates (not dominated by current)
   c. Among non-dominated: pick the one farthest from current in objective space
   d. Move to that candidate
3. Repeat until budget exhausted

**Dominance test:** b dominates a iff `b >= a` on all objectives and `b > a` on at least one.

---

## Key Method

### `pareto_front() -> list[dict]`

Extract the non-dominated set from the full evaluation history. Returns entries where no other entry in the history dominates them.

```python
rw = RidgeWalker(my_fitness, objectives=('fitness', 'diversity'))
rw.run(budget=500)
front = rw.pareto_front()
```

---

## Usage

```python
from ea_toolkit import RidgeWalker

# Single-objective (degenerates to hill climbing with diversification)
rw = RidgeWalker(my_fitness, seed=42)
rw.run(budget=500)

# Multi-objective
class BiObjective(FitnessFunction):
    def evaluate(self, params):
        return {'fitness': -params['x']**2,
                'diversity': abs(params['x'] - params['y'])}
    def param_spec(self):
        return {'x': (-5, 5), 'y': (-5, 5)}

rw = RidgeWalker(BiObjective(),
                  objectives=('fitness', 'diversity'),
                  n_candidates=5, seed=42)
rw.run(budget=1000)
front = rw.pareto_front()
```

---

## Source

`ea_toolkit/algorithms/ridge_walker.py` — 151 lines.
