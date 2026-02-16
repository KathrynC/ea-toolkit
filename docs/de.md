# DifferentialEvolution

Population-based metaheuristic using vector differences for mutation.

---

## Overview

Differential Evolution (DE) maintains a population of candidate solutions and creates trial vectors by adding scaled differences between randomly selected population members. The DE/rand/1/bin variant implemented here is the standard workhorse that works well across a wide range of problems without tuning.

Storn & Price (1997). One of the most widely used evolutionary algorithms for continuous optimization.

Supports both `run()` and the ask-tell interface natively.

---

## Constructor

```python
DifferentialEvolution(
    fitness_fn: FitnessFunction,
    pop_size: int = 50,
    F: float = 0.8,
    CR: float = 0.9,
    seed: int | None = None,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fitness_fn` | — | Fitness function to optimize |
| `pop_size` | 50 | Population size (NP). Larger = more exploration, slower convergence |
| `F` | 0.8 | Differential weight (scaling factor). Controls mutation step magnitude. Range [0, 2] |
| `CR` | 0.9 | Crossover probability. Controls how many parameters come from trial vs. target. Range [0, 1] |
| `seed` | None | Random seed for reproducibility |

---

## Algorithm: DE/rand/1/bin

Each generation, for each target vector x_i in the population:

1. **Select** 3 random distinct vectors r1, r2, r3 from the population (none equal to i)
2. **Mutate:** `v = r1 + F * (r2 - r3)` — the mutant vector
3. **Crossover (binomial):** For each parameter j:
   - If `random() < CR` or `j == j_rand`: take v_j (from mutant)
   - Else: take x_i,j (from target)
   - `j_rand` is a random index guaranteeing at least one mutant parameter
4. **Select:** If trial is better than target, replace it. Otherwise keep the target.

This is greedy selection — the population never gets worse, only stays the same or improves.

---

## Methods

### `run(budget) -> list[dict]`

Run DE for the given evaluation budget.

```python
de = DifferentialEvolution(my_fitness, pop_size=30, F=0.8, CR=0.9, seed=42)
history = de.run(budget=3000)
best = de.best()
```

### `ask() -> list[dict]`

Returns parameter dicts to evaluate:
- **First call:** Returns the initial random population (pop_size candidates)
- **Subsequent calls:** Returns pop_size trial vectors generated via DE/rand/1/bin

### `tell(evaluations)`

Accepts a list of `(params_dict, result_dict)` tuples. Performs greedy selection — each trial replaces its target if the trial's fitness is higher.

```python
de = DifferentialEvolution(my_fitness, pop_size=20, seed=42)

# Initialize
candidates = de.ask()
evals = [(c, my_fitness.evaluate(c)) for c in candidates]
de.tell(evals)

# Main loop
for _ in range(100):
    trials = de.ask()
    evals = [(t, my_fitness.evaluate(t)) for t in trials]
    de.tell(evals)
```

### `best() -> dict`

Returns the best individual found so far: `{'params': {...}, 'fitness': float}`.

---

## Tuning Guide

| F | CR | Character |
|---|-----|-----------|
| 0.5 | 0.1 | Conservative: small steps, few trial params. Good for smooth unimodal. |
| 0.8 | 0.9 | Standard: large steps, many trial params. Good general-purpose starting point. |
| 1.0 | 1.0 | Aggressive: full mutation, complete crossover. Good for escaping local optima. |

**Population size:** 5–10× the number of parameters is a common rule of thumb. Too small risks premature convergence; too large wastes budget.

---

## Comparison with Other Algorithms

| vs. | DE advantage | DE disadvantage |
|-----|-------------|-----------------|
| HillClimber | Population avoids local optima | Uses more evaluations per generation |
| OnePlusLambdaES | No mutation operator to configure | More parameters (F, CR, pop_size) |
| CMA-ES | Simpler, works in high dimensions | Doesn't learn landscape geometry |
| EnsembleExplorer | More principled recombination | No teleportation between basins |

---

## Tests

6 tests in `test_new_algorithms.py`:

| Test | What it verifies |
|------|-----------------|
| `test_converges_on_sphere` | Finds near-optimal on 5D sphere |
| `test_budget_respected` | Never exceeds evaluation budget |
| `test_history_records` | All entries have params and fitness |
| `test_ask_tell_interface` | ask/tell cycle works correctly |
| `test_ask_tell_matches_run` | Ask-tell produces comparable results to run() |
| `test_on_rastrigin` | Handles multimodal functions |
