# population

Population management utilities.

---

## Overview

Provides a `PopulationManager` class for tracking individuals, plus standalone functions for population generation, elitism, and diversity measurement. Individuals are plain dicts with `'params'` and `'fitness'` keys.

---

## Class

### `PopulationManager`

Tracks a collection of individuals with params and fitness.

```python
manager = PopulationManager()
manager.add(params={'x': 1.0}, fitness=-1.0, extra='metadata')
top = manager.best(n=3)
manager.size()     # Current count
manager.clear()    # Remove all
manager.replace(new_list)  # Replace entire population
```

| Method | Returns | Description |
|--------|---------|-------------|
| `add(params, fitness, **extra)` | `dict` | Add individual, returns the created entry |
| `best(n=1)` | `list[dict]` | Top n by fitness, sorted descending |
| `size()` | `int` | Current population size |
| `clear()` | `None` | Remove all individuals |
| `replace(new_individuals)` | `None` | Replace entire population |

---

## Functions

### `random_population(n, param_spec, rng) -> list[dict]`

Generate n random individuals with parameters uniformly sampled within bounds.

| Argument | Type | Description |
|----------|------|-------------|
| `n` | `int` | Number of individuals |
| `param_spec` | `dict` | `{name: (low, high)}` bounds |
| `rng` | `np.random.Generator` | Random generator |

Returns list of dicts, each with a `'params'` key (no fitness yet — must be evaluated).

---

### `elitism(population, n_elite) -> list[dict]`

Keep the top `n_elite` individuals from a population, sorted by fitness descending.

---

### `diversity_metric(population, param_spec) -> float`

Mean pairwise normalized Euclidean distance. Parameters are normalized to [0, 1] using their bounds so all dimensions contribute equally.

Returns 0.0 for populations with fewer than 2 individuals.

---

### `behavioral_diversity(population, behavior_key='behavior') -> float`

Mean pairwise distance between behavioral descriptors (after min-max normalization). Requires individuals to have a `behavior_key` entry containing a list or array of floats.

Returns 0.0 if fewer than 2 individuals have behavioral descriptors.

---

## Source

`ea_toolkit/population.py` — 213 lines.
