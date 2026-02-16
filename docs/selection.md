# selection

Three selection strategies for choosing individuals from a population.

---

## Overview

Implements 3 selection strategies, each implementing the `SelectionStrategy` protocol: `select(population, n) -> list`. Population individuals are dicts with at least a `'fitness'` key.

---

## Strategies

### `TournamentSelection`

Randomly sample k individuals, keep the best. Repeat n times (with replacement) to produce n selected individuals. Higher tournament sizes increase selection pressure.

```python
TournamentSelection(tournament_size: int = 3, seed: int | None = None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tournament_size` | 3 | Individuals per tournament |
| `seed` | None | Random seed |

**Selection pressure:** k=2 is mild (binary tournament), k=pop_size always selects the best. Tournament size is clamped to `min(k, pop_size)`.

---

### `TruncationSelection`

Sort the population by fitness (descending), keep the top fraction. If n exceeds the truncated pool, individuals are recycled from the pool.

```python
TruncationSelection(fraction: float = 0.5)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fraction` | 0.5 | Fraction of population to keep (0, 1] |

**Behavior:** Deterministic — always returns the same individuals for the same population. Strong selection pressure at low fractions.

---

### `EpsilonGreedy`

With probability epsilon, pick a random individual (explore). Otherwise, pick the individual with the highest fitness (exploit).

```python
EpsilonGreedy(epsilon: float = 0.1, seed: int | None = None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon` | 0.1 | Exploration probability |
| `seed` | None | Random seed |

**Behavior:** At epsilon=0, always exploits (pure greedy). At epsilon=1, always explores (pure random). The default 0.1 gives 10% exploration.

---

## Source

`ea_toolkit/selection.py` — 148 lines.
