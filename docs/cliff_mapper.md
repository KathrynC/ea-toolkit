# CliffMapper

High-sensitivity region search — deliberately seeks "cliffs" in the fitness landscape.

---

## Overview

At each step, probes `n_probes` random directions at a small radius, measures the absolute fitness change in each direction, and moves toward the direction with the largest |delta fitness|. This deliberately walks toward regions where small parameter changes produce large fitness changes — the "cliffs" that are often the most interesting features of a landscape.

Provides a `cliff_map()` method returning the (params, cliffiness) trajectory.

Extracted from `walker_competition.py` `run_cliff_mapper()`.

---

## Constructor

```python
CliffMapper(fitness_fn: FitnessFunction,
            mutation: MutationOperator | None = None,
            n_probes: int = 10,
            probe_radius: float = 0.05,
            seed: int | None = None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fitness_fn` | required | Fitness function to probe |
| `mutation` | `GaussianMutation(sigma=probe_radius)` | Mutation operator |
| `n_probes` | 10 | Probe directions per step |
| `probe_radius` | 0.05 | Perturbation radius |
| `seed` | None | Random seed |

---

## Algorithm

1. Initialize at random point, evaluate
2. Each step:
   a. Generate `n_probes` random unit directions
   b. Perturb current point by `probe_radius` in each direction
   c. Evaluate all probes
   d. Record cliffiness = max |delta fitness| across probes
   e. Move to the probe with largest |delta fitness|
3. Repeat until budget exhausted

**Steps per budget:** approximately `budget / n_probes` (each step consumes `n_probes` evaluations).

---

## Key Method

### `cliff_map() -> list[tuple[dict, float]]`

Return the cliff map: list of (params, cliffiness) from the search trajectory. The first entry has cliffiness 0.0 (the random starting point).

```python
cm = CliffMapper(my_fitness, n_probes=10, seed=42)
cm.run(budget=500)
for params, cliffiness in cm.cliff_map():
    print(f"cliffiness={cliffiness:.4f}")
```

---

## Usage

```python
from ea_toolkit import CliffMapper

# Find cliff regions in a landscape
cm = CliffMapper(my_fitness, n_probes=10, probe_radius=0.05, seed=42)
cm.run(budget=500)

cliff_trajectory = cm.cliff_map()
max_cliff = max(c for _, c in cliff_trajectory)
cliff_location = max(cliff_trajectory, key=lambda x: x[1])[0]
```

---

## Relationship to landscape.probe_cliffiness

`CliffMapper` is an algorithm that *walks toward* cliffs, building a trajectory of increasingly cliff-like regions. `probe_cliffiness()` in `landscape.py` is a single-point measurement. CliffMapper uses the same directional probing internally but chains probes into a directed walk.

---

## Source

`ea_toolkit/algorithms/cliff_mapper.py` — 130 lines.
