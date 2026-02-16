# callbacks

Event-driven callback system for monitoring and controlling algorithm runs.

---

## Overview

The callback system lets you plug monitoring, logging, and early-stopping logic into any algorithm without modifying the algorithm itself. Callbacks are attached to an algorithm via its `callbacks` list attribute and receive notifications at key points during optimization.

All callbacks extend the `Callback` base class from `base.py`.

---

## Callback Protocol

The `Callback` base class defines 4 event hooks:

```python
class Callback:
    def on_start(self, algorithm):
        """Called before the first generation."""
        pass

    def on_generation(self, algorithm, generation, best_fitness):
        """Called after each generation. Return False to request early stopping."""
        pass

    def on_improvement(self, algorithm, old_fitness, new_fitness):
        """Called when a new best individual is found."""
        pass

    def on_finish(self, algorithm):
        """Called after the last generation."""
        pass
```

**Early stopping:** If any callback's `on_generation` returns `False`, the algorithm stops after that generation. All other return values (including `None`) are ignored.

**Multiple callbacks:** Algorithms check all callbacks in order. If any returns `False`, execution stops.

---

## Built-in Callbacks

### `ConvergenceChecker`

Stops optimization when fitness plateaus — no improvement of at least `min_delta` for `patience` consecutive generations.

```python
ConvergenceChecker(patience: int = 20, min_delta: float = 1e-8)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patience` | 20 | Generations without improvement before stopping |
| `min_delta` | 1e-8 | Minimum fitness change to count as improvement |

**Behavior:**
- Tracks the best fitness seen so far
- Increments a stale counter each generation where improvement < min_delta
- Resets counter when improvement ≥ min_delta
- Returns `False` from `on_generation` when stale count ≥ patience
- Resets fully on `on_start` (safe for multiple runs)

```python
cc = ConvergenceChecker(patience=50, min_delta=0.001)
algo.callbacks = [cc]
algo.run(budget=10000)  # Stops early if fitness plateaus
```

---

### `ProgressPrinter`

Prints generation number, best fitness, and evaluation count at regular intervals.

```python
ProgressPrinter(every: int = 10, prefix: str = "")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `every` | 10 | Print every N generations |
| `prefix` | `""` | String prefix for each line |

**Output format:**
```
[prefix]gen 10 | best 0.1234 | evals 70
[prefix]gen 20 | best 0.0567 | evals 140
```

---

### `TelemetryCallback`

Bridges the callback system with the `Telemetry` logger. Automatically logs generation events and improvements to a JSONL file.

```python
TelemetryCallback(telemetry: Telemetry)
```

Calls `telemetry.start()` on `on_start`, logs each generation and improvement as telemetry events, and calls `telemetry.finish()` on `on_finish`.

---

### `HistoryRecorder`

Records per-generation statistics in a list for later analysis.

```python
HistoryRecorder()
```

After a run, `recorder.generations` contains:

```python
[
    {'generation': 0, 'best_fitness': -5.2, 'n_evals': 10, 'improved': True},
    {'generation': 1, 'best_fitness': -3.1, 'n_evals': 20, 'improved': True},
    {'generation': 2, 'best_fitness': -3.1, 'n_evals': 30, 'improved': False},
    ...
]
```

---

## Usage

### Attaching callbacks

```python
from ea_toolkit import DifferentialEvolution, CMAES
from ea_toolkit.callbacks import ConvergenceChecker, HistoryRecorder

hr = HistoryRecorder()
cc = ConvergenceChecker(patience=30)
de = DifferentialEvolution(my_fitness, pop_size=50, seed=42)
de.callbacks = [hr, cc]
de.run(budget=10000)

# Inspect history
print(f"Ran {len(hr.generations)} generations")
print(f"Best: {hr.generations[-1]['best_fitness']}")
```

### Early stopping with impossible threshold

```python
# Force early stop to test callback integration
cc = ConvergenceChecker(patience=5, min_delta=1000.0)  # Impossible threshold
algo.callbacks = [cc]
history = algo.run(budget=100000)
# Will stop after ~5 generations
```

---

## Tests

7 tests in `test_callbacks.py`:

| Test | What it verifies |
|------|-----------------|
| `test_callback_has_all_methods` | Base class has all 4 hooks |
| `test_stops_when_stale` | ConvergenceChecker returns False after patience |
| `test_resets_on_improvement` | Counter resets on fitness improvement |
| `test_on_start_resets` | on_start clears state for reuse |
| `test_records_generations` | HistoryRecorder tracks generation data |
| `test_callbacks_on_algorithm` | Callbacks work when attached to DE |
| `test_convergence_early_stop` | Early stopping works end-to-end |
