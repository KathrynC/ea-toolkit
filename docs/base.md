# base

Abstract base classes for evolutionary algorithm components.

---

## Overview

Defines 6 abstract base classes that form the protocol contracts for all toolkit components. Every algorithm, mutation operator, crossover operator, selection strategy, callback, and fitness function implements one of these interfaces. Extracted and generalized from `walker_competition.py` and `temporal_optimizer.py` in the Evolutionary-Robotics project.

---

## Classes

### `FitnessFunction`

Abstract fitness function that users must implement to define their optimization problem.

```python
class FitnessFunction(ABC):
    @abstractmethod
    def evaluate(self, params: dict) -> dict: ...

    @abstractmethod
    def param_spec(self) -> dict: ...
```

| Method | Returns | Contract |
|--------|---------|----------|
| `evaluate(params)` | `dict` | Must include `'fitness'` key (float). May include additional metrics. |
| `param_spec()` | `dict` | Maps parameter names to `(low, high)` bound tuples. |

**Example:**

```python
class SphereFitness(FitnessFunction):
    def evaluate(self, params):
        values = [params[f'x{i}'] for i in range(5)]
        return {'fitness': -sum(v**2 for v in values)}

    def param_spec(self):
        return {f'x{i}': (-5.0, 5.0) for i in range(5)}
```

---

### `MutationOperator`

Abstract mutation operator for perturbing parameter vectors.

```python
class MutationOperator(ABC):
    @abstractmethod
    def mutate(self, params: dict, param_spec: dict,
               rng: np.random.Generator) -> dict: ...
```

| Argument | Type | Description |
|----------|------|-------------|
| `params` | `dict` | Current parameter values |
| `param_spec` | `dict` | Parameter bounds `{name: (low, high)}` |
| `rng` | `np.random.Generator` | Random generator for reproducibility |

Returns a new dict with mutated values clamped to bounds.

---

### `CrossoverOperator`

Abstract crossover operator for recombining two parent parameter vectors.

```python
class CrossoverOperator(ABC):
    @abstractmethod
    def crossover(self, parent1: dict, parent2: dict,
                  param_spec: dict, rng: np.random.Generator) -> tuple[dict, dict]: ...
```

| Argument | Type | Description |
|----------|------|-------------|
| `parent1` | `dict` | First parent parameter values |
| `parent2` | `dict` | Second parent parameter values |
| `param_spec` | `dict` | Parameter bounds `{name: (low, high)}` |
| `rng` | `np.random.Generator` | Random generator for reproducibility |

Returns a tuple of two child parameter dicts.

See [`crossover`](crossover.md) for SBXCrossover and UniformCrossover implementations.

---

### `SelectionStrategy`

Abstract selection strategy for choosing individuals from a population.

```python
class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population: list, n: int) -> list: ...
```

| Argument | Type | Description |
|----------|------|-------------|
| `population` | `list[dict]` | Individuals, each with at least `'fitness'` key |
| `n` | `int` | Number of individuals to select |

Returns a list of n selected individuals.

---

### `Algorithm`

Base class for all evolutionary algorithms. Provides shared infrastructure.

```python
class Algorithm(ABC):
    def __init__(self, fitness_fn, mutation=None, selection=None, seed=None): ...

    @abstractmethod
    def run(self, budget: int) -> list[dict]: ...
    def best(self) -> dict | None: ...
    def _record(self, params: dict, result: dict) -> dict: ...
```

| Method | Description |
|--------|-------------|
| `run(budget)` | Run algorithm for `budget` fitness evaluations. Returns history. |
| `best()` | Return the entry with highest fitness, or None. |
| `ask()` | Return candidates to evaluate (for ask-tell interface). Default raises NotImplementedError. |
| `tell(evaluations)` | Report (params, result) tuples. Default raises NotImplementedError. |
| `_record(params, result)` | Record evaluation in history, update best. Triggers `on_improvement` callback. |
| `_notify(event, **kwargs)` | Dispatch callback event. Returns False if any callback requests stop. |

**Constructor arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `fitness_fn` | `FitnessFunction` | required | Problem to optimize |
| `mutation` | `MutationOperator` | `None` | Optional mutation operator |
| `selection` | `SelectionStrategy` | `None` | Optional selection strategy |
| `seed` | `int` | `None` | Random seed for reproducibility |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `history` | `list[dict]` | All evaluation records |
| `callbacks` | `list[Callback]` | Attached callbacks (default empty) |

**History entry format:**

```python
{'params': {'x0': 1.23, 'x1': -0.45}, 'fitness': -1.67, ...extra_metrics...}
```

---

### `Callback`

Base class for algorithm event callbacks. Override any method to receive that event.

```python
class Callback:
    def on_start(self, algorithm): ...
    def on_generation(self, algorithm, generation, best_fitness): ...
    def on_improvement(self, algorithm, old_fitness, new_fitness): ...
    def on_finish(self, algorithm): ...
```

If `on_generation` returns `False`, the algorithm stops (early stopping).

See [`callbacks`](callbacks.md) for ConvergenceChecker, ProgressPrinter, TelemetryCallback, and HistoryRecorder.

---

## Source

`ea_toolkit/base.py`
