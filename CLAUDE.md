# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Standalone evolutionary algorithms toolkit extracted from the [Evolutionary-Robotics](../pybullet_test/Evolutionary-Robotics/) and [how-to-live-much-longer](../how-to-live-much-longer/) projects. Provides abstract base classes, mutation operators, selection strategies, population management, 6 optimization algorithms, landscape analysis tools, and telemetry logging.

All numerical operations use **numpy only** (no scipy, no sklearn).

## Commands

```bash
# Run the full test suite (47 tests)
pytest tests/ -v

# Run algorithm tests only
pytest tests/test_algorithms.py -v

# Run landscape analysis tests only
pytest tests/test_landscape.py -v

# Run a single test
pytest tests/test_algorithms.py::TestHillClimber::test_converges_on_sphere -v
```

## Architecture

### Dependency Graph

```
base.py                 ← Abstract base classes (no dependencies except numpy)
    ↓
mutation.py             ← 3 mutation operators (imports base)
selection.py            ← 3 selection strategies (imports base)
population.py           ← Population management (numpy only)
landscape.py            ← Landscape analysis tools (imports base)
telemetry.py            ← JSON-lines logging (stdlib only)
    ↓
algorithms/
    hill_climber.py     ← HillClimber (imports base, mutation)
    es.py               ← OnePlusLambdaES (imports base, mutation)
    ridge_walker.py     ← RidgeWalker (imports base, mutation)
    cliff_mapper.py     ← CliffMapper (imports base, mutation)
    novelty_seeker.py   ← NoveltySeeker (imports base, mutation)
    ensemble.py         ← EnsembleExplorer (imports base, mutation)
```

### Protocol-Based Design

All components implement abstract base classes from `base.py`:

| ABC | Method | Contract |
|-----|--------|----------|
| `FitnessFunction` | `evaluate(params) -> dict` | Must return dict with `'fitness'` key |
| `FitnessFunction` | `param_spec() -> dict` | Returns `{name: (low, high)}` bounds |
| `MutationOperator` | `mutate(params, param_spec, rng) -> dict` | Returns mutated params, clamped to bounds |
| `SelectionStrategy` | `select(population, n) -> list` | Returns n selected individuals |
| `Algorithm` | `run(budget) -> list[dict]` | Returns evaluation history within budget |

### Individuals and History

Individuals are plain dicts with at least `'params'` and `'fitness'` keys. Algorithm history entries merge params with evaluation results. All algorithms share `best()` and `_record()` from the `Algorithm` base class.

## Key Patterns

### Implementing a fitness function

```python
from ea_toolkit.base import FitnessFunction

class MyFitness(FitnessFunction):
    def evaluate(self, params: dict) -> dict:
        x = params['x']
        return {'fitness': -x**2, 'extra_metric': abs(x)}

    def param_spec(self) -> dict:
        return {'x': (-10.0, 10.0)}
```

### Running an algorithm

```python
from ea_toolkit import HillClimber, GaussianMutation

hc = HillClimber(MyFitness(), mutation=GaussianMutation(sigma=0.1), seed=42)
history = hc.run(budget=500)
best = hc.best()
```

### Using telemetry

```python
from ea_toolkit import Telemetry, load_telemetry

tel = Telemetry("run_001.jsonl")
tel.start()
for gen in range(100):
    tel.log_generation(gen, best_fitness=f, pop_size=n)
tel.finish()

entries = load_telemetry("run_001.jsonl")
```

## Conventions

- All parameter vectors are plain `dict[str, float]` — no numpy arrays in the API surface
- Internal vector operations use sorted key ordering for consistency
- All randomness flows through `np.random.Generator` instances for reproducibility
- Fitness is always maximized — negate for minimization problems
- Budget means total fitness evaluations, not generations
- Bounds clamping is applied inside mutation operators, not by algorithms
- Telemetry uses JSON-lines (.jsonl) format with numpy type serialization

## Relationship to Parent Projects

| Component | ER Project Source | This Toolkit |
|-----------|-------------------|--------------|
| Hill climbing | `walker_competition.py` | `HillClimber` |
| (1+lambda) ES | `temporal_optimizer.py` | `OnePlusLambdaES` |
| Ridge walking | `walker_competition.py` | `RidgeWalker` |
| Cliff mapping | `walker_competition.py` + `atlas_cliffiness.py` | `CliffMapper` + `landscape.py` |
| Novelty search | `walker_competition.py` | `NoveltySeeker` |
| Ensemble exploration | `walker_competition.py` | `EnsembleExplorer` |
| Mutation operators | `walker_competition.py` perturb() | `mutation.py` |
| Landscape analysis | `atlas_cliffiness.py` | `landscape.py` |

The ER project uses this toolkit via `search_v2.py` (adapter layer) and `walker_competition.py` (benchmark harness).
