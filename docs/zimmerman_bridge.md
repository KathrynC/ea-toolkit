# zimmerman_bridge

Bidirectional bridge between the EA toolkit and the Zimmerman toolkit.

---

## Overview

The ea-toolkit and [zimmerman-toolkit](../../zimmerman-toolkit/) share nearly identical simulator protocols but name the methods differently:

| | ea-toolkit | zimmerman-toolkit |
|---|---|---|
| Execute | `evaluate(params) → dict` | `run(params) → dict` |
| Bounds | `param_spec() → dict` | `param_spec() → dict` |
| Required key | `'fitness'` in result | None |
| Convention | Maximization | No convention |

This module provides adapters that cross-connect the two, plus convenience functions that combine evolutionary optimization with Zimmerman's black-box interrogation tools.

The zimmerman-toolkit is an **optional dependency**. The bridge module imports it lazily: adapter classes always work, convenience functions raise `ImportError` with a clear message if zimmerman is not installed.

---

## Adapters

### `FitnessAsSimulator`

Wraps an ea-toolkit `FitnessFunction` as a Zimmerman `Simulator`.

```python
from ea_toolkit.benchmarks import SphereFitness
from ea_toolkit.zimmerman_bridge import FitnessAsSimulator

sf = SphereFitness(n_dims=5)
sim = FitnessAsSimulator(sf)

# Now usable with any Zimmerman tool
from zimmerman import sobol_sensitivity
result = sobol_sensitivity(sim, n_base=128)
```

All keys returned by `evaluate()` (including `'fitness'`) are visible to Zimmerman analysis tools.

---

### `SimulatorAsFitness`

Wraps a Zimmerman `Simulator` as an ea-toolkit `FitnessFunction`.

```python
from zimmerman.base import SimulatorWrapper
from ea_toolkit.zimmerman_bridge import SimulatorAsFitness
from ea_toolkit import CMAES

def my_model(params):
    return {'cost': params['x']**2 + params['y']**2}

sim = SimulatorWrapper(my_model, {'x': (-5, 5), 'y': (-5, 5)})
fitness = SimulatorAsFitness(sim, fitness_key='cost', negate=True)

cma = CMAES(fitness, sigma0=2.0, seed=42)
cma.run(budget=1000)
print(cma.best()['fitness'])  # Near 0 (minimized cost)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `simulator` | required | Object with `run()` and `param_spec()` |
| `fitness_key` | `'fitness'` | Which output key becomes the fitness value |
| `negate` | `False` | Negate the value (for minimization → maximization) |

---

## Convenience Functions

### `sobol_on_fitness`

Run Zimmerman's Sobol sensitivity analysis on any `FitnessFunction`.

```python
from ea_toolkit.benchmarks import RastriginFitness
from ea_toolkit.zimmerman_bridge import sobol_on_fitness

rf = RastriginFitness(n_dims=5)
result = sobol_on_fitness(rf, n_base=256)

# Which parameters drive the most variance?
for param, s1 in result['fitness']['S1'].items():
    print(f"  {param}: S1={s1:.3f}, ST={result['fitness']['ST'][param]:.3f}")
```

---

### `falsify_fitness`

Run Zimmerman's 3-phase falsification on any `FitnessFunction`.

```python
from ea_toolkit.zimmerman_bridge import falsify_fitness

report = falsify_fitness(my_fitness, n_random=200)
if report['summary']['violations_found'] > 0:
    print(f"Found {len(report['violations'])} violations!")
    for v in report['violations']:
        print(f"  Strategy: {v['strategy']}, params: {v['params']}")
```

Useful for verifying numerical stability before running expensive optimization campaigns.

---

### `contrastive_around_best`

After optimization, find the smallest parameter change that flips the outcome near the best solution.

```python
from ea_toolkit import DifferentialEvolution
from ea_toolkit.zimmerman_bridge import contrastive_around_best

de = DifferentialEvolution(my_fitness, pop_size=30, seed=42)
de.run(budget=3000)

result = contrastive_around_best(
    de, my_fitness,
    outcome_fn=lambda r: 'good' if r['fitness'] > -1.0 else 'bad',
)
if result['found']:
    print(f"Fragility: {result['perturbation_magnitude']:.4f}")
    print(f"Flip caused by: {result['delta']}")
```

This reveals **fragility**: if a tiny perturbation flips the outcome, the optimizer converged to a knife-edge region — exactly the behavioral cliffs studied in the ER project.

---

### `optimize_and_interrogate`

Full pipeline: optimize with an EA, then interrogate with all Zimmerman tools.

```python
from ea_toolkit import CMAES
from ea_toolkit.benchmarks import RosenbrockFitness
from ea_toolkit.zimmerman_bridge import optimize_and_interrogate

rf = RosenbrockFitness(n_dims=5)
report = optimize_and_interrogate(
    rf, CMAES,
    algorithm_kwargs={'sigma0': 3.0},
    budget=5000,
    sobol_n_base=128,
)

print(f"Best fitness: {report['best']['fitness']:.4f}")
print(f"Violations: {report['falsification']['summary']['violations_found']}")
print(f"Most influential: {report['sobol']['rankings']['fitness_most_influential_S1']}")
if report['contrastive']['found']:
    print(f"Fragility: {report['contrastive']['perturbation_magnitude']:.4f}")
```

---

## Design Philosophy: Why Two Toolkits?

The ea-toolkit and zimmerman-toolkit address different questions about the same simulator:

| Question | Tool |
|----------|------|
| **What parameters optimize the outcome?** | ea-toolkit (HillClimber, DE, CMA-ES, ...) |
| **Which parameters drive the most variance?** | zimmerman (Sobol sensitivity) |
| **Where does the simulator break?** | zimmerman (Falsifier) |
| **How fragile is this solution?** | zimmerman (ContrastiveGenerator) |
| **Does the system do what we intend?** | zimmerman (POSIWIDAuditor) |
| **What does the landscape look like here?** | ea-toolkit (LandscapeAnalyzer, CliffMapper) |
| **Can we navigate via abstract dimensions?** | zimmerman (PDSMapper) |

The bridge lets you compose these into research workflows:

1. **Pre-screen** → Falsify to ensure stability, Sobol to identify important params
2. **Optimize** → Run EA (DE, CMA-ES) with full budget
3. **Post-analyze** → Contrastive pairs around optimum to measure fragility, LandscapeAnalyzer for local topology

This matches the workflow used in the Evolutionary-Robotics project (`zimmerman_analysis.py`), where the robot simulator is wrapped as a `SimulatorWrapper` and interrogated with all six Zimmerman tools alongside the ea-toolkit's optimization algorithms.

---

## Tests

15 tests in `test_zimmerman_bridge.py`:

**Always run (no zimmerman dependency):**

| Test | What it verifies |
|------|-----------------|
| FitnessAsSimulator: run delegates | evaluate() is called via run() |
| FitnessAsSimulator: param_spec | Bounds pass through unchanged |
| FitnessAsSimulator: keys preserved | Extra result keys visible to Zimmerman |
| FitnessAsSimulator: multiple calls | Stateless operation |
| SimulatorAsFitness: evaluate wraps run | run() is called via evaluate() |
| SimulatorAsFitness: negate | Minimization→maximization conversion |
| SimulatorAsFitness: param_spec | Bounds pass through unchanged |
| SimulatorAsFitness: EA algorithm | Works with HillClimber end-to-end |
| SimulatorAsFitness: default key | 'fitness' key works without config |
| Round trip | FitnessFunction → Simulator → FitnessFunction preserves values |

**Require zimmerman-toolkit (skipped otherwise):**

| Test | What it verifies |
|------|-----------------|
| Sobol on sphere | Sensitivity analysis produces S1/ST |
| Falsify sphere | Clean fitness function has zero violations |
| Contrastive after optimization | Finds flip point near DE optimum |
| Full pipeline | optimize_and_interrogate end-to-end |
| Protocol compliance | FitnessAsSimulator satisfies Simulator protocol |
