# EA Toolkit — Reference Guide

Standalone evolutionary algorithms library for black-box optimization.

---

## From Ludobots to Toolkit

This toolkit grew out of work inspired by Josh Bongard's [Ludobots](https://www.reddit.com/r/ludobots/) evolutionary robotics course at the University of Vermont. Ludobots uses Pyrosim to co-evolve morphology and neural control of simulated robots — the conceptual foundation for the [Evolutionary-Robotics](../../pybullet_test/Evolutionary-Robotics/) project by Kathryn Cramer, which extends that framework to a 3-link PyBullet robot with 116 gaits, ~25k simulations, and LLM-mediated neural control.

The same optimization algorithms were also needed for the [how-to-live-much-longer](../../how-to-live-much-longer/) project (mitochondrial aging ODE simulation based on John G. Cramer's 2025 book). Rather than duplicate code, the shared algorithms were extracted into this standalone library.

The toolkit is numpy-only (no scipy, no sklearn) and uses a protocol-based design where all components implement abstract base classes. See [References](references.md) for full citations.

---

## Overview

The toolkit provides 40 public API symbols organized into 9 layers:

| Layer | Components | Question |
|-------|-----------|----------|
| [`base`](base.md) | `FitnessFunction`, `MutationOperator`, `CrossoverOperator`, `SelectionStrategy`, `Algorithm`, `Callback` | What interfaces must components implement? |
| [`mutation`](mutation.md) | `GaussianMutation`, `CauchyMutation`, `AdaptiveMutation` | How do we perturb parameter vectors? |
| [`crossover`](crossover.md) | `SBXCrossover`, `UniformCrossover` | How do we recombine parent parameter vectors? |
| [`selection`](selection.md) | `TournamentSelection`, `TruncationSelection`, `EpsilonGreedy` | How do we choose individuals from a population? |
| [`population`](population.md) | `PopulationManager`, `random_population`, `elitism`, `diversity_metric`, `behavioral_diversity` | How do we manage and analyze populations? |
| [`landscape`](landscape.md) | `probe_cliffiness`, `roughness_ratio`, `sign_flip_rate`, `gradient_estimate`, `LandscapeAnalyzer` | What does the fitness landscape look like? |
| [`atlas`](atlas.md) | `AtlasBuilder` | Multi-phase landscape atlas (probe, slice, profile) |
| [`telemetry`](telemetry.md) | `Telemetry`, `load_telemetry` | How do we log optimization progress? |
| [`benchmarks`](benchmarks.md) | `SphereFitness`, `RosenbrockFitness`, `RastriginFitness`, `AckleyFitness`, `ZDT1Fitness` | Standard test functions for algorithm comparison |
| [`callbacks`](callbacks.md) | `ConvergenceChecker`, `ProgressPrinter`, `TelemetryCallback`, `HistoryRecorder` | How do we monitor and control algorithm runs? |
| [`zimmerman_bridge`](zimmerman_bridge.md) | `FitnessAsSimulator`, `SimulatorAsFitness` | How do we connect to the Zimmerman toolkit? |

Plus 8 algorithms:

| Algorithm | Strategy | Source |
|-----------|----------|--------|
| [`HillClimber`](hill_climber.md) | Greedy local search with parallel restarts | `walker_competition.py` |
| [`OnePlusLambdaES`](es.md) | (1+lambda) evolution strategy with adaptive sigma | `temporal_optimizer.py` |
| [`RidgeWalker`](ridge_walker.md) | Multi-objective Pareto exploration | `walker_competition.py` |
| [`CliffMapper`](cliff_mapper.md) | High-sensitivity region search | `walker_competition.py` |
| [`NoveltySeeker`](novelty_seeker.md) | k-NN novelty-driven search | `walker_competition.py` |
| [`EnsembleExplorer`](ensemble.md) | Multi-walker ensemble with convergence teleportation | `walker_competition.py` |
| [`DifferentialEvolution`](de.md) | DE/rand/1/bin with population-based vector differences | Standard (Storn & Price 1997) |
| [`CMAES`](cma_es.md) | Covariance matrix adaptation with step-size control | Standard (Hansen 2001) |

---

## Quick Start

```python
from ea_toolkit import HillClimber, GaussianMutation
from ea_toolkit.base import FitnessFunction

# 1. Define your fitness function
class MyProblem(FitnessFunction):
    def evaluate(self, params: dict) -> dict:
        x, y = params['x'], params['y']
        return {'fitness': -(x**2 + y**2)}  # Maximize (optimum at origin)

    def param_spec(self) -> dict:
        return {'x': (-5.0, 5.0), 'y': (-5.0, 5.0)}

# 2. Choose an algorithm
hc = HillClimber(MyProblem(), mutation=GaussianMutation(sigma=0.3), seed=42)

# 3. Run
history = hc.run(budget=1000)
best = hc.best()
print(f"Best fitness: {best['fitness']:.4f}")
print(f"Best params: {best['params']}")
```

---

## Design Principles

### Protocol-Based Architecture

All components implement abstract base classes from `base.py`. Any algorithm can use any mutation operator. Any selection strategy works with any population. This composability means you can mix and match freely:

```python
# Cauchy mutation + ES
es = OnePlusLambdaES(my_fitness, mutation=CauchyMutation(scale=0.1))

# Adaptive mutation + Hill Climber
hc = HillClimber(my_fitness, mutation=AdaptiveMutation(sigma_init=0.5))
```

### Budget-Based Execution

All algorithms take an evaluation `budget` — the total number of fitness function calls allowed. This makes comparison fair: 500 evaluations of HillClimber vs. 500 evaluations of OnePlusLambdaES vs. 500 evaluations of NoveltySeeker.

### Maximization Convention

Fitness is always maximized. For minimization problems, negate the objective:

```python
def evaluate(self, params):
    cost = compute_cost(params)
    return {'fitness': -cost}
```

### Reproducibility

All randomness flows through `np.random.Generator` instances seeded at construction. Same seed = identical results.

---

## Module Pages

### Core Infrastructure

- **[`base`](base.md)** — Abstract base classes: FitnessFunction, MutationOperator, CrossoverOperator, SelectionStrategy, Algorithm, Callback
- **[`mutation`](mutation.md)** — GaussianMutation, CauchyMutation, AdaptiveMutation
- **[`crossover`](crossover.md)** — SBXCrossover, UniformCrossover
- **[`selection`](selection.md)** — TournamentSelection, TruncationSelection, EpsilonGreedy
- **[`population`](population.md)** — PopulationManager, random_population, elitism, diversity metrics
- **[`landscape`](landscape.md)** — Cliffiness probing, roughness ratio, gradient estimation, LandscapeAnalyzer
- **[`atlas`](atlas.md)** — Multi-phase landscape atlas builder (random probing → 2D slices → cliff anatomy)
- **[`telemetry`](telemetry.md)** — JSON-lines logging and loading
- **[`benchmarks`](benchmarks.md)** — Sphere, Rosenbrock, Rastrigin, Ackley, ZDT1
- **[`callbacks`](callbacks.md)** — ConvergenceChecker, ProgressPrinter, TelemetryCallback, HistoryRecorder

### Algorithms

- **[`HillClimber`](hill_climber.md)** — Parallel hill climber with restarts
- **[`OnePlusLambdaES`](es.md)** — (1+lambda) evolution strategy
- **[`RidgeWalker`](ridge_walker.md)** — Multi-objective Pareto search
- **[`CliffMapper`](cliff_mapper.md)** — High-sensitivity region search
- **[`NoveltySeeker`](novelty_seeker.md)** — Novelty-driven exploration
- **[`EnsembleExplorer`](ensemble.md)** — Multi-walker ensemble with teleportation
- **[`DifferentialEvolution`](de.md)** — DE/rand/1/bin with ask-tell interface
- **[`CMAES`](cma_es.md)** — Covariance matrix adaptation evolution strategy

### Integration

- **[`zimmerman_bridge`](zimmerman_bridge.md)** — Bidirectional adapters and combined workflows with the [Zimmerman toolkit](../../zimmerman-toolkit/)
- **[References](references.md)** — Full academic citations

---

## Zimmerman Toolkit Integration

The ea-toolkit optimizes; the [Zimmerman toolkit](../../zimmerman-toolkit/) interrogates. Together they answer complementary questions about the same simulator:

```
Optimize (ea-toolkit)                Interrogate (zimmerman-toolkit)
─────────────────────                ──────────────────────────────
HillClimber, DE, CMA-ES  ←bridge→   Sobol sensitivity
CliffMapper, LandscapeAnalyzer       Falsifier (numerical stability)
NoveltySeeker                        ContrastiveGenerator (fragility)
RidgeWalker (Pareto)                 POSIWIDAuditor (alignment)
```

The bridge works because both toolkits share the same protocol — `evaluate`/`run` a dict of parameters, get back a dict of results, with `param_spec()` defining bounds. The `FitnessAsSimulator` and `SimulatorAsFitness` adapters cross-connect them.

**Typical research workflow:**

1. `falsify_fitness()` — verify the fitness function doesn't produce NaN/Inf
2. `sobol_on_fitness()` — identify which parameters drive variance
3. Run DE or CMA-ES — find the optimum
4. `contrastive_around_best()` — measure fragility of the solution

Or use `optimize_and_interrogate()` for the full pipeline in one call.

See [`zimmerman_bridge`](zimmerman_bridge.md) for complete API and examples.

---

## Test Suite

113 tests across 7 modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_algorithms.py` | 27 | All 6 original algorithms: convergence, budget, history, reproducibility, cross-algorithm |
| `test_landscape.py` | 20 | Cliffiness, roughness, sign flips, gradient, LandscapeAnalyzer |
| `test_benchmarks.py` | 14 | All 5 benchmark functions: optima, bounds, properties |
| `test_crossover.py` | 10 | SBX and Uniform crossover: bounds, reproducibility, edge cases |
| `test_callbacks.py` | 7 | Callback protocol, convergence checker, history recorder, integration |
| `test_new_algorithms.py` | 20 | DE and CMA-ES: convergence, ask-tell, budget, cross-algorithm with all 8 |
| `test_zimmerman_bridge.py` | 15 | Adapters (always), Zimmerman integration (when available) |

Test fixtures provide benchmark functions via `conftest.py` (SphereFitness, RastriginFitness, MultiObjectiveSphere, StepFitness, LinearFitness) and via `ea_toolkit.benchmarks` (Sphere, Rosenbrock, Rastrigin, Ackley, ZDT1).

Zimmerman integration tests require `zimmerman-toolkit` on `PYTHONPATH` and are skipped otherwise.
