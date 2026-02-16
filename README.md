# EA Toolkit

Standalone evolutionary algorithms library for black-box optimization. Numpy-only, protocol-based, extracted from real research projects.

## What's Inside

**8 Algorithms**

| Algorithm | Strategy |
|-----------|----------|
| `HillClimber` | Greedy local search with parallel restarts |
| `OnePlusLambdaES` | (1+lambda) evolution strategy with adaptive sigma |
| `RidgeWalker` | Multi-objective Pareto exploration along fitness ridges |
| `CliffMapper` | Seeks high-sensitivity regions ("cliffs") in the landscape |
| `NoveltySeeker` | k-NN novelty-driven search (ignores fitness entirely) |
| `EnsembleExplorer` | Parallel hill climbers with convergence detection and teleportation |
| `DifferentialEvolution` | DE/rand/1/bin with population-based vector differences |
| `CMAES` | Covariance Matrix Adaptation — learns landscape geometry via covariance |

**3 Mutation Operators** — Gaussian (isotropic, unit sphere direction), Cauchy (heavy-tailed jumps), Adaptive (1/5th success rule)

**2 Crossover Operators** — SBX (simulated binary, continuous), Uniform (per-parameter swap)

**3 Selection Strategies** — Tournament, Truncation, Epsilon-greedy

**5 Benchmark Functions** — Sphere, Rosenbrock, Rastrigin, Ackley, ZDT1 (bi-objective)

**Callback System** — ConvergenceChecker (early stopping), ProgressPrinter, TelemetryCallback, HistoryRecorder

**Ask-Tell Interface** — DE and CMA-ES support external evaluation loops for distributed/async optimization

**Landscape Analysis** — Cliffiness probing, roughness ratio, gradient estimation, sign flip rate, comprehensive `LandscapeAnalyzer`

**Atlas Builder** — Multi-phase landscape mapping: random probing → 2D slice heatmaps → cliff anatomy profiling, with matplotlib visualization and JSON export for D3/P5 dashboards

**Population Management** — Random generation, elitism, parameter-space and behavioral diversity metrics

**Telemetry** — JSON-lines logging with numpy serialization

**Zimmerman Bridge** — Bidirectional adapters connecting to the [Zimmerman toolkit](https://github.com/KathrynC/zimmerman-toolkit) for Sobol sensitivity, falsification, contrastive analysis, and POSIWID auditing

## Quick Start

```python
from ea_toolkit import HillClimber, GaussianMutation
from ea_toolkit.base import FitnessFunction

class MyProblem(FitnessFunction):
    def evaluate(self, params: dict) -> dict:
        x, y = params['x'], params['y']
        return {'fitness': -(x**2 + y**2)}

    def param_spec(self) -> dict:
        return {'x': (-5.0, 5.0), 'y': (-5.0, 5.0)}

hc = HillClimber(MyProblem(), mutation=GaussianMutation(sigma=0.3), seed=42)
history = hc.run(budget=1000)
best = hc.best()
print(f"Best: {best['fitness']:.4f} at {best['params']}")
```

## Design

**Protocol-based.** All components implement abstract base classes. Any algorithm works with any mutation operator. Implement `FitnessFunction` with two methods (`evaluate` and `param_spec`) and you're done.

**Budget-based.** All algorithms take an evaluation budget (total fitness function calls), making head-to-head comparison fair.

**Maximization.** Fitness is always maximized. Negate for minimization.

**Reproducible.** All randomness flows through seeded `np.random.Generator` instances. Same seed = identical results.

**Numpy-only.** No scipy, no sklearn. Just numpy.

## Requirements

- Python 3.11+
- NumPy

## Tests

```bash
pytest tests/ -v   # 113 tests, ~0.6s

# With Zimmerman integration tests (5 additional)
PYTHONPATH=../zimmerman-toolkit pytest tests/ -v
```

## Origin

Inspired by Josh Bongard's [Ludobots](https://www.reddit.com/r/ludobots/) evolutionary robotics course (University of Vermont, 2014) and extracted from two research projects by Kathryn Cramer:

- **[Evolutionary-Robotics](https://github.com/KathrynC/Evolutionary-Robotics)** — PyBullet 3-link robot locomotion optimization (116 gaits, ~25k simulations), created in the context of Bongard's Ludobots framework. The hill climber, ridge walker, cliff mapper, novelty seeker, and ensemble explorer all originated in `walker_competition.py`. Landscape analysis came from `atlas_cliffiness.py`.

- **how-to-live-much-longer** — Mitochondrial aging ODE simulator based on John G. Cramer's 2025 book, with LLM-mediated intervention design. The (1+lambda) ES and adaptive mutation came from `temporal_optimizer.py`.

Both projects needed the same core optimization machinery. Rather than duplicate code, the shared algorithms were extracted into this standalone library.

Works alongside the [Zimmerman toolkit](https://github.com/KathrynC/zimmerman-toolkit) for black-box simulator interrogation (Sobol sensitivity, falsification, contrastive analysis). See [`docs/zimmerman_bridge.md`](docs/zimmerman_bridge.md) for the integration API.

## Documentation

Full Wolfram-style reference documentation in [`docs/`](docs/):

- [`docs/guide.md`](docs/guide.md) — Main reference guide
- Per-module pages for all [base classes](docs/base.md), [mutation operators](docs/mutation.md), [crossover operators](docs/crossover.md), [selection strategies](docs/selection.md), [population tools](docs/population.md), [landscape analysis](docs/landscape.md), [telemetry](docs/telemetry.md), [benchmarks](docs/benchmarks.md), [callbacks](docs/callbacks.md), [Zimmerman bridge](docs/zimmerman_bridge.md), and each of the [8 algorithms](docs/guide.md#algorithms)
- [`docs/references.md`](docs/references.md) — Academic citations (Bongard, Storn & Price, Hansen, Deb, Lehman & Stanley, Beer, Zitzler, Zimmerman)

## License

Research code. Contact the author for licensing terms.
