# benchmarks

Five standard optimization test functions.

---

## Overview

Standard benchmark functions commonly used in the evolutionary computation literature for algorithm comparison. All implement the `FitnessFunction` protocol with configurable dimensionality and bounds.

Following the toolkit convention, fitness is negated (maximization): the optimal value is 0.0 for all single-objective benchmarks.

---

## Single-Objective Benchmarks

### `SphereFitness`

The simplest unimodal test function. A single global optimum at the origin with no local optima. Any algorithm that can do gradient-free optimization should solve this easily.

```python
SphereFitness(n_dims: int = 5, bounds: tuple = (-5.12, 5.12))
```

**Function:** `fitness = -sum(x_i^2)`

**Optimum:** `f(0, ..., 0) = 0`

**Landscape:** Smooth, convex, isotropic. The difficulty scales linearly with dimension.

---

### `RosenbrockFitness`

The "banana function." A narrow, curved valley connects a broad flat region to the global optimum at (1, 1, ..., 1). Easy to find the valley floor, hard to follow it to the optimum.

```python
RosenbrockFitness(n_dims: int = 5, bounds: tuple = (-5.0, 10.0))
```

**Function:** `fitness = -sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)`

**Optimum:** `f(1, ..., 1) = 0`

**Landscape:** Non-separable, moderately multimodal in higher dimensions. Tests ability to follow curved ridges. CMA-ES excels here because it learns the valley's covariance structure.

---

### `RastriginFitness`

Highly multimodal with a regular pattern of local optima. The global structure is a bowl (like Sphere), but cosine terms create ~10^n local optima in n dimensions.

```python
RastriginFitness(n_dims: int = 5, bounds: tuple = (-5.12, 5.12), A: float = 10.0)
```

**Function:** `fitness = -(A*n + sum(x_i^2 - A*cos(2*pi*x_i)))`

**Optimum:** `f(0, ..., 0) = 0`

**Landscape:** Massively multimodal but globally convex. Tests ability to escape local optima. The spacing between local optima is ~1.0 in each dimension. Returns `raw_rastrigin` in the result dict (un-negated value).

---

### `AckleyFitness`

Multimodal with a nearly flat outer region and a deep hole near the origin. The exponential terms create a distinctive needle-in-haystack character at larger scales.

```python
AckleyFitness(n_dims: int = 5, bounds: tuple = (-5.0, 5.0),
              a: float = 20.0, b: float = 0.2, c: float = 2*pi)
```

**Function:** `fitness = -(a + e - a*exp(-b*sqrt(mean(x_i^2))) - exp(mean(cos(c*x_i))))`

**Optimum:** `f(0, ..., 0) = 0`

**Landscape:** Nearly flat far from origin, sharply peaked near it. Tests algorithms that can handle search in flat regions. Returns `raw_ackley` in the result dict.

---

## Multi-Objective Benchmark

### `ZDT1Fitness`

The simplest Zitzler-Deb-Thiele bi-objective test problem. Has a convex Pareto front where x_1 through x_{n-1} are all zero.

```python
ZDT1Fitness(n_dims: int = 30)
```

**Function:**
- `f1 = x_0`
- `g = 1 + 9 * mean(x_1, ..., x_{n-1})`
- `f2 = g * (1 - sqrt(f1/g))`
- `fitness = -f1` (scalar for algorithm compatibility)

**Pareto front:** `f2 = 1 - sqrt(f1)` when all `x_i = 0` for `i > 0`

**Returns:** `f1`, `f2`, `neg_f2`, `g`, `fitness`

**Bounds:** All parameters in [0, 1].

Use with `RidgeWalker` for multi-objective exploration — the Pareto front is the trade-off curve between f1 and f2.

---

## Usage

```python
from ea_toolkit.benchmarks import SphereFitness, RosenbrockFitness
from ea_toolkit import CMAES

# Single objective
sf = SphereFitness(n_dims=10)
cma = CMAES(sf, sigma0=2.0, seed=42)
cma.run(budget=5000)
print(f"Best: {cma.best()['fitness']:.6f}")

# Multi-objective
from ea_toolkit import RidgeWalker
zdt = ZDT1Fitness(n_dims=10)
rw = RidgeWalker(zdt, seed=42)
rw.run(budget=1000)
# Pareto front is in rw.pareto_front
```

---

## Tests

14 tests in `test_benchmarks.py`:

| Test | What it verifies |
|------|-----------------|
| Sphere: optimum at origin | f(0,...,0) ≈ 0 |
| Sphere: negative away | f(1,1,1) = -3.0 |
| Sphere: param_spec | Correct bounds and dimensions |
| Rosenbrock: optimum at ones | f(1,...,1) ≈ 0 |
| Rosenbrock: valley structure | On-valley > off-valley |
| Rastrigin: optimum at origin | f(0,...,0) ≈ 0 |
| Rastrigin: local optima | f(0) > f(1) (local optima worse) |
| Rastrigin: raw value | Returns `raw_rastrigin` key |
| Ackley: optimum at origin | f(0,...,0) ≈ 0 |
| Ackley: negative away | f(1,1,1) < 0 |
| Ackley: raw value | Returns `raw_ackley` key |
| ZDT1: param_spec | All bounds [0, 1] |
| ZDT1: Pareto optimal | Correct f1, f2 on front |
| ZDT1: dominated point | Off-front has worse f2 |
| All benchmarks: param_spec | All return valid spec dicts |
| All benchmarks: fitness | All return `fitness` key with float value |
