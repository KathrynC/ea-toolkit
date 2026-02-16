# CMAES

Covariance Matrix Adaptation Evolution Strategy — the gold standard for continuous optimization.

---

## Overview

CMA-ES maintains a multivariate normal distribution over the search space and adapts both the covariance matrix (search direction and shape) and the step size (search magnitude) based on the success of recent generations.

The key insight: by learning the covariance structure of successful steps, CMA-ES automatically discovers and exploits the fitness landscape's local geometry. On Rosenbrock's banana function, it learns the curved valley. On ill-conditioned problems, it stretches along the easy axis and compresses along the hard one.

Implements the (mu/mu_w, lambda)-CMA-ES with:
- Weighted recombination of the mu best individuals
- Cumulative step-size adaptation (CSA)
- Rank-one and rank-mu covariance matrix updates
- Eigendecomposition for efficient sampling

Reference: Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial." arXiv:1604.00772.

Supports both `run()` and the ask-tell interface natively.

---

## Constructor

```python
CMAES(
    fitness_fn: FitnessFunction,
    sigma0: float = 0.5,
    pop_size: int | None = None,
    seed: int | None = None,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fitness_fn` | — | Fitness function to optimize |
| `sigma0` | 0.5 | Initial step size. Should cover ~1/4 of the expected distance to optimum |
| `pop_size` | `4 + floor(3*ln(n))` | Offspring population size (lambda). Default scales with dimension |
| `seed` | None | Random seed for reproducibility |

---

## Algorithm

### Initialization

The mean starts at a random point within bounds. The covariance matrix starts as identity (isotropic). All strategy parameters are computed from the dimension n and population size lambda:

| Parameter | Formula | Role |
|-----------|---------|------|
| lambda | `4 + floor(3*ln(n))` | Population size |
| mu | `lambda // 2` | Number of parents (best half) |
| weights | Log-linear, normalized | Recombination weights for mu best |
| mu_eff | `1 / sum(w_i^2)` | Variance effective selection mass |
| c_sigma, d_sigma | From n, mu_eff | Step-size control |
| c_c, c_1, c_mu | From n, mu_eff | Covariance control |

### Generation Cycle

1. **Sample:** Draw lambda candidates from `N(mean, sigma^2 * C)` using the eigendecomposition `x = mean + sigma * B * D * z` where `z ~ N(0, I)`
2. **Evaluate:** All candidates
3. **Sort:** By fitness (descending)
4. **Recombine:** New mean = weighted sum of mu best candidates
5. **Update p_sigma:** Cumulative step-size path (conjugate evolution path)
6. **Update p_c:** Cumulative covariance path (with Heaviside stalling detection)
7. **Update C:** Rank-one (from p_c) + rank-mu (from mu best steps) covariance update
8. **Update sigma:** Step-size adaptation via `sigma *= exp((||p_sigma||/chi_n - 1) * c_sigma/d_sigma)`
9. **Eigendecompose:** Update B, D from C (every n/10 generations for efficiency)

### Budget Handling

When the remaining budget is less than a full generation (lambda candidates), CMA-ES evaluates what it can and records those evaluations, but skips the distribution update (which requires at least mu individuals). This ensures strict budget compliance.

---

## Methods

### `run(budget) -> list[dict]`

```python
cma = CMAES(my_fitness, sigma0=2.0, seed=42)
history = cma.run(budget=5000)
best = cma.best()
```

### `ask() -> list[dict]`

Sample lambda candidates from the current distribution.

### `tell(evaluations)`

Update the distribution based on `(params, result)` tuples.

```python
cma = CMAES(my_fitness, sigma0=1.0, seed=42)
for _ in range(100):
    candidates = cma.ask()
    evals = [(c, my_fitness.evaluate(c)) for c in candidates]
    cma.tell(evals)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `sigma` | `float` | Current step size |
| `mean` | `np.ndarray` or None | Current distribution mean |
| `covariance` | `np.ndarray` or None | Current covariance matrix (n×n) |

Properties return `None` before the first `ask()` call.

---

## When to Use CMA-ES

**Good for:**
- Continuous optimization in moderate dimensions (2–100)
- Ill-conditioned problems (stretched, rotated landscapes)
- Problems with curved ridges (Rosenbrock-like)
- When you want a single algorithm that works well without tuning

**Not ideal for:**
- Very high dimensions (>100) — covariance matrix becomes O(n^2)
- Discrete/combinatorial problems
- Very noisy fitness functions
- Problems where evaluations are extremely expensive (small budgets)

---

## Comparison with DE

| Aspect | CMA-ES | DE |
|--------|--------|-----|
| Adaptation | Learns full covariance structure | No adaptation beyond greedy selection |
| Parameters | Just sigma0 (pop_size auto-scales) | F, CR, pop_size all matter |
| Dimensions | Best at 2–100 | Works at any dimension |
| Memory | O(n^2) for covariance matrix | O(pop_size × n) |
| Multimodal | Can get stuck (unimodal search) | Population provides diversity |
| Convergence | Fast on unimodal/ill-conditioned | Steady on multimodal |

---

## Tests

9 tests in `test_new_algorithms.py`:

| Test | What it verifies |
|------|-----------------|
| `test_converges_on_sphere` | Finds near-optimal on 5D sphere (fitness > -0.01) |
| `test_budget_respected` | Never exceeds evaluation budget |
| `test_history_records` | All entries have params and fitness |
| `test_ask_tell_interface` | ask/tell cycle works, consistent pop size |
| `test_sigma_adapts` | Step size changes during optimization |
| `test_properties` | mean, covariance have correct shapes after running |
| `test_custom_pop_size` | Override pop_size works |
| `test_rosenbrock` | Makes progress on Rosenbrock (curved valley) |
| `test_callbacks` | HistoryRecorder works with CMA-ES |
