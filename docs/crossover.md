# crossover

Two crossover operators for recombining parameter vectors.

---

## Overview

Implements 2 crossover strategies commonly used in evolutionary algorithms. Both operators implement the `CrossoverOperator` protocol: `crossover(parent1, parent2, param_spec, rng) -> (child1, child2)`.

All operators work on `dict[str, float]` parameter vectors with sorted-key ordering for deterministic behavior.

---

## Operators

### `SBXCrossover`

Simulated Binary Crossover. Produces two children whose spread around the parents is controlled by a distribution index eta — the real-valued analogue of single-point crossover in binary GAs.

From Deb & Agrawal (1995). Standard crossover operator in NSGA-II and most modern real-coded GAs.

```python
SBXCrossover(eta: float = 20.0, probability: float = 0.9)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eta` | 20.0 | Distribution index. Higher = children closer to parents. Lower = wider spread. |
| `probability` | 0.9 | Per-parameter probability of crossover (vs. copying parent value) |

**Algorithm (per parameter):**
1. If `random() > probability` or parents have same value, copy parent values directly
2. Draw u ~ Uniform(0, 1)
3. Compute spread factor beta from u and eta:
   - If u ≤ 0.5: `beta = (2u)^(1/(eta+1))`
   - Else: `beta = (1/(2(1-u)))^(1/(eta+1))`
4. `child1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)`
5. `child2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)`
6. Clamp both children to bounds

**Properties:**
- Identical parents produce identical children (no spread when parent distance = 0)
- Higher eta → tighter distribution around parents (eta=100 is nearly copying)
- Lower eta → children can be far from parents (eta=1 is very wide)
- Bounded: children are always clamped to `param_spec` bounds

```python
sbx = SBXCrossover(eta=20.0)
rng = np.random.default_rng(42)
child1, child2 = sbx.crossover(parent1, parent2, spec, rng)
```

---

### `UniformCrossover`

Per-parameter coin flip to swap values between parents. Each parameter independently comes from one parent or the other.

```python
UniformCrossover(swap_probability: float = 0.5)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `swap_probability` | 0.5 | Probability of swapping each parameter (0 = copy parents, 1 = swap all) |

**Algorithm (per parameter):**
1. If `random() < swap_probability`: child1 gets parent2's value, child2 gets parent1's
2. Else: child1 gets parent1's value, child2 gets parent2's

**Properties:**
- Children are complementary: every value goes to exactly one child
- `swap_probability=0.0` copies parents exactly
- `swap_probability=1.0` swaps all parameters (children = reversed parents)
- No new values are created — children contain only parent values

```python
ux = UniformCrossover(swap_probability=0.5)
child1, child2 = ux.crossover(parent1, parent2, spec, rng)
```

---

## When to Use Which

| Situation | Recommendation |
|-----------|---------------|
| Continuous optimization with smooth landscape | SBX (creates intermediate values) |
| Combinatorial or discrete-like parameters | Uniform (preserves parent values exactly) |
| Need tight exploration near parents | SBX with high eta (≥ 50) |
| Need wide exploration from parents | SBX with low eta (1–5) |
| Need maximum mixing | Uniform with probability 0.5 |

---

## Tests

10 tests in `test_crossover.py`:

| Test | What it verifies |
|------|-----------------|
| `test_produces_two_children` (SBX) | Returns two dicts with correct keys |
| `test_children_within_bounds` (SBX) | 50 rounds, all values within spec bounds |
| `test_high_eta_tight` (SBX) | eta=100 keeps children near parents |
| `test_identical_parents` (SBX) | Same parents → same children |
| `test_reproducibility` (SBX) | Same seed → same results |
| `test_produces_two_children` (Uniform) | Returns two dicts with correct keys |
| `test_children_from_parents` (Uniform) | Each value is from p1 or p2 |
| `test_complementary_children` (Uniform) | c1 and c2 are complementary |
| `test_swap_zero_copies_parents` (Uniform) | probability=0 → exact copies |
| `test_swap_one_swaps_all` (Uniform) | probability=1 → fully swapped |
