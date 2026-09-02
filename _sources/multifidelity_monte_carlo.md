---
file_format: mystnb
kernelspec:
  name: python3
---

# Multifidelity Monte Carlo Methods

{mod}`elyza.multifidelity.montecarlo` provides four estimators for approximating the expectation of an expensive {class}`~elyza.core.evaluator.Evaluator` using low-fidelity models. All four estimators share a
common interface (`get_pilots`, `budget_alloc`, `get_entry_variance`, `evaluate`) inherited from
{class}`~elyza.multifidelity.montecarlo.MultifidelityMonteCarlo`. This page introduces them in order of sophistication: the standard high-fidelity
baseline ({class}`~elyza.multifidelity.montecarlo.HFMC`),
multilevel MC ({class}`~elyza.multifidelity.montecarlo.MLMC`), classic
multifidelity MC ({class}`~elyza.multifidelity.montecarlo.MFMC`), and
finally {class}`~elyza.multifidelity.montecarlo.RMFMC` -- Recursive
Multifidelity Monte Carlo, a generalized estimator original to `elyza`.

Each multifidelity Monte Carlo estimator is fully generalized and works with *vector-valued functions*; we also use a novel greedy algorithms to allocate integer-valued sample allocations given a fixed computational budget.

Every estimator below follows the same four-step workflow:

1. **Define** the estimator from a fidelity hierarchy.
2. **Pilot** it -- draw pilot samples to estimate cross-level covariances (`set_costs=False` so the evaluators' declared costs are used rather than re-timed).
3. **Allocate** a fixed computational budget across fidelity levels.
4. **Analyze** -- compute the analytical estimator variance for that allocation, without spending any real samples.

We then compare all four resulting variances at the end.

## 1. Define a fidelity hierarchy

A fidelity hierarchy is just a list of {class}`~elyza.core.evaluator.Evaluator`
instances that share the same input(s), ordered from lowest to highest
fidelity, each carrying a `cost` attribute:

```{code-cell} python
import jax.numpy as jnp
import jax.random as jrand

from elyza.core.random import Uniform
from elyza.core.evaluator import Evaluator

x = Uniform(
    name="x",
    dim=1,
    lower=0.0,
    upper=1.0,
)

hf = Evaluator(
    name="high-fidelity",
    inputs=[x],
    output_dim=1,
    evaluation_func=lambda x: jnp.sin(2 * jnp.pi * x),
    cost=1.0,
)

lf = Evaluator(
    name="low-fidelity",
    inputs=[x],
    output_dim=1,
    evaluation_func=lambda x: jnp.sin(2 * jnp.pi * x) + 0.3 * jnp.cos(6 * jnp.pi * x),
    cost=1e-2,
)
```

`lf` is a cheap (100x lower cost) but biased approximation of `hf` -- exactly the setting multifidelity methods are built for. Every estimator below is constructed the same way, `Estimator(evaluators=[lf, hf])`, with index 0 the lowest fidelity and the last entry always the high-fidelity
model. We'll draw a million pilot samples (so the covariance estimates are accurate) and allocate the same 200-unit budget to every estimator, so their variances are directly comparable:

```{code-cell} python
n_pilots = int(1e6) # designating a million pilot samples for accurate statistics
budget = 200.0
```

## 2. Baseline: high-fidelity-only Monte Carlo (HFMC)

{class}`~elyza.multifidelity.montecarlo.HFMC` is the naive estimator which ignores every lower-fidelity level and spends the entire budget on evaluating `hf`. {meth}`~elyza.multifidelity.montecarlo.HFMC.budget_alloc` puts the whole budget toward `hf` (zero for every lower level), and {meth}`~elyza.multifidelity.montecarlo.HFMC.get_entry_variance` reports the resulting estimator variance:

```{code-cell} python
from elyza.multifidelity.montecarlo import HFMC

# 1. define
hfmc = HFMC(evaluators=[lf, hf])

# 2. pilot samples
hfmc.get_pilots(jrand.PRNGKey(0), n_pilots=n_pilots, set_costs=False)

# 3. allocate the budget
ms_hfmc = hfmc.budget_alloc(budget=budget)
print("budget allocation:", ms_hfmc)

# 4. analytical estimator variance
var_hfmc = hfmc.get_entry_variance(ms_hfmc)
print("variance:", var_hfmc)
```

## 3. MLMC: telescoping multilevel Monte Carlo

{class}`~elyza.multifidelity.montecarlo.MLMC` (Giles, 2015 [^giles2015]) takes a different approach from the estimators below it: rather than regression coefficients, it sums independent telescoping corrections `E[f_l] - E[f_{l-1}]` between adjacent levels, plus the lowest level's own mean. Each level's correction is estimated from its own independent batch of samples -- unlike `MFMC`/`RMFMC` below, there is no nesting constraint between levels. {meth}`~elyza.multifidelity.montecarlo.MLMC.budget_alloc` allocates an independent per-level sample count, weighing each level's cost (`lf` alone at level 0, `lf` and `hf` together at level 1) against its variance reduction:

```{code-cell} python
from elyza.multifidelity.montecarlo import MLMC

# 1. define
mlmc = MLMC(evaluators=[lf, hf])

# 2. pilot samples
mlmc.get_pilots(jrand.PRNGKey(0), n_pilots=n_pilots, set_costs=False)

# 3. allocate the budget
ms_mlmc = mlmc.budget_alloc(budget=budget)
print("budget allocation:", ms_mlmc)

# 4. analytical estimator variance
var_mlmc = mlmc.get_entry_variance(ms_mlmc)
print("variance:", var_mlmc)
```

## 4. MFMC: multifidelity Monte Carlo

{class}`~elyza.multifidelity.montecarlo.MFMC` (Peherstorfer, Willcox, &
Gunzburger, 2016 [^peherstorfer2016]) instead treats each lower-fidelity
level as a control variate: it computes a single regression coefficient
tying that level's covariance to `hf` and forms a nested correction. Unlike MLMC's independent batches, MFMC's levels are sampled with a *nested* design -- level 0's batch is reused by every level above it. Pilot samples estimate the cross-level covariance, which {meth}`~elyza.multifidelity.montecarlo.MFMC.get_matrix_coefs` converts into regression coefficients before the budget can be allocated:

```{code-cell} python
from elyza.multifidelity.montecarlo import MFMC

# 1. define
mfmc = MFMC(evaluators=[lf, hf])

# 2. pilot samples
mfmc.get_pilots(jrand.PRNGKey(0), n_pilots=n_pilots, set_costs=False)
mfmc.get_matrix_coefs()

# 3. allocate the budget
ms_mfmc = mfmc.budget_alloc(budget=budget)
print("budget allocation:", ms_mfmc)

# 4. analytical estimator variance
var_mfmc = mfmc.get_entry_variance(ms_mfmc)
print("variance:", var_mfmc)
```

## 5. R-MFMC: Recursive Multifidelity Monte Carlo

{class}`~elyza.multifidelity.montecarlo.RMFMC` -- Recursive Multifidelity
Monte Carlo -- is the flagship estimator in `elyza`, a generalized
estimator developed by the author as part of PhD thesis research. Where
MFMC ties each lower-fidelity level to a single control-variate
coefficient against `hf` alone, R-MFMC solves, at every level, a full
regression against the *joint* covariance of every level below it --
recursively building up coefficients from the lowest fidelity upward. This
lets it exploit correlations between the lower-fidelity levels themselves,
not just each level's individual correlation with `hf`. R-MFMC is original
to `elyza`, and this is its first public release.

```{code-cell} python
from elyza.multifidelity.montecarlo import RMFMC

# 1. define
rmfmc = RMFMC(evaluators=[lf, hf])   # index 0 is lowest-fidelity

# 2. pilot samples
rmfmc.get_pilots(jrand.PRNGKey(0), n_pilots=n_pilots, set_costs=False)
rmfmc.get_matrix_coefs()

# 3. allocate the budget
ms_rmfmc = rmfmc.budget_alloc(budget=budget)
print("budget allocation:", ms_rmfmc)

# 4. analytical estimator variance
var_rmfmc = rmfmc.get_entry_variance(ms_rmfmc)
print("variance:", var_rmfmc)
```

For evaluators that all share the same output dimension,
{meth}`~elyza.multifidelity.montecarlo.RMFMC.get_vector_coefs` and
{meth}`~elyza.multifidelity.montecarlo.RMFMC.get_scalar_coefs` are cheaper
alternatives to {meth}`~elyza.multifidelity.montecarlo.RMFMC.get_matrix_coefs`,
restricting the regression coefficients to a diagonal or scalar multiple of
the identity, respectively. `MFMC` inherits these too, though only
`get_matrix_coefs` is overridden to enforce the single-coefficient-per-level
restriction described above.

## 6. Comparing all four estimator variances

With the same pilot samples and the same 200-unit budget behind each
estimator, their analytical variances are directly comparable -- lower is
better:

```{code-cell} python
results = [
    ("HFMC", var_hfmc),
    ("MLMC", var_mlmc),
    ("MFMC", var_mfmc),
    ("RMFMC", var_rmfmc),
]

print(f"{'estimator':<10}{'variance':>15}")
for name, var in results:
    print(f"{name:<10}{float(var.squeeze()):>15.6e}")
```

As expected, each additional layer of sophistication -- from the naive
high-fidelity-only baseline, through independent telescoping corrections,
single-coefficient control variates, and finally the fully recursive joint
regression -- squeezes more variance reduction out of the same budget.

## References

[^peherstorfer2016]: Peherstorfer, B., Willcox, K., & Gunzburger, M. (2016).
    Optimal Model Management for Multifidelity Monte Carlo Estimation.
    *SIAM Journal on Scientific Computing*, 38(5), A3163-A3194.
    https://doi.org/10.1137/15M1046472

[^giles2015]: Giles, M. B. (2015). Multilevel Monte Carlo methods.
    *Acta Numerica*, 24, 259-328.
    https://doi.org/10.1017/S096249291500001X
