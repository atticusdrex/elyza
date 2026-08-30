# Multifidelity Quickstart

`elyza.multifidelity` provides two tools for combining
models of varying cost and accuracy: Monte Carlo estimators
({mod}`elyza.multifidelity.montecarlo`) that blend cheap and expensive
{class}`~elyza.core.evaluator.Evaluator` samples into a low-variance
estimate of a high-fidelity expectation, and
{class}`~elyza.multifidelity.surrogate.MAGPI`, a hierarchical surrogate
that chains per-level models together.

Every example below runs as-is against a clean checkout.

## 1. Define a fidelity hierarchy

A fidelity hierarchy is just a list of {class}`~elyza.core.evaluator.Evaluator`
instances that share the same input(s), ordered from lowest to highest
fidelity, each carrying a `cost`:

```python
import jax.numpy as jnp
import jax.random as jrand

from elyza.core.data import ScalarInput
from elyza.core.evaluator import Evaluator

x = ScalarInput(
    name="x",
    dim=1,
    sampling_func=lambda key: jrand.uniform(key, minval=0.0, maxval=1.0),
    minval=0.0,
    maxval=1.0,
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

`lf` is a cheap (100x lower cost) but biased approximation of `hf` --
exactly the setting multifidelity methods are built for.

## 2. Regression-based Multifidelity Monte Carlo (R-MFMC)

{class}`~elyza.multifidelity.montecarlo.RMFMC` is the flagship estimator in
`elyza`. Where classic MFMC gives each lower-fidelity level a single
control-variate coefficient tying it to the high-fidelity model, R-MFMC
solves for a full regression against the *joint* covariance of every level
below it -- letting it exploit correlations between the lower-fidelity
levels themselves, not just their individual correlation with `hf`. This
implementation is original to `elyza`, and this is its first public
release.

Like every estimator in this module, R-MFMC starts from pilot samples that
estimate the cross-level covariance, then converts that covariance into
regression coefficients:

```python
from elyza.multifidelity.montecarlo import RMFMC, HFMC

rmfmc = RMFMC(evaluators=[lf, hf])   # index 0 is lowest-fidelity
rmfmc.get_pilots(jrand.PRNGKey(0), n_pilots=2000)
rmfmc.get_matrix_coefs()
```

With coefficients in hand, {meth}`~elyza.multifidelity.montecarlo.RMFMC.budget_alloc`
turns a total evaluation budget into a per-level sample allocation, and
{meth}`~elyza.multifidelity.montecarlo.RMFMC.get_entry_variance` reports the
resulting estimator variance -- useful for comparing allocations before
spending any real samples:

```python
ms = rmfmc.budget_alloc(budget=200.0)     # e.g. [5000, 150]
rmfmc.get_entry_variance(ms)              # per-output-dimension variance
```

Finally, {meth}`~elyza.multifidelity.montecarlo.RMFMC.evaluate` draws fresh
samples and computes the estimate. Its `sample_sizes` are *incremental*
per-level batch sizes for a nested design -- level 0's batch is shared with
every higher level, so the total number of `lf` evaluations here is
`150 + 50 = 200` while `hf` is only evaluated 50 times:

```python
estimate = rmfmc.evaluate(jrand.PRNGKey(1), sample_sizes=[150, 50])
```

Compare against a plain high-fidelity-only baseline
({class}`~elyza.multifidelity.montecarlo.HFMC`) spending the same 50 `hf`
evaluations, with no `lf` samples at all:

```python
hfmc = HFMC(evaluators=[lf, hf])
baseline = hfmc.evaluate(jrand.PRNGKey(2), sample_sizes=[0, 50])
```

For evaluators that all share the same output dimension,
{meth}`~elyza.multifidelity.montecarlo.RMFMC.get_vector_coefs` and
{meth}`~elyza.multifidelity.montecarlo.RMFMC.get_scalar_coefs` are cheaper
alternatives to {meth}`~elyza.multifidelity.montecarlo.RMFMC.get_matrix_coefs`,
restricting the regression coefficients to a diagonal or scalar multiple of
the identity, respectively.

## 3. Related estimators: MFMC and MLMC

{class}`~elyza.multifidelity.montecarlo.MFMC` (Peherstorfer, Willcox, &
Gunzburger, 2016 [^peherstorfer2016]) is the special case of R-MFMC above
in which each level's coefficient is restricted to its own
covariance/cross-covariance with `hf`, ignoring correlations between the
lower-fidelity levels. It shares the exact same interface as `RMFMC`, so
the section above applies unchanged with `RMFMC` swapped for `MFMC`:

```python
from elyza.multifidelity.montecarlo import MFMC

mfmc = MFMC(evaluators=[lf, hf])
mfmc.get_pilots(jrand.PRNGKey(0), n_pilots=2000)
mfmc.get_matrix_coefs()
```

{class}`~elyza.multifidelity.montecarlo.MLMC` (Giles, 2015 [^giles2015])
takes a different approach entirely: rather than regression coefficients,
it sums independent telescoping corrections `E[f_l] - E[f_{l-1}]` between
adjacent levels. It is a drop-in replacement for `RMFMC`/`MFMC` above,
sharing the same `get_pilots` / `budget_alloc` / `evaluate` interface
(but no `get_matrix_coefs` step, since it has no regression coefficients
to compute).

## 4. MAGPI: a hierarchical surrogate

Where the estimators above target a single expectation,
{class}`~elyza.multifidelity.surrogate.MAGPI` fits a full surrogate at each
fidelity level, using the lower-fidelity surrogates' *predictions* as extra
input features for the level above. Each level needs its own
{class}`~elyza.surrogate.abstract.SupervisedDataset` and surrogate model:

```python
from elyza.surrogate import SupervisedDataset
from elyza.surrogate.gp import GaussianProcess, ARD, Constant
from elyza.optim import ADAM, ADAMOptions

lf_inputs = x.sample(jrand.PRNGKey(0), 60)
lf_outputs = lf.evaluate(lf_inputs)
lf_data = SupervisedDataset(input_data=[lf_inputs], output_data=lf_outputs, noise_var=1e-4)

hf_inputs = x.sample(jrand.PRNGKey(1), 15)
hf_outputs = hf.evaluate(hf_inputs)
hf_data = SupervisedDataset(input_data=[hf_inputs], output_data=hf_outputs, noise_var=1e-4)

lf_gp = GaussianProcess(input_dim=1, kernel_cls=ARD, mean_cls=Constant)
lf_gp.set_optimizer(ADAM, ADAMOptions(lr=1e-2, epochs=300))

# input_dim=2: this level's own input x, plus the lf surrogate's prediction
hf_gp = GaussianProcess(input_dim=2, kernel_cls=ARD, mean_cls=Constant)
hf_gp.set_optimizer(ADAM, ADAMOptions(lr=1e-2, epochs=300))
```

Assemble the {class}`~elyza.multifidelity.surrogate.MAGPI` model from the
datasets and evaluators, and assign each level's surrogate with
{meth}`~elyza.multifidelity.surrogate.MAGPI.set_surrogate`:

```python
from elyza.multifidelity.surrogate import MAGPI

magpi = MAGPI(data=[lf_data, hf_data], evaluators=[lf, hf])
magpi.set_surrogate(level=0, surrogate=lf_gp, full_cov=False)
magpi.set_surrogate(level=1, surrogate=hf_gp, full_cov=False)
```

Levels must be fit from lowest to highest fidelity, since fitting level
`l` needs predictions from every surrogate below it:

```python
magpi.fit(0)
magpi.fit(1)
```

{meth}`~elyza.multifidelity.surrogate.MAGPI.predict` chains through the same
lower-fidelity predictions automatically, so predicting at the top level
only requires the raw input:

```python
x_test = jnp.linspace(0.0, 1.0, 200).reshape(-1, 1)
mu, var = magpi.predict(x_test, level=1)   # each shape (200,)
```

## References

[^peherstorfer2016]: Peherstorfer, B., Willcox, K., & Gunzburger, M. (2016).
    Optimal Model Management for Multifidelity Monte Carlo Estimation.
    *SIAM Journal on Scientific Computing*, 38(5), A3163-A3194.
    https://doi.org/10.1137/15M1046472

[^giles2015]: Giles, M. B. (2015). Multilevel Monte Carlo methods.
    *Acta Numerica*, 24, 259-328.
    https://doi.org/10.1017/S096249291500001X
