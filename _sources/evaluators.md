---
file_format: mystnb
kernelspec:
  name: python3
---

# Variables and Evaluators

In this page, we give an overview of the main functional building blocks within `elyza`. These include the {class}`~elyza.core.data.Variable` class for defining arbitrary scalar- and vector-valued system inputs and wrapping existing functions as an {class}`~elyza.core.evaluator.Evaluator` class. 

## 1. Defining Variables

Every surrogate-modeling workflow starts with a {class}`~elyza.core.data.Variable`.
For a single scalar variable sampled uniformly over some range, use
{class}`~elyza.core.random.Uniform`:

```{code-cell} python
import jax.numpy as jnp
import jax.random as jrand

from elyza.core.random import Uniform

x = Uniform(
    name="x",
    dim=1,
    lower=0.0,
    upper=1.0,
)

y = Uniform(
    name="y",
    dim=1,
    lower=0.0,
    upper=1.0,
)
```

This is a simple example, but the input sampling could be a highly complex predefined distribution (e.g., quasi Monte Carlo strategies). The `RandomVariable` abstract class provides a standard interface to manage different inputs which may have their own unique sampling routines. `x.sample(key, n_points)` draws a batch of `n_points` samples, splitting
`key` internally so every point gets its own subkey:

```{code-cell} python
key = jrand.PRNGKey(0)
X_train = x.sample(key, 20)   # shape (20, 1)
Y_train = y.sample(key, 20)

print(X_train.shape, Y_train.shape)
```

## 2. Defining evaluators

An {class}`~elyza.core.evaluator.Evaluator` pairs an `evaluation_func` with
the inputs it consumes, and vectorizes it over a batch via `vmap` under the
hood. Here it stands in for a "ground truth" computer model you'd like to
approximate:

```{code-cell} python
from elyza.core.evaluator import Evaluator

evaluator = Evaluator(
    name="sinusoid",
    inputs=[x, y],
    output_dim=1,
    evaluation_func=lambda x, y: jnp.sin(2 * jnp.pi * x) + jnp.cos(2*jnp.pi * y),
)

Z_train = evaluator.evaluate(X_train, Y_train)   # shape (20, 1)
print(Z_train)
```

The evaluator is meant to serve as a standard wrapper for any function which maps a set of inputs to some numerical quantity of interest. This is meant to wrap complex engineering simulations and integrate them into the standardized `elyza` environment. Each evaluator has a built-in `print()` method which lists all the attributes associated with it:

```{code-cell} python
evaluator.print()
```
