---
file_format: mystnb
kernelspec:
  name: python3
---

# The Basics: Inputs and Evaluators

In this page, we give an overview of the main functional building blocks within `elyza`. These include the {class}`~elyza.core.data.Input` class for defining arbitrary scalar- and vector-valued system inputs and wrapping existing functions as an {class}`~elyza.core.evaluator.Evaluator` class. 

## 1. Defining inputs

Every surrogate-modeling workflow starts with an {class}`~elyza.core.data.Input`.
For a single scalar variable, use {class}`~elyza.core.data.ScalarInput` and give
it a `sampling_func` that turns a JAX PRNG key into one draw:

```{code-cell} python
import jax.numpy as jnp
import jax.random as jrand

from elyza.core.data import ScalarInput

x = ScalarInput(
    name="x",
    dim=1,
    sampling_func=lambda key: jrand.uniform(key, minval=0.0, maxval=1.0),
    minval=0.0,
    maxval=1.0,
)

y = ScalarInput(
    name="y",
    dim=1,
    sampling_func=lambda key: jrand.uniform(key, minval=0.0, maxval=1.0),
    minval=0.0,
    maxval=1.0,
)
```

This is a simple example, but the input sampling could be a highly complex predefined function (e.g., quasi Monte Carlo strategies). The `Input` abstract class provides a standard interface to manage different inputs which may have their own unique sampling routines. `x.sample(key, n_points)` draws a batch of `n_points` samples, splitting
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
