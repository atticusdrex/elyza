Surrogate Modeling Quickstart
==============================

This page walks through the core surrogate-modeling workflow in ``elyza``:
describing an input, wrapping a model as an :class:`~elyza.core.evaluator.Evaluator`,
generating training data, and fitting a surrogate. It ends with a short
comparison of the three surrogate models the library ships with.

Every example below runs as-is against a clean checkout.

1. Describe the input
----------------------

Every surrogate-modeling workflow starts with an :class:`~elyza.core.data.Input`.
For a single scalar variable, use :class:`~elyza.core.data.ScalarInput` and give
it a ``sampling_func`` that turns a JAX PRNG key into one draw:

.. code-block:: python

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

``x.sample(key, n_points)`` draws a batch of ``n_points`` samples, splitting
``key`` internally so every point gets its own subkey:

.. code-block:: python

    key = jrand.PRNGKey(0)
    X_train = x.sample(key, 20)   # shape (20, 1)

2. Wrap the model as an Evaluator
----------------------------------

An :class:`~elyza.core.evaluator.Evaluator` pairs an ``evaluation_func`` with
the inputs it consumes, and vectorizes it over a batch via ``vmap`` under the
hood. Here it stands in for a "ground truth" computer model you'd like to
approximate:

.. code-block:: python

    from elyza.core.evaluator import Evaluator

    evaluator = Evaluator(
        name="sinusoid",
        inputs=[x],
        output_dim=1,
        evaluation_func=lambda x: jnp.sin(2 * jnp.pi * x),
    )

    Y_train = evaluator.evaluate(X_train)   # shape (20, 1)

3. Fit a Gaussian Process surrogate
-------------------------------------

:class:`~elyza.surrogate.gp.gp.GaussianProcess` is the flagship surrogate:
it takes a kernel and a mean function, and fits their hyperparameters by
maximizing the log-marginal-likelihood with a pluggable
:mod:`~elyza.optim` optimizer.

.. code-block:: python

    from elyza.surrogate.gp import GaussianProcess, ARD, Constant
    from elyza.optim import ADAM, ADAMOptions

    gp = GaussianProcess(input_dim=1, kernel_cls=ARD, mean_cls=Constant)
    gp.set_optimizer(ADAM, ADAMOptions(lr=1e-2, epochs=300))
    gp.fit(X_train, Y_train)

Every surrogate in ``elyza`` shares the same ``fit`` / ``predict`` /
``sample`` interface (see :class:`~elyza.surrogate.abstract.Surrogate`).
``predict`` returns a mean and, for a GP, either the marginal variance or the
full posterior covariance:

.. code-block:: python

    X_test = jnp.linspace(0.0, 1.0, 5).reshape(-1, 1)
    mu, var = gp.predict(X_test)              # marginal variance
    mu, cov = gp.predict(X_test, full_cov=True)  # full posterior covariance

    samples = gp.sample(jrand.PRNGKey(1), X_test, n_samples=3)  # (5, 3)

On this problem the fit is essentially exact -- with 20 training points the
posterior mean tracks ``sin(2*pi*x)`` to within ``1e-5`` and the posterior
variance collapses to near zero.

New data can be folded in afterward without refitting from scratch, via a
rank-``m`` Cholesky update:

.. code-block:: python

    X_new = x.sample(jrand.PRNGKey(2), 5)
    Y_new = evaluator.evaluate(X_new)
    gp.update(X_new, Y_new)

4. Other surrogates
----------------------

Every :class:`~elyza.surrogate.abstract.Surrogate` subclass is a drop-in
replacement for the GP above -- only construction and fitting differ.

**MLPRegressor** -- a feedforward network, also trained with
:mod:`~elyza.optim`:

.. code-block:: python

    from elyza.surrogate.dnn import MLPRegressor

    mlp = MLPRegressor(input_dim=1, output_dim=1, hidden_dims=(32, 32))
    mlp.set_optimizer(ADAM, ADAMOptions(lr=3e-2, epochs=3000))
    mlp.fit(X_train, Y_train)
    mlp.predict(X_test)

``MLPRegressor`` does not implement ``sample`` -- call ``predict`` for a
point estimate.

**Ridge** -- closed-form L2-regularized linear regression, with no optimizer
to configure:

.. code-block:: python

    from elyza.surrogate.linear import Ridge

    ridge = Ridge(l2_reg=1e-3)
    ridge.fit(X_train, Y_train)
    ridge.predict(X_test)

``Ridge`` fits a straight line, so it underfits a sinusoid by construction --
a reminder to reach for the GP or MLP once the response is nonlinear.
``Ridge`` also has no optimizer or posterior: ``set_optimizer``, ``update``,
and ``sample`` all raise ``NotImplementedError``.
