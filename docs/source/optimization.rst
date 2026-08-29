Optimization Quickstart
==============================

``elyza.optim`` provides the gradient-based optimizers used to fit
surrogate model parameters -- :class:`~elyza.optim.adam.ADAM` and
:class:`~elyza.optim.lbfgs.LBFGS`, both implementing the shared
:class:`~elyza.optim.abstract.BatchGradientOptimizer` interface. Surrogates
like :class:`~elyza.surrogate.gp.gp.GaussianProcess` use them internally via
``set_optimizer`` (see the :doc:`surrogate quickstart <surrogate>`), but they
work standalone on any pytree of parameters too.

Every example below runs as-is against a clean checkout.

1. Fit parameters with ADAM
------------------------------

An optimizer needs two things: a pytree of initial parameters (``p_init``)
and a ``loss_grad_fn`` in the form ``(Xbatch, Ybatch, p) -> (loss, grad)``.
Here's a plain linear fit ``y = w*x + b``:

.. code-block:: python

    import jax.numpy as jnp
    import jax.random as jrand
    from jax import value_and_grad, jit

    from elyza.optim import ADAM, ADAMOptions

    key = jrand.PRNGKey(0)
    X = jrand.uniform(key, (200, 1), minval=0.0, maxval=1.0)
    Y = 3.0 * X + 0.5 + 0.05 * jrand.normal(jrand.PRNGKey(1), (200, 1))

    def loss_fn(p, X, Y):
        preds = p["w"] * X + p["b"]
        return jnp.mean((preds - Y) ** 2)

    p_init = {"w": jnp.array(0.0), "b": jnp.array(0.0)}

    adam = ADAM(opts=ADAMOptions(p_init=p_init, lr=1e-1, epochs=200))
    adam.loss_grad_fn = jit(value_and_grad(lambda X, Y, p: loss_fn(p, X, Y), argnums=2))

    p_fit = adam.run(X, Y)   # {'w': ~3.0, 'b': ~0.5}

``ADAMOptions.batch_size`` defaults to the full dataset (full-batch
gradient descent); set it to mini-batch instead. ``run`` shuffles and
batches ``X``/``Y`` internally for every epoch.

2. Freezing and constraining parameters
------------------------------------------

``active_params`` freezes individual leaves of the parameter pytree, and
``constraints`` applies a function to a leaf after every update -- useful
for e.g. keeping a noise variance positive. Both accept a value for any
subset of ``p_init``'s structure; anything unspecified defaults to
active/unconstrained:

.. code-block:: python

    p_init = {"w": jnp.array(-1.0), "b": jnp.array(10.0)}

    adam = ADAM(opts=ADAMOptions(
        p_init=p_init, lr=1e-1, epochs=200,
        active_params={"w": True, "b": False},          # freeze b
        constraints={"w": lambda w: jnp.maximum(w, 0.0)},  # keep w >= 0
    ))
    adam.loss_grad_fn = jit(value_and_grad(lambda X, Y, p: loss_fn(p, X, Y), argnums=2))

    p_fit = adam.run(X, Y)   # b stays 10.0; w is clamped at 0.0

3. LBFGS as a drop-in replacement
-------------------------------------

:class:`~elyza.optim.lbfgs.LBFGS` shares the exact same
``opts``/``loss_grad_fn``/``run`` interface, so switching from ADAM only
means swapping the class and its options. It uses a two-loop recursion
over recent curvature pairs plus a backtracking line search rather than
ADAM's per-parameter moment estimates, and its ``lr`` is an initial step
scale for that line search rather than a fixed learning rate:

.. code-block:: python

    from elyza.optim import LBFGS, LBFGSOptions

    lbfgs = LBFGS(opts=LBFGSOptions(p_init=p_init, lr=1.0, epochs=20))
    lbfgs.loss_grad_fn = jit(value_and_grad(lambda X, Y, p: loss_fn(p, X, Y), argnums=2))

    p_fit = lbfgs.run(X, Y)

LBFGS typically needs far fewer epochs than ADAM to reach the same
accuracy, at the cost of a pricier per-step line search -- a good default
to reach for once ADAM's convergence feels too slow.

4. Inside a surrogate model
-------------------------------

Every surrogate's ``set_optimizer(optimizer_cls, optimizer_opts)`` builds
``loss_grad_fn`` for you from its own internal objective (e.g. negative
log-marginal-likelihood for a GP) and stores it for ``fit``/``update`` to
call -- so the pattern above is exactly what's running under the hood:

.. code-block:: python

    from elyza.surrogate.gp import GaussianProcess, ARD, Constant

    gp = GaussianProcess(input_dim=1, kernel_cls=ARD, mean_cls=Constant)
    gp.set_optimizer(ADAM, ADAMOptions(lr=1e-2, epochs=300))
    gp.fit(X, Y)
