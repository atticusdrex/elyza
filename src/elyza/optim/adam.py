"""Batched ADAM optimizer built on ``jax.lax.scan``.

Defines :class:`ADAMOptions` and :class:`ADAM`, a
:class:`~elyza.optim.abstract.BatchGradientOptimizer` implementation of the
ADAM update rule with support for per-parameter active/constraint masks
(via :func:`~elyza.optim.abstract.fill_pytree_spec`) and epoch-wise
mini-batching.
"""
# imports
from elyza.optim.abstract import OptimizerOptions, BatchGradientOptimizer, fill_pytree_spec
from elyza.util.imports import *
from jax.tree_util import tree_map, tree_leaves
from jax import lax

class ADAMOptions(OptimizerOptions):
    """Options which parameterize an ADAM optimizer.

    Attributes:
        p_init: Initial dictionary (pytree) of parameters.
        lr: Learning rate for gradient descent.
        epochs: Number of times we pass through the training data.
        batch_size: Number of training datapoints in a specific loss
            function evaluation.
        beta1: First momentum parameter.
        beta2: Second momentum parameter.
        active_params: A dictionary of the active parameters to optimize.
        constraints: A dictionary of constraints mapping from
            parameter to constraint function.
        verbose: Whether or not to print the results of the optimizer.
        eps: Small positive number to prevent division by zero.
        random_state: Random seed for replication.
        unroll: Whether or not to unroll the ``jax.lax.scan`` operation
            (``unroll=True``: long compilation times, faster execution
            times, high memory; ``unroll=k``: unroll in size-``k`` blocks;
            ``unroll=False``: short compile times, slower execution times,
            lower memory).
    """
    p_init : dict | jax.Array | None = Field(default = None, description = "initial dictionary of parameters")
    lr : float = Field(default = 1e-3, description = "learning rate for gradient descent")
    epochs : int = Field(default = 1, description = "number of times we pass through the training data")
    batch_size : int | None = Field(default = None, description = "number of training datapoints in a specific loss function evaluation")
    beta1 : float = Field(default = 0.9, description = "first momentum parameter")
    beta2 : float = Field(default = 0.999, description = "second momentum parameter")
    active_params : dict | None = Field(default = None, description = "a dictionary of the active parameters to optimize")
    constraints : dict | None = Field(default = None, description = "a dictionary of constraints mapping from parameter:constraint function")
    verbose : bool = Field(default = False, description = "whether or not to print the reuslts of the optimizer")
    eps : float = Field(default = 1e-8, description = "small positive number to prevent division by zero")
    random_state : int = Field(default = 42, description = "random seed for replication")
    unroll : int | bool = Field(default = False, description = "whether or not to unroll the jax.lax.scan operation (unroll=True: long compilation times, faster execution times, high memory, unroll = k: unroll for set size-k blocks of k loop steps, unroll = False: short compile times, slower execution times, lower memory)")

    def model_post_init(self, __context):
        """Validate that the option values are internally consistent.

        Raises:
            AssertionError: If ``lr``, ``epochs``, ``batch_size``, ``beta1``,
                or ``beta2`` hold invalid values.
        """
        assert self.lr > 0, "learning rate cannot be negative"
        assert self.epochs >= 1, "need at least one epoch"
        assert self.batch_size is None or self.batch_size >= 1, "batch size must be None or at least 1"
        assert self.beta1 > 0, "beta1 must be positive"
        assert self.beta2 > 0, "beta2 must be positive"


class ADAM(BatchGradientOptimizer):
    """Batch ADAM optimizer for running general gradient-based optimization scripts.

    Attributes:
        opts: Options for the optimizer.
    """
    opts : ADAMOptions | None = Field(default = None, description = "options for the optimizer")

    def model_post_init(self, __context):
        """Validate that valid options have been supplied.

        Raises:
            AssertionError: If ``opts`` is ``None``.
        """
        super().model_post_init(__context)
        assert self.opts is not None, "you must pass a valid instance of ADAMOptions()"

    def run(self, *data : list[jax.Array]):
        """Run ADAM over the given data for ``opts.epochs`` epochs.

        Everything needed to parameterize the estimator must already be in
        ``self.opts``. The ``*data`` is purely passed through to
        ``loss_grad_fn`` after the parameter pytree.

        Args:
            *data: Training arrays (e.g. ``X, Y``) sharing a leading sample
                dimension, batched internally according to
                ``opts.batch_size``.

        Returns:
            The optimized parameter pytree, with the same structure as
            ``opts.p_init``.

        Raises:
            AssertionError: If ``loss_grad_fn`` or ``opts.p_init`` has not
                been set.
        """
        # asserting the loss function has been set
        assert self.loss_grad_fn is not None, "you must specify a loss function"
        assert self.opts.p_init is not None, "must give initial parameter guess"


        # determining the batch size for this run (without mutating shared opts)
        batch_size = self.opts.batch_size if self.opts.batch_size is not None else data[0].shape[0]

        # generating the PRNG key
        key = jrand.PRNGKey(self.opts.random_state)

        # filling in unspecified constraints/active_params (identity constraint, active=True), matching the full p_init pytree structure
        self.opts.constraints = fill_pytree_spec(self.opts.p_init, self.opts.constraints, lambda y: y)
        self.opts.active_params = fill_pytree_spec(self.opts.p_init, self.opts.active_params, True)

        # initialize parameters
        p = deepcopy(self.opts.p_init)

        # initialize moment estimates
        m, s = {}, {}
        m, s = tree_map(lambda x: jnp.zeros_like(x), p), tree_map(lambda x: jnp.zeros_like(x), p)

        # initializing progress bar
        progress_bar = tqdm(range(self.opts.epochs)) if self.opts.verbose else range(self.opts.epochs)

        # setting per-epoch PRNG keys
        keys = jrand.split(key, self.opts.epochs)

        # initializing the carry object
        carry = {'p':p, 'm':m, 's':s, 'loss':0.0, 'iter_num':0}

        # define the scan function
        scan_fn = jit(lambda carry, batch: _batch_adam_scan(
            carry, batch, self.loss_grad_fn, self.opts.lr, self.opts.beta1, self.opts.beta2, self.opts.eps, self.opts.active_params, self.opts.constraints
        ))

        # main optimization loop
        for iter in progress_bar:
            batches = self._get_batches(keys[iter], batch_size, *data)
            unzipped_batches = zip(*batches)
            stacked_batches = tuple([jnp.stack(arg) for arg in unzipped_batches])

            # performing the lax scan
            carry, batch_losses = jax.lax.scan(scan_fn, carry, xs=stacked_batches, unroll = self.opts.unroll)

            # displaying the loss
            self.opts.verbose and progress_bar.set_postfix_str(f"avg. batch objective: {batch_losses.mean():.4e}")

        return deepcopy(carry['p'])

def _adam_update(active, constraint, param, m_val, s_val, step, lr, beta1, beta2, eps):
    """Apply the ADAM update rule to a single parameter leaf, then constrain it.

    Args:
        active: Whether this parameter should be updated at all.
        constraint: Function applied to the (possibly updated) parameter to
            enforce a domain constraint.
        param: Current parameter value.
        m_val: First-moment (mean) estimate for this parameter.
        s_val: Second-moment (uncentered variance) estimate for this parameter.
        step: Current step count (1-indexed), used for bias correction.
        lr: Learning rate.
        beta1: First momentum decay rate.
        beta2: Second momentum decay rate.
        eps: Small positive number to prevent division by zero.

    Returns:
        The updated (and constrained) parameter value.
    """
    # only updating the parameter if it's active
    if active:
        m_hat = m_val / (1 - beta1 ** step)
        s_hat = s_val / (1 - beta2 ** step)
        param -= lr * m_hat / (jnp.sqrt(s_hat) + eps)

    return constraint(param)

def _batch_adam_scan(carry, batch, loss_grad_fn : Callable, lr:float, beta1:float, beta2:float, eps:float, active_params:dict, constraints:dict):
    """Perform one ADAM parameter update over a single batch, for use in ``lax.scan``.

    Args:
        carry: Scan carry dict with keys ``p`` (parameters), ``m``
            (first-moment estimates), ``s`` (second-moment estimates),
            ``loss``, and ``iter_num``.
        batch: A tuple ``(Xbatch, Ybatch)`` for this scan step.
        loss_grad_fn: Function returning ``(loss, grad)`` for the current
            parameters and batch.
        lr: Learning rate.
        beta1: First momentum decay rate.
        beta2: Second momentum decay rate.
        eps: Small positive number to prevent division by zero.
        active_params: Pytree of booleans, matching ``p``, indicating which
            parameters are updated.
        constraints: Pytree of constraint functions, matching ``p``, applied
            after each update.

    Returns:
        tuple: ``(carry, loss)`` -- the updated carry dict and this batch's
        loss, as required by ``lax.scan``.
    """
    # extract batches
    Xbatch, Ybatch = batch

    # obtaining the loss function and gradient
    loss, grad = loss_grad_fn(Xbatch, Ybatch, carry['p'])

    # setting the loss for this batch
    carry['loss'] = loss

    # updating the moment estimates
    carry['m'] = tree_map(lambda m_val, grad_val: beta1 * m_val + (1 - beta1) * grad_val, carry['m'], grad)
    carry['s'] = tree_map(lambda s_val, grad_val: beta2 * s_val + (1 - beta2) * grad_val**2, carry['s'], grad)

    # updating the parameter estimates with constraints
    carry['p'] = tree_map(
        lambda active, constraint, param, m_val, s_val: _adam_update(
            active, constraint, param, m_val, s_val, carry['iter_num']+1, lr, beta1, beta2, eps
        ),
        active_params, constraints, carry['p'], carry['m'], carry['s'])

    # updating the iter number
    carry['iter_num'] += 1

    # displaying the loss
    return carry, carry['loss']
