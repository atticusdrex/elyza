# imports
from elyza.optim.abstract import OptimizerOptions, BatchGradientOptimizer, fill_pytree_spec
from elyza.util.imports import *
from jax.tree_util import tree_map, tree_leaves
from jax import lax

'''
LBFGSOptions
------------
the options which parameterize an L-BFGS optimizer
'''
class LBFGSOptions(OptimizerOptions):
    p_init : dict | jax.Array | None = Field(default = None, description = "initial dictionary of parameters")
    lr : float = Field(default = 1.0, description = "initial step size scale for the backtracking line search")
    epochs : int = Field(default = 1, description = "number of times we pass through the training data")
    batch_size : int | None = Field(default = None, description = "number of training datapoints in a specific loss function evaluation")
    m : int = Field(default = 10, description = "number of (s,y) curvature pairs retained for the two-loop recursion")
    max_backtracks : int = Field(default = 30, description = "maximum number of step-halving attempts in the backtracking line search")
    active_params : dict[str,bool] | None = Field(default = None, description = "a dictionary of the active parameters to optimize")
    constraints : dict[str,Callable] | None = Field(default = None, description = "a dictionary of constraints mapping from parameter:constraint function")
    verbose : bool = Field(default = False, description = "whether or not to print the reuslts of the optimizer")
    eps : float = Field(default = 1e-8, description = "small positive number to prevent division by zero")
    random_state : int = Field(default = 42, description = "random seed for replication")
    unroll : int | bool = Field(default = False, description = "whether or not to unroll the jax.lax.scan operation (unroll=True: long compilation times, faster execution times, high memory, unroll = k: unroll for set size-k blocks of k loop steps, unroll = False: short compile times, slower execution times, lower memory)")

    def model_post_init(self, __context):
        assert self.p_init is not None, "must give initial parameter guess"
        assert self.lr > 0, "learning rate cannot be negative"
        assert self.epochs >= 1, "need at least one epoch"
        assert self.batch_size is None or self.batch_size >= 1, "batch size must be None or at least 1"
        assert self.m >= 1, "history size m must be at least 1"
        assert self.max_backtracks >= 1, "must allow at least one backtracking attempt"


'''
Batch LBFGS optimizer
----------------------
class for running quasi-Newton (L-BFGS) optimization scripts in batches
'''
class LBFGS(BatchGradientOptimizer):
    opts : LBFGSOptions | None = Field(default = None, description = "options for the optimizer")

    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert self.opts is not None, "you must pass a valid instance of LBFGSOptions()"

    '''
    everything needed to parameterize the estimator must already be in the self.opts variable. the *args is purely just to pass into the loss functions after p
    '''
    def run(self, *data : list[jax.Array]):
        # asserting the loss function has been set
        assert self.loss_grad_fn is not None, "you must specify a loss function"

        # setting the default batch size
        if self.opts.batch_size is None:
            self.opts.batch_size = data[0].shape[0]

        # generating the PRNG key
        key = jrand.PRNGKey(self.opts.random_state)

        # filling in unspecified constraints/active_params (identity constraint, active=True), matching the full p_init pytree structure
        self.opts.constraints = fill_pytree_spec(self.opts.p_init, self.opts.constraints, lambda y: y)
        self.opts.active_params = fill_pytree_spec(self.opts.p_init, self.opts.active_params, True)

        # flattening the parameter pytree into a single vector
        p_flat, unravel_fn = flatten_util.ravel_pytree(deepcopy(self.opts.p_init))

        # flattening the active-parameter mask to match
        mask_flat, _ = flatten_util.ravel_pytree(
            tree_map(lambda active, param: jnp.full_like(param, 1.0 if active else 0.0), self.opts.active_params, self.opts.p_init)
        )

        # initializing the curvature-pair history buffers
        n = p_flat.shape[0]
        s_buf, y_buf = jnp.zeros((self.opts.m, n), dtype=p_flat.dtype), jnp.zeros((self.opts.m, n), dtype=p_flat.dtype)
        rho_buf = jnp.zeros((self.opts.m,), dtype=p_flat.dtype)

        # initializing progress bar
        progress_bar = tqdm(range(self.opts.epochs)) if self.opts.verbose else range(self.opts.epochs)

        # setting per-epoch PRNG keys
        keys = jrand.split(key, self.opts.epochs)

        # initializing the carry object
        carry = {'p_flat':p_flat, 's_buf':s_buf, 'y_buf':y_buf, 'rho_buf':rho_buf, 'loss':0.0}

        # define the scan function
        scan_fn = jit(lambda carry, batch: _batch_lbfgs_scan(
            carry, batch, self.loss_grad_fn, unravel_fn, mask_flat, self.opts.lr, self.opts.max_backtracks, self.opts.eps, self.opts.constraints
        ))

        # main optimization loop
        for iter in progress_bar:
            batches = self._get_batches(keys[iter], self.opts.batch_size, *data)
            unzipped_batches = zip(*batches)
            stacked_batches = tuple([jnp.stack(arg) for arg in unzipped_batches])

            # performing the lax scan
            carry, batch_losses = jax.lax.scan(scan_fn, carry, xs=stacked_batches, unroll = self.opts.unroll)

            # displaying the loss
            self.opts.verbose and progress_bar.set_postfix_str(f"avg. batch objective: {batch_losses.mean():.4e}")

        return deepcopy(unravel_fn(carry['p_flat']))

'''
Standard L-BFGS two-loop recursion (Nocedal & Wright, Algorithm 7.4) approximating
-H_k @ grad_flat from the last `m` (s, y) curvature pairs, without ever forming the
(inverse) Hessian explicitly. Operates on fixed-size (m, n) buffers so the whole step
can be compiled into a single XLA program via lax.scan. Slot i holds a real pair iff
rho_buf[i] != 0; empty/invalid slots contribute exactly zero (alpha=0, beta=0), so no
separate validity mask is needed. Index -1 is always the most recently added pair
(see _lbfgs_update_history).
'''
def _lbfgs_two_loop_recursion(grad_flat, s_buf, y_buf, rho_buf, eps):
    m = s_buf.shape[0]

    def backward(i, carry):
        q, alphas = carry
        idx = m - 1 - i
        alpha_i = rho_buf[idx] * jnp.dot(s_buf[idx], q)
        q = q - alpha_i * y_buf[idx]
        alphas = alphas.at[idx].set(alpha_i)
        return q, alphas

    q, alphas = lax.fori_loop(0, m, backward, (grad_flat, jnp.zeros((m,), dtype=grad_flat.dtype)))

    has_history = rho_buf[-1] != 0.0
    gamma = jnp.where(
        has_history,
        jnp.dot(s_buf[-1], y_buf[-1]) / (jnp.dot(y_buf[-1], y_buf[-1]) + eps),
        jnp.asarray(1.0, dtype=grad_flat.dtype)
    )

    def forward(idx, r):
        beta_i = rho_buf[idx] * jnp.dot(y_buf[idx], r)
        return r + s_buf[idx] * (alphas[idx] - beta_i)

    r = lax.fori_loop(0, m, forward, gamma * q)

    # no curvature pairs yet -- normalize the raw gradient instead of using it directly,
    # since it can be astronomically large near an ill-conditioned objective
    r = jnp.where(has_history, r, grad_flat / (jnp.linalg.norm(grad_flat) + eps))

    return -r

'''
the function for folding a new curvature pair into the fixed-size L-BFGS history
buffers, shifting out the oldest pair and appending (s_k, y_k) at the end -- unless
the curvature condition s^T y > eps is violated, in which case the buffers are left
completely untouched
'''
def _lbfgs_update_history(s_buf, y_buf, rho_buf, s_k, y_k, eps):
    sy = jnp.dot(s_k, y_k)
    curvature_ok = sy > eps

    s_shifted = jnp.concatenate([s_buf[1:], s_k[None]], axis=0)
    y_shifted = jnp.concatenate([y_buf[1:], y_k[None]], axis=0)
    safe_sy = jnp.where(curvature_ok, sy, jnp.asarray(1.0, dtype=sy.dtype))
    rho_shifted = jnp.concatenate([rho_buf[1:], (1.0 / safe_sy)[None]], axis=0)

    s_buf = jnp.where(curvature_ok, s_shifted, s_buf)
    y_buf = jnp.where(curvature_ok, y_shifted, y_buf)
    rho_buf = jnp.where(curvature_ok, rho_shifted, rho_buf)

    return s_buf, y_buf, rho_buf

'''
the function for performing parameter updates on a single batch: forms the L-BFGS
search direction from the current curvature history, backtracks along it (Armijo
sufficient-decrease line search) until a step is accepted, and folds the accepted
step into the curvature-pair history for the next call
'''
def _batch_lbfgs_scan(carry, batch, loss_grad_fn : Callable, unravel_fn : Callable, mask_flat : jax.Array, lr:float, max_backtracks:int, eps:float, constraints:dict):
    # extract batches
    Xbatch, Ybatch = batch

    # evaluates the loss/gradient at a flattened parameter vector, masking out inactive parameters
    def _evaluate(vec):
        loss, grad = loss_grad_fn(Xbatch, Ybatch, unravel_fn(vec))
        grad_flat, _ = flatten_util.ravel_pytree(grad)
        return loss, grad_flat * mask_flat

    p_flat = carry['p_flat']
    loss, grad_flat = _evaluate(p_flat)

    # setting the loss for this batch
    carry['loss'] = loss

    # approximate Newton direction via the two-loop recursion
    direction = _lbfgs_two_loop_recursion(grad_flat, carry['s_buf'], carry['y_buf'], carry['rho_buf'], eps) * mask_flat
    directional_derivative = jnp.dot(grad_flat, direction)
    direction_ok = directional_derivative < 0.0  # only search along genuine descent directions

    # backtracking the step scale until we find a sufficient decrease in the objective
    def bt_cond(state):
        _, attempt, accepted, *_ = state
        return direction_ok & jnp.logical_not(accepted) & (attempt < max_backtracks)

    def bt_body(state):
        step_scale, attempt, _, _, _, _ = state
        p_new = tree_map(lambda c, prm: c(prm), constraints, unravel_fn(p_flat + step_scale * lr * direction))
        p_flat_trial, _ = flatten_util.ravel_pytree(p_new)
        loss_trial, grad_flat_trial = _evaluate(p_flat_trial)

        sufficient_decrease = loss_trial <= loss + 1e-4 * step_scale * lr * directional_derivative
        next_step_scale = jnp.where(sufficient_decrease, step_scale, step_scale * 0.5)
        return (next_step_scale, attempt + 1, sufficient_decrease, p_flat_trial, loss_trial, grad_flat_trial)

    init_bt = (
        jnp.asarray(1.0, dtype=p_flat.dtype), jnp.asarray(0),
        jnp.asarray(False), p_flat, loss, grad_flat
    )
    _, _, accepted, p_flat_trial, loss_trial, grad_flat_trial = lax.while_loop(bt_cond, bt_body, init_bt)

    # updating the curvature-pair history with the step actually taken
    s_k, y_k = p_flat_trial - p_flat, grad_flat_trial - grad_flat
    s_buf_new, y_buf_new, rho_buf_new = _lbfgs_update_history(carry['s_buf'], carry['y_buf'], carry['rho_buf'], s_k, y_k, eps)

    # this search direction was unusable -- drop the curvature history so the next
    # direction is a fresh steepest-descent guess, and hold the last known-good parameters
    carry['s_buf'] = jnp.where(accepted, s_buf_new, jnp.zeros_like(carry['s_buf']))
    carry['y_buf'] = jnp.where(accepted, y_buf_new, jnp.zeros_like(carry['y_buf']))
    carry['rho_buf'] = jnp.where(accepted, rho_buf_new, jnp.zeros_like(carry['rho_buf']))
    carry['p_flat'] = jnp.where(accepted, p_flat_trial, p_flat)
    carry['loss'] = jnp.where(accepted, loss_trial, loss)

    # displaying the loss
    return carry, carry['loss']
