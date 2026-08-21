from elyza.util.imports import *
from jax.tree_util import tree_map, tree_leaves
from jax import lax


'''
True iff every leaf of a scalar/array/pytree is finite. Used to guard optimizer
state against a NaN/Inf objective or gradient (e.g. from a near-singular kernel
matrix) so that a single bad step can't permanently corrupt the parameters --
a rejected step just leaves the last known-good parameters untouched.

This eager version forces a concrete bool and is for use OUTSIDE any jit/scan
(e.g. a one-time check on p_init before entering the compiled loop). Inside a
traced/scanned step, use _is_finite_traced instead, which stays a JAX value.
'''
def _is_finite(tree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in tree_leaves(tree))

def _is_finite_traced(tree) -> jax.Array:
    result = jnp.array(True)
    for leaf in tree_leaves(tree):
        result = jnp.logical_and(result, jnp.all(jnp.isfinite(leaf)))
    return result


class GradientOptimizer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    loss_grad_fn : SkipValidation[callable] | None = Field(default = None, description = "loss and gradient function from jax.value_and_grad() which returns the parameter gradients based on some inputs")
    constraints: dict | None = Field(default = None, description = "a dictionary which maps param_name : function to determine how to constrain the parameter e.g. lambda x: jnp.clip(x, 0.0, 1.0)")

    def run(self, **kwargs): 
        raise NotImplementedError("this method is for abstract purposes only")

class BatchGradientOptimizer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    loss_grad_fn : SkipValidation[callable] 
    '''
    The main difference between the batch gradient optimizer is that the loss_grad_fn takes in Xbatch and Ybatch arguments such that a call should resemble: 

    loss, grad = loss_grad_fn(p, Xbatch, Ybatch) 

    This way the gradients are computed only on a subset of the full data
    '''
    constraints: dict | None = None 

    '''Function to break up data into batches'''
    def _get_batches(self, key, X: jax.Array, Y: jax.Array, batch_size: int) -> list[tuple[jax.Array]]:
        n = X.shape[0]
        perm = jax.random.permutation(key, n)
        X_shuffled = X[perm]
        Y_shuffled = Y[perm]

        n_batches = n // batch_size  # drop last incomplete batch
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batches.append((X_shuffled[start:end], Y_shuffled[start:end]))

        return batches

class ADAM(GradientOptimizer):
    eps: float = 1e-8

    def run(self, lr : float, steps : int, p_init : dict, beta1 : float = 0.9, beta2 : float = 0.999, active_params : dict | None = None, verbose = True) -> dict:
        def param_update(active, constraint, param, m_val, s_val, step):
            # only updating the parameter if it's active 
            if active: 
                m_hat = m_val / (1 - beta1 ** step)
                s_hat = s_val / (1 - beta2 ** step)
                param -= lr * m_hat / (jnp.sqrt(s_hat) + self.eps)

            return constraint(param)
        
        # fill in identity constraints for any params not given an explicit constraint
        default_constraints = tree_map(lambda x: lambda y: y, p_init) # use identity constraints
        self.constraints = {**default_constraints, **(self.constraints or {})}

        # determining which parameters are active
        if active_params is None:
            active_params = tree_map(lambda _: True, p_init)

        # initialize parameters
        p = deepcopy(p_init)

        # initialize moment estimates
        m, s = {}, {}
        m, s = tree_map(lambda x: jnp.zeros_like(x), p), tree_map(lambda x: jnp.zeros_like(x), p)

        # initializing iterator object
        iterator = tqdm(range(steps)) if verbose else range(steps)

        # main optimization loop
        for iter in iterator:
            # obtaining the loss function and gradient
            loss, grad = self.loss_grad_fn(p)

            # a non-finite objective/gradient (e.g. from a near-singular kernel matrix)
            # can't be trusted to produce a sane update -- reject the step and keep the
            # last known-good p/m/s rather than letting NaNs propagate into them
            if not (_is_finite(loss) and _is_finite(grad)):
                verbose and iterator.set_postfix_str(f"Objective: {loss:.4e} (step rejected: non-finite)")
                continue

            # updating the moment estimates
            m_new = tree_map(lambda m_val, grad_val: beta1 * m_val + (1 - beta1) * grad_val, m, grad)
            s_new = tree_map(lambda s_val, grad_val: beta2 * s_val + (1 - beta2) * grad_val**2, s, grad)

            # updating the parameter estimates with constraints
            p_new = tree_map(
                lambda active, constraint, param, m_val, s_val: param_update(
                    active, constraint, param, m_val, s_val, iter+1
                ),
                active_params, self.constraints, p, m_new, s_new)

            # guard against the update itself (e.g. a constraint fn) introducing non-finite values
            if not _is_finite(p_new):
                verbose and iterator.set_postfix_str(f"Objective: {loss:.4e} (step rejected: non-finite update)")
                continue

            p, m, s = p_new, m_new, s_new

            # displaying the loss
            verbose and iterator.set_postfix_str(f"Objective: {loss:.4e}")

        # returning the final parameter estimates
        return deepcopy(p)

'''
Standard L-BFGS two-loop recursion (Nocedal & Wright, Algorithm 7.4) approximating
-H_k @ grad_flat from the last `m` (s, y) curvature pairs, without ever forming
the (inverse) Hessian explicitly. Shared by LBFGS and BatchLBFGS below.
'''
def _lbfgs_two_loop_recursion(grad_flat, s_history, y_history, rho_history, eps):
    q = grad_flat
    alphas = []
    for s, y, rho in zip(reversed(s_history), reversed(y_history), reversed(rho_history)):
        alpha = rho * jnp.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)

    if s_history:
        s_last, y_last = s_history[-1], y_history[-1]
        gamma = jnp.dot(s_last, y_last) / (jnp.dot(y_last, y_last) + eps)
        r = gamma * q
    else:
        # no curvature pairs yet (the very first iteration, or right after a reset) --
        # q is just the raw gradient here, which can be astronomically large near an
        # ill-conditioned objective (e.g. a near-singular kernel matrix). Normalize to
        # a unit-length direction so `lr` controls an actual step length instead of
        # being multiplied by however large the raw gradient happens to be.
        r = q / (jnp.linalg.norm(q) + eps)
    for (s, y, rho), alpha in zip(zip(s_history, y_history, rho_history), reversed(alphas)):
        beta = rho * jnp.dot(y, r)
        r = r + s * (alpha - beta)

    return -r

'''
Appends a new curvature pair to the L-BFGS history (in place), skipping pairs that
violate the curvature condition s^T y > eps, and evicting the oldest pair once the
history exceeds `m` entries.
'''
def _lbfgs_update_history(s_history, y_history, rho_history, s_k, y_k, m, eps):
    sy = jnp.dot(s_k, y_k)
    if sy > eps:
        s_history.append(s_k)
        y_history.append(y_k)
        rho_history.append(1.0 / sy)

        if len(s_history) > m:
            s_history.pop(0)
            y_history.pop(0)
            rho_history.pop(0)

'''
Fixed-size-buffer analog of _lbfgs_two_loop_recursion: same recursion, but the
curvature history lives in (m, n) arrays instead of Python lists, so the whole
step -- including this recursion -- can be compiled into a single XLA program
via lax.scan/lax.while_loop rather than looping in Python. Slot i holds a real
pair iff rho_buf[i] != 0; empty/invalid slots contribute exactly zero (alpha=0,
beta=0), so no separate validity mask is needed. Index -1 is always the most
recently added pair (see _lbfgs_update_history_buf).
'''
def _lbfgs_two_loop_recursion_buf(grad_flat, s_buf, y_buf, rho_buf, eps):
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
Fixed-size-buffer analog of _lbfgs_update_history: shifts out the oldest pair and
appends (s_k, y_k) at the end if the curvature condition s^T y > eps holds;
otherwise leaves the buffers completely untouched (matching the list version's
silent skip of pairs that violate the curvature condition).
'''
def _lbfgs_update_history_buf(s_buf, y_buf, rho_buf, s_k, y_k, eps):
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

class LBFGS(GradientOptimizer):
    m: int = 10
    eps: float = 1e-8

    '''
    Same algorithm as before (two-loop recursion + Armijo backtracking line search +
    NaN/no-decrease step rejection), but compiled into a single XLA program instead of
    a Python for-loop: the curvature history is a fixed-size (m, n) buffer rather than
    Python lists, the backtracking search is a lax.while_loop, and the outer per-step
    loop is a lax.scan. This turns "up to steps * 30 separate jitted-function dispatches
    from Python" into one compiled call. Progress reporting (tqdm) still works via
    jax.debug.callback, which fires a host-side callback from inside the compiled loop.
    '''
    def run(self, lr : float, steps : int, p_init : dict, active_params : dict | None = None, verbose = True):
        # fill in identity constraints for any params not given an explicit constraint
        default_constraints = tree_map(lambda x: lambda y: y, p_init) # use identity constraints
        self.constraints = {**default_constraints, **(self.constraints or {})}

        # determining which parameters are active
        if active_params is None:
            active_params = tree_map(lambda _: True, p_init)

        # flattening the parameter pytree into a single vector
        p_flat, unravel_fn = flatten_util.ravel_pytree(deepcopy(p_init))

        # flattening the active-parameter mask to match
        mask_flat, _ = flatten_util.ravel_pytree(
            tree_map(lambda active, param: jnp.full_like(param, 1.0 if active else 0.0), active_params, p_init)
        )

        # evaluates the loss/gradient at a flattened parameter vector, masking out inactive parameters
        def _evaluate(vec):
            loss, grad = self.loss_grad_fn(unravel_fn(vec))
            grad_flat, _ = flatten_util.ravel_pytree(grad)
            return loss, grad_flat * mask_flat

        loss, grad_flat = _evaluate(p_flat)

        # can't optimize from a broken starting point -- fail loudly rather than silently
        # returning/propagating NaN parameters. This check is eager/outside the compiled
        # loop, so a concrete bool() is fine here.
        if not (_is_finite(loss) and _is_finite(grad_flat)):
            raise FloatingPointError(
                "LBFGS.run: objective/gradient at p_init is non-finite (NaN/Inf); refusing to start."
            )

        n = p_flat.shape[0]
        s_buf = jnp.zeros((self.m, n), dtype=p_flat.dtype)
        y_buf = jnp.zeros((self.m, n), dtype=p_flat.dtype)
        rho_buf = jnp.zeros((self.m,), dtype=p_flat.dtype)

        constraints, max_backtracks, eps = self.constraints, 30, self.eps

        bar = tqdm(total=steps) if verbose else None

        def step(carry, _):
            p_flat, grad_flat, loss, s_buf, y_buf, rho_buf = carry

            # approximate Newton direction via the two-loop recursion
            direction = _lbfgs_two_loop_recursion_buf(grad_flat, s_buf, y_buf, rho_buf, eps) * mask_flat
            directional_derivative = jnp.dot(grad_flat, direction)
            direction_ok = directional_derivative < 0.0  # only search along genuine descent directions

            # backtrack the step size (Armijo sufficient-decrease line search) so a step is only
            # accepted if it's both finite AND actually improves the objective by a meaningful
            # amount. Without this, a technically-finite step near an ill-conditioned objective
            # (e.g. a near-singular kernel matrix) can still send the loss up by many orders of
            # magnitude and get accepted anyway, since "finite" alone doesn't mean "good".
            def bt_cond(state):
                _, attempt, accepted, *_ = state
                return direction_ok & jnp.logical_not(accepted) & (attempt < max_backtracks)

            def bt_body(state):
                step_scale, attempt, _, _, _, _ = state
                p_new = tree_map(lambda c, prm: c(prm), constraints, unravel_fn(p_flat + step_scale * lr * direction))
                p_flat_trial, _ = flatten_util.ravel_pytree(p_new)
                loss_trial, grad_flat_trial = _evaluate(p_flat_trial)

                sufficient_decrease = loss_trial <= loss + 1e-4 * step_scale * lr * directional_derivative
                ok = _is_finite_traced(loss_trial) & _is_finite_traced(grad_flat_trial) & sufficient_decrease
                next_step_scale = jnp.where(ok, step_scale, step_scale * 0.5)
                return (next_step_scale, attempt + 1, ok, p_flat_trial, loss_trial, grad_flat_trial)

            init_bt = (
                jnp.asarray(1.0, dtype=p_flat.dtype), jnp.asarray(0),
                jnp.asarray(False), p_flat, loss, grad_flat
            )
            _, _, accepted, p_flat_trial, loss_trial, grad_flat_trial = lax.while_loop(bt_cond, bt_body, init_bt)

            # updating the curvature-pair history (only takes effect if accepted, else
            # dropped entirely below -- matching "reset on reject" from the list version)
            s_k, y_k = p_flat_trial - p_flat, grad_flat_trial - grad_flat
            s_buf_new, y_buf_new, rho_buf_new = _lbfgs_update_history_buf(s_buf, y_buf, rho_buf, s_k, y_k, eps)

            # this search direction was unusable everywhere we tried -- drop the curvature
            # history (so the next direction is a fresh steepest-descent guess) and hold
            # the last known-good, finite parameters rather than corrupting them
            s_buf_out = jnp.where(accepted, s_buf_new, jnp.zeros_like(s_buf))
            y_buf_out = jnp.where(accepted, y_buf_new, jnp.zeros_like(y_buf))
            rho_buf_out = jnp.where(accepted, rho_buf_new, jnp.zeros_like(rho_buf))
            p_flat_out = jnp.where(accepted, p_flat_trial, p_flat)
            grad_flat_out = jnp.where(accepted, grad_flat_trial, grad_flat)
            loss_out = jnp.where(accepted, loss_trial, loss)

            new_carry = (p_flat_out, grad_flat_out, loss_out, s_buf_out, y_buf_out, rho_buf_out)
            return new_carry, (loss_out, accepted)

        # compiled once per distinct chunk length (almost always just once, for the common
        # full-size chunk, plus a second time for a shorter final remainder chunk) and reused
        # across chunks -- `length` must be static since lax.scan needs a concrete trip count.
        def _run_chunk(carry, length):
            return lax.scan(step, carry, xs=None, length=length)
        run_chunk = jit(_run_chunk, static_argnames=('length',))

        # Run the scan in chunks from the host instead of one single lax.scan over all
        # `steps`. Each chunk still compiles/executes as one XLA program (so we keep almost
        # all of the speedup), but progress reporting happens on the main thread between
        # chunks rather than via a per-iteration jax.debug.callback -- the callback approach
        # crashes long runs inside Jupyter, since ipykernel's stdout redirection opens a new
        # ZMQ socket per distinct thread that touches it, and JAX dispatches host callbacks
        # from its own runtime thread(s), so the sockets pile up until the process runs out
        # of file descriptors.
        chunk_size = min(50, steps) if steps > 0 else 0
        carry = (p_flat, grad_flat, loss, s_buf, y_buf, rho_buf)
        remaining = steps
        while remaining > 0:
            this_len = min(chunk_size, remaining)
            carry, (chunk_losses, chunk_accepted) = run_chunk(carry, this_len)
            remaining -= this_len

            if bar is not None:
                n_accepted = int(jnp.sum(chunk_accepted))
                bar.set_postfix_str(f"Objective: {float(carry[2]):.4e} (accepted {n_accepted}/{this_len} this chunk)")
                bar.update(this_len)

        p_flat = carry[0]

        if bar is not None:
            bar.close()

        # returning the final parameter estimates
        return unravel_fn(p_flat)


class BatchADAM(BatchGradientOptimizer):
    eps: float = 1e-8

    def run(self, key, X, Y, lr : float, epochs : int, batch_size : int, p_init : dict, beta1 : float = 0.9, beta2 : float = 0.999, active_params : dict | None = None, verbose = True):
        def param_update(active, constraint, param, m_val, s_val, step):
            # only updating the parameter if it's active 
            if active: 
                m_hat = m_val / (1 - beta1 ** step)
                s_hat = s_val / (1 - beta2 ** step)
                param -= lr * m_hat / (jnp.sqrt(s_hat) + self.eps)

            return constraint(param)
        
        # fill in identity constraints for any params not given an explicit constraint
        default_constraints = tree_map(lambda x: lambda y: y, p_init) # use identity constraints
        self.constraints = {**default_constraints, **(self.constraints or {})}

        # determining which parameters are active
        if active_params is None:
            active_params = tree_map(lambda _: True, p_init)

        # initialize parameters
        p = deepcopy(p_init)

        # initialize moment estimates
        m, s = {}, {}
        m, s = tree_map(lambda x: jnp.zeros_like(x), p), tree_map(lambda x: jnp.zeros_like(x), p)

        # initializing iterator object
        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        keys = jrand.split(key, epochs)



        # main optimization loop
        for iter in iterator:
            batches = self._get_batches(keys[iter], X, Y, batch_size) 

            batch_losses = np.zeros(len(batches))
            for i, (Xbatch, Ybatch) in enumerate(batches):
                # obtaining the loss function and gradient
                loss, grad = self.loss_grad_fn(p, Xbatch, Ybatch)

                # setting the loss for this batch
                batch_losses[i] = float(loss)

                # a non-finite objective/gradient on this batch can't be trusted -- reject the
                # step and keep the last known-good p/m/s rather than letting NaNs propagate
                if not (_is_finite(loss) and _is_finite(grad)):
                    continue

                # updating the moment estimates
                m_new = tree_map(lambda m_val, grad_val: beta1 * m_val + (1 - beta1) * grad_val, m, grad)
                s_new = tree_map(lambda s_val, grad_val: beta2 * s_val + (1 - beta2) * grad_val**2, s, grad)

                # updating the parameter estimates with constraints
                p_new = tree_map(
                    lambda active, constraint, param, m_val, s_val: param_update(
                        active, constraint, param, m_val, s_val, iter+1
                    ),
                    active_params, self.constraints, p, m_new, s_new)

                # guard against the update itself (e.g. a constraint fn) introducing non-finite values
                if not _is_finite(p_new):
                    continue

                p, m, s = p_new, m_new, s_new

            # displaying the loss
            verbose and iterator.set_postfix_str(f"Mean Objective: {batch_losses.mean():.4e}")

        # returning the final parameter estimates
        return deepcopy(p) 


class BatchSGD(BatchGradientOptimizer):
    lr: float = 1e-3

    def run(self, key, X, Y, lr : float, epochs : int, batch_size : int, p_init : dict, active_params : dict | None = None, verbose = True):
        def param_update(active, constraint, param, grad_val):
            # only updating the parameter if it's active 
            if active: 
                param -= lr * grad_val

            return constraint(param)
        
        # fill in identity constraints for any params not given an explicit constraint
        default_constraints = tree_map(lambda x: lambda y: y, p_init) # use identity constraints
        self.constraints = {**default_constraints, **(self.constraints or {})}

        # determining which parameters are active
        if active_params is None:
            active_params = tree_map(lambda _: True, p_init)

        # initialize parameters
        p = deepcopy(p_init)

        # initializing iterator object
        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        keys = jrand.split(key, epochs)

        # main optimization loop
        for iter in iterator:
            batches = self._get_batches(keys[iter], X, Y, batch_size)

            batch_losses = np.zeros(len(batches))
            for i, (Xbatch, Ybatch) in enumerate(batches):
                # obtaining the loss function and gradient
                loss, grad = self.loss_grad_fn(p, Xbatch, Ybatch)

                # setting the loss for this batch
                batch_losses[i] = float(loss)

                # a non-finite objective/gradient on this batch can't be trusted -- reject the
                # step and keep the last known-good p rather than letting NaNs propagate
                if not (_is_finite(loss) and _is_finite(grad)):
                    continue

                # updating the parameter estimates with constraints
                p_new = tree_map(
                    lambda active, constraint, param, grad_val: param_update(
                        active, constraint, param, grad_val
                    ),
                    active_params, self.constraints, p, grad)

                # guard against the update itself (e.g. a constraint fn) introducing non-finite values
                if not _is_finite(p_new):
                    continue

                p = p_new

            # displaying the loss
            verbose and iterator.set_postfix_str(f"Mean Objective: {batch_losses.mean():.4e}")

        # returning the final parameter estimates
        return deepcopy(p)


class BatchLBFGS(BatchGradientOptimizer):
    '''
    Stochastic L-BFGS for large datasets: the same two-loop recursion as LBFGS,
    but the loss/gradient (and each curvature pair) are evaluated on a fresh
    mini-batch at every step. Curvature pairs are only kept when the curvature
    condition s^T y > eps holds, which keeps noisy mini-batch gradients from
    corrupting the inverse-Hessian approximation.
    '''
    m: int = 10
    eps: float = 1e-8

    def run(self, key, X, Y, lr : float, epochs : int, batch_size : int, p_init : dict, active_params : dict | None = None, verbose = True):
        # fill in identity constraints for any params not given an explicit constraint
        default_constraints = tree_map(lambda x: lambda y: y, p_init) # use identity constraints
        self.constraints = {**default_constraints, **(self.constraints or {})}

        # determining which parameters are active
        if active_params is None:
            active_params = tree_map(lambda _: True, p_init)

        # flattening the parameter pytree into a single vector
        p_flat, unravel_fn = flatten_util.ravel_pytree(deepcopy(p_init))

        # flattening the active-parameter mask to match
        mask_flat, _ = flatten_util.ravel_pytree(
            tree_map(lambda active, param: jnp.full_like(param, 1.0 if active else 0.0), active_params, p_init)
        )

        # evaluates the loss/gradient at a flattened parameter vector on a given batch
        def _evaluate(vec, Xbatch, Ybatch):
            loss, grad = self.loss_grad_fn(unravel_fn(vec), Xbatch, Ybatch)
            grad_flat, _ = flatten_util.ravel_pytree(grad)
            return loss, grad_flat * mask_flat

        # curvature-pair history for the two-loop recursion, shared across batches/epochs
        s_history, y_history, rho_history = [], [], []

        # initializing iterator object
        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        keys = jrand.split(key, epochs)

        # main optimization loop
        for iter in iterator:
            batches = self._get_batches(keys[iter], X, Y, batch_size)

            batch_losses = np.zeros(len(batches))
            for i, (Xbatch, Ybatch) in enumerate(batches):
                # obtaining the loss/gradient on this batch
                loss, grad_flat = _evaluate(p_flat, Xbatch, Ybatch)

                if not (_is_finite(loss) and _is_finite(grad_flat)):
                    # can't form a trustworthy direction from a non-finite loss/gradient -- drop
                    # the curvature history and hold the last known-good parameters for this batch
                    s_history.clear()
                    y_history.clear()
                    rho_history.clear()
                    batch_losses[i] = float('nan')
                    continue

                # approximate Newton direction via the two-loop recursion
                direction = _lbfgs_two_loop_recursion(grad_flat, s_history, y_history, rho_history, self.eps) * mask_flat
                directional_derivative = jnp.dot(grad_flat, direction)

                # backtrack the step size (Armijo sufficient-decrease line search) so a step is
                # only accepted if it's both finite AND actually improves the objective by a
                # meaningful amount -- "finite" alone doesn't mean "good" near an ill-conditioned
                # objective (e.g. a near-singular kernel matrix)
                step_scale = 1.0
                accepted = False
                if directional_derivative < 0:  # only search along genuine descent directions
                    for _ in range(30):
                        p_new = tree_map(lambda constraint, param: constraint(param), self.constraints, unravel_fn(p_flat + step_scale * lr * direction))
                        p_flat_new, _ = flatten_util.ravel_pytree(p_new)
                        loss_new, grad_flat_new = _evaluate(p_flat_new, Xbatch, Ybatch)

                        sufficient_decrease = loss_new <= loss + 1e-4 * step_scale * lr * directional_derivative
                        if _is_finite(loss_new) and _is_finite(grad_flat_new) and sufficient_decrease:
                            accepted = True
                            break

                        step_scale *= 0.5

                if not accepted:
                    s_history.clear()
                    y_history.clear()
                    rho_history.clear()
                    batch_losses[i] = float('nan')
                    continue

                # setting the loss for this batch
                batch_losses[i] = float(loss_new)

                # updating the curvature-pair history (measured on this same batch)
                _lbfgs_update_history(s_history, y_history, rho_history, p_flat_new - p_flat, grad_flat_new - grad_flat, self.m, self.eps)

                p_flat = p_flat_new

            # displaying the loss
            verbose and iterator.set_postfix_str(f"Mean Objective: {batch_losses.mean():.4e}")

        # returning the final parameter estimates
        return unravel_fn(p_flat)

