from elyza.util.imports import *
from jax.tree_util import tree_map, tree_leaves


'''
True iff every leaf of a scalar/array/pytree is finite. Used to guard optimizer
state against a NaN/Inf objective or gradient (e.g. from a near-singular kernel
matrix) so that a single bad step can't permanently corrupt the parameters --
a rejected step just leaves the last known-good parameters untouched.
'''
def _is_finite(tree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in tree_leaves(tree))


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

class LBFGS(GradientOptimizer):
    m: int = 10
    eps: float = 1e-8

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

        # curvature-pair history for the two-loop recursion
        s_history, y_history, rho_history = [], [], []

        loss, grad_flat = _evaluate(p_flat)

        # can't optimize from a broken starting point -- fail loudly rather than silently
        # returning/propagating NaN parameters
        if not (_is_finite(loss) and _is_finite(grad_flat)):
            raise FloatingPointError(
                "LBFGS.run: objective/gradient at p_init is non-finite (NaN/Inf); refusing to start."
            )

        # initializing iterator object
        iterator = tqdm(range(steps)) if verbose else range(steps)

        # main optimization loop
        for iter in iterator:
            # approximate Newton direction via the two-loop recursion
            direction = _lbfgs_two_loop_recursion(grad_flat, s_history, y_history, rho_history, self.eps) * mask_flat
            directional_derivative = jnp.dot(grad_flat, direction)

            # backtrack the step size (Armijo sufficient-decrease line search) so a step is only
            # accepted if it's both finite AND actually improves the objective by a meaningful
            # amount. Without this, a technically-finite step near an ill-conditioned objective
            # (e.g. a near-singular kernel matrix) can still send the loss up by many orders of
            # magnitude and get accepted anyway, since "finite" alone doesn't mean "good".
            step_scale = 1.0
            accepted = False
            if directional_derivative < 0:  # only search along genuine descent directions
                for _ in range(30):
                    p_new = tree_map(lambda constraint, param: constraint(param), self.constraints, unravel_fn(p_flat + step_scale * lr * direction))
                    p_flat_new, _ = flatten_util.ravel_pytree(p_new)
                    loss_new, grad_flat_new = _evaluate(p_flat_new)

                    sufficient_decrease = loss_new <= loss + 1e-4 * step_scale * lr * directional_derivative
                    if _is_finite(loss_new) and _is_finite(grad_flat_new) and sufficient_decrease:
                        accepted = True
                        break

                    step_scale *= 0.5

            if not accepted:
                # this search direction is unusable everywhere we tried -- drop the curvature
                # history (so the next direction is a fresh steepest-descent guess) and hold
                # the last known-good, finite parameters rather than corrupting them
                s_history.clear()
                y_history.clear()
                rho_history.clear()
                verbose and iterator.set_postfix_str(f"Objective: {loss:.4e} (step rejected: no sufficient decrease)")
                continue

            # updating the curvature-pair history
            _lbfgs_update_history(s_history, y_history, rho_history, p_flat_new - p_flat, grad_flat_new - grad_flat, self.m, self.eps)

            p_flat, grad_flat, loss = p_flat_new, grad_flat_new, loss_new

            # displaying the loss
            verbose and iterator.set_postfix_str(f"Objective: {loss:.4e}")

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

