from elyza.util.imports import *
from jax.tree_util import tree_map

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
    else:
        gamma = 1.0

    r = gamma * q
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

class GradientOptimizer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    loss_grad_fn : SkipValidation[callable]
    constraints: dict | None = None 

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
    beta1: float = 0.9
    beta2: float = 0.999 
    lr: float = 1e-3
    eps: float = 1e-8

    def run(self, lr : float, steps : int, p_init : dict, active_params : dict | None = None, verbose = True):
        def param_update(active, constraint, param, m_val, s_val, step):
            # only updating the parameter if it's active 
            if active: 
                m_hat = m_val / (1 - self.beta1 ** step)
                s_hat = s_val / (1 - self.beta2 ** step)
                param -= lr * m_hat / (jnp.sqrt(s_hat) + self.eps)

            return constraint(param)
        
        # default identity constraints pytree if not specified
        if self.constraints is None: 
            self.constraints = tree_map(lambda x: lambda y: y, p_init) # use identity constraints
        
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

            # updating the moment estimates
            m = tree_map(lambda m_val, grad_val: self.beta1 * m_val + (1 - self.beta1) * grad_val, m, grad)
            s = tree_map(lambda s_val, grad_val: self.beta2 * s_val + (1 - self.beta2) * grad_val**2, s, grad)

            # updating the parameter estimates with constraints 
            p = tree_map(
                lambda active, constraint, param, m_val, s_val: param_update(
                    active, constraint, param, m_val, s_val, iter+1
                ), 
                active_params, self.constraints, p, m, s)

            # displaying the loss 
            verbose and iterator.set_postfix_str(f"Objective: {loss:.4e}")

        # returning the final parameter estimates
        return deepcopy(p)


class LBFGS(GradientOptimizer):
    m: int = 10
    eps: float = 1e-8

    def run(self, lr : float, steps : int, p_init : dict, active_params : dict | None = None, verbose = True):
        # default identity constraints pytree if not specified
        if self.constraints is None:
            self.constraints = tree_map(lambda x: lambda y: y, p_init) # use identity constraints

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

        # initializing iterator object
        iterator = tqdm(range(steps)) if verbose else range(steps)

        # main optimization loop
        for iter in iterator:
            # approximate Newton direction via the two-loop recursion
            direction = _lbfgs_two_loop_recursion(grad_flat, s_history, y_history, rho_history, self.eps) * mask_flat

            # taking a step and applying per-leaf constraints
            p_new = tree_map(lambda constraint, param: constraint(param), self.constraints, unravel_fn(p_flat + lr * direction))
            p_flat_new, _ = flatten_util.ravel_pytree(p_new)

            loss_new, grad_flat_new = _evaluate(p_flat_new)

            # updating the curvature-pair history
            _lbfgs_update_history(s_history, y_history, rho_history, p_flat_new - p_flat, grad_flat_new - grad_flat, self.m, self.eps)

            p_flat, grad_flat, loss = p_flat_new, grad_flat_new, loss_new

            # displaying the loss
            verbose and iterator.set_postfix_str(f"Objective: {loss:.4e}")

        # returning the final parameter estimates
        return unravel_fn(p_flat)


class BatchADAM(BatchGradientOptimizer):
    beta1: float = 0.9
    beta2: float = 0.999 
    lr: float = 1e-3
    eps: float = 1e-8

    def run(self, key, X, Y, lr : float, epochs : int, batch_size : int, p_init : dict, active_params : dict | None = None, verbose = True):
        def param_update(active, constraint, param, m_val, s_val, step):
            # only updating the parameter if it's active 
            if active: 
                m_hat = m_val / (1 - self.beta1 ** step)
                s_hat = s_val / (1 - self.beta2 ** step)
                param -= lr * m_hat / (jnp.sqrt(s_hat) + self.eps)

            return constraint(param)
        
        # default identity constraints pytree if not specified
        if self.constraints is None: 
            self.constraints = tree_map(lambda x: lambda y: y, p_init) # use identity constraints
        
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

                # updating the moment estimates
                m = tree_map(lambda m_val, grad_val: self.beta1 * m_val + (1 - self.beta1) * grad_val, m, grad)
                s = tree_map(lambda s_val, grad_val: self.beta2 * s_val + (1 - self.beta2) * grad_val**2, s, grad)

                # updating the parameter estimates with constraints 
                p = tree_map(
                    lambda active, constraint, param, m_val, s_val: param_update(
                        active, constraint, param, m_val, s_val, iter+1
                    ), 
                    active_params, self.constraints, p, m, s)

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
        
        # default identity constraints pytree if not specified
        if self.constraints is None: 
            self.constraints = tree_map(lambda x: lambda y: y, p_init) # use identity constraints
        
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

                # updating the parameter estimates with constraints 
                p = tree_map(
                    lambda active, constraint, param, grad_val: param_update(
                        active, constraint, param, grad_val
                    ), 
                    active_params, self.constraints, p, grad)

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
        # default identity constraints pytree if not specified
        if self.constraints is None:
            self.constraints = tree_map(lambda x: lambda y: y, p_init) # use identity constraints

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
                # obtaining the gradient on this batch
                _, grad_flat = _evaluate(p_flat, Xbatch, Ybatch)

                # approximate Newton direction via the two-loop recursion
                direction = _lbfgs_two_loop_recursion(grad_flat, s_history, y_history, rho_history, self.eps) * mask_flat

                # taking a step and applying per-leaf constraints
                p_new = tree_map(lambda constraint, param: constraint(param), self.constraints, unravel_fn(p_flat + lr * direction))
                p_flat_new, _ = flatten_util.ravel_pytree(p_new)

                loss_new, grad_flat_new = _evaluate(p_flat_new, Xbatch, Ybatch)

                # setting the loss for this batch
                batch_losses[i] = float(loss_new)

                # updating the curvature-pair history (measured on this same batch)
                _lbfgs_update_history(s_history, y_history, rho_history, p_flat_new - p_flat, grad_flat_new - grad_flat, self.m, self.eps)

                p_flat = p_flat_new

            # displaying the loss
            verbose and iterator.set_postfix_str(f"Mean Objective: {batch_losses.mean():.4e}")

        # returning the final parameter estimates
        return unravel_fn(p_flat)

