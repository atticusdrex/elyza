from util.imports import * 
from jax.tree_util import tree_map

class GradientOptimizer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    loss_grad_fn : callable

    constraints: dict | None = None 
    


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





        
