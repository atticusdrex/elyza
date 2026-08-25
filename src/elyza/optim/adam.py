# imports 
from elyza.optim.abstract import OptimizerOptions, BatchGradientOptimizer
from elyza.util.imports import * 
from jax.tree_util import tree_map, tree_leaves
from jax import lax

'''
ADAMOptions 
------------
the options which parameterize an ADAM optimizer
'''
class ADAMOptions(OptimizerOptions):
    p_init : dict[str|int,jax.Array] | None = Field(default = None, description = "initial dictionary of parameters")
    lr : float = Field(default = 1e-3, description = "learning rate for gradient descent")
    epochs : int = Field(default = 1, description = "number of times we pass through the training data")
    batch_size : int | None = Field(default = None, description = "number of training datapoints in a specific loss function evaluation")
    beta1 : float = Field(default = 0.9, description = "first momentum parameter")
    beta2 : float = Field(default = 0.999, description = "second momentum parameter")
    active_params : dict[str,bool] | None = Field(default = None, description = "a dictionary of the active parameters to optimize")
    constraints : dict[str,Callable] | None = Field(default = None, description = "a dictionary of constraints mapping from parameter:constraint function")
    verbose : bool = Field(default = False, description = "whether or not to print the reuslts of the optimizer")
    eps : float = Field(default = 1e-8, description = "small positive number to prevent division by zero")
    random_state : int = Field(default = 42, description = "random seed for replication")
    unroll : int | bool = Field(default = False, description = "whether or not to unroll the jax.lax.scan operation (unroll=True: long compilation times, faster execution times, high memory, unroll = k: unroll for set size-k blocks of k loop steps, unroll = False: short compile times, slower execution times, lower memory)")

    def model_post_init(self, __context):
        assert self.lr > 0, "learning rate cannot be negative" 
        assert self.epochs >= 1, "need at least one epoch"
        assert self.batch_size is None or self.batch_size >= 1, "batch size must be None or at least 1"
        assert self.beta1 > 0, "beta1 must be positive" 
        assert self.beta2 > 0, "beta2 must be positive" 


'''
Batch ADAM optimizer
--------------------
class for running general gradient-based optimization scripts 
'''
class ADAM(BatchGradientOptimizer):
    opts : ADAMOptions | None = Field(default = None, description = "options for the optimizer")

    def model_post_init(self, __context):
        super().model_post_init(__context) 
        assert self.opts is not None, "you must pass a valid instance of ADAMOptions()"

    '''
    everything needed to parameterize the estimator must already be in the self.opts variable. the *args is purely just to pass into the loss functions after p 
    '''
    def run(self, *data : list[jax.Array]):
        # asserting the loss function has been set 
        assert self.loss_grad_fn is not None, "you must specify a loss function" 
        assert self.opts.p_init is not None, "must give initial parameter guess"


        # determining the batch size for this run (without mutating shared opts)
        batch_size = self.opts.batch_size if self.opts.batch_size is not None else data[0].shape[0]

        # generating the PRNG key 
        key = jrand.PRNGKey(self.opts.random_state) 

        # fill in identity constraints for any params not given an explicit constraint
        default_constraints = tree_map(lambda x: lambda y: y, self.opts.p_init) # use identity constraints
        self.opts.constraints = {**default_constraints, **(self.opts.constraints or {})}

        # determining which parameters are active
        if self.opts.active_params is None:
            self.opts.active_params = tree_map(lambda _: True, self.opts.p_init)

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

'''
the function for updating a single parameter 
'''
def _adam_update(active, constraint, param, m_val, s_val, step, lr, beta1, beta2, eps):
    # only updating the parameter if it's active 
    if active: 
        m_hat = m_val / (1 - beta1 ** step)
        s_hat = s_val / (1 - beta2 ** step)
        param -= lr * m_hat / (jnp.sqrt(s_hat) + eps)

    return constraint(param)

'''
the function for performing parameter updates on a single batch
'''
def _batch_adam_scan(carry, batch, loss_grad_fn : Callable, lr:float, beta1:float, beta2:float, eps:float, active_params:dict, constraints:dict):
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

