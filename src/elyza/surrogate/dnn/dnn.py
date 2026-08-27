from elyza.util.imports import * 
from elyza.surrogate import Surrogate
from elyza.optim.abstract import Optimizer, OptimizerOptions
from elyza.util.helpers import softplus, inv_softplus, ensure_2d
from jax.nn import relu 

class MLPRegressor(Surrogate): 
    # public fields
    input_dim: int = Field(description = "input dimension")
    output_dim : int = Field(description = "output dimension") 
    hidden_dims : tuple[int, ...] = Field(description = "hidden layer dimensions starting from first hidden layer") 
    activation : SkipValidation[callable] = Field(default = relu, description = "activation function")
    l2_reg : float | None = Field(default = None, description = "ridge regularization parameter") 
    l1_reg : float | None = Field(default = None, description = "lasso regularization parameter") 
    eps: float = Field(default = 1e-12, description = "small positive jitter value to avoid singular kernel matrices and divide-by-zero errors")
    verbose: bool = Field(default = False, description = "whether or not to print the training and calibration progress") 
    p: dict | None = Field(default = None, description = "an optional value depending on whether the user wants to instantiate the GP with predefined model parameters")
    random_state : int = Field(default = 42, description = "a PRNG seed for initializing the weights") 
    init_scale : float = Field(default = 1e-4, description = "scalar for the weight initialization. larger init_scale means larger initial weight variance") 

    # private fields 
    _n_layers : int | None = PrivateAttr(default = None) 
    _optimizer : Optimizer | None = PrivateAttr(default = None) 
    def model_post_init(self, __context): 
        # storing the number of layers 
        self._n_layers = len(self.hidden_dims) + 1 

        if self.p is None: 
            # initializing prng keys 
            init_keys = jrand.split(jrand.PRNGKey(self.random_state), self._n_layers)

            # building the parameter pytree 
            self.p = {'weights':{}, 'biases':{}}

            # building the weights and biases 
            prev_dim = self.input_dim 

            for layer, _ in enumerate(self.hidden_dims): 
                # splitting the weight and bias into different RNG keys 
                weight_key, bias_key = jrand.split(init_keys[layer]) 

                # storing the weight matrix 
                self.p['weights'][layer] = self.init_scale * jrand.normal(
                    weight_key, shape = (self.hidden_dims[layer], prev_dim) 
                )

                # storing the bias vector
                self.p['biases'][layer] = self.init_scale * jrand.normal(
                    bias_key, shape = (self.hidden_dims[layer], 1) 
                )

                # setting prev_dim to this hidden layer dimension 
                prev_dim = self.hidden_dims[layer]  

            # storing the weights and biases for the output layer
            layer += 1
            weight_key, bias_key = jrand.split(init_keys[layer]) 
            self.p['weights'][layer] = self.init_scale * jrand.normal(
                weight_key, shape = (self.output_dim, prev_dim) 
            )
            self.p['biases'][layer] = self.init_scale * jrand.normal(
                bias_key, shape = (self.output_dim, 1) 
            )

    def _pred(self, p, X:jax.Array) -> jax.Array:
        # storing the hidden state
        hidden_state = X.T

        # iterating through the hidden layers, applying the activation function
        for layer in range(self._n_layers - 1):
            hidden_state = self.activation(
                p['weights'][layer] @ hidden_state + p['biases'][layer]
            )

        # output layer stays linear so predictions aren't clamped by the activation
        output_layer = self._n_layers - 1
        hidden_state = p['weights'][output_layer] @ hidden_state + p['biases'][output_layer]

        return hidden_state


    def predict(self, X:jax.Array) -> jax.Array: 
        return self._pred(self.p, X) 

    def fit(
            self,
            X: np.ndarray | jax.Array,
            Y: np.ndarray | jax.Array,
        ):
        # making sure an optimizer has been declared 
        assert self._optimizer is not None, "must declare an optimizer" 

        # making sure p_init is specified 
        self._optimizer.opts.p_init = deepcopy(self.p)

        # converting training data to jax arrays
        X, Y = ensure_2d(jnp.array(X)), ensure_2d(jnp.array(Y))

        self._optimizer.loss_grad_fn = jit(value_and_grad(lambda p, X, Y: self._objective(p, X, Y), argnums=2))

        # run the optimizer
        new_params = self._optimizer.run(X, Y)

        # setting the new params
        self.p = deepcopy(new_params)

    def sample(self, X: jax.Array) -> jax.Array: 
        pass 

    def set_optimizer(self, optimizer:Optimizer, optimizer_options:OptimizerOptions): 
        self._optimizer = optimizer(opts = optimizer_options) 


    def update(self, X:jax.Array, Y: jax.Array):
        pass 

    def _objective(self, X, Y, p:dict) -> float: 
        # making predictions on this dataset
        Yhat = self._pred(p, X)

        # computing the loss term (Yhat is (output_dim, batch); transpose to match Y's (batch, output_dim))
        loss_term = ((Yhat.T - Y)**2).ravel().sum()

        # computing the regularization
        regularization = 0.0 
        if self.l2_reg is not None: 
            for weight in p['weights'].values(): 
                regularization += self.l2_reg * jnp.sum(weight**2)
        if self.l1_reg is not None: 
            for weight in p['weights'].values(): 
                regularization += self.l1_reg * jnp.sum(jnp.abs(weight))

        return loss_term + regularization 




            




            


