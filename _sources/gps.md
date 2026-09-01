---
file_format: mystnb
kernelspec:
  name: python3
---

# Gaussian process regression

This page walks through the core surrogate-modeling workflow in `elyza`:
describing an input, wrapping a model as an {class}`~elyza.core.evaluator.Evaluator`,
generating training data, and fitting a surrogate. It ends with a short
comparison of the three surrogate models the library ships with.

Every example below runs as-is against a clean checkout.

## 1. Vanilla Gaussian Process Regression

Every surrogate modeling workflow typically starts with a {class}`~elyza.core.data.Variable`.
For a single scalar variable sampled uniformly over some range, use
{class}`~elyza.core.random.Uniform`. This isn't required, you can call surrogate models using `Xtrain` and `Ytrain` jax arrays exactly like `scikit-learn` does, but for active learning schemes or routines in which we must call an underlying function on new data, it helps to fit everything into the input/evaluator schema. 

```{code-cell} python
import jax.numpy as jnp
import jax.random as jrand

from elyza.core.random import Uniform

# instantiate the input variable
x = Uniform(
    name="x",
    dim=1,
    lower=0.0,
    upper=1.0,
)

# generate input training data 
key = jrand.PRNGKey(0)
X_train = x.sample(key, 20)   # shape (20, 1)

from elyza.core.evaluator import Evaluator
import jax.random as jrand 

# instantiate the output function 
evaluator = Evaluator(
    name="sinusoid",
    inputs=[x],
    output_dim=1,
    evaluation_func=lambda x: jnp.sin(2 * jnp.pi * x),
)

# generate the output training data 
Y_train = evaluator.evaluate(X_train)  + 1e-1 * jrand.normal(jrand.PRNGKey(42), shape = X_train.shape) # shape (20, 1)
```

{class}`~elyza.surrogate.gp.gp.GaussianProcess` is the class for the vanilla dense kernel matrix Gaussian process regression as described in Rasmussen \& Williams et al. 2008. It takes a kernel and a mean function, and fits their hyperparameters by maximizing the log-marginal-likelihood with an 
{mod}`~elyza.optim` optimizer.

```{code-cell} python
from elyza.surrogate.gp import GaussianProcess, ARD, Constant
from elyza.optim import ADAM, ADAMOptions

gp = GaussianProcess(input_dim=1, kernel_cls=ARD, mean_cls=Constant, noise_var = 1e-2)
gp.set_optimizer(ADAM, ADAMOptions(lr=1e-1, epochs=500))
gp.fit(X_train, Y_train)
```

Every surrogate in `elyza` shares the same `fit` / `predict` /
`sample` interface (see {class}`~elyza.surrogate.abstract.Surrogate`).
`predict` returns a mean and, for a GP, either the marginal variance or the
full posterior covariance:

```{code-cell} python
X_test = jnp.linspace(0.0, 1.0, 1000).reshape(-1, 1)
Y_test = evaluator.evaluate(X_test)
mu, var = gp.predict(X_test)              # analytical posterior mean and variance
conf = 2 * jnp.sqrt(var) # plus/minus two standard deviation confidence interval 

samples = gp.sample(jrand.PRNGKey(1), X_test, n_samples=3)  # sampling from the GP posterior

from matplotlib.pyplot import * 
%matplotlib inline
figure(figsize=(6,4), dpi=100)
plot(X_test.ravel(), Y_test.ravel(), color = "black", linestyle = 'dotted', label = "Target function") 
fill_between(X_test.ravel(), mu - conf, mu + conf, alpha = 0.3, color = 'green', label = "$\\pm 2 \\sigma$ confidence interval")
scatter(X_train.ravel(), Y_train.ravel(), s = 10.0, color = 'red', marker = '+') 
legend() 
xlabel("x") 
ylabel("f(x)") 
title("GP Approximation of Target Function")
show()
```

New data can be folded in afterward without refitting from scratch, via a
rank-`m` Cholesky update:

```{code-cell} python
X_new = x.sample(jrand.PRNGKey(2), 5)
Y_new = evaluator.evaluate(X_new)
gp.update(X_new, Y_new)
```

## Sparse Gaussian Process Regression

In this section, we describe how to implement a sparse Gaussian Process, which scales to much larger datasets. First, we generate training data: 

```{code-cell} python 
from elyza.core import Uniform, Evaluator
from elyza.util.imports import * 
from matplotlib.pyplot import * 

# define a uniform input on [0,5] 
x = Uniform(
        name = "x",
        dim = 1,
        lower = 0.0,
        upper = 5.0
    )

# define a jit-compiled evaluator on the input 
y = Evaluator(
    name = "y=f(x)",
    inputs = [x],
    output_dim = 1,
    evaluation_func = jit(
        lambda x: jnp.exp(-x) * jnp.sin(2*pi*x)
    ),
    jit_compile = True
)

# generate noisy training data 
n_train = int(1e5)
key_input, key_noise = jrand.split(jrand.PRNGKey(42))
input_data = x.sample(key_input, n_train)
output_data = y.evaluate(input_data) + 2e-2 * jrand.normal(key_noise, shape = (n_train,1))

# visualize the training data 
figure(figsize=(16,9), dpi = 300)
scatter(input_data.ravel(), output_data.ravel(), s = 10.0, color = 'black', alpha = 0.5)
show()
```

Next, we instantiate a `SparseGP` and `ADAM` optimizer. 

```{code-cell} python 
from elyza.surrogate.gp import SparseGP, ARD, Constant
from elyza.optim import ADAM, ADAMOptions

# instantiate a sparse Gaussian Process 
model = SparseGP(
    input_dim = 1, 
    kernel_cls = ARD, 
    mean_cls = Constant, 
    n_inducing_points=25, # number of inducing points
    calibrate_noise = True, 
    noise_var = 4e-4, 
    eps = 1e-6,
    max_cond = 1e5, 
    verbose = True
)

# declaring the optimizer options 
adam_opts = ADAMOptions(
    p_init = deepcopy(model.p), 
    lr = 1e-2, 
    epochs = 50, 
    batch_size = 2500, 
    beta1 = 0.9, 
    beta2 = 0.999, 
    active_params = {'mean':True, 'kernel':True, 'noise':True, 'inducing':False, 'q_mu':True, 'q_L':True}, # q_mu and q_L are the mean and cholesky factor of the variacne of the variational distribution. 
    constraints = None, 
    verbose = False, 
    eps = 1e-8, 
    random_state = 42, 
    unroll = False
)

# setting the optimizer to the model 
model.set_optimizer(ADAM, adam_opts)

# fitting to the data 
model.fit(input_data, output_data, n_monte_carlo = 250, random_state = 42)

# predicting on new data 
test_inputs = jnp.linspace(0,5,1000).reshape(-1,1)
ymean, yvar = model.predict(test_inputs) 
yconf = 2 * jnp.sqrt(yvar) 

# plotting the model predictions 
figure()
plot(test_inputs.ravel(), y.evaluate(test_inputs).ravel(), linestyle = 'dotted', color = 'black', label = "True function")
fill_between(test_inputs.ravel(), ymean-yconf, ymean+yconf, alpha = 0.3, color =  'green', label = "SparseGP $\\pm 2 \\sigma$ confidence interval")
legend()
xlabel("x"); ylabel("y=f(x)")
show()
```

