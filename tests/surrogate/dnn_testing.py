# %% 
from elyza.surrogate import MLPRegressor
from jax.nn import relu
from elyza.benchmarks.multifidelity.magpi_analytical import x, hf_evaluator
from elyza.util.imports import * 
from matplotlib.pyplot import * 
from elyza.optim import ADAM, ADAMOptions

n_train = 10000
train_inputs = x.sample(jrand.PRNGKey(42), n_points = n_train)
train_outputs = hf_evaluator.evaluate(train_inputs)

test_inputs = jnp.linspace(0, 5, num = 1000).reshape(-1,1)
test_outputs = hf_evaluator.evaluate(test_inputs) 

model = MLPRegressor(
    input_dim = 1, 
    output_dim = 1, 
    hidden_dims = (100, 50), 
    activation = relu, 
    l2_reg = 1e-3, 
    l1_reg = 1e-3, 
    eps = 1e-12, 
    verbose = True, 
    random_state = 42, 
    init_scale = 1e-3
)


# declaring the model options 
adam_opts = ADAMOptions(
    p_init = model.p, 
    lr = 1e-2, 
    epochs = 500, 
    batch_size = 1000, 
    beta1 = 0.9, 
    beta2 = 0.999, 
    active_params = None,
    constraints = None, 
    verbose = True, 
    eps = 1e-8, 
    random_state = 42, 
    unroll = 5
)

# setting the optimizer
model.set_optimizer(ADAM, adam_opts)

# %% 
model.fit(train_inputs, train_outputs)

# %%
yhat_test = model.predict(test_inputs)

figure() 
plot(test_inputs.ravel(),test_outputs.ravel())
scatter(train_inputs.ravel(), train_outputs.ravel(), s = 5.0, color = 'black', marker = '.') 
plot(test_inputs.ravel(), yhat_test.ravel(), color = 'green')

