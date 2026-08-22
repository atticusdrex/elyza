# %%
from elyza.benchmarks.multifidelity.magpi_analytical import x, hf_evaluator 
from elyza.util.imports import * 
from matplotlib.pyplot import * 
from elyza.surrogate.gp import GaussianProcess, ARD, Constant
from elyza.optim.adam import ADAM, ADAMOptions

# %%
n_train = 250 
input_data = x.sample(jrand.PRNGKey(42), n_train)
output_data = hf_evaluator.evaluate(input_data) + 1e-2 * jrand.uniform(jrand.PRNGKey(42), shape = (n_train,1))

model = GaussianProcess(
    input_dim = 1, 
    kernel_cls = ARD, 
    mean_cls = Constant, 
    calibrate_noise = True, 
    noise_var = 1e-6, 
    eps = 1e-12, 
    max_cond = 1e5, 
    verbose = True
)

# declaring the model options 
adam_opts = ADAMOptions(
    p_init = model.p, 
    lr = 1e-1, 
    epochs = 100, 
    batch_size = None, 
    beta1 = 0.9, 
    beta2 = 0.999, 
    active_params = {'mean':True, 'kernel':True, 'noise':False}, 
    constraints = None, 
    verbose = True, 
    eps = 1e-8, 
    random_state = 42 
)

# setting the optimizer
model.set_optimizer(ADAM, adam_opts)

# fitting to the data 
model.fit(input_data, output_data)

# predicting on new data 
test_inputs = jnp.linspace(0,5,1000).reshape(-1,1)
ymean, yvar = model.predict(test_inputs) 
yconf = 2 * jnp.sqrt(yvar) 

figure()
plot(test_inputs.ravel(), hf_evaluator.evaluate(test_inputs).ravel(), linestyle = 'dotted', color = 'black')
fill_between(test_inputs.ravel(), ymean-yconf, ymean+yconf, alpha = 0.3, color =  'green')
show()

# %% 
from elyza.optim.lbfgs import LBFGS, LBFGSOptions

n_train = 250 
input_data = x.sample(jrand.PRNGKey(42), n_train)
output_data = hf_evaluator.evaluate(input_data) + 1e-2 * jrand.uniform(jrand.PRNGKey(42), shape = (n_train,1))

model = GaussianProcess(
    input_dim = 1, 
    kernel_cls = ARD, 
    mean_cls = Constant, 
    calibrate_noise = True, 
    noise_var = 1e-6, 
    eps = 1e-12, 
    max_cond = 1e5, 
    verbose = True
)

# declaring the model options 
lbfgs_opts = LBFGSOptions(
    p_init = model.p, 
    lr = 1e-1, 
    epochs = 100, 
    batch_size = None, 
    m = 10, 
    max_backtracks = 30, 
    active_params = {'mean':True, 'kernel':True, 'noise':False}, 
    constraints = None, 
    verbose = True, 
    eps = 1e-8, 
    random_state = 42 
)

# setting the optimizer
model.set_optimizer(LBFGS, lbfgs_opts)

# fitting to the data 
model.fit(input_data, output_data)

# predicting on new data 
test_inputs = jnp.linspace(0,5,1000).reshape(-1,1)
ymean, yvar = model.predict(test_inputs) 
yconf = 2 * jnp.sqrt(yvar) 

figure()
plot(test_inputs.ravel(), hf_evaluator.evaluate(test_inputs).ravel(), linestyle = 'dotted', color = 'black')
fill_between(test_inputs.ravel(), ymean-yconf, ymean+yconf, alpha = 0.3, color =  'green')
show()
