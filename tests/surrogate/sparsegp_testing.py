# %%
from elyza.benchmarks.multifidelity.magpi_analytical import x, hf_evaluator 
from elyza.util.imports import * 
from matplotlib.pyplot import * 
from elyza.surrogate.gp import SparseGP, ARD, Constant
from elyza.optim import ADAM, ADAMOptions
from elyza.optim import LBFGS, LBFGSOptions 
# 
n_train = int(1e5)
input_data = x.sample(jrand.PRNGKey(42), n_train)
output_data = hf_evaluator.evaluate(input_data) + 2e-2 * jrand.normal(jrand.PRNGKey(42), shape = (n_train,1))

figure(figsize=(16,9), dpi = 300)
scatter(input_data.ravel(), output_data.ravel(), s = 10.0, color = 'black', alpha = 0.5)
show()

model = SparseGP(
    input_dim = 1, 
    kernel_cls = ARD, 
    mean_cls = Constant, 
    n_inducing_points=25, 
    calibrate_noise = True, 
    noise_var = 4e-4, 
    eps = 1e-6,
    max_cond = 1e5, 
    verbose = True
)

# %% declaring the model options 
adam_opts = ADAMOptions(
    p_init = deepcopy(model.p), 
    lr = 1e-2, 
    epochs = 25, 
    batch_size = 2500, 
    beta1 = 0.9, 
    beta2 = 0.999, 
    active_params = {'mean':True, 'kernel':True, 'noise':True, 'inducing':False, 'q_mu':True, 'q_L':True}, 
    constraints = None, 
    verbose = True, 
    eps = 1e-8, 
    random_state = 42, 
    unroll = False
)

# setting the optimizer
model.set_optimizer(ADAM, adam_opts)

# fitting to the data 
model.fit(input_data, output_data, n_monte_carlo = 250, random_state = 42)

# predicting on new data 
test_inputs = jnp.linspace(0,5,1000).reshape(-1,1)
ymean, yvar = model.predict(test_inputs) 
yconf = 2 * jnp.sqrt(yvar) 

# %% 
figure(figsize=(10,4), dpi = 200)
plot(test_inputs.ravel(), hf_evaluator.evaluate(test_inputs).ravel(), linestyle = 'dotted', color = 'black')
fill_between(test_inputs.ravel(), ymean-yconf, ymean+yconf, alpha = 0.3, color =  'green')
# scatter(input_data.ravel(), output_data.ravel(), s = 10.0, color = 'black', alpha = 0.75)
show()