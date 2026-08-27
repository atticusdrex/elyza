# %% 
from elyza.multifidelity.surrogate import MAGPI 
from elyza.benchmarks.multifidelity.magpi_analytical import * 
from elyza.surrogate.gp import GaussianProcess, ARD, Linear, Constant
from elyza.optim import ADAM, ADAMOptions
from elyza.surrogate import SupervisedDataset
from elyza.util.imports import * 

from matplotlib.pyplot import * 

# declaring the ADAM optimizer options 
adam_opts = ADAMOptions(
    lr = 3e-1, 
    epochs = 500, 
    batch_size = None, 
    beta1 = 0.9, 
    beta2 = 0.999, 
    active_params = {'mean':True, 'kernel':True, 'noise':False}, 
    constraints = None, 
    verbose = True, 
    eps = 1e-8, 
    random_state = 42, 
    unroll = 25
)

# building the lowest-fidelity GP 
lf_inputs = x.sample(jrand.PRNGKey(42), 250)
lf_outputs = lf_evaluator.evaluate(lf_inputs) 

lf_data = SupervisedDataset(
    input_data = [lf_inputs], 
    output_data = lf_outputs, 
    noise_var = 1e-4
)

lf_gp = GaussianProcess(
    input_dim = 1, 
    kernel_cls = ARD, 
    mean_cls = Constant, 
    noise_var = lf_data.noise_var, 
    eps = 1e-12, 
    max_cond = 1e5, 
    verbose = True, 
    calibrate_noise = True
)
 
lf_gp.set_optimizer(ADAM, adam_opts) 



# building the medium-fidelity gp 
mf_inputs = x.sample(jrand.PRNGKey(42), 100)
mf_outputs = mf_evaluator.evaluate(mf_inputs) 

mf_data = SupervisedDataset(
    input_data = [mf_inputs], 
    output_data = mf_outputs, 
    noise_var = 1e-4
)

mf_gp = GaussianProcess(
    input_dim = 2, 
    kernel_cls = ARD, 
    mean_cls = Constant, 
    noise_var = mf_data.noise_var, 
    eps = 1e-12, 
    max_cond = 1e5, 
    verbose = True
)
 

# constraint set to zero out the input part of the linear mean parameters 
mf_gp.set_optimizer(ADAM, adam_opts)

hf_inputs = x.sample(jrand.PRNGKey(42), 25)
hf_outputs = hf_evaluator.evaluate(hf_inputs) 

hf_data = SupervisedDataset(
    input_data = [hf_inputs], 
    output_data = hf_outputs, 
    noise_var = 1e-4
)

hf_gp = GaussianProcess(
    input_dim = 3, 
    kernel_cls = ARD, 
    mean_cls = Constant, 
    noise_var = hf_data.noise_var, 
    eps = 1e-12, 
    max_cond = 1e5, 
    verbose = True, 
    calibrate_noise = True
)

hf_gp.set_optimizer(ADAM, adam_opts)

# %% instantiating MAGPI model
magpi = MAGPI(
    data = [lf_data, mf_data, hf_data], 
    evaluators = [lf_evaluator, mf_evaluator, hf_evaluator]
)

# setting the surrogates for each level with prediction keyword arguments
magpi.set_surrogate(
    level = 0,
    surrogate = lf_gp, 
    full_cov = False
)

magpi.set_surrogate(
    level = 1,
    surrogate = mf_gp, 
    full_cov = False
)

# 
magpi.set_surrogate(
    level = 2,
    surrogate = hf_gp, 
    full_cov = False
)

# %%training the lowest-fidelity surrogate 
magpi.fit(0)

figure()
x_test = jnp.linspace(0,5,1000).reshape(-1,1) 
ymean, yvar = magpi.predict(x_test, level = 0) 
yconf = 2 * jnp.sqrt(yvar) 
ytrue = lf_evaluator.evaluate(x_test) 

fill_between(x_test.ravel(), (ymean - yconf).ravel(), (ymean + yconf).ravel(), alpha = 0.3, color = 'green')
plot(x_test.ravel(), ytrue.ravel(), linestyle = 'dotted', color = 'black')
scatter(lf_inputs.ravel(), lf_outputs.ravel())
show()

# training the medium-fidelity surrogate
magpi.fit(1)

figure()
x_test = jnp.linspace(0,5,1000).reshape(-1,1) 
ymean, yvar = magpi.predict(x_test, level = 1) 
yconf = 2 * jnp.sqrt(yvar) 
ytrue = mf_evaluator.evaluate(x_test) 

fill_between(x_test.ravel(), (ymean - yconf).ravel(), (ymean + yconf).ravel(), alpha = 0.3, color = 'green')
plot(x_test.ravel(), ytrue.ravel(), linestyle = 'dotted', color = 'black')
scatter(mf_inputs.ravel(), mf_outputs.ravel())
show() 
# training the medium-fidelity surrogate
magpi.fit(2)

figure()
x_test = jnp.linspace(0,5,1000).reshape(-1,1) 
ymean, yvar = magpi.predict(x_test, level = 2) 
yconf = 2 * jnp.sqrt(yvar) 
ytrue = hf_evaluator.evaluate(x_test) 

fill_between(x_test.ravel(), (ymean - yconf).ravel(), (ymean + yconf).ravel(), alpha = 0.3, color = 'green')
plot(x_test.ravel(), ytrue.ravel(), linestyle = 'dotted', color = 'black')
scatter(hf_inputs.ravel(), hf_outputs.ravel())
show()
