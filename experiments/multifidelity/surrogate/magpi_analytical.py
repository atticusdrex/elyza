# %% 
from elyza.multifidelity.surrogate import MAGPI 
from elyza.benchmarks.multifidelity.magpi_analytical import * 
from elyza.surrogate.gp import GaussianProcess, ARD, Linear, Constant
from elyza.optim.gradient import ADAM, LBFGS
from elyza.surrogate import SupervisedDataset
from elyza.util.imports import * 

from matplotlib.pyplot import * 

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
    verbose = True
)
 
lf_gp.set_optimizer(LBFGS, m=25, constraints = None) 


# building the medium-fidelity gp 
mf_inputs = x.sample(jrand.PRNGKey(42), 100)
mf_outputs = mf_evaluator.evaluate(mf_inputs) 

mf_data = SupervisedDataset(
    input_data = [mf_inputs], 
    output_data = mf_outputs, 
    noise_var = 1e-4
)

mf_gp = GaussianProcess(
    input_dim = 2, # note the +1 dimensional input 
    kernel_cls = ARD, 
    mean_cls = Linear, 
    noise_var = mf_data.noise_var, 
    eps = 1e-12, 
    max_cond = 1e5, 
    verbose = True, 
    calibrate_noise = True
)

# constraint set to zero out the input part of the linear mean parameters 
mf_gp.set_optimizer(LBFGS, m=25, constraints = {'mean':lambda a: a.at[1:1+x.dim].set(0)})

# 
hf_inputs = x.sample(jrand.PRNGKey(42), 25)
hf_outputs = hf_evaluator.evaluate(hf_inputs) 

hf_data = SupervisedDataset(
    input_data = [hf_inputs], 
    output_data = hf_outputs, 
    noise_var = 1e-2
)

hf_gp = GaussianProcess(
    input_dim = 3, # note the +1 dimensional input 
    kernel_cls = ARD, 
    mean_cls = Linear, 
    noise_var = lf_data.noise_var, 
    eps = 1e-12, 
    max_cond = 1e5, 
    verbose = True, 
    calibrate_noise = True
)

# constraint set to zero out the input part of the linear mean parameters 
hf_gp.set_optimizer(LBFGS, m = 25, constraints = {'mean':lambda a: a.at[1:1+x.dim].set(0)})

# 
magpi = MAGPI(
    data = [lf_data, mf_data, hf_data], 
    evaluators = [lf_evaluator, mf_evaluator, hf_evaluator]
)

# setting the surrogates for each level 
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
magpi.fit(
    0, 
    p_init = magpi._surrogates[0].p,
    active_params = {'kernel':True, 'mean':True, 'noise':False}, 
    lr = 1e-1, 
    steps = 100, 
    verbose = True
)

figure()
x_test = jnp.linspace(0,5,1000).reshape(-1,1) 
ymean, yvar = magpi.predict(x_test, level = 0) 
yconf = 2 * jnp.sqrt(yvar) 
ytrue = lf_evaluator.evaluate(x_test) 

fill_between(x_test.ravel(), (ymean - yconf).ravel(), (ymean + yconf).ravel(), alpha = 0.3, color = 'green')
plot(x_test.ravel(), ytrue.ravel(), linestyle = 'dotted', color = 'black')
scatter(lf_inputs.ravel(), lf_outputs.ravel())
show()

# %% training the medium-fidelity surrogate
magpi.fit(
    1, 
    p_init = magpi._surrogates[1].p,
    active_params = {'kernel':True, 'mean':True, 'noise':False}, 
    lr = 1e-1, 
    steps = 100, 
    verbose = True
)

figure()
x_test = jnp.linspace(0,5,1000).reshape(-1,1) 
ymean, yvar = magpi.predict(x_test, level = 1) 
yconf = 2 * jnp.sqrt(yvar) 
ytrue = mf_evaluator.evaluate(x_test) 

fill_between(x_test.ravel(), (ymean - yconf).ravel(), (ymean + yconf).ravel(), alpha = 0.3, color = 'green')
plot(x_test.ravel(), ytrue.ravel(), linestyle = 'dotted', color = 'black')
scatter(mf_inputs.ravel(), mf_outputs.ravel())
show() 
# %% training the medium-fidelity surrogate
magpi.fit(
    2, 
    p_init = magpi._surrogates[2].p,
    active_params = {'kernel':True, 'mean':True, 'noise':False}, 
    lr = 1.0, 
    steps = 250, 
    verbose = True
)


figure()
x_test = jnp.linspace(0,5,1000).reshape(-1,1) 
ymean, yvar = magpi.predict(x_test, level = 2) 
yconf = 2 * jnp.sqrt(yvar) 
ytrue = hf_evaluator.evaluate(x_test) 

fill_between(x_test.ravel(), (ymean - yconf).ravel(), (ymean + yconf).ravel(), alpha = 0.3, color = 'green')
plot(x_test.ravel(), ytrue.ravel(), linestyle = 'dotted', color = 'black')
scatter(hf_inputs.ravel(), hf_outputs.ravel())
show()
# %%
