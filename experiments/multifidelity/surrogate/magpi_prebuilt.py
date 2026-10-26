# %%
from elyza.multifidelity.surrogate.magpi import MAGPI_prebuilt
from elyza.benchmarks.multifidelity.magpi_analytical import *
from elyza.surrogate.gp import ARD, Constant
from elyza.surrogate import SupervisedDataset
from elyza.util.imports import *

from matplotlib.pyplot import *

# ADAM optimizer options, passed to MAGPI_prebuilt.fit() for each level
fit_opts = dict(
    lr = 3e-1,
    epochs = 1000,
    batch_size = None,
    beta1 = 0.9,
    beta2 = 0.999,
    active_params = {'mean':True, 'kernel':True, 'noise':False},
    constraints = None,
    random_state = 42,
    unroll = 25
)

# lowest-fidelity data
lf_inputs = x.sample(jrand.PRNGKey(42), 250)
lf_outputs = lf_evaluator.evaluate(lf_inputs)

lf_data = SupervisedDataset(
    input_data = [lf_inputs],
    output_data = lf_outputs,
    noise_var = 1e-4
)

# medium-fidelity data
mf_inputs = x.sample(jrand.PRNGKey(42), 100)
mf_outputs = mf_evaluator.evaluate(mf_inputs)

mf_data = SupervisedDataset(
    input_data = [mf_inputs],
    output_data = mf_outputs,
    noise_var = 1e-4
)

# highest-fidelity data
hf_inputs = x.sample(jrand.PRNGKey(42), 25)
hf_outputs = hf_evaluator.evaluate(hf_inputs)

hf_data = SupervisedDataset(
    input_data = [hf_inputs],
    output_data = hf_outputs,
    noise_var = 1e-4
)

# %% instantiating the prebuilt MAGPI model (builds one GP per level internally)
magpi = MAGPI_prebuilt(
    data = [lf_data, mf_data, hf_data],
    evaluators = [lf_evaluator, mf_evaluator, hf_evaluator],
    input_dim = 1,
    kernel_cls = ARD,
    mean_cls = Constant,
    eps = 1e-8,
    max_cond = 1e5,
    verbose = True,
    calibrate_noise = True
)

# number of posterior samples used to estimate the mean/confidence band
n_samples = 500

# %% training the lowest-fidelity surrogate
magpi.fit(0, **fit_opts)

figure()
x_test = jnp.linspace(0,5,1000).reshape(-1,1)
samples = magpi.sample(x_test, key = jrand.PRNGKey(0), level = 0, n_points = n_samples)
ymean = samples.mean(axis=1)
yconf = 2 * samples.std(axis=1)
ytrue = lf_evaluator.evaluate(x_test)

fill_between(x_test.ravel(), (ymean - yconf).ravel(), (ymean + yconf).ravel(), alpha = 0.3, color = 'green')
plot(x_test.ravel(), ytrue.ravel(), linestyle = 'dotted', color = 'black')
scatter(lf_inputs.ravel(), lf_outputs.ravel())
show()

# %% training the medium-fidelity surrogate
magpi.fit(1, **fit_opts)

figure()
x_test = jnp.linspace(0,5,1000).reshape(-1,1)
samples = magpi.sample(x_test, key = jrand.PRNGKey(1), level = 1, n_points = n_samples)
ymean = samples.mean(axis=1)
yconf = 2 * samples.std(axis=1)
ytrue = mf_evaluator.evaluate(x_test)

fill_between(x_test.ravel(), (ymean - yconf).ravel(), (ymean + yconf).ravel(), alpha = 0.3, color = 'green')
plot(x_test.ravel(), ytrue.ravel(), linestyle = 'dotted', color = 'black')
scatter(mf_inputs.ravel(), mf_outputs.ravel())
show()

# %% training the highest-fidelity surrogate
magpi.fit(2, **fit_opts)

figure()
x_test = jnp.linspace(0,5,1000).reshape(-1,1)
samples = magpi.sample(x_test, key = jrand.PRNGKey(2), level = 2, n_points = n_samples)
ymean = samples.mean(axis=1)
yconf = 2 * samples.std(axis=1)
ytrue = hf_evaluator.evaluate(x_test)

fill_between(x_test.ravel(), (ymean - yconf).ravel(), (ymean + yconf).ravel(), alpha = 0.3, color = 'green')
plot(x_test.ravel(), ytrue.ravel(), linestyle = 'dotted', color = 'black')
scatter(hf_inputs.ravel(), hf_outputs.ravel())
show()
