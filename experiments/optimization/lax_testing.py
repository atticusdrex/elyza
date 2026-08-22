from elyza.benchmarks.multifidelity.magpi_analytical import x, hf_evaluator 
from elyza.util.imports import * 
from matplotlib.pyplot import * 
from elyza.surrogate.gp import GaussianProcess, ARD, Constant
from elyza.optim.gradient import LaxBatchADAM, ADAM

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

# setting the optimizer
model.set_optimizer(LaxBatchADAM, eps = 1e-8)

# fitting to the data 
model.fit(
    input_data, 
    output_data, 
    p_init = model.p, 
    lr = 1e-1, 
    epochs = 250, 
    batch_size = 25,
    key = jrand.PRNGKey(42),
    beta1 = 0.9, 
    beta2 = 0.999, 
    active_params = {'mean':True, 'kernel':True, 'noise':True}, 
    verbose = True
)

# predicting on new data 
test_inputs = jnp.linspace(0,5,1000).reshape(-1,1)
ymean, yvar = model.predict(test_inputs) 
yconf = 2 * jnp.sqrt(yvar) 

figure()
plot(test_inputs.ravel(), hf_evaluator.evaluate(test_inputs).ravel(), linestyle = 'dotted', color = 'black')
fill_between(test_inputs.ravel(), ymean-yconf, ymean+yconf, alpha = 0.3, color =  'green')
show()
