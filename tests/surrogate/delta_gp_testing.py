# %% 
from elyza.util.imports import * 
from elyza.surrogate.gp import DeltaGP, ARD, Constant
from elyza.core import Uniform 
from elyza.optim import ADAM, ADAMOptions 
from matplotlib.pyplot import * 

n_train = 10
x = Uniform(name = "x", dim = 1, lower = 0.0, upper = 1.0) 
x_train = x.sample(jrand.PRNGKey(42), n_train)
y1_train = jnp.sin(2*pi*x_train)
y2_train = jnp.exp(x_train) 

x_test = jnp.linspace(0,1,1000).reshape(-1,1) 
y1_test = jnp.sin(2*pi*x_test)
y2_test = jnp.exp(x_test)

# %%
model = DeltaGP(
    input_dim = 1, 
    mean_cls = Constant, 
    kernel_cls = ARD, 
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
    epochs = 250, 
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

# setting the optimizer
model.set_optimizer(ADAM, adam_opts)

# fitting to the data 
model.fit(x_train, y1_train, y2_train)



# %%
# predicting on new data 
delta_mean, delta_var = model.predict(x_test) 
delta_conf = 2 * jnp.sqrt(delta_var) 

figure()
plot(x_test.ravel(), y1_test - model.p['rho'] * y2_test, linestyle = 'dotted', color = 'black')
fill_between(x_test.ravel(), delta_mean-delta_conf, delta_mean+delta_conf, alpha = 0.3, color =  'green')
scatter(x_train.ravel(), y1_train - model.p['rho'] * y2_train)