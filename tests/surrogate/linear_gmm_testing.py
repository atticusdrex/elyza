# %% 
from elyza.util.imports import * 
from elyza.surrogate.linear.gmm import GMMRegression
from elyza.util.preprocessing import KernelFeatures 
from elyza.core import Uniform, Evaluator
from matplotlib.pyplot import * 
from elyza.optim import ADAM, ADAMOptions
from elyza.surrogate.gp import ARD

# defining the input 
n_train = 1000
x = Uniform(name = "x", dim = 1, lower = 0.0, upper = 1.0) 
y = Evaluator(name = "y", inputs = [x], output_dim = 1, evaluation_func = lambda x: jnp.sin(2*jnp.pi*x)**2, jit_compile=False)
x_train = x.sample(jrand.PRNGKey(42), n_train)
y_train = y.evaluate(x_train) + 5e-2 * jrand.normal(jrand.PRNGKey(43), shape = x_train.shape) 

x_train = jnp.concatenate((x_train, x_train), axis=0) 
y_train = jnp.concatenate((y_train, -y_train), axis=0) 

# making kernel features 
features = KernelFeatures(
    input_dim = 1, kernel_cls = ARD, 
    eps = 1e-12
)

centers = jnp.linspace(0,1,50).reshape(-1,1) 

f_train = features.fit_transform(x_train, jnp.array([1.0, 1e-2]), centers) 

figure() 
scatter(x_train.ravel(), y_train.ravel(), s = 5.0, color = 'black') 

# declaring the GMM regression model 
model = GMMRegression(input_dim = 50, n_dist = 1, scale = 1e0, random_state = 42)

# %% declaring the model options 
adam_opts = ADAMOptions(
    p_init = None, 
    lr = 1e-3, 
    epochs = 10000, 
    batch_size = None, 
    beta1 = 0.95, 
    beta2 = 0.999, 
    active_params = None, 
    constraints = None, 
    verbose = True, 
    eps = 1e-8, 
    random_state = 42, 
    unroll = 25
)

# defining the model 

model.set_optimizer(
    ADAM, adam_opts 
)
# 
model.fit(f_train, y_train)

# %% making predictions from the model 
x_test = jnp.linspace(0, 1, 1000).reshape(-1,1) 
f_test = features.transform(x_test) 
ysamp = model.sample(
    jrand.PRNGKey(45), f_test, n_samples = 100
)

figure()
scatter(x_train.ravel(), y_train.ravel(), s = 5.0, color = 'black', alpha = 0.1) 

for i in range(ysamp.shape[1]): 
    scatter(x_test.ravel(), ysamp[:,i], s = 5.0, marker = ".", color = 'green', alpha = 0.01)

ylim(-0.1, 1.1)



