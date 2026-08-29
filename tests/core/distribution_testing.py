# %% 
from elyza.util.imports import * 
from elyza.core.distribution import Gaussian, GaussianMixture
from seaborn import kdeplot

# testing gaussian distribution 
x = Gaussian(0.0, 1.0) 

print(x.log_pdf(0.5,p=None))

x = Gaussian(jnp.array([0.0, 0.0]), jnp.array([[1.0, 0.0], [0.0, 1.0]]))

print(x.log_pdf(jnp.array([0.5, 0.5]), p=None))

# %% testing gaussian mixture model 

x = GaussianMixture([0.0, 5.0, 10.0], [1.0, 1.0, 2.0], [0.25, 0.5, 0.25]) 

sample = x.sample(jrand.PRNGKey(42), int(1e6)) 

kdeplot(sample.ravel(), fill=True, alpha = 0.3, bw_adjust = 0.25) 
