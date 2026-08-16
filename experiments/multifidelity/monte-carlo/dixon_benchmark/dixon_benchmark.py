# %% 
from elyza.core.data import ScalarInput 
from elyza.core.evaluator import Evaluator 

from elyza.multifidelity.montecarlo import MFMC

import jax.random as jrand 
import jax.numpy as jnp
from jax import jit


# defining the data sources
x = ScalarInput(name = "x", dim = 1, sampling_func = lambda key: jrand.uniform(key, minval = 0.0, maxval = 1.0))

hf = Evaluator(
    name = "high-fidelity", 
    inputs = [x], 
    output_dim = 3, 
    evaluation_func = lambda x: jnp.array([jnp.sqrt(11) * x ** 5, x **4, jnp.sin(2 * jnp.pi * x)]), 
    cost = 1.0 
)

mf = Evaluator(
    name = "medium-fidelity", 
    inputs = [x], 
    output_dim = 3, 
    evaluation_func = lambda x: jnp.array([jnp.sqrt(7) * x ** 3, x ** 2, jnp.cos(2 * jnp.pi * x + jnp.pi /2)]), 
    cost = 1e-2
)

lf = Evaluator(
    name = "low-fidelity", 
    inputs = [x], 
    output_dim = 3, 
    evaluation_func = lambda x: jnp.array([jnp.sqrt(3) * x ** 2/2, jnp.sqrt(3) * x / 2, jnp.cos(2 * jnp.pi * x + jnp.pi /4)]), 
    cost = 1e-3
)


# 
big_sample = x.sample(jrand.PRNGKey(42), int(1e6))
true_hf_mean = hf.evaluate(big_sample).mean(axis=0) 

# 

mfmc = MFMC(
    evaluators = [lf, mf, hf], 
    l2_reg = 1e-8, 
    rcond = 1e-6
)

mfmc.get_pilots(jrand.PRNGKey(42), n_pilots = int(1e5), set_costs = False) 
mfmc.get_alphas()
mfmc.evaluate(jrand.PRNGKey(42), sample_sizes = [2000, 1000, 500])
mfmc._get_equiv_costs()
mfmc._get_info_coefs()
print(jnp.array(mfmc._info_coefs) / jnp.array(mfmc._equiv_costs))
mfmc.budget_sample_allocation()