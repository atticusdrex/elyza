# %% 
from elyza.core.data import ScalarInput 
from elyza.core.evaluator import Evaluator 

from elyza.multifidelity.montecarlo import RMFMC, MFMC

import jax.random as jrand 
import jax.numpy as jnp
from jax import jit


# defining the data sources
x = ScalarInput(name = "x", dim = 1, sampling_func = lambda key: jrand.uniform(key, minval = 0.0, maxval = 1.0))

# hf = Evaluator(
#     name = "high-fidelity", 
#     inputs = [x], 
#     output_dim = 3, 
#     evaluation_func = lambda x: jnp.array([jnp.sqrt(11) * x ** 5, x **4, jnp.sin(2 * jnp.pi * x)]), 
#     cost = 1.0 
# )

# mf = Evaluator(
#     name = "medium-fidelity", 
#     inputs = [x], 
#     output_dim = 3, 
#     evaluation_func = lambda x: jnp.array([jnp.sqrt(7) * x ** 3, x ** 2, jnp.cos(2 * jnp.pi * x + jnp.pi /2)]), 
#     cost = 1e-2
# )

# lf = Evaluator(
#     name = "low-fidelity", 
#     inputs = [x], 
#     output_dim = 3, 
#     evaluation_func = lambda x: jnp.array([jnp.sqrt(3) * x ** 2/2, jnp.sqrt(3) * x / 2, jnp.cos(2 * jnp.pi * x + jnp.pi /4)]), 
#     cost = 1e-3
# )

# alternative levels of fidelity: hf is built as the sum of two mutually
# ORTHOGONAL components (shifted Legendre polynomials on [0,1], which are
# exactly uncorrelated under uniform sampling). lf only sees one component
# and mf only sees the other, so individually each is moderately correlated
# with hf, but the linear combination lf + mf reproduces hf exactly.
def L1(x): return 2*x - 1
def L2(x): return 6*x**2 - 6*x + 1
def L3(x): return 20*x**3 - 30*x**2 + 12*x - 1
def L4(x): return 70*x**4 - 140*x**3 + 90*x**2 - 20*x + 1

hf = Evaluator(
    name = "high-fidelity",
    inputs = [x],
    output_dim = 3,
    evaluation_func = lambda x: jnp.array([L2(x) + L3(x), L1(x) + L4(x), L2(x) - 1.5*L3(x)]),
    cost = 1.0
)

mf = Evaluator(
    name = "medium-fidelity",
    inputs = [x],
    output_dim = 3,
    evaluation_func = lambda x: jnp.array([L2(x), L1(x), L2(x)]),
    cost = 1e-2
)

lf = Evaluator(
    name = "low-fidelity",
    inputs = [x],
    output_dim = 3,
    
    evaluation_func = lambda x: jnp.array([L3(x), L4(x), -1.5*L3(x)]),
    cost = 1e-3
)


# 
big_sample = x.sample(jrand.PRNGKey(42), int(1e6))
true_hf_mean = hf.evaluate(big_sample).mean(axis=0) 

# 
rmfmc = RMFMC(
    evaluators = [lf, mf, hf], 
    l2_reg = 1e-6, 
    rcond = 1e-6
)
rmfmc.get_pilots(jrand.PRNGKey(42), n_pilots = int(1e6), set_costs = False) 

rmfmc_sizes = rmfmc.budget_alloc(10.0, warm_start = False)
print(rmfmc_sizes)
print(rmfmc._get_variance(rmfmc_sizes))


# 
mfmc = MFMC(
    evaluators = [lf, mf, hf], 
    l2_reg = 1e-6, 
    rcond = 1e-6
)
mfmc.get_pilots(jrand.PRNGKey(42), n_pilots = int(1e6), set_costs = False) 

mfmc_sizes = mfmc.budget_alloc(10.0, warm_start = False)
print(mfmc_sizes)
print(mfmc._get_variance(mfmc_sizes))