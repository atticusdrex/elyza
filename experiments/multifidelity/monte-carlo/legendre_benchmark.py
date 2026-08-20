# %% 
from elyza.core.data import ScalarInput 
from elyza.core.evaluator import Evaluator 

from elyza.multifidelity.montecarlo import RMFMC, MFMC, MLMC, HFMC

import jax.random as jrand 
import jax.numpy as jnp


# defining the data sources
x = ScalarInput(name = "x", dim = 1, sampling_func = lambda key: jrand.uniform(key, minval = -1.0, maxval = 1.0))

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

# alternative levels of fidelity: hf is built as the sum of two mutually
# ORTHOGONAL components (the standard Legendre polynomials, which are exactly
# uncorrelated under uniform sampling on [-1,1]). lf only sees one component
# and mf only sees the other, so individually each is moderately correlated
# with hf, but the linear combination lf + mf reproduces hf exactly.
def P1(x): return x
def P2(x): return (3*x**2 - 1) / 2
def P3(x): return (5*x**3 - 3*x) / 2
def P4(x): return (35*x**4 - 30*x**2 + 3) / 8
def P5(x): return (63*x**5 - 70*x**3 + 15*x) / 8
def P6(x): return (231*x**6 - 315*x**4 + 105*x**2 - 5) / 16

# mf/lf component i is built from a distinct orthogonal Legendre basis
# function so cross-component correlations can be controlled exactly.
# hf[0]=mf[0]+lf[0] and hf[1]=-mf[1]+lf[1] are same-index (diagonal-visible),
# but with OPPOSITE signs on the mf term -- a single shared scalar can't
# satisfy both, which is what makes scalar coefs noticeably worse.
# hf[2]=mf[0]+lf[1] is a purely CROSS-index combination: vector (diagonal)
# coefficients only ever see mf[2]/lf[2] when predicting hf[2], which are
# orthogonal to it, so vector gets zero benefit there while matrix
# coefficients -- which can mix across components -- reconstruct it exactly.
hf = Evaluator(
    name = "high-fidelity",
    inputs = [x],
    output_dim = 3,
    evaluation_func = lambda x: jnp.array([P1(x) + P4(x), -P2(x) + P5(x), P1(x) + P5(x)]),
    cost = 1.0
)

mf = Evaluator(
    name = "medium-fidelity",
    inputs = [x],
    output_dim = 3,
    evaluation_func = lambda x: jnp.array([P1(x), P2(x), P3(x)]),
    cost = 1e-2
)

lf = Evaluator(
    name = "low-fidelity",
    inputs = [x],
    output_dim = 3,
    evaluation_func = lambda x: jnp.array([P4(x), P5(x), P6(x)]),
    cost = 1e-3
)

import numpy as np 
from tqdm import tqdm 

budgets = jnp.logspace(jnp.log10(5), jnp.log10(300), num = 8) 

estimator_vars = np.zeros((6, budgets.shape[0]))

l2_reg, rcond = 1e-6, 1e-8 

rmfmc = RMFMC(
    evaluators = [lf, mf, hf], 
    l2_reg = l2_reg, 
    rcond = rcond
)
rmfmc.get_pilots(jrand.PRNGKey(42), n_pilots = int(1e7), set_costs = False)

# storing true covariance matrices 
true_covs = rmfmc.covs 

# re-computing rmfmc pilots with fewer pilot samples
rmfmc.get_pilots(jrand.PRNGKey(41), n_pilots = int(1e7), set_costs = False) 
pilot_covs = rmfmc.covs

mfmc = MFMC(
    evaluators = [lf, mf, hf], 
    l2_reg = l2_reg, 
    rcond = rcond
)
mfmc.covs = rmfmc.covs

mlmc = MLMC(evaluators = [lf, mf, hf])
mlmc.covs = rmfmc.covs 
hfmc = HFMC(evaluators = [lf, mf, hf])
hfmc.covs = rmfmc.covs 

for i, budget in tqdm(enumerate(budgets), total = budgets.shape[0]):
    vars = [] 

    # hfmc 
    # ms = hfmc.budget_alloc(budget, warm_start = False)
    ms = hfmc._budget_fractional_alloc(budget) 
    vars.append(hfmc._get_variance(ms)) 

    # mlmc 
    mlmc._get_info_coefs() 
    # ms = mlmc.budget_alloc(budget, warm_start = False) 
    ms = mlmc._budget_fractional_alloc(budget)
    vars.append(mlmc._get_variance(ms))

    # mfmc with matrix coefficients
    mfmc.covs = pilot_covs 
    mfmc.get_matrix_coefs() 
    mfmc.covs = true_covs 
    mfmc._get_info_coefs() 
    # ms = mfmc.budget_alloc(budget, warm_start = False) 
    ms = mfmc._budget_fractional_alloc(budget) 
    vars.append(mfmc._get_variance(ms)) 

    # rmfmc with matrix coefficients 
    rmfmc.covs = pilot_covs  
    rmfmc.get_matrix_coefs()
    rmfmc.covs = true_covs 
    rmfmc._get_info_coefs() 
    # ms = rmfmc.budget_alloc(budget, warm_start = False) 
    ms = rmfmc._budget_fractional_alloc(budget) 
    vars.append(rmfmc._get_variance(ms)) 

    # rmfmc with vector coefficients 
    rmfmc.covs = pilot_covs
    rmfmc.get_vector_coefs()
    rmfmc.covs = true_covs
    rmfmc._get_info_coefs() 
    # ms = rmfmc.budget_alloc(budget, warm_start = False) 
    ms = rmfmc._budget_fractional_alloc(budget) 
    vars.append(rmfmc._get_variance(ms)) 

    # rmfmc with scalar coefficients 
    rmfmc.covs = pilot_covs
    rmfmc.get_scalar_coefs()
    rmfmc.covs = true_covs
    rmfmc._get_info_coefs() 
    # ms = rmfmc.budget_alloc(budget, warm_start = False) 
    ms = rmfmc._budget_fractional_alloc(budget)
    vars.append(rmfmc._get_variance(ms)) 



    # storing the estimator variances for the specific budget
    estimator_vars[:,i] = jnp.array(vars) 


# %% plotting 
from matplotlib.pyplot import * 


rcParams.update(
    {
        "text.usetex": False,  # Turn off external LaTeX
        "mathtext.fontset": "cm",  # Use built-in Computer Modern math font
        "font.family": "serif",  # Use generic serif font for standard labels
        "font.serif": ["DejaVu Serif"],  # Python's built-in serif font
        "axes.labelsize": 14,
        "font.size": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)

labels = ["HFMC (baseline)", "MLMC", "MFMC", "RMFMC w/ matrix coefs.", "RMFMC w/ vector coefs.", "RMFMC w/ scalar coefs." ]
markers = ['.', "D", "P", '*', 's', "^", ]
# the three RMFMC variants share a blue family (dark -> light, richest -> simplest
# coefficients), while HFMC/MFMC/MLMC each get a distinct matplotlib default color
colors = ['#7f7f7f', '#2ca02c', '#ff7f0e', '#08306b', '#4292c6', '#9ecae1']

figure(figsize=(12,7), dpi = 300) 
for i in range(estimator_vars.shape[0]):
    loglog(budgets, estimator_vars[i], linestyle = 'dashed', marker = markers[i], label = labels[i], markersize=12, color = colors[i])

legend() 
xlabel("Computational budget")
ylabel("Trace of estimator covariance matrix")
title("Convergence Comparison for Multivariate Monte Carlo Estimators")
savefig("figs\\analytical_multivariate_benchmark.png")
tight_layout() 