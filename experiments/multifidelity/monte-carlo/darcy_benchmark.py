# %% importing evaluators and dependencies
from elyza.benchmarks.pde.darcy2d import GRFInput, DarcyFlowEvaluator 

from elyza.multifidelity.montecarlo import RMFMC, MFMC, MLMC, HFMC

import jax.random as jrand 
import jax.numpy as jnp

from matplotlib.pyplot import * 


# declaring the permeability fields
REFERENCE_DIM, HF_DIM, MF_DIM, LF_DIM = 64, 64, 32, 16
LENGTH_SCALE, KL_TERMS, GRF_MEAN, GRF_STD = 0.15, 64, 0.0, 1.0 

lf_perm = GRFInput(name='permeability', grid_dim=LF_DIM, length_scale=LENGTH_SCALE, n_kl_terms=KL_TERMS, grf_mean=GRF_MEAN, grf_std=GRF_STD, reference_grid_dim=REFERENCE_DIM)
mf_perm = GRFInput(name='permeability', grid_dim=MF_DIM, length_scale=LENGTH_SCALE, n_kl_terms=KL_TERMS, grf_mean=GRF_MEAN, grf_std=GRF_STD, reference_grid_dim=REFERENCE_DIM)
hf_perm = GRFInput(name='permeability', grid_dim=HF_DIM, length_scale=LENGTH_SCALE, n_kl_terms=KL_TERMS, grf_mean=GRF_MEAN, grf_std=GRF_STD, reference_grid_dim=REFERENCE_DIM)

# declaring the darcy field 
lf_darcy = DarcyFlowEvaluator(name='low-fidelity darcy flow', inputs=[lf_perm], grid_dim=LF_DIM, cost=1.4149e-04, source_term = "figs/Pi2.png")

mf_darcy = DarcyFlowEvaluator(name='medium-fidelity darcy flow', inputs=[mf_perm], grid_dim=MF_DIM, cost=2.8317e-03, source_term = "figs/Pi2.png")

hf_darcy = DarcyFlowEvaluator(name='high-fidelity darcy flow', inputs=[hf_perm], grid_dim=HF_DIM, cost=4.4440e-02, source_term = "figs/Pi2.png")


# visualizing the permeability, forcing, and pressure
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
seed = 48
# 
perm_eval = hf_perm.sample(jrand.PRNGKey(seed), 1)

press = hf_darcy.evaluate(perm_eval) 

# high-fidelity sets the color norm (vmin/vmax) shared by every row in each column
perm_hf = perm_eval[0].reshape((HF_DIM, HF_DIM))
force_hf = hf_darcy._f.reshape((HF_DIM, HF_DIM))
press_hf = press.reshape((HF_DIM, HF_DIM))

perm_norm = dict(vmin = float(perm_hf.min()), vmax = float(perm_hf.max()))
force_norm = dict(vmin = float(force_hf.min()), vmax = float(force_hf.max()))
press_norm = dict(vmin = float(press_hf.min()), vmax = float(press_hf.max()))

fig, axes = subplots(3, 3, figsize=(14,12), dpi = 300)

im_perm = axes[0,0].imshow(perm_hf, cmap = 'viridis', **perm_norm)
axes[0,0].set_title("Random sample of $\\kappa(\\boldsymbol{x}; \\xi)$")
axes[0,0].set_ylabel("High-Fidelity", fontsize=18)
axes[0,0].set_xticks([]); axes[0,0].set_yticks([])

im_force = axes[0,1].imshow(force_hf, cmap = 'GnBu', **force_norm)
axes[0,1].set_title("Forcing term $f(\\boldsymbol{x})$")
axes[0,1].set_xticks([]); axes[0,1].set_yticks([])

im_press = axes[0,2].imshow(press_hf, cmap = 'rainbow', **press_norm)
axes[0,2].set_title("Resulting $p(\\boldsymbol{x})$")
axes[0,2].set_xticks([]); axes[0,2].set_yticks([])

perm_eval = mf_perm.sample(jrand.PRNGKey(seed), 1)
press = mf_darcy.evaluate(perm_eval)

axes[1,0].imshow(perm_eval[0].reshape((MF_DIM, MF_DIM)), cmap = 'viridis', **perm_norm)
axes[1,0].set_ylabel("Medium-Fidelity", fontsize=18)
axes[1,0].set_xticks([]); axes[1,0].set_yticks([])

axes[1,1].imshow(mf_darcy._f.reshape((MF_DIM, MF_DIM)), cmap = 'GnBu', **force_norm)
axes[1,1].set_xticks([]); axes[1,1].set_yticks([])

axes[1,2].imshow(press.reshape((MF_DIM, MF_DIM)), cmap = 'rainbow', **press_norm)
axes[1,2].set_xticks([]); axes[1,2].set_yticks([])

perm_eval = lf_perm.sample(jrand.PRNGKey(seed), 1)
press = lf_darcy.evaluate(perm_eval)

axes[2,0].imshow(perm_eval[0].reshape((LF_DIM, LF_DIM)), cmap = 'viridis', **perm_norm)
axes[2,0].set_xlabel("$x$")
axes[2,0].set_ylabel("Low-Fidelity", fontsize=18)
axes[2,0].set_xticks([]); axes[2,0].set_yticks([])

axes[2,1].imshow(lf_darcy._f.reshape((LF_DIM, LF_DIM)), cmap = 'GnBu', **force_norm)
axes[2,1].set_xlabel("$x$")
axes[2,1].set_xticks([]); axes[2,1].set_yticks([])

axes[2,2].imshow(press.reshape((LF_DIM, LF_DIM)), cmap = 'rainbow', **press_norm)
axes[2,2].set_xlabel("$x$")
axes[2,2].set_xticks([]); axes[2,2].set_yticks([])

# one shared, vertically-spanning colorbar per column instead of one per subplot.
# placed via explicit add_axes (rather than colorbar(ax=[...])) since the
# automatic gridspec-based placement collapses to a single row's height when
# the source axes have imshow's equal-aspect box adjustment applied.
cbar_pad, cbar_width = 0.012, 0.012
for col, im in enumerate([im_perm, im_force, im_press]):
    col_positions = [axes[row, col].get_position() for row in range(3)]
    x_right = max(pos.x1 for pos in col_positions)
    y_bottom = min(pos.y0 for pos in col_positions)
    y_top = max(pos.y1 for pos in col_positions)
    cax = fig.add_axes([x_right + cbar_pad, y_bottom, cbar_width, y_top - y_bottom])
    fig.colorbar(im, cax = cax)

# %% running R-MFMC on the darcy flow 

rmfmc = RMFMC(
    evaluators = [lf_darcy, mf_darcy, hf_darcy], 
    l2_reg = 1e-8, 
    rcond = 1e-8
)

rmfmc.get_pilots(jrand.PRNGKey(42), n_pilots = 200, set_costs = True)
true_covs = rmfmc.covs 

#%%
rmfmc.get_pilots(jrand.PRNGKey(43), n_pilots = 175, set_costs = False, noise_std = 1e-10)
bad_covs = rmfmc.covs 

# %% examining regularized least squares solve
budget = 1.0

rmfmc.l2_reg, rmfmc.rcond = 1e-8, 1e-8
rmfmc.covs = bad_covs
rmfmc.get_matrix_coefs() 
rmfmc._get_info_coefs()
good_ms = rmfmc.budget_alloc(budget, warm_start = False)

rmfmc.covs = true_covs 
rmfmc._get_info_coefs() 
rmfmc_var = rmfmc.get_entry_variance(good_ms).reshape(HF_DIM, HF_DIM)
print(good_ms)

#%%  examining unregularized least squares solve 
rmfmc.l2_reg, rmfmc.rcond = 3e-16, 1e-300
rmfmc.covs = bad_covs
rmfmc.get_matrix_coefs() 
rmfmc._get_info_coefs()
bad_ms = rmfmc.budget_alloc(budget, warm_start = False)
print(bad_ms) 
rmfmc.covs = true_covs
rmfmc._get_info_coefs() 
unreg_var = rmfmc.get_entry_variance(bad_ms).reshape(HF_DIM, HF_DIM)



hfmc = HFMC(
    evaluators = [lf_darcy, hf_darcy]
)
hfmc._costs = jnp.copy(rmfmc._costs)

# copying over covariances so don't have to recompute
hfmc.covs = true_covs 

hf_ms = hfmc.budget_alloc(budget)

hfmc_var = hfmc.get_entry_variance(hf_ms).reshape(HF_DIM, HF_DIM)
hf_ms 
# %% 
from matplotlib import colors

# Define norm with custom min (-10), center (0), and max (80)

# 2. Create the continuous colormap

# norm = colors.TwoSlopeNorm(vcenter=1.0, vmin=0, vmax=(hfmc_var / rmfmc_var).max())

figure(figsize=(16,4), dpi = 300)
subplot(1,3,1)
imshow(jnp.diag(true_covs[-1][-1]).reshape(HF_DIM, HF_DIM) / jnp.diag(true_covs[-1][-1]).max(), cmap = "Oranges", vmin=0.0, vmax = 1.0)
colorbar()
title("High-Fidelity Variance")
xticks([]); yticks([])

subplot(1,3,2)
imshow(hfmc_var / unreg_var, cmap = "GnBu", vmin = 1.0, vmax = (hfmc_var/rmfmc_var).max())
colorbar()
title("R-MFMC Variance Reduction\nWithout Regularization")
xticks([]); yticks([])

subplot(1,3,3)
imshow(hfmc_var / rmfmc_var, cmap = "GnBu", vmin = 1.0, vmax = (hfmc_var/rmfmc_var).max())
colorbar()
title("R-MFMC Variance Reduction\nwith Regularization")
xticks([]); yticks([])
# %%
