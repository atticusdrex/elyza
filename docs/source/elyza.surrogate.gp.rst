elyza.surrogate.gp package
==========================

Gaussian Process regression surrogate. :class:`~elyza.surrogate.gp.gp.GaussianProcess`
pairs a pluggable :mod:`~elyza.surrogate.gp.kernel` with a pluggable
:mod:`~elyza.surrogate.gp.mean`, fits hyperparameters by maximizing the
log-marginal-likelihood, and supports incremental low-rank (block Cholesky)
updates so new data can be ingested without a full refit.

Submodules
----------

.. toctree::
   :maxdepth: 4

   elyza.surrogate.gp.gp
   elyza.surrogate.gp.kernel
   elyza.surrogate.gp.mean

Module contents
---------------

.. automodule:: elyza.surrogate.gp
   :members:
   :show-inheritance:
   :undoc-members:
