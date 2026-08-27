"""Gaussian Process surrogate model, kernels, and mean functions.

Exposes :class:`~elyza.surrogate.gp.gp.GaussianProcess` along with the
:class:`~elyza.surrogate.gp.kernel.BaseKernel` and
:class:`~elyza.surrogate.gp.mean.BaseMean` families it composes.
"""
from elyza.surrogate.gp.gp import GaussianProcess
from elyza.surrogate.gp.kernel import BaseKernel, RBF, ARD, Laplace
from elyza.surrogate.gp.mean import BaseMean, Zero, Constant, Linear
