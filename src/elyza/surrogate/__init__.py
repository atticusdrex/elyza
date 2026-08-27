"""Surrogate (ML) models sharing the common :class:`Surrogate` interface.

Exposes the :class:`~elyza.surrogate.abstract.Surrogate` /
:class:`~elyza.surrogate.abstract.SupervisedDataset` base classes along with
the :class:`~elyza.surrogate.gp.gp.GaussianProcess` and
:class:`~elyza.surrogate.dnn.dnn.MLPRegressor` implementations.
"""
from elyza.surrogate.abstract import Surrogate, SupervisedDataset
from elyza.surrogate.gp import GaussianProcess
from elyza.surrogate.dnn import MLPRegressor