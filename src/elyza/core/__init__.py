"""Core building blocks: model inputs and the evaluator that wraps a computer model.

Exposes :class:`~elyza.core.data.Variable`, plus
:class:`~elyza.core.evaluator.Evaluator`. See :mod:`elyza.core.random` for
sampleable :class:`~elyza.core.random.RandomVariable` subclasses.
"""
from elyza.core.data import Variable
from elyza.core.evaluator import Evaluator
from elyza.core.random import RandomVariable, Uniform, Gaussian, GaussianMixture