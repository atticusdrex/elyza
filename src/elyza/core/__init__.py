"""Core building blocks: model inputs and the evaluator that wraps a computer model.

Exposes :class:`~elyza.core.data.Input` and its
:class:`~elyza.core.data.ScalarInput`/:class:`~elyza.core.data.VectorInput`
specializations, plus :class:`~elyza.core.evaluator.Evaluator`.
"""
from elyza.core.data import Input, ScalarInput, VectorInput
from elyza.core.evaluator import Evaluator