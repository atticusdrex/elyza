"""Hierarchical (multifidelity-augmented) surrogate modeling.

Defines :class:`HierarchicalSurrogate`, a container for one
:class:`~elyza.surrogate.abstract.Surrogate` and one
:class:`~elyza.surrogate.abstract.SupervisedDataset` per level of fidelity,
and :class:`MAGPI` (Multifidelity-Augmented GP Inputs), which fits each
level's surrogate on the concatenation of its own features with the
lower-fidelity surrogates' predictions.
"""
from elyza.surrogate.gp import GaussianProcess, ARD, Linear
from elyza.surrogate.abstract import Surrogate, SupervisedDataset

from elyza.core.evaluator import Evaluator

from elyza.util.imports import *
from elyza.util.helpers import ensure_2d
from elyza.util.preprocessing import StandardScaler

from elyza.optim.abstract import Optimizer, OptimizerOptions

class HierarchicalSurrogate(BaseModel):
    """Base class holding one surrogate/dataset pair per level of fidelity.

    Attributes:
        data: List of individual supervised datasets, one per fidelity level.
        evaluators: List of evaluators, in case data needs to be generated
            on the fly.
        _K: Number of levels of fidelity.
        _surrogates: Per-level surrogate models, set via
            :meth:`MAGPI.set_surrogate`.
        _pred_kwargs: Per-level keyword arguments forwarded to that level's
            ``predict`` call.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # public fields
    data : list[SupervisedDataset] = Field(default = None, description = "list of individual supervised datasets")
    evaluators : list[Evaluator] | None = Field(default = None, description = "list of evaluators in case we want to generate data on the fly")

    # private fields
    _K : int | None = PrivateAttr(default = None)
    _surrogates : list[Surrogate] | None = PrivateAttr(default = None)
    _pred_kwargs : list[list] | None = PrivateAttr(default = None)
    _samp_kwargs : list[list] | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        """Validate ``data``/``evaluators`` and initialize per-level slots.
        """

        # setting the number of levels of fidelity
        self._K = len(self.data)

        # initializing the list of surrogates
        self._surrogates = [None] * self._K

        # initializing prediction keyword arguments
        self._pred_kwargs = [[]] * self._K

        # initializing sampling keyword arguments
        self._samp_kwargs = [[]] * self._K