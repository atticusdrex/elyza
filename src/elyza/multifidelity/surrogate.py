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
    data : list[SupervisedDataset] = Field(default = None, description = "list of individual supervised datasets ")
    evaluators : list[Evaluator] | None = Field(default = None, description = "list of evaluators in case we want to generate data on the fly")

    # private fields
    _K : int | None = PrivateAttr(default = None)
    _surrogates : list[Surrogate] | None = PrivateAttr(default = None)
    _pred_kwargs : list[list] | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        """Validate ``data``/``evaluators`` and initialize per-level slots.

        Raises:
            AssertionError: If the number of datasets doesn't match the
                number of evaluators.
        """
        assert len(self.evaluators) == len(self.data), "number of datasets doesn't match number of evaluators"

        # setting the number of levels of fidelity
        self._K = len(self.evaluators)

        # initializing the list of surrogates
        self._surrogates = [None] * self._K

        # initializing prediction keyword arguments
        self._pred_kwargs = [[]] * self._K

class MAGPI(HierarchicalSurrogate):
    """Multifidelity-augmented Gaussian Process inputs model.

    Fits a chain of level-specific surrogates where level ``l``'s features
    are its own inputs concatenated with the predictions of every surrogate
    at levels ``< l``.
    """
    def model_post_init(self, __context):
        """Initialize base hierarchical-surrogate state."""
        super().model_post_init(__context)

    def set_surrogate(self, level : int, surrogate : Surrogate, **pred_kwargs):
        """Assign the surrogate model used for a given level of fidelity.

        Args:
            level: Fidelity level index to assign the surrogate to.
            surrogate: A pre-constructed :class:`~elyza.surrogate.abstract.Surrogate`
                instance for this level.
            **pred_kwargs: Keyword arguments forwarded to this surrogate's
                ``predict`` calls when it is used as a lower-fidelity input
                to another level.
        """
        # declaring the surrogate using the keyword arguments
        self._surrogates[level] = surrogate
        # setting the prediction keyword arguments
        self._pred_kwargs[level] = pred_kwargs

    def set_optimizer(self, level:int, optimizer:Optimizer, optimizer_opts:OptimizerOptions):
        """Assign the optimizer used to fit a given level's surrogate.

        Args:
            level: Fidelity level index.
            optimizer: An :class:`~elyza.optim.abstract.Optimizer` class.
            optimizer_opts: An :class:`~elyza.optim.abstract.OptimizerOptions`
                instance configuring that optimizer.

        Raises:
            AssertionError: If :meth:`set_surrogate` has not been called for
                ``level`` yet.
        """
        assert self._surrogates[level] is not None, "you must use set_surrogate() to assign a surrogate model to this level of fidelity"

        self._surrogates[level].set_optimizer()


    def fit(self, level:int):
        """Fit the surrogate model at a given level of fidelity.

        Requires making predictions at the lower-fidelity surrogate models,
        so those surrogate models must already be fit before calling this
        for ``level``.

        Args:
            level: Fidelity level index to fit.
        """
        # compute the level inputs
        features = self.data[level].concatenate_inputs()
        level_outputs = self.data[level].output_data

        # lower-fidelity outputs
        for lower_level in range(level):
            # obtain the level-specific outputs
            outputs = self._surrogates[lower_level].predict(
                features,
                **self._pred_kwargs[lower_level]
            )

            # if the model returns multiple outputs always take the first arguments
            if type(outputs) is tuple:
                outputs = outputs[0]

            # append the model output to the lf_outputs
            features= jnp.concatenate(
                (features, ensure_2d(outputs)), axis=1
            )

        # fitting the level-specific surrogate models
        self._surrogates[level].fit(
            features,
            level_outputs
        )

    def update(self, new_data : SupervisedDataset, level : int, **kwargs):
        """Append new observations at a level and update its surrogate.

        Args:
            new_data: New observations to append, matching the structure of
                ``self.data[level]``.
            level: Fidelity level index to update.
            **kwargs: Forwarded to the level's surrogate ``update`` call.
        """
        # updating the data with new data
        self.data[level].update(*new_data.input_data, new_data.output_data)

        # updating the surrogate model with the new data
        self._surrogates[level].update(
            new_data.concatenate_inputs(),
            new_data.output_data,
            **kwargs
        )

    def predict(self, *new_inputs : jax.Array, level : int, **pred_kwargs) -> jax.Array | tuple[jax.Array]:
        """Predict at a given level of fidelity for new inputs.

        Chains predictions through every lower-fidelity level first, since
        this level's surrogate expects those predictions concatenated onto
        its own input features.

        Args:
            *new_inputs: Raw input arrays for the query points.
            level: Fidelity level to predict at.
            **pred_kwargs: Keyword arguments forwarded to this level's
                surrogate ``predict`` call.

        Returns:
            jax.Array | tuple[jax.Array]: The prediction returned by the
            level-``level`` surrogate's ``predict`` method.
        """
        # compute the level inputs
        features = jnp.concatenate(new_inputs)

        # lower-fidelity outputs
        for lower_level in range(level):
            outputs = self._surrogates[lower_level].predict(
                features, **self._pred_kwargs[lower_level]
            )

            # if the model returns multiple outputs always take the first arguments
            if type(outputs) is tuple:
                outputs = outputs[0]

            # append the model output to the lf_outputs
            features = jnp.concatenate((features, ensure_2d(outputs)), axis=1)

        # making the prediction at this level
        return self._surrogates[level].predict(
            features, **pred_kwargs
        )

    def sample(self, *new_inputs : jax.Array, level:int):
        """Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("this method has not been implemented yet")
