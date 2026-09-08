from elyza.multifidelity.surrogate.abstract import *
from elyza.surrogate.gp import DeltaGP

class KOH(HierarchicalSurrogate):
    """Kennedy-O'Hagan autoregressive multifidelity Gaussian Process.

    Models level 0 directly with a standalone
    :class:`~elyza.surrogate.abstract.Surrogate` (e.g. a plain
    :class:`~elyza.surrogate.gp.gp.GaussianProcess`), and every level
    ``l > 0`` as the autoregressive recursion
    ``y_l(x) = rho_l * y_{l-1}(x) + delta_l(x)``, where ``delta_l`` is a
    :class:`~elyza.surrogate.gp.deltagp.DeltaGP` fit to the scaled
    discrepancy between level ``l`` and level ``l - 1``. Unlike
    :class:`~elyza.multifidelity.surrogate.magpi.MAGPI`, which augments a
    level's inputs with every lower-fidelity prediction, KOH's Markov
    property means each level only ever depends on the one immediately
    below it.
    """
    def model_post_init(self, __context):
        """Initialize base hierarchical-surrogate state."""
        super().model_post_init(__context)

    def set_surrogate(self, level : int, surrogate : Surrogate, samp_kwargs : dict | None = None, **pred_kwargs):
        """Assign the surrogate model used for a given level of fidelity.

        Args:
            level: Fidelity level index to assign the surrogate to. Level
                ``0`` must be a standalone :class:`Surrogate` (e.g.
                :class:`~elyza.surrogate.gp.gp.GaussianProcess`) fit
                directly to that level's data; every level ``> 0`` must be
                a :class:`~elyza.surrogate.gp.deltagp.DeltaGP` modeling the
                discrepancy with the level below it.
            samp_kwargs: Keyword arguments forwarded to this surrogate's
                ``sample`` calls when it is used as the lower-fidelity input
                to the level above it.
            **pred_kwargs: Keyword arguments forwarded to this surrogate's
                ``predict`` calls when it is used as the lower-fidelity input
                to the level above it.
        """
        # declaring the surrogate using the keyword arguments
        self._surrogates[level] = surrogate
        # setting the prediction keyword arguments
        self._pred_kwargs[level] = pred_kwargs
        # setting the sampling keyword arguments
        self._samp_kwargs[level] = samp_kwargs if samp_kwargs is not None else {}

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

        self._surrogates[level].set_optimizer(optimizer, optimizer_opts)

    def fit(self, level:int):
        """Fit the surrogate model at a given level of fidelity.

        Level ``0`` is fit directly to its own data. Level ``l > 0``
        requires a posterior-mean prediction from level ``l - 1`` at this
        level's inputs, so that level's surrogate must already be fit
        before calling this for ``level``.

        Args:
            level: Fidelity level index to fit.
        """
        # compute the level inputs
        features = self.data[level].concatenate_inputs()
        level_outputs = self.data[level].output_data

        if level == 0:
            self._surrogates[0].fit(features, level_outputs)
            return

        # predicted low-fidelity signal y_{l-1}(x) at this level's inputs
        lower_outputs = self._surrogates[level - 1].predict(
            features, **self._pred_kwargs[level - 1]
        )

        # if the model returns multiple outputs always take the first argument
        if type(lower_outputs) is tuple:
            lower_outputs = lower_outputs[0]

        # fitting the discrepancy model delta_l(x) = y_l(x) - rho_l * y_{l-1}(x)
        self._surrogates[level].fit(
            features, level_outputs, lower_outputs
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
        self.data[level].update(*new_data.input_data, new_outputs=new_data.output_data)

        features = new_data.concatenate_inputs()

        if level == 0:
            self._surrogates[0].update(features, new_data.output_data, **kwargs)
            return

        # predicted low-fidelity signal y_{l-1}(x) at the new inputs
        lower_outputs = self._surrogates[level - 1].predict(
            features, **self._pred_kwargs[level - 1]
        )

        if type(lower_outputs) is tuple:
            lower_outputs = lower_outputs[0]

        # updating the discrepancy model with the new observations
        self._surrogates[level].update(
            features, new_data.output_data, lower_outputs, **kwargs
        )

    def predict(self, *new_inputs : jax.Array, level : int, **pred_kwargs) -> tuple[jax.Array]:
        """Predict at a given level of fidelity for new inputs.

        Builds up the posterior mean/variance of ``y_level`` by propagating
        the autoregressive recursion ``y_l = rho_l * y_{l-1} + delta_l``
        from level 0 up to ``level``, combining each level's variance
        additively since ``y_{l-1}`` and ``delta_l`` are modeled as
        independent GPs.

        Args:
            *new_inputs: Raw input arrays for the query points.
            level: Fidelity level to predict at.
            **pred_kwargs: Keyword arguments forwarded to this level's
                surrogate ``predict`` call.

        Returns:
            tuple[jax.Array]: ``(mu, var)``, the posterior mean and
            (co)variance of ``y_level`` at ``new_inputs``.
        """
        # compute the level inputs
        features = jnp.concatenate(new_inputs)

        # posterior of the lowest-fidelity level, modeled directly
        mu, var = self._surrogates[0].predict(
            features, **(pred_kwargs if level == 0 else self._pred_kwargs[0])
        )

        # propagate the autoregressive recursion up through every intermediate level
        for cur_level in range(1, level + 1):
            kwargs = pred_kwargs if cur_level == level else self._pred_kwargs[cur_level]

            mu_delta, var_delta = self._surrogates[cur_level].predict(features, **kwargs)
            rho = self._surrogates[cur_level].p['rho']

            mu = mu_delta + rho * mu
            var = var_delta + rho**2 * var

        return mu, var

    def sample(self, *new_inputs : jax.Array, key: jax.Array, level:int, n_points:int, **kwargs) -> jax.Array:
        """Draw posterior samples at a given level of fidelity for new inputs.

        Builds up posterior samples of ``y_level`` by propagating the
        autoregressive recursion ``y_l = rho_l * y_{l-1} + delta_l`` from
        level 0 up to ``level``, drawing independent samples of ``y_{l-1}``
        and ``delta_l`` at each step and combining them.

        Args:
            *new_inputs: Raw input arrays for the query points.
            key: A JAX PRNG key, split into ``self._K`` per-level keys.
            level: Fidelity level to sample at.
            n_points: Number of posterior samples to draw at ``level``.
            **kwargs: Keyword arguments forwarded to this level's
                surrogate ``sample`` call.

        Returns:
            jax.Array: Samples of ``y_level``, shape ``(n_points_X, n_points)``.
        """
        # split the key into one unique key per level of fidelity
        level_keys = jrand.split(key, self._K)

        # compute the level inputs
        features = jnp.concatenate(new_inputs)

        # posterior samples of the lowest-fidelity level, modeled directly
        samples = self._surrogates[0].sample(
            level_keys[0], features, n_points, **(kwargs if level == 0 else self._samp_kwargs[0])
        )

        # propagate the autoregressive recursion up through every intermediate level
        for cur_level in range(1, level + 1):
            level_kwargs = kwargs if cur_level == level else self._samp_kwargs[cur_level]

            delta_samples = self._surrogates[cur_level].sample(
                level_keys[cur_level], features, n_points, **level_kwargs
            )
            rho = self._surrogates[cur_level].p['rho']

            samples = delta_samples + rho * samples

        return samples
