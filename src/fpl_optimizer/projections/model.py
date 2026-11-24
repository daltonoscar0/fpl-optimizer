"""The own projection model: a three-part generalized linear model in JAX.

Why three parts rather than one regression on points
----------------------------------------------------
In FPL, most of the variance in a player's score is *whether they played*. A
£12m midfielder on the bench scores 0 or 1; the same player starting scores 2 to
15. A single regression on points spends its capacity blending those two
regimes. Splitting them keeps the model honest about what it knows:

    E[points] = P(start) · E[points | start] + P(cameo) · E[points | cameo]

Three heads, all linear in the features, all fitted jointly:

``p60``
    Logistic. Probability the player plays 60 minutes or more — FPL's threshold
    for the second appearance point and for clean-sheet eligibility.
``p_cameo``
    Logistic, parameterized as ``(1 - p60) · σ(w·x)`` so that
    ``P(cameo) + P(start) ≤ 1`` holds by construction rather than by luck.
``mu60`` / ``mu_cameo``
    Poisson means (log link) for points conditional on each regime, each fitted
    on the subset of rows in that regime.

Every head is linear, so the fitted model is fully explainable: the standardized
coefficients printed by :meth:`JaxPointsModel.coefficient_table` say directly
how much each feature moves playing-time probability and conditional points.

Leakage
-------
:meth:`fit` takes a :class:`~fpl_optimizer.features.FeatureCache` and asks it
only for gameweeks before ``upto_gw``. The cache built each of those gameweeks'
features from *its* own past. No path exists from a gameweek's outcome to a
decision made for it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from ..features import FEATURE_COLUMNS, FeatureCache
from ..schema import SeasonData
from .base import aggregate_to_gw, empty_predictions
from .baselines import (
    EP_HINDSIGHT_THRESHOLD,
    EPProjection,
    ep_looks_like_hindsight,
)

log = logging.getLogger(__name__)

jax.config.update("jax_enable_x64", False)

#: Points are clipped below at zero for the Poisson heads. Negative scores (own
#: goals, red cards) are rare; the share clipped is reported in `fit_report`.
_MIN_POINTS = 0.0


@dataclass
class ModelParams:
    """Fitted weights. Each head is ``(weights, bias)`` over standardized features."""

    w_p60: np.ndarray
    b_p60: float
    w_cameo: np.ndarray
    b_cameo: float
    w_mu60: np.ndarray
    b_mu60: float
    w_mucameo: np.ndarray
    b_mucameo: float
    mean: np.ndarray
    std: np.ndarray


def _init(n_features: int, key: jax.Array) -> dict[str, jnp.ndarray]:
    keys = jax.random.split(key, 4)
    scale = 0.01
    return {
        "w_p60": jax.random.normal(keys[0], (n_features,)) * scale,
        "b_p60": jnp.array(0.0),
        "w_cameo": jax.random.normal(keys[1], (n_features,)) * scale,
        "b_cameo": jnp.array(0.0),
        "w_mu60": jax.random.normal(keys[2], (n_features,)) * scale,
        # Opening guess of roughly e^1.1 ≈ 3 points for a player who starts.
        "b_mu60": jnp.array(1.1),
        "w_mucameo": jax.random.normal(keys[3], (n_features,)) * scale,
        # Roughly e^0 = 1 point for a substitute appearance.
        "b_mucameo": jnp.array(0.0),
    }


def _heads(params: dict[str, jnp.ndarray], X: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    """``(p60, p_cameo, mu60, mu_cameo)`` for standardized features ``X``."""
    logit60 = X @ params["w_p60"] + params["b_p60"]
    p60 = jax.nn.sigmoid(logit60)
    # Nested parameterization guarantees p60 + p_cameo <= 1.
    p_cameo = (1.0 - p60) * jax.nn.sigmoid(X @ params["w_cameo"] + params["b_cameo"])
    mu60 = jnp.exp(jnp.clip(X @ params["w_mu60"] + params["b_mu60"], -6.0, 4.0))
    mu_cameo = jnp.exp(
        jnp.clip(X @ params["w_mucameo"] + params["b_mucameo"], -6.0, 4.0)
    )
    return p60, p_cameo, mu60, mu_cameo


def _loss(
    params: dict[str, jnp.ndarray],
    X: jnp.ndarray,
    y_points: jnp.ndarray,
    is60: jnp.ndarray,
    is_cameo: jnp.ndarray,
    l2: float,
) -> jnp.ndarray:
    """Joint negative log-likelihood of the three heads, plus L2."""
    p60, p_cameo, mu60, mu_cameo = _heads(params, X)
    eps = 1e-6

    # Head 1: did they reach 60 minutes? Over every row.
    nll_p60 = -jnp.mean(
        is60 * jnp.log(p60 + eps) + (1.0 - is60) * jnp.log(1.0 - p60 + eps)
    )

    # Head 2: given they did not reach 60, did they appear at all? The nested
    # form means the conditional probability is sigmoid(w_cameo . x), so the
    # target is `is_cameo` restricted to rows with is60 == 0.
    not60 = 1.0 - is60
    cond_cameo = p_cameo / (1.0 - p60 + eps)
    denom2 = jnp.maximum(jnp.sum(not60), 1.0)
    nll_cameo = -jnp.sum(
        not60
        * (
            is_cameo * jnp.log(cond_cameo + eps)
            + (1.0 - is_cameo) * jnp.log(1.0 - cond_cameo + eps)
        )
    ) / denom2

    # Heads 3 and 4: Poisson points within each regime.
    def poisson_nll(mu: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        return jnp.sum(mask * (mu - y_points * jnp.log(mu + eps))) / denom

    nll_mu60 = poisson_nll(mu60, is60)
    nll_mucameo = poisson_nll(mu_cameo, is_cameo)

    penalty = l2 * sum(
        jnp.sum(params[k] ** 2)
        for k in ("w_p60", "w_cameo", "w_mu60", "w_mucameo")
    )
    return nll_p60 + nll_cameo + nll_mu60 + nll_mucameo + penalty


@jax.jit
def _expected_points(
    params: dict[str, jnp.ndarray], X: jnp.ndarray
) -> jnp.ndarray:
    p60, p_cameo, mu60, mu_cameo = _heads(params, X)
    return p60 * mu60 + p_cameo * mu_cameo


@dataclass
class JaxPointsModel:
    """Walk-forward three-part points model.

    Parameters
    ----------
    min_train_gws:
        Gameweeks of history required before the model's own predictions are
        used. Below this it defers to FPL's published expected points — but only
        where that column is a genuine forecast, which on the archive it is not;
        see :meth:`_fallback_is_a_forecast`. Either way the hand-off is reported
        rather than hidden, in :attr:`fallback_gws` and :attr:`no_view_gws`.
    """

    name: str = "model"
    learning_rate: float = 0.05
    steps: int = 400
    l2: float = 1e-3
    min_train_gws: int = 3
    seed: int = 0
    form_window: int = 6

    params: dict[str, jnp.ndarray] | None = field(default=None, repr=False)
    _cache: FeatureCache | None = field(default=None, repr=False)
    _mean: np.ndarray | None = field(default=None, repr=False)
    _std: np.ndarray | None = field(default=None, repr=False)
    _fallback: EPProjection = field(default_factory=EPProjection, repr=False)
    fitted_upto: int | None = None
    #: Gameweeks where the model had too little history and deferred to the
    #: baseline. Printed by the backtester so the hand-off is visible.
    fallback_gws: list[int] = field(default_factory=list)
    #: Gameweeks where it had too little history *and* the baseline could not be
    #: trusted, so it reported no view. See :meth:`_fallback_is_a_forecast`.
    no_view_gws: list[int] = field(default_factory=list)
    _fallback_trusted: bool | None = field(default=None, repr=False)
    fit_report: dict[str, float] = field(default_factory=dict)

    def attach(self, cache: FeatureCache) -> None:
        """Reuse a prebuilt feature cache instead of building a private one."""
        self._cache = cache

    def _ensure_cache(self, data: SeasonData) -> FeatureCache:
        if self._cache is None or self._cache.data is not data:
            self._cache = FeatureCache(data, form_window=self.form_window)
        return self._cache

    # --- training -----------------------------------------------------------

    def fit(self, data: SeasonData, upto_gw: int) -> None:
        """Fit on every gameweek before ``upto_gw``."""
        cache = self._ensure_cache(data)
        X_df, y_df = cache.training_set(upto_gw)
        self.fitted_upto = upto_gw
        n_gws = int(y_df["gw"].nunique()) if len(y_df) else 0
        if len(X_df) < 200 or n_gws < self.min_train_gws:
            self.params = None
            log.info(
                "gw%d: %d training rows over %d gameweeks — deferring to the "
                "FPL expected-points baseline",
                upto_gw,
                len(X_df),
                n_gws,
            )
            return

        X = X_df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        minutes = y_df["minutes"].to_numpy(dtype=np.float32)
        points = y_df["points"].to_numpy(dtype=np.float32)

        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-6] = 1.0
        Xs = jnp.asarray((X - self._mean) / self._std)

        is60 = jnp.asarray((minutes >= 60).astype(np.float32))
        is_cameo = jnp.asarray(
            ((minutes > 0) & (minutes < 60)).astype(np.float32)
        )
        y_clipped = np.maximum(points, _MIN_POINTS)
        self.fit_report = {
            "rows": float(len(X)),
            "gameweeks": float(n_gws),
            "share_60plus": float(np.mean(minutes >= 60)),
            "share_cameo": float(np.mean((minutes > 0) & (minutes < 60))),
            "share_points_clipped": float(np.mean(points < _MIN_POINTS)),
        }
        y = jnp.asarray(y_clipped)

        params = _init(len(FEATURE_COLUMNS), jax.random.PRNGKey(self.seed))
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)
        loss_and_grad = jax.value_and_grad(_loss)

        @jax.jit
        def step(params, opt_state):
            loss, grads = loss_and_grad(params, Xs, y, is60, is_cameo, self.l2)
            updates, opt_state = optimizer.update(grads, opt_state)
            return optax.apply_updates(params, updates), opt_state, loss

        loss = jnp.array(float("nan"))
        for _ in range(self.steps):
            params, opt_state, loss = step(params, opt_state)
        self.params = params
        self.fit_report["final_loss"] = float(loss)
        log.info(
            "gw%d: fitted on %d rows over %d gameweeks (loss %.4f)",
            upto_gw,
            len(X),
            n_gws,
            float(loss),
        )

    # --- prediction ---------------------------------------------------------

    def _fallback_is_a_forecast(self, data: SeasonData) -> bool:
        """Whether deferring to FPL's expected points would import hindsight.

        Live, ``ep_next`` is a genuine pre-deadline forecast and deferring to it
        is the right thing to do with too little history. The archive's ``xP``
        is the same field recorded after the fact — it rates two thirds of
        non-appearances at exactly zero — so deferring to it there would hand
        the model an oracle on playing time for its first few gameweeks and
        quietly flatter its season total.

        Where that is the case the model reports no view at all, and the
        backtester holds the squad rather than acting on a number neither the
        model nor FPL actually produced in advance.
        """
        if self._fallback_trusted is None:
            share, _ = ep_looks_like_hindsight(data)
            self._fallback_trusted = share <= EP_HINDSIGHT_THRESHOLD
            if not self._fallback_trusted:
                log.info(
                    "expected-points column rates %.0f%% of non-appearances at "
                    "zero: not deferring to it",
                    share * 100,
                )
        return self._fallback_trusted

    def predict(self, data: SeasonData, gw: int) -> pd.DataFrame:
        if self.params is None:
            if not self._fallback_is_a_forecast(data):
                if gw not in self.no_view_gws:
                    self.no_view_gws.append(gw)
                return empty_predictions()
            if gw not in self.fallback_gws:
                self.fallback_gws.append(gw)
            return self._fallback.predict(data, gw)

        cache = self._ensure_cache(data)
        ff = cache.at(gw)
        if not len(ff):
            return empty_predictions()
        X = ff.X[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        assert self._mean is not None and self._std is not None
        Xs = jnp.asarray((X - self._mean) / self._std)
        xpts = np.asarray(_expected_points(self.params, Xs), dtype=np.float64)

        per_fixture = pd.DataFrame(
            {
                "player_id": ff.meta["player_id"].astype("int64"),
                "gw": int(gw),
                # No separate injury multiplier: `availability` is already one
                # of the features, so the p60 head has learned its effect.
                "xpts": np.clip(xpts, 0.0, None),
            }
        )
        return aggregate_to_gw(per_fixture)

    def predict_components(self, data: SeasonData, gw: int) -> pd.DataFrame:
        """Per-fixture head outputs, for inspecting *why* a player is rated.

        Returns ``player_id``, ``p60``, ``p_cameo``, ``mu60``, ``mu_cameo`` and
        ``xpts``. Empty if the model is running on the baseline fallback.
        """
        if self.params is None:
            return pd.DataFrame(
                columns=["player_id", "p60", "p_cameo", "mu60", "mu_cameo", "xpts"]
            )
        cache = self._ensure_cache(data)
        ff = cache.at(gw)
        if not len(ff):
            return pd.DataFrame(
                columns=["player_id", "p60", "p_cameo", "mu60", "mu_cameo", "xpts"]
            )
        X = ff.X[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        assert self._mean is not None and self._std is not None
        Xs = jnp.asarray((X - self._mean) / self._std)
        p60, p_cameo, mu60, mu_cameo = _heads(self.params, Xs)
        return pd.DataFrame(
            {
                "player_id": ff.meta["player_id"].astype("int64").to_numpy(),
                "p60": np.asarray(p60, dtype=np.float64),
                "p_cameo": np.asarray(p_cameo, dtype=np.float64),
                "mu60": np.asarray(mu60, dtype=np.float64),
                "mu_cameo": np.asarray(mu_cameo, dtype=np.float64),
                "xpts": np.asarray(
                    _expected_points(self.params, Xs), dtype=np.float64
                ),
            }
        )

    def coefficient_table(self) -> pd.DataFrame:
        """Standardized coefficients per head — the model's explanation of itself.

        Because features are standardized before fitting, a coefficient is the
        effect of a one-standard-deviation move in that feature, on the log-odds
        (probability heads) or log-points (Poisson heads) scale.
        """
        if self.params is None:
            return pd.DataFrame(columns=["feature", "p60", "p_cameo", "mu60", "mu_cameo"])
        return pd.DataFrame(
            {
                "feature": list(FEATURE_COLUMNS),
                "p60": np.asarray(self.params["w_p60"], dtype=np.float64),
                "p_cameo": np.asarray(self.params["w_cameo"], dtype=np.float64),
                "mu60": np.asarray(self.params["w_mu60"], dtype=np.float64),
                "mu_cameo": np.asarray(self.params["w_mucameo"], dtype=np.float64),
            }
        )
