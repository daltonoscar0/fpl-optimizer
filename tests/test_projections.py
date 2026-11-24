"""The three projection strategies and the interface they share.

The backtester runs all three through identical machinery, so these tests check
two kinds of thing: that each strategy computes what it claims to, and that all
three agree on the contract — one row per player per gameweek, doubles worth two
fixtures, and no reading of a result before it has happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.features import FeatureCache
from fpl_optimizer.projections import (
    PREDICTION_COLUMNS,
    STRATEGIES,
    EPProjection,
    JaxPointsModel,
    NaiveProjection,
    Projection,
    aggregate_to_gw,
    empty_predictions,
)
from fpl_optimizer.projections.baselines import EP_COVERAGE_THRESHOLD, ep_coverage

from .fixtures import BLANK_GW, DOUBLE_GW, corrupt_outcomes, make_season


@pytest.fixture(scope="module")
def season():
    return make_season(n_gws=12)


@pytest.fixture(scope="module")
def cache(season):
    """One warm feature cache for the module.

    Building features is the slow part of these tests and the result is the same
    every time, so the cache is shared rather than rebuilt per model.
    """
    c = FeatureCache(season)
    c.warm()
    return c


def _fast_model(cache: FeatureCache | None = None, **kwargs) -> JaxPointsModel:
    """A model with the step count turned down, for tests about behaviour."""
    model = JaxPointsModel(steps=60, **kwargs)
    if cache is not None:
        model.attach(cache)
    return model


#: Gameweek the shared fitted model is trained up to.
FIT_GW = 11


@pytest.fixture(scope="module")
def fitted(season, cache):
    """A model fitted once, for the tests that only read its output."""
    model = _fast_model(cache)
    model.fit(season, FIT_GW)
    assert model.params is not None
    return model


# --- the shared contract ----------------------------------------------------


def test_every_registered_strategy_satisfies_the_protocol():
    assert set(STRATEGIES) == {"model", "ep", "naive"}
    for name, factory in STRATEGIES.items():
        proj = factory()
        assert isinstance(proj, Projection)
        assert proj.name == name


@pytest.mark.parametrize("name", ["ep", "naive", "model"])
def test_predictions_are_one_row_per_player(season, cache, name):
    proj = _fast_model(cache) if name == "model" else STRATEGIES[name]()
    proj.fit(season, 8)
    preds = proj.predict(season, 8)

    assert list(preds.columns) == list(PREDICTION_COLUMNS)
    assert preds["player_id"].is_unique
    assert set(preds["gw"]) == {8}
    assert preds["xpts"].notna().all()


@pytest.mark.parametrize("name", ["ep", "naive", "model"])
def test_blank_gameweek_leaves_the_absent_clubs_out(season, cache, name):
    proj = _fast_model(cache) if name == "model" else STRATEGIES[name]()
    proj.fit(season, BLANK_GW)
    preds = proj.predict(season, BLANK_GW)
    playing = set(season.pre_deadline_at(BLANK_GW)["player_id"])
    assert set(preds["player_id"]) <= playing
    assert len(preds) <= len(playing)


@pytest.mark.parametrize("name", ["ep", "naive", "model"])
def test_a_gameweek_with_no_fixtures_predicts_nothing(season, cache, name):
    proj = _fast_model(cache) if name == "model" else STRATEGIES[name]()
    proj.fit(season, 8)
    assert len(proj.predict(season, 99)) == 0


# --- aggregation ------------------------------------------------------------


def test_aggregate_sums_a_players_fixtures():
    per_fixture = pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "gw": [5, 5, 5],
            "xpts": [2.5, 3.0, 4.0],
        }
    )
    out = aggregate_to_gw(per_fixture)
    assert out.set_index("player_id")["xpts"].to_dict() == {1: 5.5, 2: 4.0}
    assert str(out["player_id"].dtype) == "int64"
    assert str(out["gw"].dtype) == "int64"


def test_aggregate_of_nothing_is_an_empty_frame():
    out = aggregate_to_gw(pd.DataFrame(columns=["player_id", "gw", "xpts"]))
    assert len(out) == 0
    assert list(out.columns) == list(PREDICTION_COLUMNS)


def test_empty_predictions_is_typed():
    empty = empty_predictions()
    assert list(empty.columns) == list(PREDICTION_COLUMNS)
    assert str(empty["xpts"].dtype) == "float64"


# --- FPL's own expected points ----------------------------------------------


def test_ep_reads_the_published_figure(season):
    proj = EPProjection()
    proj.fit(season, 7)
    preds = proj.predict(season, 7).set_index("player_id")["xpts"]
    expected = season.pre_deadline_at(7).groupby("player_id")["ep"].sum()

    assert len(preds) == len(expected)
    assert preds.sort_index().to_numpy() == pytest.approx(
        expected.sort_index().to_numpy(), abs=1e-5
    )


def test_ep_counts_both_fixtures_of_a_double(season):
    proj = EPProjection()
    rows = season.pre_deadline_at(DOUBLE_GW)
    doubled = rows.groupby("player_id").size()
    doubled = doubled[doubled == 2].index
    preds = proj.predict(season, DOUBLE_GW).set_index("player_id")["xpts"]
    per_fixture = rows.groupby("player_id")["ep"].sum()

    assert len(doubled) > 0
    assert preds.loc[doubled].to_numpy() == pytest.approx(
        per_fixture.loc[doubled].to_numpy(), abs=1e-5
    )


def test_ep_reports_itself_unavailable_rather_than_predicting_zero():
    season = make_season(n_gws=6, with_ep=False)
    proj = EPProjection()
    assert not proj.available_for(season, 3)
    assert len(proj.predict(season, 3)) == 0
    assert proj.missing_gws == [3]


def test_ep_coverage_is_measured_over_the_season():
    full = make_season(n_gws=6)
    covered, missing = ep_coverage(full)
    assert covered == pytest.approx(1.0)
    assert missing == []

    none = make_season(n_gws=6, with_ep=False)
    covered, missing = ep_coverage(none)
    assert covered == pytest.approx(0.0)
    assert missing == none.gameweeks
    assert covered < EP_COVERAGE_THRESHOLD


def test_ep_fills_a_single_missing_player_rather_than_dropping_them(season):
    """A legal squad needs every candidate priced; one null must not remove one."""
    data = make_season(n_gws=6)
    df = data.player_gws.copy()
    victim = int(df["player_id"].iloc[0])
    df.loc[df["player_id"] == victim, "ep"] = np.nan
    data.player_gws = df

    preds = EPProjection().predict(data, 3)
    assert victim in set(preds["player_id"])
    assert float(preds.set_index("player_id").loc[victim, "xpts"]) == 0.0


# --- the naive baseline -----------------------------------------------------


def test_naive_carries_forward_points_per_fixture(season):
    proj = NaiveProjection()
    proj.fit(season, 7)
    hist = season.before(7)
    agg = hist.groupby("player_id")["points"].agg(["sum", "size"])

    preds = proj.predict(season, 7).set_index("player_id")["xpts"]
    for pid, row in agg.iterrows():
        if pid not in preds.index:
            continue
        assert preds.loc[pid] == pytest.approx(float(row["sum"]) / int(row["size"]))


def test_naive_values_a_double_gameweek_at_two_fixtures(season):
    proj = NaiveProjection()
    proj.fit(season, DOUBLE_GW)
    preds = proj.predict(season, DOUBLE_GW).set_index("player_id")["xpts"]

    rows = season.pre_deadline_at(DOUBLE_GW).groupby("player_id").size()
    doubled = [p for p in rows[rows == 2].index if preds.loc[p] > 0]
    single = [p for p in rows[rows == 1].index if preds.loc[p] > 0]
    assert doubled and single

    hist = season.before(DOUBLE_GW).groupby("player_id")["points"].agg(["sum", "size"])
    for pid in doubled:
        rate = float(hist.loc[pid, "sum"]) / int(hist.loc[pid, "size"])
        assert preds.loc[pid] == pytest.approx(2 * rate)


def test_naive_rates_a_player_with_no_history_at_zero(season):
    proj = NaiveProjection()
    proj.fit(season, 1)
    preds = proj.predict(season, 1)
    assert preds["xpts"].eq(0.0).all()


# --- the model --------------------------------------------------------------


def test_model_defers_to_the_baseline_before_it_has_history(season, cache):
    model = _fast_model(cache, min_train_gws=3)
    model.fit(season, 2)
    assert model.params is None

    preds = model.predict(season, 2)
    expected = EPProjection().predict(season, 2)
    assert model.fallback_gws == [2]
    pd.testing.assert_frame_equal(preds, expected)


def test_model_fits_once_enough_gameweeks_exist(fitted):
    assert fitted.fallback_gws == []
    report = fitted.fit_report
    assert report["gameweeks"] == 9.0
    assert report["rows"] > 200
    assert 0.0 < report["share_60plus"] < 1.0
    assert 0.0 < report["share_cameo"] < 1.0
    assert np.isfinite(report["final_loss"])


def test_model_predictions_are_non_negative_and_bounded(season, fitted):
    preds = fitted.predict(season, FIT_GW)
    assert preds["xpts"].ge(0.0).all()
    # Nobody is projected a hat-trick-plus haul; the Poisson link is clipped.
    assert preds["xpts"].lt(25.0).all()


def test_model_separates_players_who_start_from_players_who_do_not(season, fitted):
    """The point of the three-part split, stated as a test.

    A player who has never appeared must be projected below a nailed starter,
    and the probability head is what should be doing that work.
    """
    preds = fitted.predict(season, FIT_GW).set_index("player_id")["xpts"]
    meta = (
        season.pre_deadline_at(FIT_GW)
        .drop_duplicates("player_id")
        .set_index("player_id")["name"]
    )
    nailed = [p for p in preds.index if meta[p].endswith("nailed")]
    absent = [p for p in preds.index if meta[p].endswith("absent")]

    assert nailed and absent
    assert preds.loc[nailed].min() > preds.loc[absent].max()

    components = fitted.predict_components(season, FIT_GW).set_index("player_id")
    assert components.loc[nailed, "p60"].min() > components.loc[absent, "p60"].max()


def test_model_components_are_probabilities_that_cannot_exceed_one(season, fitted):
    comp = fitted.predict_components(season, FIT_GW)

    assert len(comp) == len(season.pre_deadline_at(FIT_GW))
    assert comp["p60"].between(0.0, 1.0).all()
    assert comp["p_cameo"].between(0.0, 1.0).all()
    # The nested parameterization is what guarantees this.
    assert (comp["p60"] + comp["p_cameo"]).le(1.0 + 1e-6).all()
    assert comp["xpts"].ge(0.0).all()


def test_model_explains_itself_with_one_coefficient_per_feature(fitted):
    from fpl_optimizer.features import FEATURE_COLUMNS

    table = fitted.coefficient_table()

    assert table["feature"].tolist() == list(FEATURE_COLUMNS)
    assert set(table.columns) == {"feature", "p60", "p_cameo", "mu60", "mu_cameo"}
    assert np.isfinite(table[["p60", "p_cameo", "mu60", "mu_cameo"]].to_numpy()).all()


def test_coefficient_table_is_empty_while_the_model_is_unfitted():
    model = _fast_model()
    assert len(model.coefficient_table()) == 0
    assert len(model.predict_components(make_season(n_gws=4), 2)) == 0


def test_model_reuses_an_attached_feature_cache(season, cache):
    model = _fast_model()
    model.attach(cache)
    model.fit(season, FIT_GW)
    assert model._cache is cache


def test_model_is_deterministic_given_the_seed(season, cache):
    first, second = _fast_model(cache, seed=7), _fast_model(cache, seed=7)
    first.fit(season, FIT_GW)
    second.fit(season, FIT_GW)
    pd.testing.assert_frame_equal(
        first.predict(season, FIT_GW), second.predict(season, FIT_GW)
    )


# --- leakage ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["ep", "naive", "model"])
def test_predictions_do_not_move_when_the_future_changes(season, cache, name):
    """The end-to-end leakage check: fit and predict against rewritten results.

    The model gets the warm cache for the clean run and builds its own for the
    corrupted one, so the comparison also covers the cache itself.
    """
    gw = 9
    corrupted = corrupt_outcomes(season, gw)

    clean = _fast_model(cache) if name == "model" else STRATEGIES[name]()
    dirty = _fast_model() if name == "model" else STRATEGIES[name]()
    clean.fit(season, gw)
    dirty.fit(corrupted, gw)

    pd.testing.assert_frame_equal(clean.predict(season, gw), dirty.predict(corrupted, gw))
