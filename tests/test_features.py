"""Features, and the leakage property they exist to preserve.

The tests that matter most here are :func:`test_features_do_not_move_when_the
_future_changes` and its training-set counterpart. They work by rewriting every
outcome column from the target gameweek onwards to nonsense and asserting the
features are bit-identical. Any path from a result to a decision made before it
would show up as a difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.features import (
    FEATURE_COLUMNS,
    FeatureCache,
    build_features,
    build_training_set,
    rolling_team_form,
)

from .fixtures import BLANK_GW, DOUBLE_GW, corrupt_outcomes, make_season


@pytest.fixture(scope="module")
def season():
    return make_season(n_gws=8)


# --- shape ------------------------------------------------------------------


def test_feature_frame_is_numeric_and_ordered(season):
    ff = build_features(season, 4)
    assert list(ff.X.columns) == list(FEATURE_COLUMNS)
    assert all(str(d) == "float64" for d in ff.X.dtypes)
    assert not ff.X.isna().any().any()
    assert np.isfinite(ff.X.to_numpy()).all()
    assert list(ff.meta.columns) == [
        "player_id",
        "gw",
        "name",
        "position",
        "club",
        "price",
        "fixture_id",
    ]
    assert len(ff.X) == len(ff.meta)


def test_one_row_per_player_fixture(season):
    ff = build_features(season, 4)
    expected = len(season.pre_deadline_at(4))
    assert len(ff) == expected


def test_double_gameweek_gives_a_player_two_rows(season):
    ff = build_features(season, DOUBLE_GW)
    counts = ff.meta.groupby("player_id").size()
    doubled = ff.meta[ff.meta["club"].isin(["Club A", "Club B"])]["player_id"].unique()
    assert counts.loc[doubled].eq(2).all()
    assert (counts.loc[~counts.index.isin(doubled)] == 1).all()


def test_blank_gameweek_omits_the_clubs_that_do_not_play(season):
    ff = build_features(season, BLANK_GW)
    assert set(ff.meta["club"]) == {"Club A", "Club B"}


def test_position_dummies_are_one_hot(season):
    ff = build_features(season, 4)
    dummies = ff.X[["is_gkp", "is_def", "is_mid", "is_fwd"]]
    assert dummies.sum(axis=1).eq(1.0).all()
    gkp = ff.meta["position"] == "GKP"
    assert ff.X.loc[gkp.to_numpy(), "is_gkp"].eq(1.0).all()


def test_an_empty_gameweek_yields_an_empty_frame(season):
    ff = build_features(season, 99)
    assert len(ff) == 0
    assert list(ff.X.columns) == list(FEATURE_COLUMNS)


# --- cold start -------------------------------------------------------------


def test_gameweek_one_has_no_history_to_read(season):
    """Nothing has been played, so the history features are zero, not undefined."""
    ff = build_features(season, 1)
    history_derived = [
        "minutes_mean_3",
        "minutes_mean_6",
        "start_rate_6",
        "played_rate_6",
        "full_game_rate_6",
        "games_available",
        "points_per_90_6",
        "points_mean_3",
        "points_mean_6",
        "bps_per_90_6",
    ]
    assert ff.X[history_derived].eq(0.0).all().all()
    # Team form falls back to a league-average prior rather than zero goals.
    assert ff.X["opp_conceded_per_game"].gt(0.0).all()
    assert ff.X["own_scored_per_game"].gt(0.0).all()


def test_form_features_reflect_the_history_actually_played(season):
    ff = build_features(season, 7)
    meta = ff.meta.set_index("player_id")
    nailed = meta.index[meta["name"].str.endswith("nailed")]
    absent = meta.index[meta["name"].str.endswith("absent")]
    X = ff.X.set_index(ff.meta["player_id"].to_numpy())

    assert X.loc[nailed, "played_rate_6"].eq(1.0).all()
    assert X.loc[absent, "played_rate_6"].eq(0.0).all()
    assert X.loc[nailed, "minutes_mean_6"].gt(60.0).all()
    assert X.loc[absent, "points_mean_6"].eq(0.0).all()


def test_points_per_90_cannot_be_inflated_by_a_single_minute():
    """The denominator is floored at one full match."""
    season = make_season(n_gws=4)
    df = season.player_gws.copy()
    victim = int(df["player_id"].iloc[0])
    mask = df["player_id"] == victim
    df.loc[mask, "minutes"] = 0
    df.loc[mask, "points"] = 0
    first = df.index[mask][0]
    df.loc[first, "minutes"] = 1
    df.loc[first, "points"] = 12
    season.player_gws = df

    ff = build_features(season, 4)
    row = ff.X[ff.meta["player_id"].to_numpy() == victim]
    assert float(row["points_per_90_6"].iloc[0]) == pytest.approx(12.0)


# --- team form --------------------------------------------------------------


def test_rolling_team_form_counts_goals_from_the_right_side():
    history = pd.DataFrame(
        {
            "club": ["X", "X", "Y", "Y"],
            "fixture_id": [1, 2, 1, 2],
            "gw": [1, 2, 1, 2],
            "was_home": [True, False, False, True],
            "team_h_score": [3, 1, 3, 1],
            "team_a_score": [0, 2, 0, 2],
        }
    )
    form = rolling_team_form(history, window=6).set_index("club")
    # X was home for a 3-0 and away for a 1-2: scored 3 + 2, conceded 0 + 1.
    assert form.loc["X", "scored_per_game"] == pytest.approx(2.5)
    assert form.loc["X", "conceded_per_game"] == pytest.approx(0.5)
    assert form.loc["Y", "scored_per_game"] == pytest.approx(0.5)
    assert form.loc["Y", "conceded_per_game"] == pytest.approx(2.5)
    assert form.loc["X", "games"] == 2


def test_rolling_team_form_keeps_only_the_window():
    history = pd.DataFrame(
        {
            "club": ["X"] * 4,
            "fixture_id": [1, 2, 3, 4],
            "gw": [1, 2, 3, 4],
            "was_home": [True] * 4,
            "team_h_score": [0, 0, 4, 4],
            "team_a_score": [0, 0, 0, 0],
        }
    )
    form = rolling_team_form(history, window=2).set_index("club")
    assert form.loc["X", "scored_per_game"] == pytest.approx(4.0)
    assert form.loc["X", "games"] == 4


def test_rolling_team_form_on_an_empty_history_is_empty():
    form = rolling_team_form(pd.DataFrame(), window=6)
    assert len(form) == 0
    assert list(form.columns) == [
        "club",
        "scored_per_game",
        "conceded_per_game",
        "games",
    ]


# --- leakage ----------------------------------------------------------------


def test_history_containing_the_target_gameweek_is_rejected(season):
    contaminated = season.player_gws[season.player_gws["gw"] <= 5]
    with pytest.raises(ValueError, match="would leak the result being predicted"):
        build_features(season, 5, history=contaminated)


@pytest.mark.parametrize("gw", [1, 3, 5, 6, 8])
def test_features_do_not_move_when_the_future_changes(season, gw):
    """Rewrite every result from ``gw`` onwards; the features must not notice."""
    corrupted = corrupt_outcomes(season, gw)
    clean = build_features(season, gw)
    dirty = build_features(corrupted, gw)

    pd.testing.assert_frame_equal(clean.X, dirty.X)
    pd.testing.assert_frame_equal(clean.meta, dirty.meta)


def test_features_do_move_when_the_past_changes(season):
    """The control for the test above: the past is supposed to matter.

    Without this, a feature builder that ignored history entirely would pass the
    leakage test perfectly.
    """
    corrupted = corrupt_outcomes(season, 2)
    clean = build_features(season, 5)
    dirty = build_features(corrupted, 5)
    assert not clean.X.equals(dirty.X)


# --- cache and training set -------------------------------------------------


def test_cache_matches_building_each_gameweek_directly(season):
    cache = FeatureCache(season)
    for gw in (2, 5, 7):
        pd.testing.assert_frame_equal(cache.at(gw).X, build_features(season, gw).X)


def test_cache_is_computed_once(season):
    cache = FeatureCache(season)
    first = cache.at(4)
    assert cache.at(4) is first
    cache.warm([2, 3])
    assert set(cache._frames) >= {2, 3, 4}


def test_training_set_stops_before_the_gameweek_being_decided(season):
    X, y = build_training_set(season, 6)
    assert int(y["gw"].max()) == 5
    assert int(y["gw"].min()) == 2
    assert len(X) == len(y)
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert set(y.columns) == {"player_id", "gw", "minutes", "points"}


def test_training_set_targets_are_the_realized_outcomes(season):
    X, y = build_training_set(season, 5)
    realized = season.before(5).groupby(["player_id", "gw"])["points"].sum()
    got = y.groupby(["player_id", "gw"])["points"].sum()
    common = got.index.intersection(realized.index)
    assert len(common) > 0
    assert (got.loc[common].astype(int) == realized.loc[common].astype(int)).all()


def test_training_set_is_empty_before_any_history_exists(season):
    X, y = build_training_set(season, 2)
    assert len(X) == 0
    assert len(y) == 0


def test_training_set_does_not_move_when_the_future_changes(season):
    """The leakage property, carried through to what the model is actually fed."""
    corrupted = corrupt_outcomes(season, 6)
    X_clean, y_clean = build_training_set(season, 6)
    X_dirty, y_dirty = build_training_set(corrupted, 6)

    pd.testing.assert_frame_equal(X_clean, X_dirty)
    pd.testing.assert_frame_equal(y_clean, y_dirty)
