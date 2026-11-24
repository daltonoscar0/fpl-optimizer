"""Double gameweeks, on a season small enough to check by hand.

A double gameweek is where per-fixture and per-gameweek thinking come apart, and
it is the case the three strategies most easily disagree about for reasons that
have nothing to do with how good their projections are. The season is
:mod:`tests.fixtures.handbuilt`, where every result is written out in full, so
the expected answers below are arithmetic done on paper. A strategy that
quietly reverts to a per-gameweek rate fails here rather than in a backtest six
layers downstream.

The figures pinned here — gameweek-2 totals of 6, 11, 8 and 12 for the first
four players — are the same ones the scorer is tested against in
:mod:`tests.test_score`, so both sides of a backtest are checked against one
set of numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_optimizer.projections import (
    EPProjection,
    JaxPointsModel,
    NaiveProjection,
    aggregate_to_gw,
)
from fpl_optimizer.projections.baselines import MIN_RATE_FIXTURES

from .fixtures import make_season
from .fixtures import DOUBLE_GW as SYNTHETIC_DOUBLE_GW
from .fixtures.handbuilt import DOUBLE_GW, hand_built_season

#: Players whose clubs play twice in :data:`DOUBLE_GW`.
DOUBLED = (1, 2, 3, 4, 5, 6)
#: The four players whose every result is pinned across this file.
PINNED = [1, 2, 3, 4]


@pytest.fixture(scope="module")
def season():
    return hand_built_season()


# --- the data layer ---------------------------------------------------------


def test_a_double_gameweek_is_two_rows_per_player(season):
    counts = season.at(DOUBLE_GW).groupby("player_id").size()
    assert set(counts[counts == 2].index) == set(DOUBLED)
    assert season.at(1).groupby("player_id").size().eq(1).all()
    assert set(season.at(DOUBLE_GW)["fixture_id"]) == {102, 103, 112, 122}


def test_realized_points_sum_over_both_fixtures(season):
    """The quantity a scorer, a captaincy pick and auto-subs all need.

    Hand-checked: keeper 5 + 1, playmaker 3 + 8, fullback 2 + 6, finisher
    12 + 0.
    """
    got = season.at(DOUBLE_GW).groupby("player_id")["points"].sum().astype(int)
    assert got.loc[PINNED].to_dict() == {1: 6, 2: 11, 3: 8, 4: 12}

    # Worth stating for the scorer: the captain multiplier applies to this
    # summed figure, so the finisher's armband is worth 24 and not 12 —
    # doubling one leg of a double is the mistake available here.
    assert int(got[4]) == 12


def test_history_before_a_double_counts_only_earlier_gameweeks(season):
    assert set(season.before(DOUBLE_GW)["fixture_id"]) == {101, 111, 121}
    assert set(season.before(3)["fixture_id"]) == {101, 111, 121, 102, 103, 112, 122}
    # 18 players once in gameweek 1, plus gameweek 2's 24 rows.
    assert len(season.before(3)) == 42


# --- the strategies, against arithmetic done on paper -----------------------


def test_fpl_expected_points_is_published_once_per_gameweek(season):
    """It does *not* add both legs, because there is only one figure to add.

    FPL quotes one ``ep_next`` per player per gameweek. The archive's ``xP`` is
    that same figure, and in all 1,278 double-gameweek player-gameweeks across
    2023-24 and 2024-25 it is identical on both fixture rows — so summing it
    would double a number nobody published. The fixture repeats the value the
    way the real source does, and the projection takes it once.
    """
    preds = EPProjection().predict(season, DOUBLE_GW).set_index("player_id")["xpts"]
    assert preds.loc[PINNED].to_dict() == pytest.approx(
        {1: 3.0, 2: 4.0, 3: 2.5, 4: 6.0}
    )

    rows = season.pre_deadline_at(DOUBLE_GW)
    doubled = rows.groupby("player_id")["ep"].nunique()
    assert doubled.loc[list(DOUBLED)].eq(1).all(), "the fixture must repeat, not vary"


def test_naive_rate_is_per_fixture_and_so_doubles(season):
    """After gameweek 1, each player has played exactly one fixture.

    One fixture is below ``MIN_RATE_FIXTURES``, so the denominator is floored at
    three and the gameweek-1 scores of 2, 6, 1 and 9 become rates of 2/3, 2, 1/3
    and 3. A two-fixture gameweek is then worth exactly twice each rate, which
    is the property under test — the floor changes the level, not the scaling.
    """
    proj = NaiveProjection()
    proj.fit(season, DOUBLE_GW)
    rates = {p: v / MIN_RATE_FIXTURES for p, v in {1: 2, 2: 6, 3: 1, 4: 9}.items()}
    assert {p: proj._ppf[p] for p in PINNED} == pytest.approx(rates)

    preds = proj.predict(season, DOUBLE_GW).set_index("player_id")["xpts"]
    assert preds.loc[PINNED].to_dict() == pytest.approx(
        {p: 2 * r for p, r in rates.items()}
    )


def test_naive_divides_by_fixtures_not_gameweeks(season):
    """Fitted after the double, the denominator is three fixtures, not two gameweeks.

    Keeper 2 + 5 + 1 = 8 over three fixtures; a per-gameweek rate would say 8/2
    = 4.0 and overstate him by half.
    """
    proj = NaiveProjection()
    proj.fit(season, 3)
    assert proj._ppf[1] == pytest.approx(8 / 3)
    assert proj._ppf[2] == pytest.approx(17 / 3)
    assert proj._ppf[3] == pytest.approx(9 / 3)
    assert proj._ppf[4] == pytest.approx(21 / 3)

    preds = proj.predict(season, 3).set_index("player_id")["xpts"]
    assert preds[1] == pytest.approx(8 / 3)


def test_the_baselines_disagree_about_doubles_and_that_is_the_source_talking(
    season,
):
    """Naive scales with fixtures; FPL's published figure cannot.

    This asymmetry is not a bug to be reconciled. A per-fixture rate can be
    doubled for a double gameweek because it is quoted per fixture. FPL's
    number is quoted per gameweek and says nothing about how many matches are
    in it, so doubling it would be inventing data. The consequence — that the
    ``ep`` baseline undervalues doubles — is a limitation of the source, and
    is recorded as one.
    """
    rows = season.pre_deadline_at(DOUBLE_GW)

    naive = NaiveProjection()
    naive.fit(season, DOUBLE_GW)
    naive_preds = naive.predict(season, DOUBLE_GW).set_index("player_id")["xpts"]
    per_fixture = rows["player_id"].map(naive._ppf).groupby(rows["player_id"]).sum()
    assert naive_preds.to_dict() == pytest.approx(per_fixture.to_dict())

    ep_preds = EPProjection().predict(season, DOUBLE_GW).set_index("player_id")["xpts"]
    published = rows.groupby("player_id")["ep"].max()
    assert ep_preds.to_dict() == pytest.approx(published.to_dict())
    # Strictly less than the sum, for exactly the players who play twice.
    summed = rows.groupby("player_id")["ep"].sum()
    assert (ep_preds.loc[list(DOUBLED)] < summed.loc[list(DOUBLED)]).all()


def test_the_model_sums_its_own_per_fixture_heads():
    """The model's contract on doubles, checked against its own components.

    A six-club hand-built season is far too small to fit on, so this uses the
    synthetic season — but the invariant is the same one: what ``predict``
    returns for a gameweek is exactly its per-fixture outputs added up.
    """
    data = make_season(n_gws=12)
    model = JaxPointsModel(steps=60)
    model.fit(data, SYNTHETIC_DOUBLE_GW + 6)
    assert model.params is not None

    preds = model.predict(data, SYNTHETIC_DOUBLE_GW)
    components = model.predict_components(data, SYNTHETIC_DOUBLE_GW)
    summed = aggregate_to_gw(
        components.assign(gw=SYNTHETIC_DOUBLE_GW)[["player_id", "gw", "xpts"]]
    )

    doubled = components.groupby("player_id").size()
    assert doubled.max() == 2
    pd.testing.assert_frame_equal(
        preds.sort_values("player_id").reset_index(drop=True),
        summed.sort_values("player_id").reset_index(drop=True),
        atol=1e-6,
    )
