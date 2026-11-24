"""Double gameweeks, on a season small enough to check by hand.

A double gameweek is where per-fixture and per-gameweek thinking come apart, and
it is the case the three strategies most easily disagree about for reasons that
have nothing to do with how good their projections are. Every number in this
module is written out in full and the expected answers are arithmetic done on
paper, so a strategy that quietly reverts to a per-gameweek rate fails here
rather than in a backtest six layers downstream.

Two fixtures, two clubs, four players, three gameweeks:

====  ========  ==================================================
GW    fixtures  note
====  ========  ==================================================
1     101       ordinary single gameweek
2     102, 103  **double**: both clubs play twice
3     104       single again, now with a double in the history
====  ========  ==================================================

What is *not* tested here, because it does not exist yet: realized-points
scoring, captaincy and auto-subs. Those will consume realized points summed per
(player, gameweek) across fixtures, which is asserted below as
:func:`test_realized_points_sum_over_both_fixtures` — so the input they will be
built on is pinned even though they are not written.
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
from fpl_optimizer.schema import (
    CLUB_COLUMNS,
    FIXTURE_COLUMNS,
    SeasonData,
    coerce_player_gws,
)

from .fixtures import DOUBLE_GW, make_season

#: ``(gw, fixture_id, club, opponent, was_home)`` for every fixture-side.
_SIDES: tuple[tuple[int, int, str, str, bool], ...] = (
    (1, 101, "Reds", "Blues", True),
    (1, 101, "Blues", "Reds", False),
    (2, 102, "Reds", "Blues", False),
    (2, 102, "Blues", "Reds", True),
    (2, 103, "Reds", "Blues", True),
    (2, 103, "Blues", "Reds", False),
    (3, 104, "Reds", "Blues", False),
    (3, 104, "Blues", "Reds", True),
)

#: ``player_id -> (name, position, club, price)``.
_PLAYERS: dict[int, tuple[str, str, str, int]] = {
    1: ("Keeper", "GKP", "Reds", 45),
    2: ("Playmaker", "MID", "Reds", 85),
    3: ("Fullback", "DEF", "Blues", 50),
    4: ("Finisher", "FWD", "Blues", 120),
}

#: ``(player_id, fixture_id) -> (minutes, points, ep)``. Written out so every
#: expected value below can be checked by adding up this table.
_RESULTS: dict[tuple[int, int], tuple[int, int, float]] = {
    (1, 101): (90, 2, 3.0),
    (2, 101): (90, 6, 5.0),
    (3, 101): (90, 1, 2.0),
    (4, 101): (90, 9, 7.0),
    # The double.
    (1, 102): (90, 5, 2.0),
    (2, 102): (90, 3, 4.0),
    (3, 102): (90, 2, 1.5),
    (4, 102): (90, 12, 6.0),
    (1, 103): (90, 1, 3.0),
    (2, 103): (90, 8, 2.0),
    (3, 103): (90, 6, 2.5),
    (4, 103): (90, 0, 4.0),
    # Back to single fixtures.
    (1, 104): (90, 4, 2.5),
    (2, 104): (90, 7, 5.5),
    (3, 104): (90, 3, 2.0),
    (4, 104): (90, 2, 6.5),
}

_SHORT = {"Reds": "RED", "Blues": "BLU"}


@pytest.fixture(scope="module")
def season() -> SeasonData:
    rows = []
    for gw, fixture_id, club, opponent, was_home in _SIDES:
        for pid, (name, position, player_club, price) in _PLAYERS.items():
            if player_club != club:
                continue
            minutes, points, ep = _RESULTS[(pid, fixture_id)]
            rows.append(
                {
                    "season": "2024-25",
                    "gw": gw,
                    "player_id": pid,
                    "name": name,
                    "position": position,
                    "club": club,
                    "club_short": _SHORT[club],
                    "opponent": opponent,
                    "opponent_short": _SHORT[opponent],
                    "was_home": was_home,
                    "price": price,
                    "fixture_id": fixture_id,
                    "kickoff_time": f"2024-09-0{gw}T14:00:00Z",
                    "ep": ep,
                    "availability": 1.0,
                    "minutes": minutes,
                    "points": points,
                    "bonus": 0,
                    "bps": points * 5,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 0,
                    "goals_conceded": 1,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "saves": 0,
                    "starts": 1,
                    "defensive_contribution": 0,
                    "team_h_score": 1,
                    "team_a_score": 1,
                    "selected": 1000,
                    "transfers_in": 0,
                    "transfers_out": 0,
                }
            )
    clubs = pd.DataFrame(
        [
            {"season": "2024-25", "club_id": 1, "club": "Reds", "club_short": "RED"},
            {"season": "2024-25", "club_id": 2, "club": "Blues", "club_short": "BLU"},
        ]
    )[list(CLUB_COLUMNS)]
    fixtures = pd.DataFrame(
        [
            {
                "season": "2024-25",
                "gw": gw,
                "fixture_id": fid,
                "home_club": club if home else opp,
                "away_club": opp if home else club,
                "kickoff_time": f"2024-09-0{gw}T14:00:00Z",
                "home_difficulty": 3,
                "away_difficulty": 3,
                "finished": True,
            }
            for gw, fid, club, opp, home in _SIDES
            if home
        ]
    )[list(FIXTURE_COLUMNS)]
    events = pd.DataFrame(
        {
            "gw": [1, 2, 3],
            "deadline_time": [f"2024-09-0{g}T11:30:00Z" for g in (1, 2, 3)],
            "average_entry_score": [50.0, 70.0, 48.0],
            "finished": [True, True, True],
        }
    )
    return SeasonData(
        season="2024-25",
        player_gws=coerce_player_gws(pd.DataFrame(rows)),
        clubs=clubs,
        fixtures=fixtures,
        events=events,
        source="archive",
    )


# --- the data layer ---------------------------------------------------------


def test_a_double_gameweek_is_two_rows_per_player(season):
    counts = season.at(2).groupby("player_id").size()
    assert counts.to_dict() == {1: 2, 2: 2, 3: 2, 4: 2}
    assert season.at(1).groupby("player_id").size().eq(1).all()
    assert set(season.at(2)["fixture_id"]) == {102, 103}


def test_realized_points_sum_over_both_fixtures(season):
    """The quantity a scorer, a captaincy pick and auto-subs all need.

    Hand-checked: keeper 5 + 1, playmaker 3 + 8, fullback 2 + 6, finisher
    12 + 0.
    """
    got = season.at(2).groupby("player_id")["points"].sum().astype(int)
    assert got.to_dict() == {1: 6, 2: 11, 3: 8, 4: 12}

    # Worth stating for whoever writes the scorer: the captain multiplier
    # applies to this summed figure, so the finisher's armband is worth 24 and
    # not 12 — doubling one leg of a double is the mistake available here.
    assert int(got[4]) == 12


def test_history_before_a_double_counts_only_earlier_gameweeks(season):
    assert set(season.before(2)["fixture_id"]) == {101}
    assert set(season.before(3)["fixture_id"]) == {101, 102, 103}
    assert len(season.before(3)) == 12


# --- the strategies, against arithmetic done on paper -----------------------


def test_fpl_expected_points_adds_both_legs(season):
    """Keeper 2.0 + 3.0, playmaker 4.0 + 2.0, fullback 1.5 + 2.5, finisher 6.0 + 4.0."""
    preds = EPProjection().predict(season, 2).set_index("player_id")["xpts"]
    assert preds.to_dict() == pytest.approx({1: 5.0, 2: 6.0, 3: 4.0, 4: 10.0})


def test_naive_rate_is_per_fixture_and_so_doubles(season):
    """After gameweek 1, each player has played exactly one fixture.

    So their points-per-fixture is their gameweek-1 score — 2, 6, 1 and 9 — and
    a two-fixture gameweek is worth twice that.
    """
    proj = NaiveProjection()
    proj.fit(season, 2)
    assert proj._ppf == pytest.approx({1: 2.0, 2: 6.0, 3: 1.0, 4: 9.0})

    preds = proj.predict(season, 2).set_index("player_id")["xpts"]
    assert preds.to_dict() == pytest.approx({1: 4.0, 2: 12.0, 3: 2.0, 4: 18.0})


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


def test_the_two_baselines_agree_on_what_a_double_is_worth(season):
    """Not that they agree on the value — that they agree on the *scaling*.

    Each strategy's total for a two-fixture gameweek must be the sum of its two
    per-fixture figures. That is the property the backtest needs in order to
    attribute a difference in results to projection quality.
    """
    ep = EPProjection()
    naive = NaiveProjection()
    naive.fit(season, 2)
    rows = season.pre_deadline_at(2)

    for proj, per_fixture in (
        (ep, rows.groupby("player_id")["ep"].sum()),
        (naive, rows["player_id"].map(naive._ppf).groupby(rows["player_id"]).sum()),
    ):
        preds = proj.predict(season, 2).set_index("player_id")["xpts"]
        assert preds.to_dict() == pytest.approx(per_fixture.to_dict())


def test_the_model_sums_its_own_per_fixture_heads():
    """The model's contract on doubles, checked against its own components.

    A two-club hand-built season is far too small to fit on, so this uses the
    synthetic season — but the invariant is the same one: what ``predict``
    returns for a gameweek is exactly its per-fixture outputs added up.
    """
    data = make_season(n_gws=12)
    model = JaxPointsModel(steps=60)
    model.fit(data, DOUBLE_GW + 6)
    assert model.params is not None

    preds = model.predict(data, DOUBLE_GW)
    components = model.predict_components(data, DOUBLE_GW)
    summed = aggregate_to_gw(components.assign(gw=DOUBLE_GW)[["player_id", "gw", "xpts"]])

    doubled = components.groupby("player_id").size()
    assert doubled.max() == 2
    pd.testing.assert_frame_equal(
        preds.sort_values("player_id").reset_index(drop=True),
        summed.sort_values("player_id").reset_index(drop=True),
        atol=1e-6,
    )
