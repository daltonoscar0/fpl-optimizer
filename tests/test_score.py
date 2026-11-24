"""The realized-points scorer, on hand-built squads.

Expected values are arithmetic on :mod:`tests.fixtures.handbuilt`, whose results
are written out in full, and the gameweek-2 figures are the same ones pinned in
:mod:`tests.test_double_gameweek` — keeper 6, playmaker 11, fullback 8, finisher
12. Both sides of a backtest are therefore checked against one set of numbers.
"""

from __future__ import annotations

import pytest

from fpl_optimizer.rules import archive_season_rules
from fpl_optimizer.score import (
    GameweekScore,
    IllegalSquad,
    Squad,
    apply_autosubs,
    formation_is_legal,
    positions_from,
    realized,
    score_gameweek,
    transfer_cost,
)

from .fixtures.handbuilt import BLANK_GW, DOUBLE_GW, hand_built_season

#: Gameweek-2 totals, summed over both legs of the double. Hand-checked in
#: test_double_gameweek and repeated here deliberately.
PINNED_GW2 = {1: 6, 2: 11, 3: 8, 4: 12}


@pytest.fixture(scope="module")
def season():
    return hand_built_season()


@pytest.fixture(scope="module")
def rules():
    return archive_season_rules("2024-25")


@pytest.fixture(scope="module")
def positions(season):
    return positions_from(season)


def _squad(
    starting: tuple[int, ...],
    bench: tuple[int, ...],
    captain: int,
    vice: int,
) -> Squad:
    return Squad(starting=starting, bench=bench, captain=captain, vice=vice)


#: A legal starting XI: keeper 1, defenders 3/8/10, midfielders 2/6/9/11/14,
#: forwards 4/12. Bench: reserve keeper 7, then 5, 13, 15.
#: That is 1-3-5-2, and three players each from Reds, Blues, Greens, Whites and
#: Blacks.
XI = (1, 3, 8, 10, 2, 6, 9, 11, 14, 4, 12)
BENCH = (7, 5, 13, 15)


@pytest.fixture
def squad():
    return _squad(XI, BENCH, captain=4, vice=2)


# --- the pieces -------------------------------------------------------------


def test_realized_sums_both_legs_of_a_double(season):
    points, minutes = realized(season, DOUBLE_GW, [1, 2, 3, 4])
    assert points == PINNED_GW2
    assert minutes == {1: 180, 2: 180, 3: 180, 4: 180}


def test_realized_treats_a_blank_as_zero_not_missing(season):
    """Blacks have no gameweek-3 fixture, so the marker has no row to score."""
    points, minutes = realized(season, BLANK_GW, [13, 2])
    assert points[13] == 0
    assert minutes[13] == 0
    assert points[2] == 7


def test_positions_are_known_even_for_a_blanked_player(season):
    """Read across the season, so a club with no fixture still has positions."""
    got = positions_from(season)
    assert got[13] == "DEF"
    assert got[1] == "GKP"
    assert len(got) == 18


def test_formation_legality_follows_the_rules(rules):
    assert formation_is_legal({"GKP": 1, "DEF": 3, "MID": 5, "FWD": 2}, rules)
    assert formation_is_legal({"GKP": 1, "DEF": 5, "MID": 3, "FWD": 2}, rules)
    # Two defenders is below the minimum of three.
    assert not formation_is_legal({"GKP": 1, "DEF": 2, "MID": 5, "FWD": 3}, rules)
    # No goalkeeper.
    assert not formation_is_legal({"GKP": 0, "DEF": 4, "MID": 5, "FWD": 2}, rules)
    # Ten players.
    assert not formation_is_legal({"GKP": 1, "DEF": 3, "MID": 4, "FWD": 2}, rules)


@pytest.mark.parametrize(
    "transfers, free, expected",
    [(0, 1, 0), (1, 1, 0), (2, 1, 4), (3, 1, 8), (5, 5, 0), (6, 5, 4), (1, 0, 4)],
)
def test_transfer_cost_charges_only_the_excess(rules, transfers, free, expected):
    assert transfer_cost(transfers, free, rules) == expected


# --- captaincy --------------------------------------------------------------


def test_the_captain_doubles_his_whole_double_gameweek(season, rules, squad):
    """The finisher scored 12 + 0 across the double, so his armband adds 12.

    The exact XI total is asserted in
    :func:`test_a_gameweek_total_is_hand_checkable`; what this test pins is that
    the bonus is computed on the summed gameweek rather than on one leg of it.
    """
    score = score_gameweek(squad, season, DOUBLE_GW, rules)

    assert score.armband == 4
    assert not score.vice_used
    assert score.captain_bonus == 12  # (multiplier - 1) x 12
    xi_total = sum(score.player_points[p] for p in score.final_xi)
    assert score.starters_points == xi_total
    assert score.points == xi_total + 12


def test_the_vice_inherits_the_armband_when_the_captain_does_not_play(
    season, rules
):
    """The anchor is dropped for the double, so a captain's armband moves."""
    squad = _squad(
        (1, 3, 5, 10, 2, 6, 9, 11, 14, 4, 12), (7, 8, 13, 15), captain=5, vice=2
    )
    score = score_gameweek(squad, season, DOUBLE_GW, rules)

    assert score.player_minutes[5] == 0
    assert score.armband == 2
    assert score.vice_used
    assert score.captain_bonus == PINNED_GW2[2]


def test_nobody_is_doubled_when_neither_captain_nor_vice_plays(season, rules):
    """Both the anchor and the marker sat out gameweek 2."""
    squad = _squad(
        (1, 3, 5, 13, 2, 6, 9, 11, 14, 4, 12), (7, 8, 10, 15), captain=5, vice=13
    )
    score = score_gameweek(squad, season, DOUBLE_GW, rules)

    assert score.armband is None
    assert score.captain_bonus == 0
    assert not score.vice_used


# --- auto-subs --------------------------------------------------------------


def test_a_dropped_starter_is_replaced_in_bench_order(season, rules, positions):
    """The anchor starts and does not play; the bench is keeper, 8, 13, 15.

    The reserve keeper cannot replace an outfielder, so the stopper (8) — next
    in bench order and a defender — comes on.
    """
    squad = _squad(
        (1, 3, 5, 10, 2, 6, 9, 11, 14, 4, 12), (7, 8, 13, 15), captain=4, vice=2
    )
    final_xi, subs = apply_autosubs(
        squad,
        realized(season, DOUBLE_GW, squad.players)[1],
        positions,
        rules,
    )
    assert subs == ((5, 8),)
    assert 5 not in final_xi and 8 in final_xi
    assert formation_is_legal(
        {
            code: sum(1 for p in final_xi if positions[p] == code)
            for code in ("GKP", "DEF", "MID", "FWD")
        },
        rules,
    )


def test_a_goalkeeper_is_only_replaced_by_a_goalkeeper(season, rules, positions):
    """The keeper is injured for gameweek 3 and the reserve is third on the bench.

    Bench order does not help an outfielder here: only the reserve keeper is a
    legal replacement, so he comes on even though two outfielders precede him.
    """
    squad = _squad(
        (1, 3, 8, 10, 2, 6, 9, 11, 14, 4, 12), (5, 15, 7, 13), captain=2, vice=11
    )
    minutes = realized(season, BLANK_GW, squad.players)[1]
    assert minutes[1] == 0

    final_xi, subs = apply_autosubs(squad, minutes, positions, rules)
    assert (1, 7) in subs
    assert 7 in final_xi
    assert sum(1 for p in final_xi if positions[p] == "GKP") == 1


def test_a_blank_gameweek_is_substituted_like_any_other_absence(
    season, rules, positions
):
    """Blacks and Golds have no gameweek-3 fixture at all."""
    squad = _squad(
        (1, 3, 13, 10, 2, 6, 9, 11, 14, 4, 12), (7, 5, 8, 15), captain=2, vice=11
    )
    minutes = realized(season, BLANK_GW, squad.players)[1]
    assert minutes[13] == 0 and minutes[14] == 0

    final_xi, subs = apply_autosubs(squad, minutes, positions, rules)
    # Three absences: the injured keeper and two blanked outfielders. The
    # reserve keeper takes the gloves, then the anchor and stopper come on.
    assert subs == ((1, 7), (13, 5), (14, 8))
    assert 13 not in final_xi and 14 not in final_xi
    assert 7 in final_xi


def test_no_substitution_is_made_if_it_would_break_the_formation(
    season, rules, positions
):
    """Three defenders start, one is dropped, and no bench defender played.

    The understudy forward did play, but bringing him on would leave two
    defenders — below the minimum — so FPL makes no substitution at all and the
    dropped defender's zero stands.
    """
    squad = _squad(
        (1, 3, 5, 10, 2, 6, 9, 11, 14, 4, 12), (7, 13, 16, 18), captain=4, vice=2
    )
    minutes = realized(season, DOUBLE_GW, squad.players)[1]
    assert minutes[5] == 0  # the dropped starter
    assert minutes[13] == 0 and minutes[16] == 0  # both bench defenders sat out
    assert minutes[18] > 0  # the understudy forward did play

    final_xi, subs = apply_autosubs(squad, minutes, positions, rules)
    assert subs == ()
    assert final_xi == squad.starting


def test_a_bench_player_who_did_not_play_cannot_come_on(season, rules, positions):
    squad = _squad(
        (1, 3, 5, 10, 2, 6, 9, 11, 14, 4, 12), (7, 13, 8, 15), captain=4, vice=2
    )
    minutes = realized(season, DOUBLE_GW, squad.players)[1]
    final_xi, subs = apply_autosubs(squad, minutes, positions, rules)

    # The marker (13) is ahead of the stopper (8) in bench order but sat out.
    assert subs == ((5, 8),)
    assert 13 not in final_xi


def test_each_bench_player_comes_on_at_most_once(season, rules, positions):
    squad = _squad(
        (1, 3, 5, 13, 2, 6, 9, 11, 14, 4, 12), (7, 8, 10, 15), captain=4, vice=2
    )
    minutes = realized(season, DOUBLE_GW, squad.players)[1]
    final_xi, subs = apply_autosubs(squad, minutes, positions, rules)

    assert len(set(p for _, p in subs)) == len(subs)
    assert len(final_xi) == rules.squad_play
    assert len(set(final_xi)) == rules.squad_play


# --- the whole thing --------------------------------------------------------


def test_the_score_is_the_final_xi_plus_the_armband_minus_the_hit(
    season, rules, squad
):
    score = score_gameweek(squad, season, DOUBLE_GW, rules, transfer_hit=4)
    assert isinstance(score, GameweekScore)
    assert score.points == score.starters_points + score.captain_bonus - 4
    assert score.transfer_hit == 4


def test_bench_points_are_counted_but_not_scored(season, rules, squad):
    score = score_gameweek(squad, season, DOUBLE_GW, rules)
    benched = set(squad.players) - set(score.final_xi)

    assert score.bench_points == sum(score.player_points[p] for p in benched)
    assert score.bench_points > 0
    assert score.points == score.starters_points + score.captain_bonus


def test_substituted_in_points_count_and_substituted_out_do_not(season, rules):
    squad = _squad(
        (1, 3, 5, 10, 2, 6, 9, 11, 14, 4, 12), (7, 8, 13, 15), captain=4, vice=2
    )
    score = score_gameweek(squad, season, DOUBLE_GW, rules)

    assert score.autosubs == ((5, 8),)
    assert 8 in score.final_xi
    # The stopper's gameweek-2 point is in the total; the anchor's zero is not.
    assert score.player_points[8] == 1
    assert score.starters_points == sum(score.player_points[p] for p in score.final_xi)


def test_a_gameweek_total_is_hand_checkable(season, rules):
    """The whole calculation, on paper.

    XI: keeper 1 (6), fullback 3 (8), stopper 8 (1), overlap 10 (9),
    playmaker 2 (11), winger 6 (7), engine 9 (2), creator 11 (11),
    runner 14 (6), finisher 4 (12), poacher 12 (6) — 79 points.
    Everyone played, so there are no substitutions. The finisher is captain:
    +12. One transfer was taken as a hit: -4. Total 87.
    """
    squad = _squad(XI, BENCH, captain=4, vice=2)
    score = score_gameweek(squad, season, DOUBLE_GW, rules, transfer_hit=4)

    assert score.autosubs == ()
    assert {p: score.player_points[p] for p in XI} == {
        1: 6,
        3: 8,
        8: 1,
        10: 9,
        2: 11,
        6: 7,
        9: 2,
        11: 11,
        14: 6,
        4: 12,
        12: 6,
    }
    assert score.starters_points == 79
    assert score.captain_bonus == 12
    assert score.points == 87


# --- squad legality ---------------------------------------------------------


def test_a_legal_squad_validates(season, rules, squad, positions):
    squad.validate(positions, rules)


@pytest.mark.parametrize(
    "squad, message",
    [
        (_squad(XI, (7, 5, 13), 4, 2), "squad has 14 players"),
        (_squad(XI[:10], (7, 5, 13, 15, 16), 4, 2), "starting XI has 10"),
        (_squad(XI, (7, 5, 13, 12), 4, 2), "appears twice"),
        (_squad(XI, BENCH, 16, 2), "captain is not in the starting XI"),
        (_squad(XI, BENCH, 4, 16), "vice-captain is not in the starting XI"),
        (_squad(XI, BENCH, 4, 4), "same player"),
    ],
)
def test_an_illegal_squad_is_rejected(season, rules, positions, squad, message):
    with pytest.raises(IllegalSquad, match=message):
        squad.validate(positions, rules)


def test_an_illegal_formation_is_rejected(season, rules, positions):
    """A strikerless XI: the quotas are all satisfied, the formation is not.

    Five defenders and five midfielders behind a goalkeeper is eleven players,
    two from each club, and every squad quota met — but FPL requires at least
    one forward on the pitch, so this XI could never have been fielded.
    """
    strikerless = _squad(
        (1, 3, 5, 8, 10, 13, 2, 6, 9, 11, 14), (7, 4, 12, 15), captain=2, vice=11
    )
    with pytest.raises(IllegalSquad, match="not a legal formation"):
        strikerless.validate(positions, rules)


def test_scoring_validates_by_default(season, rules, positions):
    broken = _squad(XI, (7, 5, 13), captain=4, vice=2)
    with pytest.raises(IllegalSquad):
        score_gameweek(broken, season, DOUBLE_GW, rules)

    # ...and can be told not to, for callers that validated once already.
    score = score_gameweek(broken, season, DOUBLE_GW, rules, validate=False)
    assert score.points > 0
