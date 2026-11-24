"""The candidate pool and the squad integer program.

Two kinds of test. The pool tests are about availability — who can be owned,
which is not the same as who is playing — and they carry the same leakage check
as the rest of the project. The program tests are about optimality under
constraints, and they use candidate frames small enough that the right answer
can be worked out by hand, because "the solver said so" is not a test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_optimizer.config import Config
from fpl_optimizer.optimize import (
    InfeasibleSquad,
    build_candidates,
    horizon_predictions,
    latest_known,
    optimize_squad,
    pick_squad,
    prune_pool,
    validate_candidates,
)
from fpl_optimizer.projections import EPProjection
from fpl_optimizer.rules import archive_season_rules
from fpl_optimizer.score import positions_from, score_gameweek

from .fixtures import corrupt_outcomes
from .fixtures.handbuilt import BLANK_GW, DOUBLE_GW, PLAYERS, hand_built_season


@pytest.fixture(scope="module")
def season():
    return hand_built_season()


@pytest.fixture(scope="module")
def rules():
    return archive_season_rules("2024-25")


@pytest.fixture
def config():
    """No club exclusions, so the fictional clubs are all selectable.

    The real default excludes three clubs; that behaviour has its own test.
    """
    return Config(excluded_clubs=())


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """Candidate frame from ``(player_id, position, club, price, xpts)`` tuples."""
    return pd.DataFrame(
        rows, columns=["player_id", "position", "club", "price", "xpts"]
    )


def _spread(
    n_per_club: int = 3,
    *,
    price: int = 40,
    xpts: float = 1.0,
    start_id: int = 1,
    clubs: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G"),
) -> list[tuple]:
    """A legal-by-construction pool: 2 GKP, 5 DEF, 5 MID, 3 FWD across clubs."""
    wanted = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    rows = []
    for i, position in enumerate(wanted):
        rows.append(
            (start_id + i, position, clubs[i // n_per_club], price, xpts)
        )
    return rows


# --- the candidate pool -----------------------------------------------------


def test_the_pool_is_the_roster_not_the_fixture_list(season):
    """Blanked players stay ownable, at zero expected points.

    Blacks and Golds have no gameweek-3 fixture, so six players have no row to
    project. Dropping them would leave four ownable defenders in the league and
    the program would report a blank gameweek as infeasible.
    """
    preds = EPProjection().predict(season, BLANK_GW)
    pool = build_candidates(season, BLANK_GW, preds)

    assert len(pool) == len(PLAYERS)
    blanked = [p for p, (_, _, club, _) in PLAYERS.items() if club in ("Blacks", "Golds")]
    assert set(blanked) <= set(pool["player_id"])
    assert pool.set_index("player_id").loc[blanked, "xpts"].eq(0.0).all()
    assert pool["position"].value_counts()["DEF"] == 6


def test_the_pool_carries_the_most_recent_known_price(season):
    """Prices are pre-deadline, so the latest row at or before the gameweek wins."""
    data = hand_built_season()
    df = data.player_gws.copy()
    df.loc[(df["player_id"] == 4) & (df["gw"] >= 2), "price"] = 125
    data.player_gws = df

    assert int(latest_known(data, 1).set_index("player_id").loc[4, "price"]) == 120
    assert int(latest_known(data, 2).set_index("player_id").loc[4, "price"]) == 125
    # Never a later gameweek's price.
    assert int(latest_known(data, 1).set_index("player_id").loc[4, "price"]) != 125


def test_selling_prices_default_to_the_market_price(season):
    preds = EPProjection().predict(season, DOUBLE_GW)
    pool = build_candidates(season, DOUBLE_GW, preds, selling_prices={4: 115})
    indexed = pool.set_index("player_id")

    assert int(indexed.loc[4, "selling_price"]) == 115
    assert int(indexed.loc[4, "price"]) == 120
    # Anyone not owned sells for what they cost.
    assert int(indexed.loc[2, "selling_price"]) == int(indexed.loc[2, "price"])


def test_the_horizon_is_a_weighted_average_not_a_sum(season):
    """On the same scale as one gameweek, which is what keeps hits honest.

    Summed, the horizon would count a transfer's gain three or four times over
    against a hit charged once, and the program would churn the squad.
    """
    proj = EPProjection()
    future = horizon_predictions(proj, season, 1, horizon=2, decay=0.5)
    direct = {
        gw: proj.predict(season, gw).set_index("player_id")["xpts"] for gw in (2, 3)
    }
    got = future.set_index("player_id")["xpts"]

    # Gameweek 2 at half weight, gameweek 3 at a quarter, divided by 0.75.
    expected = (direct[2][4] * 0.5 + direct[3][4] * 0.25) / 0.75
    assert got[4] == pytest.approx(expected)
    # Bounded by the gameweeks it averages, which a sum would not be.
    assert got[4] <= max(direct[2][4], direct[3][4]) + 1e-9
    assert got[4] >= min(direct[2][4], direct[3][4]) - 1e-9


def test_the_horizon_runs_out_at_the_end_of_the_season(season):
    future = horizon_predictions(EPProjection(), season, 3, horizon=4, decay=0.9)
    assert future.empty


def test_the_pool_does_not_move_when_the_future_changes(season):
    """The leakage property, carried into the decision layer."""
    gw = DOUBLE_GW
    corrupted = corrupt_outcomes(season, gw)
    proj = EPProjection()

    clean = build_candidates(season, gw, proj.predict(season, gw))
    dirty = build_candidates(corrupted, gw, proj.predict(corrupted, gw))
    pd.testing.assert_frame_equal(clean, dirty)


def test_pruning_keeps_the_best_and_whatever_is_held(config):
    rows = _spread(xpts=1.0) + _spread(start_id=100, xpts=9.0)
    pool = prune_pool(
        _frame(rows),
        Config(pool_per_position={"GKP": 1, "DEF": 2, "MID": 2, "FWD": 1}),
        keep=(1,),
    )
    # The nine-point players win their slots, and the held player survives.
    assert set(pool["player_id"]) >= {1}
    assert pool["position"].value_counts().to_dict() == {
        "DEF": 2,
        "MID": 2,
        "GKP": 2,  # one by rank, one because it is held
        "FWD": 1,
    }
    assert pool.loc[pool["position"] == "MID", "xpts"].eq(9.0).all()


def test_candidate_validation_catches_malformed_frames():
    with pytest.raises(ValueError, match="missing columns"):
        validate_candidates(pd.DataFrame({"player_id": [1]}))
    with pytest.raises(ValueError, match="duplicate players"):
        validate_candidates(_frame([(1, "GKP", "A", 40, 1.0), (1, "DEF", "A", 40, 1.0)]))
    with pytest.raises(ValueError, match="unknown positions"):
        validate_candidates(_frame([(1, "SWEEPER", "A", 40, 1.0)]))


# --- the program: legality --------------------------------------------------


def test_the_chosen_squad_satisfies_every_rule(season, rules, config):
    preds = EPProjection().predict(season, DOUBLE_GW)
    pool = build_candidates(season, DOUBLE_GW, preds)
    plan = pick_squad(pool, rules, config=config)

    # The scorer's own validation is the arbiter, since it is what a backtest
    # would hit if the optimizer produced something illegal.
    plan.squad.validate(positions_from(season), rules)
    assert plan.spend <= rules.budget
    assert plan.bank == rules.budget - plan.spend

    chosen = plan.squad.players
    by_club = pd.Series([PLAYERS[p][2] for p in chosen]).value_counts()
    assert by_club.max() <= rules.team_limit


def test_the_armband_goes_to_the_best_starter(rules, config):
    rows = _spread()
    # Positions are kept as _spread laid them out: index 7 is a midfielder and
    # index 12 a forward. Changing one would break a squad quota, not the test.
    rows[7] = (8, "MID", "C", 40, 9.0)  # the standout
    rows[12] = (13, "FWD", "E", 40, 7.0)  # the runner-up
    plan = pick_squad(_frame(rows), rules, config=config)

    assert plan.squad.captain == 8
    assert plan.squad.vice == 13
    assert plan.squad.captain in plan.squad.starting
    assert plan.squad.vice in plan.squad.starting


def test_the_bench_is_ordered_by_expected_points(rules, config):
    rows = _spread(xpts=1.0) + [
        (100 + i, pos, "H", 40, 2.0 + i)
        for i, pos in enumerate(("DEF", "MID", "FWD"))
    ]
    plan = pick_squad(_frame(rows), rules, config=config)
    bench_xpts = [
        float(_frame(rows).set_index("player_id").loc[p, "xpts"])
        for p in plan.squad.bench
    ]
    assert bench_xpts == sorted(bench_xpts, reverse=True)


def test_a_legal_squad_is_preferred_to_a_better_illegal_one(rules, config):
    """Fifteen cheap players, and four stars nobody can afford.

    The stars are worth ten points each and cost 100.0m, so a greedy pick by
    expected points buys one and then cannot field a squad. The only legal
    fifteen is the cheap one, and that is what the program must return.
    """
    cheap = _spread(price=40, xpts=1.0)
    stars = [
        (100, "FWD", "Z", 1000, 10.0),
        (101, "MID", "Z", 1000, 10.0),
        (102, "DEF", "Y", 1000, 10.0),
        (103, "GKP", "Y", 1000, 10.0),
    ]
    plan = pick_squad(_frame(cheap + stars), rules, config=config)

    assert set(plan.squad.players) == {row[0] for row in cheap}
    assert plan.spend == 15 * 40


def test_the_club_limit_binds(rules, config):
    """Three per club, so the fourth-best player at a loaded club is passed over."""
    rows = _spread(clubs=("A",) * 7, xpts=1.0)  # all fifteen from one club
    rows += _spread(start_id=200, clubs=("B", "C", "D", "E", "F", "G", "H"), xpts=0.5)
    plan = pick_squad(_frame(rows), rules, config=config)

    frame = _frame(rows).set_index("player_id")
    counts = pd.Series(
        [frame.loc[p, "club"] for p in plan.squad.players]
    ).value_counts()
    assert counts.max() <= rules.team_limit
    assert counts.get("A", 0) == rules.team_limit


def test_excluded_clubs_are_never_bought(rules):
    """Config names clubs that are off-limits under every strategy."""
    rows = _spread(clubs=("Arsenal",) * 7, xpts=9.0)
    rows += _spread(start_id=200, clubs=("B", "C", "D", "E", "F", "G", "H"), xpts=0.5)
    frame = _frame(rows)
    config = Config()  # the real default: Arsenal, Spurs and Man Utd excluded

    plan = pick_squad(frame, rules, config=config)
    chosen = frame.set_index("player_id").loc[list(plan.squad.players), "club"]
    assert "Arsenal" not in set(chosen)

    # ...and the exclusion can be lifted deliberately.
    lifted = pick_squad(frame, rules, config=config, allow_excluded_clubs=True)
    lifted_clubs = frame.set_index("player_id").loc[list(lifted.squad.players), "club"]
    assert "Arsenal" in set(lifted_clubs)


def test_must_include_and_must_exclude_are_honoured(rules, config):
    rows = _spread(xpts=1.0) + [(50, "FWD", "H", 45, 0.1)]
    plan = pick_squad(_frame(rows), rules, config=config, must_include=(50,))
    assert 50 in plan.squad.players

    # Exclude a forward: the extra candidate leaves four, so three remain and
    # the squad stays feasible. Excluding a goalkeeper would just be infeasible.
    without = pick_squad(_frame(rows), rules, config=config, must_exclude=(15,))
    assert 15 not in without.squad.players
    assert len(without.squad.players) == rules.squad_size


# --- the program: infeasibility ---------------------------------------------


def test_too_few_candidates_in_a_position_says_which(rules, config):
    rows = [r for r in _spread() if r[1] != "FWD"]
    with pytest.raises(InfeasibleSquad, match="only 0 FWD candidates"):
        pick_squad(_frame(rows), rules, config=config)


def test_too_few_clubs_says_so(rules, config):
    rows = _spread(clubs=("A",) * 7)
    with pytest.raises(InfeasibleSquad, match="players per club"):
        pick_squad(_frame(rows), rules, config=config)


def test_an_unaffordable_pool_says_so(rules, config):
    rows = _spread(price=200)  # fifteen players at 20.0m each
    with pytest.raises(InfeasibleSquad, match="no legal squad exists"):
        pick_squad(_frame(rows), rules, config=config)


def test_must_include_outside_the_pool_is_an_error(rules, config):
    with pytest.raises(InfeasibleSquad, match="must_include"):
        pick_squad(_frame(_spread()), rules, config=config, must_include=(9999,))


# --- the program: transfers -------------------------------------------------


def test_a_free_transfer_is_taken_when_it_improves_the_squad(rules, config):
    rows = _spread(xpts=1.0) + [(60, "FWD", "H", 40, 6.0)]
    current = tuple(row[0] for row in _spread())

    plan = optimize_squad(
        _frame(rows), rules, config=config, current=current, free_transfers=1, bank=0
    )
    assert plan.n_transfers == 1
    assert plan.transfers_in == (60,)
    assert plan.transfer_hit == 0


def test_a_transfer_that_does_not_pay_for_itself_is_declined(rules, config):
    """One point better, four points to buy it: the program says no.

    The hit is in the objective rather than deducted afterwards, which is what
    makes this decision rather than arithmetic after the fact.
    """
    rows = _spread(xpts=1.0) + [(60, "FWD", "H", 40, 2.0)]
    current = tuple(row[0] for row in _spread())

    plan = optimize_squad(
        _frame(rows), rules, config=config, current=current, free_transfers=0, bank=0
    )
    assert plan.n_transfers == 0
    assert plan.transfer_hit == 0


def test_a_transfer_worth_more_than_the_hit_is_taken(rules, config):
    rows = _spread(xpts=1.0) + [(60, "FWD", "H", 40, 9.0)]
    current = tuple(row[0] for row in _spread())

    plan = optimize_squad(
        _frame(rows), rules, config=config, current=current, free_transfers=0, bank=0
    )
    assert plan.transfers_in == (60,)
    assert plan.transfer_hit == rules.transfer_hit_cost


def test_transfers_are_paid_for_out_of_the_bank_and_the_proceeds(rules, config):
    """A 9.0m replacement for a 4.0m player needs 5.0m from somewhere."""
    rows = _spread(price=40, xpts=1.0) + [(60, "FWD", "H", 90, 9.0)]
    current = tuple(row[0] for row in _spread(price=40))

    broke = optimize_squad(
        _frame(rows), rules, config=config, current=current, free_transfers=1, bank=0
    )
    assert broke.n_transfers == 0

    funded = optimize_squad(
        _frame(rows), rules, config=config, current=current, free_transfers=1, bank=50
    )
    assert funded.transfers_in == (60,)
    assert funded.bank == 0


def test_a_price_rise_is_banked_when_selling(rules, config):
    """Selling at 5.0m what cost 4.0m funds a 5.0m buy with nothing in the bank."""
    rows = _spread(price=40, xpts=1.0) + [(60, "FWD", "H", 50, 9.0)]
    frame = _frame(rows)
    current = tuple(row[0] for row in _spread(price=40))
    frame["selling_price"] = frame["price"]
    frame.loc[frame["player_id"] == 13, "selling_price"] = 50  # a FWD who rose

    plan = optimize_squad(
        frame, rules, config=config, current=current, free_transfers=1, bank=0
    )
    assert plan.transfers_in == (60,)
    assert 13 in plan.transfers_out


def test_the_number_of_transfers_matches_what_changed(rules, config):
    rows = _spread(xpts=1.0) + [
        (60, "FWD", "H", 40, 9.0),
        (61, "MID", "H", 40, 9.0),
        (62, "DEF", "I", 40, 9.0),
    ]
    current = tuple(row[0] for row in _spread())

    plan = optimize_squad(
        _frame(rows), rules, config=config, current=current, free_transfers=5, bank=0
    )
    assert plan.n_transfers == len(plan.transfers_in) == len(plan.transfers_out)
    assert set(plan.transfers_in) <= {60, 61, 62}
    assert plan.transfer_hit == 0


# --- determinism and the handover to the scorer -----------------------------


def test_the_same_inputs_give_the_same_squad(season, rules, config):
    pool = build_candidates(season, DOUBLE_GW, EPProjection().predict(season, DOUBLE_GW))
    first = pick_squad(pool, rules, config=config)
    second = pick_squad(pool, rules, config=config)

    assert first.squad == second.squad
    assert first.expected_points == pytest.approx(second.expected_points)


def test_what_the_optimizer_returns_is_what_the_scorer_accepts(season, rules, config):
    """The two halves meet here: no adapter, no reshaping, one Squad type."""
    pool = build_candidates(season, DOUBLE_GW, EPProjection().predict(season, DOUBLE_GW))
    plan = pick_squad(pool, rules, config=config)

    score = score_gameweek(plan.squad, season, DOUBLE_GW, rules)
    assert score.points > 0
    assert score.armband in plan.squad.starting


# --- ties -------------------------------------------------------------------


def test_a_tied_problem_is_solved_the_same_way_every_time(rules, config):
    """Ties are common and a solver may return any optimum it likes.

    Two defenders on identical projections is unremarkable, and the solver is
    free to pick either. That showed up as a backtest total moving by six
    points between identical runs, because the two tied players went on to
    score differently. The decision is genuinely arbitrary; which arbitrary
    answer it is should not be.
    """
    rows = [
        (i, position, f"C{i // 3}", 45, 3.0)
        for i, position in enumerate(
            ["GKP"] * 2 + ["DEF"] * 8 + ["MID"] * 8 + ["FWD"] * 5, start=1
        )
    ]
    frame = _frame(rows)

    solved = {
        (
            tuple(plan.squad.starting),
            tuple(plan.squad.bench),
            plan.squad.captain,
        )
        for plan in (pick_squad(frame, rules, config=config) for _ in range(5))
    }
    assert len(solved) == 1


def test_the_tie_break_cannot_outweigh_a_real_difference(rules, config):
    """It is a nudge, not a preference: a tenth of a point still wins.

    The whole tie-break term is bounded well under a thousandth of a point, so
    it decides only between options the objective already considers equal.
    """
    rows = _spread(xpts=1.0)
    # The highest player id, which the tie-break disfavours most, is also the
    # best player by a margin the tie-break cannot touch.
    rows[-1] = (15, "FWD", "E", 40, 1.1)
    plan = pick_squad(_frame(rows), rules, config=config)

    assert 15 in plan.squad.starting
    assert plan.squad.captain == 15
