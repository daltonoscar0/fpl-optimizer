"""The walk-forward backtest.

The test that matters most here is
:func:`test_decisions_do_not_move_when_the_results_do`. Everywhere else in this
project the leakage property is asserted about features and projections, which
is as far downstream as there was code to assert it about. There is now a
decision, so the claim can be made in the form it was always meant to take: the
squad chosen for gameweek *g* is unchanged when every result from *g* onward is
rewritten to nonsense. Only the score moves, because only the score is entitled
to read them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.backtest import (
    BacktestResult,
    _next_free_transfers,
    backtest,
    compare,
)
from fpl_optimizer.config import Config
from fpl_optimizer.features import FeatureCache
from fpl_optimizer.rules import archive_season_rules
from fpl_optimizer.score import positions_from

from .fixtures import corrupt_outcomes, make_season
from .fixtures.handbuilt import hand_built_season


@pytest.fixture(scope="module")
def rules():
    return archive_season_rules("2024-25")


@pytest.fixture
def config():
    """Fictional clubs, so nothing is excluded, and a short horizon."""
    return Config(excluded_clubs=(), horizon=2)


@pytest.fixture(scope="module")
def season():
    return hand_built_season()


# --- the loop ---------------------------------------------------------------


def test_a_season_is_scored_gameweek_by_gameweek(season, rules, config):
    result = backtest(season, "ep", rules, config=config)

    assert isinstance(result, BacktestResult)
    assert [r.gw for r in result.records] == season.gameweeks
    assert result.total_points == sum(r.points for r in result.records)
    assert result.records[-1].cumulative == result.total_points
    assert result.gameweeks == 3


def test_the_naive_strategy_cannot_start_before_a_gameweek_has_been_played(
    season, rules, config
):
    """Its rate is zero for everyone in gameweek 1, which is not a view.

    An all-zero projection used to be allowed to pick, and the squad it chose —
    arbitrary, and well under budget — then cost points in hits to unwind.
    """
    result = backtest(season, "naive", rules, config=config)

    assert [r.gw for r in result.records] == [2, 3]
    assert 1 not in result.decided_gws


def test_every_gameweeks_squad_is_legal(season, rules, config):
    """Checked per gameweek, because an illegal squad scores points nobody could."""
    from fpl_optimizer.score import Squad

    positions = positions_from(season)
    result = backtest(season, "ep", rules, config=config)

    for record in result.records:
        bench = tuple(p for p in record.squad if p not in set(record.starting))
        Squad(
            starting=record.starting,
            bench=bench,
            captain=record.captain,
            vice=next(p for p in record.starting if p != record.captain),
        ).validate(positions, rules)
        assert len(record.squad) == rules.squad_size
        assert record.squad_value + record.bank >= 0


def test_the_first_gameweek_is_a_fresh_pick_not_a_transfer(season, rules, config):
    result = backtest(season, "ep", rules, config=config)
    first = result.records[0]

    assert first.n_transfers == 0
    assert first.transfer_hit == 0
    assert first.squad_value <= rules.budget


def test_the_squad_carries_between_gameweeks(season, rules, config):
    """Only the transfers change the squad; the rest is held."""
    result = backtest(season, "ep", rules, config=config)

    for previous, current in zip(result.records, result.records[1:]):
        changed = set(current.squad) - set(previous.squad)
        assert changed == set(current.transfers_in)
        assert set(previous.squad) - set(current.squad) == set(current.transfers_out)


def test_money_is_conserved_across_a_transfer(season, rules, config):
    result = backtest(season, "naive", rules, config=config)
    for record in result.records:
        assert record.bank >= 0


# --- free transfers ---------------------------------------------------------


@pytest.mark.parametrize(
    "free, used, expected",
    [
        (1, 0, 2),  # unused, so it banks
        (1, 1, 1),  # spent one, gained one
        (1, 3, 1),  # two were hits; the free one was spent, one accrues
        (5, 0, 5),  # already at the cap, which is as high as it goes
        (5, 2, 4),
        (2, 2, 1),
    ],
)
def test_free_transfers_accrue_and_cap(rules, free, used, expected):
    assert _next_free_transfers(free, used, rules) == expected


def test_free_transfers_are_tracked_through_a_season(season, rules, config):
    result = backtest(season, "ep", rules, config=config)
    for record in result.records:
        assert 0 <= record.free_transfers_after <= rules.max_free_transfers


# --- a strategy with no view ------------------------------------------------


def test_a_gameweek_with_no_projection_holds_the_squad(rules, config):
    """FPL published no expected points for the last gameweek here.

    The strategy does not get to decide a gameweek it has no view of: the squad
    stands, no transfers are made, and the gameweek is recorded as held.
    """
    data = hand_built_season()
    df = data.player_gws.copy()
    df.loc[df["gw"] == 3, "ep"] = np.nan
    data.player_gws = df

    result = backtest(data, "ep", rules, config=config)

    assert result.held_gws == [3]
    assert result.decided_gws == [1, 2]
    held = result.records[-1]
    assert held.n_transfers == 0
    assert held.transfer_hit == 0
    assert held.squad == result.records[-2].squad
    # It still scores: the players still played.
    assert held.points != 0
    # And a free transfer still accrues — no transfer was made, but the
    # gameweek happened.
    assert held.free_transfers_after == min(
        rules.max_free_transfers, result.records[-2].free_transfers_after + 1
    )


def test_a_strategy_with_no_view_at_all_records_nothing(rules, config):
    """It cannot start a season it cannot see, and says so rather than guessing."""
    data = make_season(n_gws=4, with_ep=False)
    result = backtest(data, "ep", rules, config=Config(excluded_clubs=(), horizon=2))

    assert result.records == []
    assert result.total_points == 0
    assert any("below the" in note for note in result.notes)


def test_coverage_is_reported_for_the_expected_points_strategy(season, rules, config):
    result = backtest(season, "ep", rules, config=config)
    assert any("cover" in note for note in result.notes)


# --- the leakage property, at the decision ----------------------------------


@pytest.mark.parametrize("gw", [2, 3])
def test_decisions_do_not_move_when_the_results_do(rules, config, gw):
    """Rewrite every result from ``gw`` onward; the squad chosen must not change.

    The score is expected to change — it is computed from the results, which is
    the one place they are allowed to be read. What must not change is the
    decision: the fifteen, the eleven, the armband and the transfers.
    """
    clean_data = hand_built_season()
    dirty_data = corrupt_outcomes(clean_data, gw)

    clean = backtest(clean_data, "naive", rules, config=config, end_gw=gw)
    dirty = backtest(dirty_data, "naive", rules, config=config, end_gw=gw)

    decision = clean.records[-1]
    corrupted = dirty.records[-1]
    assert decision.gw == corrupted.gw == gw
    assert decision.squad == corrupted.squad
    assert decision.starting == corrupted.starting
    assert decision.captain == corrupted.captain
    assert decision.transfers_in == corrupted.transfers_in
    assert decision.transfers_out == corrupted.transfers_out
    assert decision.transfer_hit == corrupted.transfer_hit


def test_the_control_the_score_does_move(rules, config):
    """Without this, a backtest that ignored results entirely would pass above."""
    clean_data = hand_built_season()
    dirty_data = corrupt_outcomes(clean_data, 2)

    clean = backtest(clean_data, "naive", rules, config=config, end_gw=2)
    dirty = backtest(dirty_data, "naive", rules, config=config, end_gw=2)
    assert clean.records[-1].points != dirty.records[-1].points


def test_an_earlier_corruption_does_change_the_decision(rules, config):
    """The past is supposed to inform the decision — the other control."""
    clean_data = hand_built_season()
    dirty_data = corrupt_outcomes(clean_data, 1)

    clean = backtest(clean_data, "naive", rules, config=config, end_gw=3)
    dirty = backtest(dirty_data, "naive", rules, config=config, end_gw=3)
    assert clean.records[-1].squad != dirty.records[-1].squad


# --- the model, and comparison ----------------------------------------------


def test_the_model_reports_the_gameweeks_it_deferred_on(season, rules, config):
    """Three gameweeks is never enough history, so it defers every week.

    The hand-off is reported rather than hidden, which is the point: a season
    total built on a baseline is not a season total built on the model.
    """
    result = backtest(season, "model", rules, config=config)

    assert result.fallback_gws == season.gameweeks
    assert any("deferred" in note for note in result.notes)
    assert result.total_points > 0


def test_the_model_fits_and_runs_once_it_has_history(rules):
    """A wider season, so the three-per-club cap leaves a squad to field.

    The synthetic season has four clubs and seven players each; at three per
    club that is twelve ownable players against a squad of fifteen, so the team
    limit is relaxed here rather than the season being reshaped around it.
    """
    from dataclasses import replace

    data = make_season(n_gws=14)
    wide = replace(rules, team_limit=7)
    config = Config(
        excluded_clubs=(),
        horizon=2,
        pool_per_position={"GKP": 8, "DEF": 12, "MID": 12, "FWD": 8},
    )
    result = backtest(data, "model", wide, config=config, cache=FeatureCache(data))

    assert result.gameweeks == len(data.gameweeks)
    assert result.total_points > 0
    # It defers early and fits later; both must show up.
    assert result.fallback_gws
    assert len(result.fallback_gws) < result.gameweeks


def test_compare_puts_the_strategies_side_by_side(season, rules, config):
    results = [
        backtest(season, name, rules, config=config) for name in ("naive", "ep")
    ]
    table = compare(results)

    assert set(table["strategy"]) == {"naive", "ep"}
    assert table.loc[table["strategy"] == "naive", "vs_naive"].isna().all()

    ep_row = table[table["strategy"] == "ep"].iloc[0]
    naive_row = table[table["strategy"] == "naive"].iloc[0]
    # The gap is measured inside the window both strategies scored, not between
    # totals over different numbers of gameweeks.
    assert int(ep_row["vs_naive"]) == int(ep_row["window_points"]) - int(
        naive_row["window_points"]
    )
    assert int(ep_row["window_gws"]) == int(naive_row["window_gws"]) == 2
    assert int(ep_row["gameweeks"]) == 3 and int(naive_row["gameweeks"]) == 2


def test_a_strategy_that_knows_who_played_is_marked_not_comparable(rules, config):
    """It still runs and still scores; what it does not get is a comparison.

    A gap against the baseline would read as a finding about projections, when
    it is a finding about a data column.
    """
    data = hand_built_season()
    df = data.player_gws.copy()
    absent = pd.to_numeric(df["minutes"], errors="coerce").fillna(0) == 0
    df.loc[absent, "ep"] = 0.0
    df.loc[~absent, "ep"] = 4.0
    data.player_gws = df

    result = backtest(data, "ep", rules, config=config)
    assert not result.comparable
    assert "knows who played" in (result.not_comparable_because or "")
    assert result.total_points > 0, "it is still run and still scored"

    naive = backtest(data, "naive", rules, config=config)
    assert naive.comparable
    table = compare([result, naive])
    ep_row = table[table["strategy"] == "ep"].iloc[0]
    assert ep_row["vs_naive"] is None or pd.isna(ep_row["vs_naive"])
    assert int(ep_row["window_points"]) > 0, "the figure is reported, not hidden"


def _wide(rules):
    """The synthetic season has four clubs; three per club cannot field fifteen."""
    from dataclasses import replace

    return replace(rules, team_limit=7), Config(
        excluded_clubs=(),
        horizon=2,
        pool_per_position={"GKP": 8, "DEF": 12, "MID": 12, "FWD": 8},
    )


def test_the_naive_baseline_never_reads_the_expected_points_column(rules):
    """Delete FPL's column outright and its season does not move."""
    data = make_season(n_gws=12)
    stripped = make_season(n_gws=12)
    df = stripped.player_gws.copy()
    df["ep"] = np.nan
    stripped.player_gws = df
    wide, config = _wide(rules)

    with_ep = backtest(data, "naive", wide, config=config)
    without = backtest(stripped, "naive", wide, config=config)

    assert with_ep.total_points == without.total_points
    assert [r.squad for r in with_ep.records] == [r.squad for r in without.records]


def test_the_model_reads_it_only_through_a_fallback_it_can_trust(rules):
    """So on archive data — where it refuses the fallback — it reads it not at all.

    This is what makes the expected-points coverage threshold a fact about one
    strategy rather than about a season. 2025-26 carries `xP` for 11 of 38
    gameweeks, which disqualifies `ep` there and says nothing about the model:
    the column is absent from the feature set, the candidate pool and the
    scorer, and the one path that did reach it — the too-little-history
    fallback — is refused on exactly the data that is contaminated.

    Checked here by giving a season the archive's signature (non-appearances
    rated zero) and then deleting the column: identical seasons either way.
    """
    hindsight = make_season(n_gws=12)
    df = hindsight.player_gws.copy()
    absent = pd.to_numeric(df["minutes"], errors="coerce").fillna(0) == 0
    df.loc[absent, "ep"] = 0.0
    df.loc[~absent, "ep"] = 4.0
    hindsight.player_gws = df

    stripped = make_season(n_gws=12)
    df2 = stripped.player_gws.copy()
    df2["ep"] = np.nan
    stripped.player_gws = df2

    wide, config = _wide(rules)
    with_ep = backtest(
        hindsight, "model", wide, config=config, cache=FeatureCache(hindsight)
    )
    without = backtest(
        stripped, "model", wide, config=config, cache=FeatureCache(stripped)
    )

    assert with_ep.total_points == without.total_points
    assert [r.squad for r in with_ep.records] == [r.squad for r in without.records]
    assert [r.captain for r in with_ep.records] == [r.captain for r in without.records]


def test_the_model_does_use_a_fallback_that_is_a_forecast(rules):
    """The control for the test above: the fallback is not dead code.

    Where the column behaves like a forecast, the model leans on it for its
    first gameweeks — and deleting it then *does* change the season.
    """
    data = make_season(n_gws=12)
    stripped = make_season(n_gws=12)
    df = stripped.player_gws.copy()
    df["ep"] = np.nan
    stripped.player_gws = df
    wide, config = _wide(rules)

    with_ep = backtest(data, "model", wide, config=config, cache=FeatureCache(data))
    without = backtest(
        stripped, "model", wide, config=config, cache=FeatureCache(stripped)
    )

    assert with_ep.fallback_gws, "the forecast fallback should have been used"
    assert with_ep.total_points != without.total_points


def test_the_frame_and_summary_describe_the_same_season(season, rules, config):
    result = backtest(season, "naive", rules, config=config)
    frame = result.frame()
    summary = result.summary()

    assert len(frame) == result.gameweeks
    assert int(frame["points"].sum()) == summary["total_points"]
    assert summary["decided_gameweeks"] == len(result.decided_gws)
    assert frame["cumulative"].is_monotonic_increasing or (frame["points"] < 0).any()


def test_an_unknown_strategy_is_rejected(season, rules, config):
    with pytest.raises(ValueError, match="unknown strategy"):
        backtest(season, "vibes", rules, config=config)
