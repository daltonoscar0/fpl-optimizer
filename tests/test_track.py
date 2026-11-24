"""The live record, whose value is entirely in how it is kept.

A backtest can be re-run with a better answer; a live record cannot, which is
what makes it worth having — and what makes the one rule load bearing:

    A gameweek is scored only from a recommendation recorded before its
    deadline.

Most of this module tests the ways that rule can be got around, and that none
of them work: a recommendation written after the deadline, a recommendation
overwritten after the result is known, a gameweek with nothing on record at
all. The honest outcome in each case is no score, not a score.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fpl_optimizer.config import Config
from fpl_optimizer.rules import archive_season_rules
from fpl_optimizer.track import (
    NotRecordedInAdvance,
    plan_track,
    Recommendation,
    append_tracked,
    deadline_for,
    is_finished,
    load_recommendation,
    markdown_table,
    record,
    recommend,
    recommendation_path,
    score_recorded,
    tracked_frame,
    tracking_path,
)

from .fixtures.handbuilt import hand_built_season

#: The hand-built season's gameweek deadlines are 2024-09-0{gw}T11:30:00Z.
BEFORE_GW2 = "2024-09-01T09:00:00+00:00"
AFTER_GW2 = "2024-09-02T18:00:00+00:00"


@pytest.fixture
def config(tmp_path):
    """Results under a temporary directory, so tests never touch results/."""
    return Config(excluded_clubs=(), horizon=2, results_dir=tmp_path)


@pytest.fixture(scope="module")
def rules():
    return archive_season_rules("2024-25")


@pytest.fixture
def season():
    """The hand-built season, with an average-manager score per gameweek."""
    return hand_built_season()


def _recommendation(season, rules, config, gw=2, now=BEFORE_GW2, **kwargs):
    return recommend(season, gw, "ep", rules, config=config, now=now, **kwargs)


# --- recording --------------------------------------------------------------


def test_a_recommendation_records_what_was_picked_and_when(season, rules, config):
    rec = _recommendation(season, rules, config)

    assert rec.gw == 2
    assert rec.strategy == "ep"
    assert rec.recorded_at == BEFORE_GW2
    assert rec.deadline == deadline_for(season, 2)
    assert len(rec.squad) == rules.squad_size
    assert len(rec.starting) == rules.squad_play
    assert rec.captain in rec.starting
    assert rec.vice in rec.starting and rec.vice != rec.captain
    assert set(rec.bench) == set(rec.squad) - set(rec.starting)
    # Readable a year later without a lookup table.
    assert all(str(p) in rec.names for p in rec.squad)
    assert rec.in_advance


def test_recording_writes_a_file_that_round_trips(season, rules, config):
    rec = _recommendation(season, rules, config)
    path = record(rec, config)

    assert path.exists()
    assert path == recommendation_path("2024-25", 2, "ep", config)
    loaded = load_recommendation("2024-25", 2, "ep", config)
    assert loaded == rec
    assert json.loads(path.read_text())["captain"] == rec.captain


def test_a_recorded_recommendation_is_never_overwritten(season, rules, config):
    """Overwriting is how a live record quietly becomes a backtest."""
    record(_recommendation(season, rules, config), config)

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        record(_recommendation(season, rules, config, now=AFTER_GW2), config)

    # The original is intact, including its timestamp.
    assert load_recommendation("2024-25", 2, "ep", config).recorded_at == BEFORE_GW2


def test_nothing_is_written_when_the_strategy_has_no_view(season, rules, config):
    data = hand_built_season()
    df = data.player_gws.copy()
    df["ep"] = float("nan")
    data.player_gws = df

    with pytest.raises(NotRecordedInAdvance, match="nothing to"):
        recommend(data, 2, "ep", rules, config=config, now=BEFORE_GW2)
    assert not recommendation_path("2024-25", 2, "ep", config).exists()


def test_a_recommendation_carries_from_the_previous_squad(season, rules, config):
    """A new gameweek is a transfer decision, not a fresh fifteen."""
    first = _recommendation(season, rules, config, gw=2)
    second = recommend(
        season,
        3,
        "ep",
        rules,
        config=config,
        now="2024-09-02T09:00:00+00:00",
        current=first.as_squad(),
        free_transfers=1,
    )

    changed = set(second.squad) - set(first.squad)
    assert changed == set(second.transfers_in)
    assert len(second.transfers_in) == len(second.transfers_out)


# --- the rule ---------------------------------------------------------------


def test_a_gameweek_is_scored_from_a_recommendation_made_in_advance(
    season, rules, config
):
    rec = _recommendation(season, rules, config)
    tracked = score_recorded(season, rec, rules)

    assert tracked.gw == 2
    assert tracked.points > 0
    # The hand-built season's gameweek 2 average is 70.
    assert tracked.average_entry_score == 70.0
    assert tracked.delta == tracked.points - 70.0
    assert tracked.recorded_at == BEFORE_GW2


def test_a_recommendation_made_after_the_deadline_cannot_be_scored(
    season, rules, config
):
    """The whole mechanism, in one test.

    Running the optimizer once the team sheets are known and calling the result
    a prediction is the failure this guards against — and it is the easy
    mistake, because the code path is otherwise identical.
    """
    late = _recommendation(season, rules, config, now=AFTER_GW2)
    assert not late.in_advance

    with pytest.raises(NotRecordedInAdvance, match="not before its deadline"):
        score_recorded(season, late, rules)


def test_a_recommendation_with_no_known_deadline_cannot_be_scored(
    season, rules, config
):
    """An unknown deadline cannot be shown to be in the future."""
    rec = _recommendation(season, rules, config)
    undated = Recommendation(**{**rec.__dict__, "deadline": None})

    assert not undated.in_advance
    with pytest.raises(NotRecordedInAdvance):
        score_recorded(season, undated, rules)


def test_an_unfinished_gameweek_is_not_scored(season, rules, config):
    data = hand_built_season()
    events = data.events.copy()
    events.loc[events["gw"] == 2, "finished"] = False
    data.events = events

    rec = _recommendation(data, rules, config)
    assert not is_finished(data, 2)
    with pytest.raises(NotRecordedInAdvance, match="not finished"):
        score_recorded(data, rec, rules)


def test_a_gameweek_with_nothing_on_record_has_no_score(config):
    """The only honest answer when nobody wrote an answer down."""
    assert load_recommendation("2024-25", 2, "ep", config) is None
    assert load_recommendation("2024-25", 99, "model", config) is None


# --- scoring matches the backtest's scorer ----------------------------------


def test_the_live_score_is_the_same_scorer_the_backtest_uses(season, rules, config):
    from fpl_optimizer.score import score_gameweek

    rec = _recommendation(season, rules, config)
    tracked = score_recorded(season, rec, rules)
    direct = score_gameweek(rec.as_squad(), season, 2, rules)

    assert tracked.points == direct.points
    assert tracked.starters_points == direct.starters_points
    assert tracked.captain_bonus == direct.captain_bonus
    assert tracked.autosubs == len(direct.autosubs)


def test_a_double_gameweek_counts_both_legs_live_too(season, rules, config):
    """Gameweek 2 is the double; the pinned totals are 6, 11, 8 and 12."""
    rec = _recommendation(season, rules, config)
    tracked = score_recorded(season, rec, rules)

    realized = season.at(2).groupby("player_id")["points"].sum()
    expected_xi = sum(int(realized.get(p, 0)) for p in rec.starting)
    # Equal unless an auto-sub changed the XI, in which case it can only rise.
    assert tracked.starters_points >= expected_xi or tracked.autosubs > 0


# --- the written record -----------------------------------------------------


def test_scored_gameweeks_accumulate_one_row_each(season, rules, config):
    for gw, when in ((2, BEFORE_GW2), (3, "2024-09-02T09:00:00+00:00")):
        rec = _recommendation(season, rules, config, gw=gw, now=when)
        append_tracked(score_recorded(season, rec, rules), config)

    frame = tracked_frame("2024-25", config)
    assert list(frame["gw"]) == [2, 3]
    assert tracking_path("2024-25", config).exists()
    assert frame["average_entry_score"].tolist() == [70.0, 48.0]


def test_rescoring_a_gameweek_replaces_its_row_rather_than_duplicating(
    season, rules, config
):
    rec = _recommendation(season, rules, config)
    append_tracked(score_recorded(season, rec, rules), config)
    append_tracked(score_recorded(season, rec, rules), config)

    frame = tracked_frame("2024-25", config)
    assert len(frame) == 1


def test_an_empty_record_says_so_rather_than_showing_a_blank_table(config):
    table = markdown_table(tracked_frame("2024-25", config))

    assert "no gameweek called in advance yet" in table
    assert table.startswith("| GW |")


def test_the_table_shows_each_gameweek_and_a_running_total(season, rules, config):
    for gw, when in ((2, BEFORE_GW2), (3, "2024-09-02T09:00:00+00:00")):
        rec = _recommendation(season, rules, config, gw=gw, now=when)
        append_tracked(score_recorded(season, rec, rules), config)

    frame = tracked_frame("2024-25", config)
    table = markdown_table(frame)
    lines = [line for line in table.splitlines() if line.startswith("|")]

    assert len(lines) == 5  # header, rule, two gameweeks, total
    assert "**2 GWs**" in lines[-1]
    total = int(frame["points"].sum())
    assert f"**{total}**" in lines[-1]


def test_an_unpublished_average_is_shown_as_absent_not_as_zero(
    season, rules, config
):
    data = hand_built_season()
    events = data.events.copy()
    events["average_entry_score"] = float("nan")
    data.events = events

    rec = _recommendation(data, rules, config)
    tracked = score_recorded(data, rec, rules)
    assert tracked.average_entry_score is None
    assert tracked.delta is None

    append_tracked(tracked, config)
    table = markdown_table(tracked_frame("2024-25", config))
    assert "| — | — |" in table


# --- the plan: what a scheduled run decides to do ---------------------------


def _events(rows):
    """``(gw, deadline, finished)`` triples into the events frame's shape."""
    return pd.DataFrame(
        [
            {"gw": gw, "deadline_time": deadline, "finished": finished}
            for gw, deadline, finished in rows
        ]
    )


EVENTS = _events(
    [
        (11, "2026-11-21T11:00:00Z", True),
        (12, "2026-11-27T18:30:00Z", False),
        (13, "2026-12-01T18:30:00Z", False),
    ]
)
BEFORE_12 = "2026-11-24T17:00:00Z"
AFTER_12 = "2026-11-28T09:00:00Z"


def _plan(**kwargs):
    base = dict(
        season="2026-27",
        events=EVENTS,
        recorded_gws=set(),
        scored_gws=set(),
        next_gw=12,
        now=BEFORE_12,
    )
    return plan_track(**{**base, **kwargs})


def test_a_run_before_the_deadline_records_the_next_gameweek():
    plan = _plan()

    assert plan.record_gw == 12
    assert plan.next_deadline == "2026-11-27T18:30:00Z"
    assert plan.needs_full_season
    assert not plan.nothing_to_do


def test_a_run_once_the_gameweek_is_on_record_does_nothing():
    """The case a four-hourly schedule hits almost every time."""
    plan = _plan(recorded_gws={12})

    assert plan.record_gw is None
    assert plan.nothing_to_do
    assert "already on record" in plan.reason
    assert not plan.needs_full_season


def test_a_run_after_the_deadline_records_nothing():
    plan = _plan(now=AFTER_12, next_gw=12)

    assert plan.record_gw is None
    assert "deadline has passed" in plan.reason


def test_a_run_exactly_on_the_deadline_records_nothing():
    """The boundary belongs to the deadline, not to the optimizer."""
    plan = _plan(now="2026-11-27T18:30:00Z")

    assert plan.record_gw is None
    assert "deadline has passed" in plan.reason


def test_a_gameweek_with_no_published_deadline_is_not_recorded():
    events = _events([(12, None, False)])
    plan = _plan(events=events)

    assert plan.record_gw is None
    assert "no published deadline" in plan.reason


def test_nothing_is_recorded_when_the_season_has_no_next_gameweek():
    plan = _plan(next_gw=None)

    assert plan.record_gw is None
    assert plan.reason == "no upcoming gameweek"


def test_a_finished_gameweek_on_record_is_scored():
    events = _events(
        [(11, "2026-11-21T11:00:00Z", True), (12, "2026-11-27T18:30:00Z", False)]
    )
    plan = _plan(events=events, recorded_gws={11, 12}, next_gw=12)

    assert plan.score_gws == [11]
    # Scoring reads fifteen players' histories, not the whole game.
    assert not plan.needs_full_season


def test_a_gameweek_already_scored_is_not_scored_again():
    events = _events([(11, "2026-11-21T11:00:00Z", True)])
    plan = _plan(events=events, recorded_gws={11}, scored_gws={11}, next_gw=None)

    assert plan.score_gws == []
    assert plan.nothing_to_do


def test_an_unfinished_gameweek_is_not_scored_even_once_recorded():
    plan = _plan(recorded_gws={12})
    assert plan.score_gws == []


def test_a_deadline_passing_with_no_record_is_reported_as_missed():
    """The alarm condition: a permanent hole in the record."""
    events = _events(
        [
            (12, "2026-11-27T18:30:00Z", True),
            (13, "2026-12-01T18:30:00Z", False),
        ]
    )
    plan = plan_track(
        season="2026-27",
        events=events,
        recorded_gws={12},
        scored_gws={12},
        next_gw=14,
        now="2026-12-02T09:00:00Z",  # GW13's deadline has been and gone
    )

    assert plan.missed_gws == [13]


def test_gameweeks_before_tracking_began_are_not_missed():
    """Eleven gameweeks were played before any of this existed.

    Calling those "missed" would mean an alarm that can never be cleared.
    """
    events = _events(
        [
            (1, "2026-08-14T17:30:00Z", True),
            (11, "2026-11-21T11:00:00Z", True),
            (12, "2026-11-27T18:30:00Z", True),
        ]
    )
    plan = plan_track(
        season="2026-27",
        events=events,
        recorded_gws={12},
        scored_gws={12},
        next_gw=13,
        now="2026-11-28T09:00:00Z",
    )

    assert plan.missed_gws == []


def test_nothing_is_missed_before_tracking_has_begun():
    plan = _plan(now=AFTER_12, recorded_gws=set())
    assert plan.missed_gws == []


def test_the_plan_reads_neither_disk_nor_network():
    """It is handed deadlines and records, and returns a decision.

    That is what makes every branch above testable, and what lets a scheduled
    run decide in one request instead of six hundred.
    """
    import inspect

    from fpl_optimizer import track

    source = inspect.getsource(track.plan_track)
    for forbidden in ("open(", "requests", "load_recommendation", "Path("):
        assert forbidden not in source


# --- recording is idempotent and never late ---------------------------------


def test_a_late_recommendation_is_never_written(season, rules, config):
    """Refused at the write boundary, not just at the scoring one.

    `score_recorded` already refuses to score a late record; this stops one
    reaching the disk at all, where a later reader could not tell it apart from
    a timely one without checking the timestamps itself.
    """
    rec = _recommendation(season, rules, config, now=AFTER_GW2)
    assert not rec.in_advance

    with pytest.raises(NotRecordedInAdvance, match="will not be written"):
        record(rec, config)
    assert not recommendation_path("2024-25", 2, "ep", config).exists()


def test_recording_twice_writes_one_file(season, rules, config):
    """What makes a four-hourly schedule safe."""
    first = record(_recommendation(season, rules, config), config)

    for _ in range(5):
        with pytest.raises(FileExistsError):
            record(_recommendation(season, rules, config), config)

    assert len(list(first.parent.glob("*-gw02-ep.json"))) == 1
