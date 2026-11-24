"""Building the current season into the archive's own files.

The payloads are recorded, so none of this needs a request. They are built from
the hand-built season, which means an ingested gameweek can be checked against
figures written out by hand — including its double gameweek and its blank.

The properties that matter most are the two that protect what a model trains
on: a gameweek is written once and never rewritten, and the gameweek being
predicted is never in the history being fitted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_optimizer.config import Config
from fpl_optimizer.ingest import (
    FIELD_SOURCES,
    MERGED_COLUMNS,
    IngestFailed,
    append_gameweek,
    ingest_gameweek,
    ingested_gameweeks,
    load_ingested_season,
    merged_path,
    season_with_upcoming,
    upcoming_rows,
    write_provenance,
    write_reference,
)
from fpl_optimizer.schema import schema_signature, validate_player_gws

from .fixtures import ingest_payloads, load_archive_fixture
from .fixtures.handbuilt import RESULTS


@pytest.fixture
def payloads():
    return ingest_payloads()


@pytest.fixture
def config(tmp_path):
    """Ingest under a temporary directory, never the repository's own data."""
    return Config(data_dir=tmp_path)


def _ingest(payloads, gw, prices=None):
    return ingest_gameweek(
        gw,
        bootstrap=payloads["bootstrap"],
        fixtures=payloads["fixtures"],
        event_live=payloads["live"][gw],
        prices=prices,
    )


# --- one gameweek -----------------------------------------------------------


def test_a_gameweek_comes_out_in_the_archives_shape(payloads):
    frame, _ = _ingest(payloads, 1)

    assert list(frame.columns) == list(MERGED_COLUMNS)
    assert set(frame["GW"]) == {1}
    assert len(frame) == 18  # one row per player, one fixture each


def test_the_scoring_stats_come_through_exactly(payloads):
    """Checked against the hand-built season's own table."""
    frame, _ = _ingest(payloads, 1)
    indexed = frame.set_index("element")

    for pid in (1, 2, 3, 4):
        minutes, points, _ = RESULTS[pid][101] if 101 in RESULTS[pid] else (0, 0, 0)
        if 101 not in RESULTS[pid]:
            continue
        assert int(indexed.loc[pid, "total_points"]) == points
        assert int(indexed.loc[pid, "minutes"]) == minutes


def test_a_double_gameweek_gives_a_player_two_rows(payloads):
    """`explain` lists both fixtures, which is where the second row comes from."""
    frame, _ = _ingest(payloads, 2)
    counts = frame.groupby("element").size()

    doubled = counts[counts == 2]
    assert len(doubled) == 6  # the two clubs that play twice, three players each
    assert set(frame[frame["element"].isin(doubled.index)]["fixture"]) == {102, 103}


def test_a_blank_gameweek_produces_no_rows_for_those_clubs(payloads):
    """The archive represents a blank by absence, not by a zero."""
    frame, _ = _ingest(payloads, 2)
    # Every row belongs to a club with a fixture in gameweek 2.
    assert len(frame) == 24  # 6 players twice, 12 once
    assert set(frame["fixture"]) == {102, 103, 112, 122}


def test_fixture_details_are_joined_from_the_fixture_list(payloads):
    frame, _ = _ingest(payloads, 1)
    row = frame[frame["element"] == 1].iloc[0]

    assert row["fixture"] == 101
    assert row["kickoff_time"] == "2024-09-01T14:00:00Z"
    assert bool(row["was_home"]) is True  # Reds are home in fixture 101
    assert int(row["team_h_score"]) == 1
    assert int(row["opponent_team"]) == 2  # Blues


def test_a_gameweek_with_no_fixtures_is_a_failure_not_an_empty_file(payloads):
    with pytest.raises(IngestFailed, match="no fixtures listed"):
        ingest_gameweek(
            99,
            bootstrap=payloads["bootstrap"],
            fixtures=payloads["fixtures"],
            event_live=payloads["live"][1],
        )


# --- prices: the one field the cheap endpoints lack -------------------------


def test_a_supplied_price_series_is_used_exactly(payloads):
    frame, notes = _ingest(payloads, 1, prices={1: 123, 2: 456})
    indexed = frame.set_index("element")

    assert int(indexed.loc[1, "value"]) == 123
    assert int(indexed.loc[2, "value"]) == 456


def test_a_missing_price_falls_back_to_today_and_says_so(payloads):
    frame, notes = _ingest(payloads, 1, prices=None)

    # Today's price, from bootstrap, for every row.
    assert int(frame.set_index("element").loc[1, "value"]) == 45
    assert any("today's price" in note for note in notes)


def test_a_partially_supplied_series_flags_only_the_gaps(payloads):
    frame, notes = _ingest(payloads, 1, prices={1: 999})
    indexed = frame.set_index("element")

    assert int(indexed.loc[1, "value"]) == 999
    assert int(indexed.loc[2, "value"]) == 85  # today's
    assert any("approximated" in note for note in notes)


def test_every_field_has_a_documented_source():
    """The provenance table is the contract; it should not fall behind."""
    for column in MERGED_COLUMNS:
        if column in ("element", "round", "GW"):
            continue  # identifiers, not data
        assert column in FIELD_SOURCES, f"{column} has no documented source"


def test_the_provenance_file_is_written_next_to_the_data(config):
    path = write_provenance("2026-27", config)

    text = path.read_text()
    assert path.name == "PROVENANCE.md"
    assert "| `value` |" in text
    assert "never" in text and "rewritten" in text
    # The absence of expected points is the thing most likely to be forgotten.
    assert "xP" in text


# --- idempotence ------------------------------------------------------------


def test_a_gameweek_is_written_once(payloads, config):
    frame, _ = _ingest(payloads, 1)
    append_gameweek(frame, "2026-27", 1, config)

    assert ingested_gameweeks("2026-27", config) == {1}
    with pytest.raises(IngestFailed, match="will not be rewritten"):
        append_gameweek(frame, "2026-27", 1, config)


def test_re_running_never_duplicates_rows(payloads, config):
    """What makes the scheduled job safe to run six times a day."""
    frame, _ = _ingest(payloads, 1)
    append_gameweek(frame, "2026-27", 1, config)
    for _ in range(4):
        with pytest.raises(IngestFailed):
            append_gameweek(frame, "2026-27", 1, config)

    on_disk = pd.read_csv(merged_path("2026-27", config))
    assert len(on_disk) == len(frame)
    assert on_disk["element"].duplicated().sum() == 0


def test_a_later_revision_cannot_overwrite_a_finished_gameweek(payloads, config):
    """FPL revises bonus points for a day or two after the whistle.

    A gameweek is ingested once its data is marked checked, and then it is
    history: a rewrite would silently change what a model was trained on.
    """
    frame, _ = _ingest(payloads, 1)
    append_gameweek(frame, "2026-27", 1, config)

    revised = frame.copy()
    revised["total_points"] = 999
    with pytest.raises(IngestFailed):
        append_gameweek(revised, "2026-27", 1, config)

    on_disk = pd.read_csv(merged_path("2026-27", config))
    assert 999 not in set(on_disk["total_points"])


def test_gameweeks_accumulate_in_order(payloads, config):
    for gw in (2, 1):  # deliberately out of order
        frame, _ = _ingest(payloads, gw)
        append_gameweek(frame, "2026-27", gw, config)

    on_disk = pd.read_csv(merged_path("2026-27", config))
    assert on_disk["GW"].is_monotonic_increasing
    assert ingested_gameweeks("2026-27", config) == {1, 2}


# --- reading it back --------------------------------------------------------


@pytest.fixture
def ingested(payloads, config):
    for gw in (1, 2):
        frame, _ = _ingest(payloads, gw, prices={p: 50 for p in range(1, 19)})
        append_gameweek(frame, "2026-27", gw, config)
    write_reference("2026-27", payloads["bootstrap"], payloads["fixtures"], config)
    write_provenance("2026-27", config)
    return config


def test_an_ingested_season_loads_through_the_archives_own_normalizer(ingested):
    """No second loader and no second schema, so the two cannot drift apart."""
    data = load_ingested_season("2026-27", ingested)

    validate_player_gws(data.player_gws)
    assert data.gameweeks == [1, 2]
    assert schema_signature(data.player_gws) == schema_signature(
        load_archive_fixture().player_gws
    )


def test_the_load_says_it_was_ingested_not_downloaded(ingested):
    data = load_ingested_season("2026-27", ingested)
    assert any("ingested from the live API" in note for note in data.notes)
    assert any("PROVENANCE.md" in note for note in data.notes)


def test_expected_points_are_absent_and_the_notes_say_why(ingested):
    data = load_ingested_season("2026-27", ingested)

    assert data.player_gws["ep"].isna().all()
    assert any("does not retain" in note for note in data.notes)


def test_the_average_manager_score_is_captured(ingested):
    """The one benchmark an ingested season has that a downloaded one lacks."""
    data = load_ingested_season("2026-27", ingested)

    assert data.average_score(1) == 50.0
    assert data.average_score(2) == 70.0
    assert any("average-manager" in note for note in data.notes)


def test_loading_a_season_that_was_never_ingested_is_an_error(config):
    with pytest.raises(IngestFailed, match="no ingested data"):
        load_ingested_season("1999-00", config)


# --- the upcoming gameweek, and walk-forward discipline ---------------------


def test_upcoming_rows_carry_pre_deadline_facts_only(payloads):
    frame = upcoming_rows(payloads["bootstrap"], payloads["fixtures"], 3, "2026-27")

    assert set(frame["gw"]) == {3}
    # Blacks and Golds have no gameweek-3 fixture, so their players have no row.
    assert len(frame) == 12
    for column in ("points", "minutes", "bps", "team_h_score"):
        assert frame[column].isna().all(), column
    for column in ("price", "ep", "opponent", "was_home", "fixture_id"):
        assert frame[column].notna().any(), column


def test_the_season_to_predict_with_holds_history_plus_the_gameweek(ingested, payloads):
    data = season_with_upcoming(
        "2026-27", payloads["bootstrap"], payloads["fixtures"], 3, ingested
    )

    assert data.gameweeks == [1, 2, 3]
    assert sorted(data.before(3)["gw"].unique()) == [1, 2]
    assert data.at(3)["points"].isna().all()
    validate_player_gws(data.player_gws)


def test_predicting_a_gameweek_already_ingested_is_refused(ingested, payloads):
    """Fitting on it would be training on the answer."""
    with pytest.raises(IngestFailed, match="training on the answer"):
        season_with_upcoming(
            "2026-27", payloads["bootstrap"], payloads["fixtures"], 2, ingested
        )


def test_the_history_only_holds_gameweeks_already_final(ingested, payloads):
    """The walk-forward property, stated against the data itself.

    Only gameweeks FPL marked checked are ingested, so every row a model fits
    on was final before the deadline being predicted.
    """
    checked = {
        int(e["id"])
        for e in payloads["bootstrap"]["events"]
        if e.get("data_checked")
    }
    data = season_with_upcoming(
        "2026-27", payloads["bootstrap"], payloads["fixtures"], 3, ingested
    )

    assert set(data.before(3)["gw"].unique()) <= checked
    assert 3 not in checked


def test_a_model_fitted_on_it_sees_no_future_results(ingested, payloads):
    """The leakage property, carried into the live path."""
    from fpl_optimizer.features import build_training_set

    data = season_with_upcoming(
        "2026-27", payloads["bootstrap"], payloads["fixtures"], 3, ingested
    )
    X, y = build_training_set(data, 3)

    assert len(y) > 0
    assert int(y["gw"].max()) < 3
