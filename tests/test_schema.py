"""The schema contract, and the parity of the two sources that produce it."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_optimizer.live import load_live_season
from fpl_optimizer.schema import (
    OUTCOME_COLUMNS,
    PLAYER_GW_COLUMNS,
    PLAYER_GW_DTYPES,
    PRE_DEADLINE_COLUMNS,
    coerce_player_gws,
    empty_player_gws,
    schema_signature,
    validate_player_gws,
)

from .fixtures import live_payloads, load_archive_fixture, make_season


def test_pre_deadline_and_outcome_columns_partition_the_schema():
    assert set(PRE_DEADLINE_COLUMNS) & set(OUTCOME_COLUMNS) == set()
    assert set(PRE_DEADLINE_COLUMNS) | set(OUTCOME_COLUMNS) == set(PLAYER_GW_COLUMNS)
    assert set(PLAYER_GW_DTYPES) == set(PLAYER_GW_COLUMNS)


def test_coerce_fills_missing_columns_in_canonical_order():
    df = pd.DataFrame({"gw": [1], "player_id": [7], "position": ["MID"]})
    out = coerce_player_gws(df)
    assert list(out.columns) == list(PLAYER_GW_COLUMNS)
    validate_player_gws(out)


def test_coerce_applies_the_declared_dtypes():
    out = coerce_player_gws(
        pd.DataFrame(
            {
                "gw": ["3"],
                "player_id": ["12"],
                "position": ["MID"],
                "minutes": ["90"],
                "points": ["6"],
                "price": ["55"],
            }
        )
    )
    for col, dtype in PLAYER_GW_DTYPES.items():
        assert str(out[col].dtype) == dtype, col
    assert int(out.loc[0, "minutes"]) == 90


def test_empty_frame_is_already_canonical():
    df = empty_player_gws()
    validate_player_gws(df)
    assert len(df) == 0


def test_outcome_columns_can_hold_missing_rather_than_zero():
    """A gameweek not yet played has no result, and the schema must say so.

    Were these columns plain ``int16``, the only way to represent "no result"
    would be 0 — indistinguishable from a player who turned out and scored
    nothing.
    """
    out = coerce_player_gws(
        pd.DataFrame({"gw": [1], "player_id": [1], "position": ["FWD"], "minutes": [None]})
    )
    assert pd.isna(out.loc[0, "minutes"])
    for col in OUTCOME_COLUMNS:
        assert PLAYER_GW_DTYPES[col][0].isupper(), f"{col} cannot hold a missing value"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda df: df.drop(columns=["minutes"]), "missing columns"),
        (lambda df: df.assign(surprise=1), "unexpected columns"),
        (lambda df: df[list(reversed(PLAYER_GW_COLUMNS))], "canonical order"),
    ],
)
def test_validate_rejects_non_canonical_frames(mutate, message):
    df = mutate(empty_player_gws())
    with pytest.raises(ValueError, match=message):
        validate_player_gws(df)


def test_validate_rejects_unknown_positions():
    df = coerce_player_gws(
        pd.DataFrame({"gw": [1], "player_id": [1], "position": ["Manager"]})
    )
    with pytest.raises(ValueError, match="non-canonical positions"):
        validate_player_gws(df)


def test_pre_deadline_at_physically_drops_outcome_columns():
    data = make_season()
    frame = data.pre_deadline_at(3)
    assert list(frame.columns) == list(PRE_DEADLINE_COLUMNS)
    assert not set(frame.columns) & set(OUTCOME_COLUMNS)
    assert set(frame["gw"]) == {3}


def test_before_excludes_the_gameweek_being_decided():
    data = make_season()
    hist = data.before(4)
    assert hist["gw"].max() == 3
    assert data.at(4)["gw"].unique().tolist() == [4]


def test_average_score_reads_the_events_table():
    data = make_season()
    assert data.average_score(1) == 41.0
    assert data.average_score(99) is None


def test_live_and_archive_normalize_to_the_identical_schema():
    """The claim the whole leakage argument rests on: one schema, one gate."""
    archive = load_archive_fixture()
    bootstrap, fixtures, summaries = live_payloads()
    live = load_live_season(bootstrap=bootstrap, fixtures=fixtures, summaries=summaries)

    assert schema_signature(live.player_gws) == schema_signature(archive.player_gws)
    assert live.source == "live"
    assert archive.source == "archive"


def test_live_leaves_unplayed_gameweeks_empty_rather_than_zero():
    bootstrap, fixtures, summaries = live_payloads()
    live = load_live_season(bootstrap=bootstrap, fixtures=fixtures, summaries=summaries)

    future = live.at(2)
    assert len(future) > 0
    for col in OUTCOME_COLUMNS:
        assert future[col].isna().all(), col
    # The gameweek already played is populated.
    assert live.at(1)["minutes"].notna().all()


def test_live_drops_players_with_no_squad_slot():
    bootstrap, fixtures, summaries = live_payloads()
    live = load_live_season(bootstrap=bootstrap, fixtures=fixtures, summaries=summaries)

    assert set(live.player_gws["position"]) <= {"GKP", "DEF", "MID", "FWD"}
    # The assistant-manager element is element 5 in the fixture payload.
    assert 5 not in set(live.player_gws["player_id"])
    assert any("no squad slot" in note for note in live.notes)


def test_live_publishes_expected_points_for_the_next_gameweek_only():
    bootstrap, fixtures, summaries = live_payloads()
    live = load_live_season(bootstrap=bootstrap, fixtures=fixtures, summaries=summaries)

    assert live.at(2)["ep"].notna().all()
    # FPL does not retain past expected points, so played gameweeks have none.
    assert live.at(1)["ep"].isna().all()
    assert any("ep_next" in note for note in live.notes)


def test_live_reads_availability_from_player_status():
    bootstrap, fixtures, summaries = live_payloads()
    live = load_live_season(bootstrap=bootstrap, fixtures=fixtures, summaries=summaries)
    nxt = live.at(2).set_index("player_id")["availability"]

    assert nxt[1] == pytest.approx(1.0)  # status "a", 100% chance
    assert nxt[2] == pytest.approx(0.5)  # doubtful, 50% chance
    assert nxt[4] == pytest.approx(0.0)  # injured, 0% chance
