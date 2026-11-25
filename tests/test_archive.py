"""Normalization of the real archive, quirks and all.

The synthetic season in :mod:`tests.fixtures` is clean by construction, which is
what makes it good for testing features. This module tests the opposite thing:
that normalization survives real data, where position labels differ from FPL's
own, expected points go missing for whole gameweeks, and columns that a scoring
rule introduced only exist in recent seasons.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_optimizer.archive import format_gw_runs, normalize_archive_season
from fpl_optimizer.projections.baselines import (
    EP_HINDSIGHT_THRESHOLD,
    ep_looks_like_hindsight,
)
from fpl_optimizer.schema import FIXTURE_COLUMNS, PLAYER_GW_COLUMNS, validate_player_gws

from .fixtures import ARCHIVE_GWS, archive_paths, load_archive_fixture


@pytest.fixture(scope="module")
def season():
    return load_archive_fixture()


@pytest.fixture(scope="module")
def raw():
    paths = archive_paths()
    return (
        pd.read_csv(paths["gws/merged_gw.csv"], low_memory=False),
        pd.read_csv(paths["teams.csv"]),
        pd.read_csv(paths["fixtures.csv"], low_memory=False),
    )


def test_loads_the_committed_slice_without_network(season):
    assert season.source == "archive"
    assert season.gameweeks == list(ARCHIVE_GWS)
    assert len(season.player_gws) > 0
    validate_player_gws(season.player_gws)
    assert list(season.player_gws.columns) == list(PLAYER_GW_COLUMNS)


def test_positions_are_mapped_to_the_canonical_four(season, raw):
    merged, _, _ = raw
    # The archive writes "GK"; the schema's canonical label is "GKP".
    assert "GK" in set(merged["position"])
    assert set(season.player_gws["position"]) <= {"GKP", "DEF", "MID", "FWD"}
    assert (season.player_gws["position"] == "GKP").any()


def test_rows_are_one_per_player_fixture(season):
    counts = season.player_gws.groupby(["gw", "player_id", "fixture_id"]).size()
    assert counts.max() == 1


def test_price_comes_from_the_gameweek_not_the_season_end(season, raw):
    """``value`` is the price as it stood at that gameweek's deadline."""
    merged, _, _ = raw
    row = merged.iloc[0]
    got = season.player_gws[
        (season.player_gws["player_id"] == int(row["element"]))
        & (season.player_gws["gw"] == int(row["GW"]))
    ]
    assert int(got.iloc[0]["price"]) == int(row["value"])


def test_all_zero_expected_points_gameweeks_become_missing(season):
    """A gameweek where every ``xP`` is 0.0 is a failed snapshot, not a forecast.

    Gameweek 7 of 2025-26 is such a gameweek in the upstream archive.
    """
    per_gw = season.player_gws.groupby("gw")["ep"].apply(lambda s: s.isna().all())
    assert per_gw[7]
    assert not per_gw[6]
    assert any("all-zero" in note and "[7]" in note for note in season.notes)


def test_a_genuinely_present_expected_points_column_is_kept(season):
    ep = season.player_gws[season.player_gws["gw"] == 1]["ep"]
    assert ep.notna().all()
    assert float(ep.max()) > 0.0


def test_missing_expected_points_column_is_reported_not_invented(raw):
    merged, teams, fixtures = raw
    season = normalize_archive_season(
        "2018-19", merged.drop(columns=["xP"]), teams, fixtures
    )
    assert season.player_gws["ep"].isna().all()
    assert any("no `xP` column" in note for note in season.notes)


def test_missing_defensive_contribution_column_is_reported(raw):
    merged, teams, fixtures = raw
    season = normalize_archive_season(
        "2018-19", merged.drop(columns=["defensive_contribution"]), teams, fixtures
    )
    assert any("defensive_contribution" in note for note in season.notes)


def test_unknown_club_names_fail_loudly(raw):
    merged, teams, fixtures = raw
    broken = merged.copy()
    broken.loc[broken.index[0], "team"] = "Wanderers"
    with pytest.raises(ValueError, match="not in teams.csv"):
        normalize_archive_season("2025-26", broken, teams, fixtures)


def test_absent_gameweeks_are_reported(season):
    """The slice stops at gameweek 8, and the notes say which are missing."""
    assert any("gameweeks absent" in note for note in season.notes)


def test_absent_gameweeks_are_reported_as_runs_not_as_a_list_of_numbers(season):
    """Mid-season every gameweek to come is absent, and the list drowns the note.

    The archive slice ends at gameweek 8, so 9–38 are missing and should read
    as one run. A hole in the middle of what has been played is the case worth
    seeing, and it only stands out if the contiguous tail is collapsed.
    """
    note = next(n for n in season.notes if "gameweeks absent" in n)
    assert "9–38" in note
    assert "[" not in note and "," not in note.split(":", 1)[1]


@pytest.mark.parametrize(
    "gws, expected",
    [
        ([], ""),
        ([5], "5"),
        ([1, 2, 3], "1–3"),
        ([12, 13, 14, 20], "12–14, 20"),
        ([3, 9, 10, 38], "3, 9–10, 38"),
        ([38, 3, 10, 9], "3, 9–10, 38"),
        ([7, 7, 8], "7–8"),
    ],
)
def test_format_gw_runs(gws, expected):
    assert format_gw_runs(gws) == expected


def test_average_manager_score_is_absent_and_said_to_be(season):
    assert season.events["average_entry_score"].isna().all()
    assert all(season.average_score(g) is None for g in season.gameweeks)
    assert any("average-manager" in note for note in season.notes)


def test_events_order_gameweeks_by_first_kickoff(season):
    deadlines = season.events["deadline_time"].tolist()
    assert deadlines == sorted(deadlines)
    assert season.events["gw"].tolist() == list(ARCHIVE_GWS)


def test_fixtures_frame_is_canonical(season):
    assert list(season.fixtures.columns) == list(FIXTURE_COLUMNS)
    assert len(season.fixtures) > 0
    known = set(season.clubs["club"])
    assert set(season.fixtures["home_club"].dropna()) <= known
    assert set(season.fixtures["away_club"].dropna()) <= known


# --- is the published forecast actually a forecast? -------------------------


def test_the_expected_points_column_is_flagged_as_knowing_who_played(season):
    """``xP`` rates non-appearances zero, which a forecast cannot do.

    FPL quotes an ``ep_next`` for every registered player before the deadline,
    including the nailed starter who turns out to be a surprise omission. A
    column that rates two thirds of non-appearances at exactly 0.0 has been
    recorded after the fact.
    """
    share, mean_rating = ep_looks_like_hindsight(season)

    assert share > EP_HINDSIGHT_THRESHOLD
    assert mean_rating < 1.0
    assert any("knows who played" in note for note in season.notes)


def test_the_flag_measures_absences_not_everyone(season):
    """Players who did turn out are rated normally; that is the contrast."""
    played = season.player_gws[
        pd.to_numeric(season.player_gws["minutes"], errors="coerce").fillna(0) >= 60
    ]["ep"]
    played = pd.to_numeric(played, errors="coerce").dropna()
    absent_share, _ = ep_looks_like_hindsight(season)

    assert float((played == 0).mean()) < 0.05
    assert absent_share > 0.5
    assert float(played.mean()) > 2.0


def test_a_genuine_forecast_is_not_flagged(raw):
    """The control: rate absences like everyone else and the flag goes quiet."""
    merged, teams, fixtures = raw
    honest = merged.copy()
    # A forecast that cannot see minutes: every player rated the same 2.5.
    honest["xP"] = 2.5
    data = normalize_archive_season("2024-25", honest, teams, fixtures)

    share, mean_rating = ep_looks_like_hindsight(data)
    assert share == 0.0
    assert mean_rating == pytest.approx(2.5)
    assert not any("knows who played" in note for note in data.notes)
