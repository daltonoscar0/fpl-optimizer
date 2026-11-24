"""Test data: a synthetic season, a slice of the real archive, a live payload.

Three fixtures, because the tests ask three different questions.

:func:`make_season`
    A synthetic season with hand-placed edge cases — a double gameweek, a blank
    gameweek, a player who never plays, a player who only ever comes off the
    bench. Features and projections are tested against this because every value
    in it is known, so an assertion can be exact rather than approximate.
:func:`archive_paths` / :func:`load_archive_fixture`
    Two clubs over the first eight gameweeks of the real 2025-26 archive,
    committed so no test needs the network. Used to check that normalization
    survives contact with real data, quirks included.
:func:`live_payloads`
    Minimal ``bootstrap-static``/``fixtures``/``element-summary`` payloads in the
    shape the FPL API returns them. Used to prove the live and archive paths
    normalize to the identical schema without calling the API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_optimizer.archive import load_season
from fpl_optimizer.schema import (
    CLUB_COLUMNS,
    FIXTURE_COLUMNS,
    OUTCOME_COLUMNS,
    SeasonData,
    coerce_player_gws,
)

FIXTURE_DIR = Path(__file__).parent

#: Clubs in the synthetic season. A and B meet each other, as do C and D, so
#: every fixture has both of its clubs inside the data and team form is defined.
CLUBS: tuple[tuple[str, str], ...] = (
    ("Club A", "CLA"),
    ("Club B", "CLB"),
    ("Club C", "CLC"),
    ("Club D", "CLD"),
)

#: Clubs A and B play twice in this gameweek: the double-gameweek case.
DOUBLE_GW = 5
#: Clubs C and D do not play in this gameweek: the blank-gameweek case.
BLANK_GW = 6

#: Minutes pattern per archetype, cycled over gameweeks. The three regimes the
#: model's heads are meant to separate, plus one player who never appears.
_ARCHETYPES: dict[str, tuple[int, ...]] = {
    "nailed": (90,),
    "rotation": (90, 0, 75, 0),
    "cameo": (18, 25, 0, 12),
    "absent": (0,),
}

#: One player per (position, archetype) per club.
_SQUAD: tuple[tuple[str, str], ...] = (
    ("GKP", "nailed"),
    ("DEF", "nailed"),
    ("DEF", "rotation"),
    ("MID", "nailed"),
    ("MID", "cameo"),
    ("FWD", "rotation"),
    ("FWD", "absent"),
)

#: Expected points per 90 by position, before noise. Forwards and midfielders
#: score more than defenders, who in turn beat goalkeepers.
_POINTS_PER_90: dict[str, float] = {"GKP": 3.0, "DEF": 3.5, "MID": 4.5, "FWD": 5.0}


def _fixtures_for_gw(gw: int) -> list[tuple[str, str]]:
    """``(home_club, away_club)`` pairs for one synthetic gameweek."""
    swap = gw % 2 == 0
    ab = ("Club B", "Club A") if swap else ("Club A", "Club B")
    cd = ("Club D", "Club C") if swap else ("Club C", "Club D")
    pairs = [ab]
    if gw != BLANK_GW:
        pairs.append(cd)
    if gw == DOUBLE_GW:
        pairs.append(("Club A", "Club B"))
    return pairs


def make_season(
    *,
    season: str = "2024-25",
    n_gws: int = 8,
    seed: int = 0,
    source: str = "archive",
    with_ep: bool = True,
) -> SeasonData:
    """A small season with known edge cases.

    Parameters
    ----------
    with_ep:
        Whether to populate the ``ep`` column. ``False`` reproduces a season
        whose source published no expected points, which the EP baseline has to
        report rather than silently treat as zero.
    """
    rng = np.random.default_rng(seed)
    names = {club: short for club, short in CLUBS}

    players: list[dict[str, Any]] = []
    pid = 1
    for club, _short in CLUBS:
        for position, archetype in _SQUAD:
            players.append(
                {
                    "player_id": pid,
                    "name": f"{club[-1]} {position} {archetype}",
                    "position": position,
                    "club": club,
                    "archetype": archetype,
                    # Prices in FPL's integer tenths-of-a-million units.
                    "price": 40 + 5 * pid % 60,
                }
            )
            pid += 1

    rows: list[dict[str, Any]] = []
    fixture_id = 1
    fixture_rows: list[dict[str, Any]] = []
    for gw in range(1, n_gws + 1):
        for home, away in _fixtures_for_gw(gw):
            home_goals = int(rng.integers(0, 4))
            away_goals = int(rng.integers(0, 3))
            kickoff = f"2024-{8 + gw // 4:02d}-{1 + gw % 28:02d}T14:00:00Z"
            fixture_rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "fixture_id": fixture_id,
                    "home_club": home,
                    "away_club": away,
                    "kickoff_time": kickoff,
                    "home_difficulty": 3,
                    "away_difficulty": 3,
                    "finished": True,
                }
            )
            for club, opponent, was_home in (
                (home, away, True),
                (away, home, False),
            ):
                for p in (q for q in players if q["club"] == club):
                    pattern = _ARCHETYPES[p["archetype"]]
                    minutes = pattern[(gw - 1) % len(pattern)]
                    conceded = away_goals if was_home else home_goals
                    scored = home_goals if was_home else away_goals
                    if minutes == 0:
                        points, bps, starts, goals = 0, 0, 0, 0
                    else:
                        base = _POINTS_PER_90[p["position"]] * minutes / 90.0
                        points = int(max(0, round(base + rng.normal(0, 1.2))))
                        bps = int(max(0, round(points * 5 + rng.normal(0, 3))))
                        starts = 1 if minutes >= 60 else 0
                        goals = int(rng.random() < 0.08)
                    rows.append(
                        {
                            "season": season,
                            "gw": gw,
                            "player_id": p["player_id"],
                            "name": p["name"],
                            "position": p["position"],
                            "club": club,
                            "club_short": names[club],
                            "opponent": opponent,
                            "opponent_short": names[opponent],
                            "was_home": was_home,
                            "price": p["price"],
                            "fixture_id": fixture_id,
                            "kickoff_time": kickoff,
                            # FPL's published figure: the truth, blurred. Good
                            # enough to beat the naive baseline, not perfect.
                            # Collapsed to one value per gameweek below.
                            "ep": (
                                round(float(points) * 0.6 + rng.normal(0, 0.8), 1)
                                if with_ep
                                else float("nan")
                            ),
                            "availability": 1.0,
                            "minutes": minutes,
                            "points": points,
                            "bonus": 0,
                            "bps": bps,
                            "goals_scored": goals,
                            "assists": 0,
                            "clean_sheets": int(minutes >= 60 and conceded == 0),
                            "goals_conceded": conceded if minutes else 0,
                            "own_goals": 0,
                            "penalties_saved": 0,
                            "penalties_missed": 0,
                            "yellow_cards": 0,
                            "red_cards": 0,
                            "saves": 0,
                            "starts": starts,
                            "defensive_contribution": 0,
                            "team_h_score": home_goals,
                            "team_a_score": away_goals,
                            "selected": 100_000,
                            "transfers_in": 0,
                            "transfers_out": 0,
                            "_scored": scored,
                        }
                    )
            fixture_id += 1

    frame = pd.DataFrame(rows).drop(columns=["_scored"])
    # One `ep` per player per gameweek, repeated across a double's two rows,
    # because that is how FPL publishes it and how the archive records it.
    frame["ep"] = frame.groupby(["player_id", "gw"])["ep"].transform("first")
    player_gws = coerce_player_gws(frame).sort_values(
        ["gw", "player_id", "fixture_id"], kind="stable"
    )
    player_gws = player_gws.reset_index(drop=True)

    clubs = pd.DataFrame(
        [
            {
                "season": season,
                "club_id": i + 1,
                "club": club,
                "club_short": short,
            }
            for i, (club, short) in enumerate(CLUBS)
        ]
    )[list(CLUB_COLUMNS)]

    fixtures = pd.DataFrame(fixture_rows)[list(FIXTURE_COLUMNS)]
    events = pd.DataFrame(
        {
            "gw": list(range(1, n_gws + 1)),
            "deadline_time": [
                f"2024-{8 + g // 4:02d}-{1 + g % 28:02d}T11:30:00Z"
                for g in range(1, n_gws + 1)
            ],
            "average_entry_score": [40.0 + g for g in range(1, n_gws + 1)],
            "finished": [True] * n_gws,
        }
    )
    return SeasonData(
        season=season,
        player_gws=player_gws,
        clubs=clubs,
        fixtures=fixtures,
        events=events,
        source=source,
        notes=[],
    )


def corrupt_outcomes(data: SeasonData, from_gw: int, *, value: int = 999) -> SeasonData:
    """Copy of ``data`` with every outcome column at ``gw >= from_gw`` replaced.

    This is the instrument of the leakage tests: anything computed for
    ``from_gw`` that reads a result it is not entitled to will move when the
    results move, and the test fails.
    """
    df = data.player_gws.copy()
    mask = (df["gw"] >= from_gw).to_numpy()
    for col in OUTCOME_COLUMNS:
        if col not in df.columns:
            continue
        filler = value * 1000 if col in ("selected", "transfers_in", "transfers_out") else value
        vals = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0)
            .astype("int64")
            .to_numpy(copy=True)
        )
        vals[mask] = filler
        df[col] = vals
    return SeasonData(
        season=data.season,
        player_gws=coerce_player_gws(df),
        clubs=data.clubs,
        fixtures=data.fixtures,
        events=data.events,
        source=data.source,
        notes=list(data.notes),
    )


# --- the real archive, sliced -------------------------------------------------

#: Season the committed archive slice came from.
ARCHIVE_SEASON = "2025-26"
#: Clubs kept in the slice.
ARCHIVE_CLUBS = ("Arsenal", "Brighton")
#: Gameweeks kept in the slice. Gameweek 7 is one where the archive's `xP`
#: column is all zeros, which is why the slice reaches that far.
ARCHIVE_GWS = tuple(range(1, 9))


def archive_paths() -> dict[str, Path]:
    """Paths keyed as :data:`fpl_optimizer.archive.ARCHIVE_FILES` expects."""
    root = FIXTURE_DIR / f"archive-{ARCHIVE_SEASON}"
    return {
        "gws/merged_gw.csv": root / "merged_gw.csv",
        "teams.csv": root / "teams.csv",
        "fixtures.csv": root / "fixtures.csv",
    }


def load_archive_fixture() -> SeasonData:
    """The committed archive slice, normalized. No network."""
    return load_season(ARCHIVE_SEASON, paths=archive_paths())


# --- the live API, faked ------------------------------------------------------


def live_payloads() -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict]]:
    """``(bootstrap, fixtures, summaries)`` in the shape the FPL API returns.

    Two clubs, four players, two gameweeks — one played, one still to come, so
    the future-fixture path is exercised too.
    """
    bootstrap = {
        "events": [
            {
                "id": 1,
                "deadline_time": "2025-08-15T17:30:00Z",
                "finished": True,
                "is_current": True,
                "is_next": False,
                "average_entry_score": 57,
            },
            {
                "id": 2,
                "deadline_time": "2025-08-22T17:30:00Z",
                "finished": False,
                "is_current": False,
                "is_next": True,
                "average_entry_score": 0,
            },
        ],
        "teams": [
            {"id": 1, "name": "Club A", "short_name": "CLA"},
            {"id": 2, "name": "Club B", "short_name": "CLB"},
        ],
        "element_types": [
            {"id": 1, "singular_name_short": "GKP"},
            {"id": 2, "singular_name_short": "DEF"},
            {"id": 3, "singular_name_short": "MID"},
            {"id": 4, "singular_name_short": "FWD"},
            # The assistant-manager slot, which has no place in a squad.
            {"id": 5, "singular_name_short": "AM"},
        ],
        "elements": [
            {
                "id": 1,
                "first_name": "Ana",
                "second_name": "Keeper",
                "web_name": "Keeper",
                "team": 1,
                "element_type": 1,
                "now_cost": 45,
                "ep_next": "3.4",
                "status": "a",
                "chance_of_playing_next_round": 100,
            },
            {
                "id": 2,
                "first_name": "Bo",
                "second_name": "Back",
                "web_name": "Back",
                "team": 1,
                "element_type": 2,
                "now_cost": 50,
                "ep_next": "4.1",
                "status": "d",
                "chance_of_playing_next_round": 50,
            },
            {
                "id": 3,
                "first_name": "Cy",
                "second_name": "Mid",
                "web_name": "Mid",
                "team": 2,
                "element_type": 3,
                "now_cost": 85,
                "ep_next": "5.9",
                "status": "a",
                "chance_of_playing_next_round": None,
            },
            {
                "id": 4,
                "first_name": "Di",
                "second_name": "Striker",
                "web_name": "Striker",
                "team": 2,
                "element_type": 4,
                "now_cost": 120,
                "ep_next": "6.3",
                "status": "i",
                "chance_of_playing_next_round": 0,
            },
            {
                "id": 5,
                "first_name": "Ed",
                "second_name": "Manager",
                "web_name": "Manager",
                "team": 1,
                "element_type": 5,
                "now_cost": 15,
                "ep_next": "0.0",
                "status": "a",
                "chance_of_playing_next_round": 100,
            },
        ],
    }
    fixtures = [
        {
            "id": 101,
            "event": 1,
            "team_h": 1,
            "team_a": 2,
            "team_h_score": 2,
            "team_a_score": 1,
            "kickoff_time": "2025-08-16T14:00:00Z",
            "finished": True,
            "team_h_difficulty": 3,
            "team_a_difficulty": 4,
        },
        {
            "id": 102,
            "event": 2,
            "team_h": 2,
            "team_a": 1,
            "team_h_score": None,
            "team_a_score": None,
            "kickoff_time": "2025-08-23T14:00:00Z",
            "finished": False,
            "team_h_difficulty": 2,
            "team_a_difficulty": 5,
        },
    ]

    def history(element: int, opponent: int, was_home: bool, points: int) -> dict:
        return {
            "history": [
                {
                    "element": element,
                    "fixture": 101,
                    "round": 1,
                    "opponent_team": opponent,
                    "was_home": was_home,
                    "kickoff_time": "2025-08-16T14:00:00Z",
                    "total_points": points,
                    "minutes": 90,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 0,
                    "goals_conceded": 1,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "saves": 2,
                    "bonus": 0,
                    "bps": 18,
                    "starts": 1,
                    "defensive_contribution": 3,
                    "value": 45,
                    "selected": 120_000,
                    "transfers_in": 10,
                    "transfers_out": 5,
                    "team_h_score": 2,
                    "team_a_score": 1,
                }
            ]
        }

    summaries = {
        1: history(1, 2, True, 6),
        2: history(2, 2, True, 2),
        3: history(3, 1, False, 8),
        4: history(4, 1, False, 1),
    }
    return bootstrap, fixtures, summaries


# --- a manager's public team --------------------------------------------------

#: The recorded entry payloads are synthetic, not a real manager's. The shapes
#: and field names come from the live endpoints — verified against a real entry
#: — but a squad and transfer history are someone's own business even when the
#: endpoint serving them is open, and this repository is public.
ENTRY_ID = 4242


def entry_payloads() -> dict[str, object]:
    """``entry``, ``history``, ``transfers`` and ``picks`` for one manager.

    Shaped like the real endpoints: a team started in gameweek 2, two transfers
    made in gameweek 3, no chips played, and picks that carry no purchase or
    selling prices — because the public endpoint does not serve them.
    """
    directory = FIXTURE_DIR / "entry"
    return {
        "entry": json.loads((directory / "entry.json").read_text()),
        "history": json.loads((directory / "history.json").read_text()),
        "transfers": json.loads((directory / "transfers.json").read_text()),
        "picks": json.loads((directory / "picks-gw3.json").read_text()),
    }


class StubEntryClient:
    """An :class:`~fpl_optimizer.api.FPLClient` that answers from the fixtures.

    Only the four entry endpoints; anything else is an error, so a test that
    starts reaching for the network says so rather than hanging.
    """

    def __init__(self, payloads: dict[str, object] | None = None) -> None:
        self.payloads = payloads or entry_payloads()
        self.calls: list[str] = []

    def entry(self, team_id: int) -> dict:
        self.calls.append(f"entry/{team_id}")
        return self.payloads["entry"]

    def entry_history(self, team_id: int) -> dict:
        self.calls.append(f"history/{team_id}")
        return self.payloads["history"]

    def entry_transfers(self, team_id: int) -> list:
        self.calls.append(f"transfers/{team_id}")
        return self.payloads["transfers"]

    def entry_picks(self, team_id: int, event: int) -> dict:
        self.calls.append(f"picks/{team_id}/{event}")
        return self.payloads["picks"]


# --- recorded payloads for ingestion -----------------------------------------


def ingest_payloads() -> dict[str, object]:
    """``bootstrap``, ``fixtures`` and one ``event/{gw}/live`` per gameweek.

    Shaped like the live endpoints and built from the hand-built season, so an
    ingested gameweek can be checked against figures written out by hand in
    :mod:`tests.fixtures.handbuilt` — including its double and its blank.
    """
    directory = FIXTURE_DIR / "ingest"
    return {
        "bootstrap": json.loads((directory / "bootstrap.json").read_text()),
        "fixtures": json.loads((directory / "fixtures.json").read_text()),
        "live": {
            gw: json.loads((directory / f"event-live-gw{gw}.json").read_text())
            for gw in (1, 2)
        },
    }
