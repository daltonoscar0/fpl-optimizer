"""A season small enough to check by hand, big enough to field a legal squad.

Six clubs, three players each, three gameweeks. Every result is written out in
:data:`RESULTS` so that any expected value in a test can be verified by adding
up a column, and the edge cases are placed deliberately rather than generated:

====  ==========================  ==================================================
GW    fixtures                    what it is for
====  ==========================  ==================================================
1     101, 111, 121               an ordinary gameweek
2     102, 103, 112, 122          **double** for Reds and Blues
3     104, 113                    **blank** for Blacks and Golds
====  ==========================  ==================================================

Players 1 to 4 carry exactly the figures the double-gameweek tests pinned when
this season was four players wide, so the scorer is checked against the same
numbers rather than against a fresh set:

* gameweek 2 totals — keeper 5 + 1, playmaker 3 + 8, fullback 2 + 6, finisher
  12 + 0, i.e. ``{1: 6, 2: 11, 3: 8, 4: 12}``.

Three kinds of non-appearance are present, because they are what auto-subs have
to cope with:

* the anchor (5), the marker (13) and the sweeper (16) are dropped for
  gameweek 2 — zero minutes with a fixture. Three of them, because the case
  where a substitution *cannot* legally be made needs every bench defender to
  have sat out;
* the keeper (1) is injured for gameweek 3 — zero minutes, which is what forces
  a goalkeeper substitution;
* Blacks and Golds have no gameweek-3 fixture at all, so players 13 to 18 have
  no row to score — a blank, not a zero.
"""

from __future__ import annotations

import pandas as pd

from fpl_optimizer.schema import (
    CLUB_COLUMNS,
    FIXTURE_COLUMNS,
    SeasonData,
    coerce_player_gws,
)

#: ``club -> short name``.
CLUBS: dict[str, str] = {
    "Reds": "RED",
    "Blues": "BLU",
    "Greens": "GRN",
    "Whites": "WHT",
    "Blacks": "BLK",
    "Golds": "GLD",
}

#: ``player_id -> (name, position, club, price in 0.1m)``.
PLAYERS: dict[int, tuple[str, str, str, int]] = {
    1: ("Keeper", "GKP", "Reds", 45),
    2: ("Playmaker", "MID", "Reds", 85),
    3: ("Fullback", "DEF", "Blues", 50),
    4: ("Finisher", "FWD", "Blues", 120),
    5: ("Anchor", "DEF", "Reds", 55),
    6: ("Winger", "MID", "Blues", 75),
    7: ("Reserve", "GKP", "Greens", 40),
    8: ("Stopper", "DEF", "Greens", 45),
    9: ("Engine", "MID", "Greens", 60),
    10: ("Overlap", "DEF", "Whites", 50),
    11: ("Creator", "MID", "Whites", 95),
    12: ("Poacher", "FWD", "Whites", 70),
    13: ("Marker", "DEF", "Blacks", 45),
    14: ("Runner", "MID", "Blacks", 55),
    15: ("Target", "FWD", "Blacks", 65),
    16: ("Sweeper", "DEF", "Golds", 40),
    17: ("Deep", "MID", "Golds", 50),
    18: ("Understudy", "FWD", "Golds", 45),
}

#: ``(gw, fixture_id, home club, away club)``.
FIXTURES: tuple[tuple[int, int, str, str], ...] = (
    (1, 101, "Reds", "Blues"),
    (1, 111, "Greens", "Whites"),
    (1, 121, "Blacks", "Golds"),
    (2, 102, "Blues", "Reds"),
    (2, 103, "Reds", "Blues"),
    (2, 112, "Whites", "Greens"),
    (2, 122, "Golds", "Blacks"),
    (3, 104, "Blues", "Reds"),
    (3, 113, "Greens", "Whites"),
)

#: ``player_id -> {fixture_id: (minutes, points, ep)}``.
#:
#: Read a row to get a player's season. Players 1 to 4 are the pinned figures;
#: the keeper's gameweek-3 zero is the injury the goalkeeper substitution test
#: needs, and the anchor's, marker's and sweeper's gameweek-2 zeros are the
#: outfield ones.
#:
#: ``ep`` repeats across both fixtures of the double, because that is what the
#: real source does: FPL quotes one expected-points figure per player per
#: gameweek, and in every double-gameweek player-gameweek in the archive the
#: two rows carry the identical value.
RESULTS: dict[int, dict[int, tuple[int, int, float]]] = {
    1: {101: (90, 2, 3.0), 102: (90, 5, 3.0), 103: (90, 1, 3.0), 104: (0, 0, 2.5)},
    2: {101: (90, 6, 5.0), 102: (90, 3, 4.0), 103: (90, 8, 4.0), 104: (90, 7, 5.5)},
    3: {101: (90, 1, 2.0), 102: (90, 2, 2.5), 103: (90, 6, 2.5), 104: (90, 3, 2.0)},
    4: {101: (90, 9, 7.0), 102: (90, 12, 6.0), 103: (90, 0, 6.0), 104: (90, 2, 6.5)},
    5: {101: (90, 3, 3.5), 102: (0, 0, 3.5), 103: (0, 0, 3.5), 104: (90, 6, 3.5)},
    6: {101: (72, 4, 4.0), 102: (90, 5, 4.5), 103: (64, 2, 4.5), 104: (90, 9, 4.5)},
    7: {111: (90, 3, 3.0), 112: (90, 7, 3.0), 113: (90, 2, 3.0)},
    8: {111: (90, 6, 3.5), 112: (90, 1, 3.5), 113: (90, 8, 3.5)},
    9: {111: (90, 5, 4.0), 112: (90, 2, 4.0), 113: (45, 1, 4.0)},
    10: {111: (90, 2, 3.0), 112: (90, 9, 3.0), 113: (90, 4, 3.0)},
    11: {111: (90, 8, 6.0), 112: (90, 11, 6.0), 113: (90, 3, 6.0)},
    12: {111: (18, 1, 4.5), 112: (90, 6, 4.5), 113: (90, 5, 4.5)},
    13: {121: (90, 7, 3.0), 122: (0, 0, 3.0)},
    14: {121: (90, 4, 3.5), 122: (90, 6, 3.5)},
    15: {121: (90, 2, 4.0), 122: (90, 10, 4.0)},
    16: {121: (90, 5, 2.5), 122: (0, 0, 2.5)},
    17: {121: (90, 1, 3.0), 122: (90, 4, 3.0)},
    18: {121: (24, 2, 2.0), 122: (90, 8, 2.0)},
}

#: The double gameweek, and the gameweek with a blank.
DOUBLE_GW = 2
BLANK_GW = 3
#: Clubs with no fixture in :data:`BLANK_GW`.
BLANKED_CLUBS = ("Blacks", "Golds")

#: A legal 15: two keepers, five defenders, five midfielders, three forwards,
#: three players per club. Used as the default squad in the scorer tests.
LEGAL_FIFTEEN: tuple[int, ...] = (1, 7, 3, 5, 8, 10, 13, 2, 6, 9, 11, 14, 4, 12, 15)


def hand_built_season(season: str = "2024-25") -> SeasonData:
    """The season described in this module's docstring."""
    by_fixture = {fid: (gw, home, away) for gw, fid, home, away in FIXTURES}

    rows = []
    for pid, (name, position, club, price) in PLAYERS.items():
        for fixture_id, (minutes, points, ep) in RESULTS[pid].items():
            gw, home, away = by_fixture[fixture_id]
            assert club in (home, away), f"player {pid} has no side in {fixture_id}"
            was_home = club == home
            opponent = away if was_home else home
            rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "player_id": pid,
                    "name": name,
                    "position": position,
                    "club": club,
                    "club_short": CLUBS[club],
                    "opponent": opponent,
                    "opponent_short": CLUBS[opponent],
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
                    "starts": 1 if minutes >= 60 else 0,
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
            {
                "season": season,
                "club_id": i + 1,
                "club": club,
                "club_short": short,
            }
            for i, (club, short) in enumerate(CLUBS.items())
        ]
    )[list(CLUB_COLUMNS)]

    fixtures = pd.DataFrame(
        [
            {
                "season": season,
                "gw": gw,
                "fixture_id": fid,
                "home_club": home,
                "away_club": away,
                "kickoff_time": f"2024-09-0{gw}T14:00:00Z",
                "home_difficulty": 3,
                "away_difficulty": 3,
                "finished": True,
            }
            for gw, fid, home, away in FIXTURES
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
        season=season,
        player_gws=coerce_player_gws(pd.DataFrame(rows)),
        clubs=clubs,
        fixtures=fixtures,
        events=events,
        source="archive",
    )
