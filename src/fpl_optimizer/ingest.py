"""Building the current season into the same files the archive ships.

The community archive publishes completed seasons months after they end. This
season has no archive, so the project builds one as it goes: each finished
gameweek is written into the archive's own CSV format, which means the existing
loader reads it and the schema is identical by construction rather than by
agreement. A model can then train on this season walk-forward exactly as the
backtest trains on 2023-24.

Where each field comes from
---------------------------
Stated per field because some of it is exact and some is not, and a reader
deserves to know which is which. :data:`FIELD_SOURCES` is the same table,
machine-readable, and is written alongside the data as ``PROVENANCE.md``.

Exact, from one request per gameweek:

``total_points`` and every other scoring stat
    ``event/{gw}/live``, which carries each player's figures for that gameweek,
    already summed over both fixtures of a double.
``fixture``
    ``event/{gw}/live`` again — the ``explain`` block names the fixtures a
    player featured in, which is what gives a double gameweek two rows.
``kickoff_time``, ``team_h_score``, ``team_a_score``, ``was_home``,
``opponent_team``
    ``fixtures``, joined on the fixture id.
``name``
    ``bootstrap-static``.

Exact, but only when a price series is supplied:

``value`` — the price at that gameweek's deadline
    ``event/{gw}/live`` does not carry a price, and ``bootstrap-static`` carries
    only today's. The per-gameweek series lives in ``element-summary``, one
    request per player: expensive, but the backfill happens once. Going
    forward, the pre-deadline snapshot :mod:`fpl_optimizer.track` already
    writes each week is the same figure for free.
``selected``, ``transfers_in``, ``transfers_out``
    Also ``element-summary`` only. Zero when no series is supplied; nothing
    downstream reads them, and a zero that is documented is better than a
    guess that is not.

**Approximated**, and flagged in the notes when it happens:

``value`` with no series available
    Falls back to the nearest known price, and finally to today's. A price that
    moved during the season is then wrong for the gameweeks before the move.
``position`` and ``team``
    ``bootstrap-static`` describes a player as they are *now*. A player's club
    is recovered per gameweek from the fixture they played in, which is exact;
    but a player who changed clubs mid-season and whose current club is not in
    that fixture cannot be placed, and their current club is used. Position
    changes are rarer still and are not recoverable at all.

``xP`` is absent. FPL does not retain its own past expected-points values, so
no backfilled gameweek can carry one — which is precisely the gap the weekly
snapshot exists to close for gameweeks from here on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .config import Config
from .rules import normalize_position

log = logging.getLogger(__name__)

#: Where ingested seasons live. Separate from ``data/raw``, which holds files
#: downloaded from the archive and is not committed: nobody else publishes
#: this, so it is committed.
INGESTED_DIR = "ingested"

#: The archive's own column order for ``merged_gw.csv``, restricted to what the
#: normalizer reads. Writing these means the archive loader reads the result.
MERGED_COLUMNS: tuple[str, ...] = (
    "name",
    "position",
    "team",
    "xP",
    "element",
    "fixture",
    "round",
    "GW",
    "kickoff_time",
    "was_home",
    "opponent_team",
    "value",
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "starts",
    "bonus",
    "bps",
    "defensive_contribution",
    "team_h_score",
    "team_a_score",
    "selected",
    "transfers_in",
    "transfers_out",
)

#: Stats copied straight out of ``event/{gw}/live``.
LIVE_STATS: tuple[str, ...] = (
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "starts",
    "bonus",
    "bps",
    "defensive_contribution",
)

#: ``field -> (source, exact?)``. Written out as ``PROVENANCE.md`` next to the
#: data, so the files can be read a year from now without reading this module.
FIELD_SOURCES: dict[str, tuple[str, str]] = {
    **{f: ("event/{gw}/live", "exact") for f in LIVE_STATS},
    "fixture": ("event/{gw}/live (explain)", "exact"),
    "kickoff_time": ("fixtures", "exact"),
    "team_h_score": ("fixtures", "exact"),
    "team_a_score": ("fixtures", "exact"),
    "was_home": ("fixtures + the player's club", "exact unless the club moved"),
    "opponent_team": ("fixtures", "exact"),
    "name": ("bootstrap-static", "exact"),
    "position": ("bootstrap-static", "today's position, not that gameweek's"),
    "team": ("fixtures + bootstrap-static", "exact unless the club moved"),
    "value": (
        "element-summary history, or the weekly pre-deadline snapshot",
        "exact when supplied, otherwise the nearest known price",
    ),
    "selected": ("element-summary history", "zero when not supplied"),
    "transfers_in": ("element-summary history", "zero when not supplied"),
    "transfers_out": ("element-summary history", "zero when not supplied"),
    "xP": ("unavailable", "FPL does not retain past expected points"),
}


class IngestFailed(RuntimeError):
    """A finished gameweek could not be turned into rows."""


@dataclass
class IngestResult:
    """What one gameweek's ingestion produced."""

    season: str
    gw: int
    rows: int
    players: int
    approximated_prices: int
    notes: list[str]


def ingested_dir(season: str, config: Config | None = None) -> Path:
    directory = (config or Config()).data_dir / INGESTED_DIR / season
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def merged_path(season: str, config: Config | None = None) -> Path:
    return ingested_dir(season, config) / "merged_gw.csv"


def ingested_gameweeks(season: str, config: Config | None = None) -> set[int]:
    """Which gameweeks are already on disk.

    The whole of idempotence rests on this: a gameweek already here is never
    rebuilt, so re-running cannot rewrite a finished gameweek's results with a
    later revision — or worse, duplicate them.
    """
    path = merged_path(season, config)
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["GW"])
    return {int(g) for g in frame["GW"].unique()}


def ingest_gameweek(
    gw: int,
    *,
    bootstrap: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    event_live: Mapping[str, Any],
    prices: Mapping[int, int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """One finished gameweek, in the archive's CSV shape.

    Returns ``(rows, notes)``. One row per (player, fixture), so a double
    gameweek yields two and a blank yields none — the same convention the
    archive uses, because the same loader has to read it.
    """
    notes: list[str] = []

    clubs = {int(t["id"]): str(t["name"]) for t in bootstrap.get("teams") or []}
    types = {
        int(t["id"]): t.get("singular_name_short")
        for t in bootstrap.get("element_types") or []
    }
    meta: dict[int, dict[str, Any]] = {}
    for element in bootstrap.get("elements") or []:
        position = normalize_position(types.get(int(element["element_type"]), ""))
        if position is None:
            continue
        meta[int(element["id"])] = {
            "name": element.get("web_name"),
            "position": position,
            "club_id": int(element["team"]),
            "now_cost": int(element["now_cost"]),
        }

    by_fixture = {
        int(f["id"]): f for f in fixtures if f.get("event") in (gw, str(gw))
    }
    if not by_fixture:
        raise IngestFailed(f"no fixtures listed for GW{gw}")

    approximated: list[int] = []
    rows: list[dict[str, Any]] = []
    for element in event_live.get("elements") or []:
        pid = int(element["id"])
        info = meta.get(pid)
        if info is None:
            continue
        stats = element.get("stats") or {}
        played_in = [
            int(x["fixture"]) for x in (element.get("explain") or []) if "fixture" in x
        ]
        if not played_in:
            # No fixture: a blank gameweek for that club, which the archive
            # represents by the absence of a row rather than by a zero.
            continue

        for fixture_id in played_in:
            fixture = by_fixture.get(fixture_id)
            if fixture is None:
                notes.append(
                    f"GW{gw}: player {pid} lists fixture {fixture_id}, which is "
                    f"not in this gameweek's fixture list; row skipped"
                )
                continue
            home_id, away_id = int(fixture["team_h"]), int(fixture["team_a"])
            if info["club_id"] == home_id:
                was_home, opponent = True, away_id
            elif info["club_id"] == away_id:
                was_home, opponent = False, home_id
            else:
                # The player's current club is not in the fixture they played,
                # so they have changed clubs since. Their side cannot be
                # recovered from this endpoint.
                was_home, opponent = True, away_id
                approximated.append(pid)
                notes.append(
                    f"GW{gw}: player {pid} now plays for a club that was not in "
                    f"fixture {fixture_id}, so their side and opponent that "
                    f"gameweek are a guess"
                )

            club_id = home_id if was_home else away_id
            if prices is not None and pid in prices:
                value = int(prices[pid])
            else:
                value = info["now_cost"]
                approximated.append(pid)

            row = {
                "name": info["name"],
                "position": info["position"],
                "team": clubs.get(club_id),
                "xP": "",
                "element": pid,
                "fixture": fixture_id,
                "round": gw,
                "GW": gw,
                "kickoff_time": fixture.get("kickoff_time"),
                "was_home": was_home,
                "opponent_team": opponent,
                "value": value,
                "team_h_score": fixture.get("team_h_score"),
                "team_a_score": fixture.get("team_a_score"),
                "selected": 0,
                "transfers_in": 0,
                "transfers_out": 0,
                **{field: stats.get(field, 0) for field in LIVE_STATS},
            }
            rows.append(row)

    if not rows:
        raise IngestFailed(f"GW{gw} produced no rows from {len(by_fixture)} fixtures")

    unique_approximated = sorted(set(approximated))
    if unique_approximated and prices is None:
        notes.append(
            f"GW{gw}: no price series supplied, so every row carries today's "
            f"price rather than the price at that deadline"
        )
    elif unique_approximated:
        notes.append(
            f"GW{gw}: {len(unique_approximated)} player(s) fell back to an "
            f"approximated price or side"
        )

    frame = pd.DataFrame(rows)
    for column in MERGED_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0
    return frame[list(MERGED_COLUMNS)], notes


def append_gameweek(
    frame: pd.DataFrame,
    season: str,
    gw: int,
    config: Config | None = None,
) -> Path:
    """Add a gameweek to the season's file. Refuses to touch one already there.

    Idempotence is enforced here rather than left to the caller: results are
    revised by FPL for a day or two after a gameweek, and a rewrite would
    silently replace what a model was trained on. A gameweek is ingested once
    its data is marked checked, and then it is history.
    """
    path = merged_path(season, config)
    if gw in ingested_gameweeks(season, config):
        raise IngestFailed(
            f"GW{gw} is already in {path.name}; it will not be rewritten"
        )
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame
    combined = combined.sort_values(["GW", "element", "fixture"], kind="stable")
    combined.to_csv(path, index=False)
    log.info("appended GW%d to %s (%d rows total)", gw, path, len(combined))
    return path


def write_reference(
    season: str,
    bootstrap: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    config: Config | None = None,
) -> list[Path]:
    """Write ``teams.csv``, ``fixtures.csv`` and ``events.csv``.

    The first two are what the archive ships. ``events.csv`` is more than it
    ships: the archive carries no per-gameweek average-manager score and the
    API serves it only for the season in progress, so capturing it now is the
    one benchmark an ingested season has that a downloaded one never will.
    """
    directory = ingested_dir(season, config)
    teams = pd.DataFrame(
        [
            {
                "id": int(t["id"]),
                "name": t["name"],
                "short_name": t["short_name"],
                "strength": t.get("strength"),
            }
            for t in bootstrap.get("teams") or []
        ]
    )
    fixture_frame = pd.DataFrame(
        [
            {
                "id": int(f["id"]),
                "event": f.get("event"),
                "team_h": f.get("team_h"),
                "team_a": f.get("team_a"),
                "team_h_score": f.get("team_h_score"),
                "team_a_score": f.get("team_a_score"),
                "kickoff_time": f.get("kickoff_time"),
                "finished": f.get("finished"),
                "team_h_difficulty": f.get("team_h_difficulty"),
                "team_a_difficulty": f.get("team_a_difficulty"),
            }
            for f in fixtures
        ]
    )
    events = pd.DataFrame(
        [
            {
                "id": int(e["id"]),
                "deadline_time": e.get("deadline_time"),
                "average_entry_score": e.get("average_entry_score"),
                "finished": e.get("finished"),
                "data_checked": e.get("data_checked"),
            }
            for e in bootstrap.get("events") or []
        ]
    )
    paths = [
        directory / "teams.csv",
        directory / "fixtures.csv",
        directory / "events.csv",
    ]
    teams.to_csv(paths[0], index=False)
    fixture_frame.to_csv(paths[1], index=False)
    events.to_csv(paths[2], index=False)
    return paths


def write_provenance(season: str, config: Config | None = None) -> Path:
    """Write the field-source table next to the data it describes."""
    directory = ingested_dir(season, config)
    lines = [
        f"# {season} — where each field came from",
        "",
        "Built by `fpl ingest` from the public FPL API, one gameweek at a time,",
        "in the archive's own CSV format so the same loader reads both.",
        "",
        "| field | source | exactness |",
        "|---|---|---|",
    ]
    for field, (source, exactness) in FIELD_SOURCES.items():
        lines.append(f"| `{field}` | {source} | {exactness} |")
    lines += [
        "",
        "A gameweek is ingested once FPL marks its data checked, and is never",
        "rewritten afterwards: results are revised for a day or two after the",
        "whistle, and rewriting them would silently change what a model was",
        "trained on.",
        "",
        "`xP` is empty throughout. FPL does not retain its own past",
        "expected-points values, so no backfilled gameweek can carry one. The",
        "pre-deadline snapshots in `results/live/` are that figure, recorded",
        "weekly from the gameweek this project started watching.",
    ]
    path = directory / "PROVENANCE.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def load_ingested_season(season: str, config: Config | None = None):
    """Read an ingested season through the archive's own normalizer.

    The point of writing the archive's format is this line: there is no second
    loader, no second schema, and no chance of the two drifting apart.
    """
    from .archive import normalize_archive_season

    directory = ingested_dir(season, config)
    merged = merged_path(season, config)
    if not merged.exists():
        raise IngestFailed(f"no ingested data for {season} at {merged}")
    data = normalize_archive_season(
        season,
        pd.read_csv(merged, low_memory=False),
        pd.read_csv(directory / "teams.csv"),
        pd.read_csv(directory / "fixtures.csv", low_memory=False),
    )

    # The archive has no average-manager score, so its normalizer leaves the
    # column empty and says so. An ingested season does have one, captured
    # while the API was still serving it.
    events_path = directory / "events.csv"
    if events_path.exists():
        events = pd.read_csv(events_path)
        averages = dict(
            zip(events["id"].astype(int), events["average_entry_score"])
        )
        data.events["average_entry_score"] = (
            data.events["gw"].map(averages).astype("float64")
        )
        data.notes = [
            note for note in data.notes if "average-manager" not in note
        ]
        covered = int(data.events["average_entry_score"].notna().sum())
        data.notes.append(
            f"per-gameweek average-manager scores captured for {covered} "
            f"gameweek(s) while the API was still serving them — the one "
            f"benchmark an ingested season has that an archive one does not"
        )

    data.notes.insert(
        0,
        f"{season} was ingested from the live API by this project, not "
        f"downloaded from the archive — see "
        f"data/{INGESTED_DIR}/{season}/PROVENANCE.md for which fields are exact "
        f"and which are approximated",
    )
    data.notes.append(
        "`xP` is empty: FPL does not retain its own past expected-points "
        "values, so the expected-points baseline is unavailable for every "
        "ingested gameweek"
    )
    return data


def upcoming_rows(
    bootstrap: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    gw: int,
    season: str,
) -> pd.DataFrame:
    """Pre-deadline rows for a gameweek not yet played.

    Everything here is published before the deadline — the fixture, today's
    price, FPL's own expected points — and every outcome column is left empty.
    Concatenated onto the ingested history it gives a season a projection can
    be asked about: eleven gameweeks of results to fit on, and a twelfth to
    predict.
    """
    from .live import _availability, _to_float
    from .schema import coerce_player_gws

    clubs = {int(t["id"]): (str(t["name"]), str(t["short_name"])) for t in bootstrap["teams"]}
    types = {
        int(t["id"]): t.get("singular_name_short")
        for t in bootstrap.get("element_types") or []
    }
    by_club: dict[int, list[Mapping[str, Any]]] = {}
    for fixture in fixtures:
        if fixture.get("event") != gw:
            continue
        for club_id in (int(fixture["team_h"]), int(fixture["team_a"])):
            by_club.setdefault(club_id, []).append(fixture)

    rows: list[dict[str, Any]] = []
    for element in bootstrap.get("elements") or []:
        position = normalize_position(types.get(int(element["element_type"]), ""))
        if position is None:
            continue
        club_id = int(element["team"])
        for fixture in by_club.get(club_id, []):
            was_home = int(fixture["team_h"]) == club_id
            opponent = int(fixture["team_a"] if was_home else fixture["team_h"])
            rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "player_id": int(element["id"]),
                    "name": element.get("web_name"),
                    "position": position,
                    "club": clubs[club_id][0],
                    "club_short": clubs[club_id][1],
                    "opponent": clubs[opponent][0],
                    "opponent_short": clubs[opponent][1],
                    "was_home": was_home,
                    "price": int(element["now_cost"]),
                    "fixture_id": int(fixture["id"]),
                    "kickoff_time": fixture.get("kickoff_time"),
                    "ep": _to_float(element.get("ep_next")),
                    "availability": _availability(element),
                }
            )
    return coerce_player_gws(pd.DataFrame(rows))


def season_with_upcoming(
    season: str,
    bootstrap: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    gw: int,
    config: Config | None = None,
):
    """The ingested season plus the gameweek about to be played.

    This is what makes a live projection walk-forward in the same sense the
    backtest is: the history is only gameweeks FPL has marked checked, so every
    row a model fits on is a result that was final before the deadline being
    predicted, and the gameweek itself contributes nothing but pre-deadline
    facts.

    Raises
    ------
    IngestFailed
        If the gameweek being predicted is already in the ingested history.
        Fitting on it would be training on the answer.
    """
    from .schema import SeasonData, coerce_player_gws

    data = load_ingested_season(season, config)
    if gw in set(data.gameweeks):
        raise IngestFailed(
            f"GW{gw} is already ingested, so it cannot also be the gameweek "
            f"being predicted — that would be training on the answer"
        )

    future = upcoming_rows(bootstrap, fixtures, gw, season)
    combined = coerce_player_gws(
        pd.concat([data.player_gws, future], ignore_index=True)
    ).sort_values(["gw", "player_id", "fixture_id"], kind="stable")

    return SeasonData(
        season=data.season,
        player_gws=combined.reset_index(drop=True),
        clubs=data.clubs,
        fixtures=data.fixtures,
        events=data.events,
        source=data.source,
        notes=[
            *data.notes,
            f"GW{gw} carries pre-deadline facts only — fixture, today's price "
            f"and FPL's published expected points — so a model fitted here has "
            f"seen nothing but gameweeks already final",
        ],
    )
