"""Load the current season from the live FPL API into :class:`SeasonData`.

Composed from three endpoints:

* ``bootstrap-static/`` — player identity, prices, clubs, positions, and the
  ``events`` block that carries each gameweek's deadline and
  ``average_entry_score`` (the average-manager benchmark).
* ``element-summary/{id}/`` — one player's realized per-gameweek rows for the
  season so far, plus their remaining fixtures. Same field names as the
  archive's ``merged_gw.csv``, which is why both sources land on one schema.
* ``fixtures/`` — the full fixture list, used to build rows for gameweeks that
  have not happened yet.

Future gameweeks get rows with outcome columns left empty and ``ep`` populated
from ``ep_next``. FPL publishes ``ep_next`` for the *next* gameweek only, so
beyond that the EP baseline has nothing to say; :func:`load_live_season`
records that in ``notes`` and leaves ``ep`` null rather than extrapolating.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

import pandas as pd

from .api import FPLClient
from .config import Config
from .rules import normalize_position
from .schema import (
    CLUB_COLUMNS,
    FIXTURE_COLUMNS,
    OUTCOME_COLUMNS,
    SeasonData,
    coerce_player_gws,
)

log = logging.getLogger(__name__)

#: ``element-summary`` history fields that map straight onto outcome columns.
_HISTORY_OUTCOMES: tuple[str, ...] = (
    "minutes",
    "bonus",
    "bps",
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
    "defensive_contribution",
    "team_h_score",
    "team_a_score",
    "selected",
    "transfers_in",
    "transfers_out",
)


def current_season(bootstrap: dict[str, Any]) -> str:
    """Season label such as ``"2026-27"``, derived from the GW1 deadline."""
    events = bootstrap.get("events") or []
    first = min((e for e in events if e.get("deadline_time")), key=lambda e: e["id"])
    year = int(str(first["deadline_time"])[:4])
    return f"{year}-{str(year + 1)[-2:]}"


def current_gw(bootstrap: dict[str, Any]) -> int | None:
    """The gameweek now in progress, or ``None`` before the season starts."""
    for e in bootstrap.get("events") or []:
        if e.get("is_current"):
            return int(e["id"])
    return None


def next_gw(bootstrap: dict[str, Any]) -> int | None:
    """The next gameweek with an open deadline — what ``fpl pick`` defaults to."""
    for e in bootstrap.get("events") or []:
        if e.get("is_next"):
            return int(e["id"])
    unfinished = [e for e in bootstrap.get("events") or [] if not e.get("finished")]
    return int(min(unfinished, key=lambda e: e["id"])["id"]) if unfinished else None


def _clubs_frame(bootstrap: dict[str, Any], season: str) -> pd.DataFrame:
    teams = bootstrap["teams"]
    return pd.DataFrame(
        {
            "season": season,
            "club_id": [int(t["id"]) for t in teams],
            "club": [str(t["name"]) for t in teams],
            "club_short": [str(t["short_name"]) for t in teams],
        }
    )[list(CLUB_COLUMNS)]


def _events_frame(bootstrap: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for e in bootstrap.get("events") or []:
        avg = e.get("average_entry_score")
        # A gameweek that has not been scored reports 0, which is not an average.
        if not e.get("finished") or avg in (None, 0):
            avg = float("nan")
        rows.append(
            {
                "gw": int(e["id"]),
                "deadline_time": e.get("deadline_time"),
                "average_entry_score": float(avg) if avg == avg else float("nan"),
                "finished": bool(e.get("finished")),
            }
        )
    return pd.DataFrame(rows).sort_values("gw").reset_index(drop=True)


def _fixtures_frame(
    fixtures: list[dict[str, Any]], season: str, id_to_name: dict[int, str]
) -> pd.DataFrame:
    rows = []
    for f in fixtures:
        rows.append(
            {
                "season": season,
                "gw": f.get("event"),
                "fixture_id": int(f["id"]),
                "home_club": id_to_name.get(int(f["team_h"])),
                "away_club": id_to_name.get(int(f["team_a"])),
                "kickoff_time": f.get("kickoff_time"),
                "home_difficulty": f.get("team_h_difficulty"),
                "away_difficulty": f.get("team_a_difficulty"),
                "finished": bool(f.get("finished")),
            }
        )
    df = pd.DataFrame(rows)
    for col in FIXTURE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[list(FIXTURE_COLUMNS)].reset_index(drop=True)


def load_live_season(
    client: FPLClient | None = None,
    config: Config | None = None,
    *,
    bootstrap: dict[str, Any] | None = None,
    fixtures: list[dict[str, Any]] | None = None,
    summaries: dict[int, dict[str, Any]] | None = None,
    include_future: bool = True,
    progress: Callable[[int, int], None] | None = None,
    player_ids: Iterable[int] | None = None,
) -> SeasonData:
    """Normalize the live season.

    Parameters
    ----------
    bootstrap, fixtures, summaries:
        Pre-fetched payloads. Tests pass small saved fixtures so no network call
        is made; omit them and they are fetched (and cached) through ``client``.
    include_future:
        Add rows for gameweeks not yet played, so the optimizer has something to
        pick for. Outcome columns stay empty on those rows.
    progress:
        Called as ``progress(done, total)`` while fetching player summaries —
        this is the slow step (one request per player, one request per second).
    """
    config = config or Config()
    client = client or FPLClient(config=config)
    bootstrap = bootstrap if bootstrap is not None else client.bootstrap_static()
    fixtures = fixtures if fixtures is not None else client.fixtures()

    season = current_season(bootstrap)
    notes: list[str] = []
    clubs = _clubs_frame(bootstrap, season)
    id_to_name = dict(zip(clubs["club_id"], clubs["club"]))
    id_to_short = dict(zip(clubs["club_id"], clubs["club_short"]))

    elements = [e for e in bootstrap["elements"] if normalize_position(
        _position_code(bootstrap, e["element_type"])
    ) is not None]
    dropped = len(bootstrap["elements"]) - len(elements)
    if dropped:
        notes.append(f"dropped {dropped} players whose position has no squad slot")

    meta = {
        int(e["id"]): {
            "name": _display_name(e),
            "position": normalize_position(
                _position_code(bootstrap, e["element_type"])
            ),
            "club": id_to_name.get(int(e["team"])),
            "club_short": id_to_short.get(int(e["team"])),
            "now_cost": int(e["now_cost"]),
            "ep_next": _to_float(e.get("ep_next")),
            "availability": _availability(e),
        }
        for e in elements
    }

    ids = (
        [int(i) for i in player_ids]
        if player_ids is not None
        else sorted(meta.keys())
    )
    if summaries is None:
        summaries = {}
        for n, pid in enumerate(ids, start=1):
            try:
                summaries[pid] = client.element_summary(pid)
            except FileNotFoundError:
                log.warning("no element-summary for player %s", pid)
            if progress is not None:
                progress(n, len(ids))

    nxt = next_gw(bootstrap)
    rows: list[dict[str, Any]] = []

    # --- gameweeks already played ------------------------------------------
    for pid, summary in summaries.items():
        info = meta.get(int(pid))
        if info is None:
            continue
        for h in summary.get("history") or []:
            opp = h.get("opponent_team")
            row = {
                "season": season,
                "gw": h.get("round"),
                "player_id": int(pid),
                "name": info["name"],
                "position": info["position"],
                "club": info["club"],
                "club_short": info["club_short"],
                "opponent": id_to_name.get(int(opp)) if opp is not None else None,
                "opponent_short": id_to_short.get(int(opp)) if opp is not None else None,
                "was_home": bool(h.get("was_home")),
                # `value` is the price that applied to that gameweek.
                "price": h.get("value"),
                "fixture_id": h.get("fixture"),
                "kickoff_time": h.get("kickoff_time"),
                # FPL does not retain its past expected-points figures; only
                # the archive has them (as `xP`).
                "ep": float("nan"),
                "availability": float("nan"),
                "points": h.get("total_points"),
            }
            for col in _HISTORY_OUTCOMES:
                row[col] = h.get(col)
            rows.append(row)

    if not summaries:
        notes.append("no player histories loaded: realized results are unavailable")
    notes.append(
        "FPL does not serve its past expected-points values, so `ep` is null on "
        "gameweeks already played; the EP baseline is only available going forward"
    )

    # --- gameweeks not yet played ------------------------------------------
    if include_future:
        played = {(r["player_id"], r["gw"]) for r in rows}
        club_fixtures = _future_fixtures_by_club(fixtures, id_to_name)
        for pid, info in meta.items():
            for fx in club_fixtures.get(info["club"], []):
                gw = fx["gw"]
                if gw is None or (pid, gw) in played:
                    continue
                rows.append(
                    {
                        "season": season,
                        "gw": gw,
                        "player_id": int(pid),
                        "name": info["name"],
                        "position": info["position"],
                        "club": info["club"],
                        "club_short": info["club_short"],
                        "opponent": fx["opponent"],
                        "opponent_short": id_to_short.get(fx["opponent_id"]),
                        "was_home": fx["was_home"],
                        "price": info["now_cost"],
                        "fixture_id": fx["fixture_id"],
                        "kickoff_time": fx["kickoff_time"],
                        "ep": info["ep_next"] if gw == nxt else float("nan"),
                        "availability": info["availability"],
                        **{col: None for col in OUTCOME_COLUMNS},
                        "points": None,
                    }
                )
        if nxt is not None:
            notes.append(
                f"`ep` (ep_next) is published for GW{nxt} only; later gameweeks "
                "in a horizon have no FPL expected-points value"
            )

    df = pd.DataFrame(rows)
    df = df[df["gw"].notna()]
    player_gws = coerce_player_gws(df)
    # Outcome columns must be genuinely empty on unplayed gameweeks, not zero,
    # or "scored 0" and "has not played" become indistinguishable.
    future_mask = player_gws["gw"] > (current_gw(bootstrap) or 0)
    for col in OUTCOME_COLUMNS + ("points",):
        player_gws.loc[future_mask, col] = pd.NA
    player_gws = player_gws.sort_values(
        ["gw", "player_id", "fixture_id"], kind="stable"
    ).reset_index(drop=True)

    return SeasonData(
        season=season,
        player_gws=player_gws,
        clubs=clubs,
        fixtures=_fixtures_frame(fixtures, season, id_to_name),
        events=_events_frame(bootstrap),
        source="live",
        notes=notes,
    )


def _future_fixtures_by_club(
    fixtures: list[dict[str, Any]], id_to_name: dict[int, str]
) -> dict[str, list[dict[str, Any]]]:
    """Unplayed fixtures grouped by club, from each club's point of view."""
    out: dict[str, list[dict[str, Any]]] = {}
    for f in fixtures:
        if f.get("finished") or f.get("event") is None:
            continue
        h, a = int(f["team_h"]), int(f["team_a"])
        for club_id, opp_id, home in ((h, a, True), (a, h, False)):
            name = id_to_name.get(club_id)
            if name is None:
                continue
            out.setdefault(name, []).append(
                {
                    "gw": int(f["event"]),
                    "fixture_id": int(f["id"]),
                    "opponent": id_to_name.get(opp_id),
                    "opponent_id": opp_id,
                    "was_home": home,
                    "kickoff_time": f.get("kickoff_time"),
                }
            )
    return out


def _position_code(bootstrap: dict[str, Any], element_type: int) -> str:
    for et in bootstrap.get("element_types") or []:
        if int(et["id"]) == int(element_type):
            return str(
                et.get("singular_name_short") or et.get("singular_name") or element_type
            )
    return str(element_type)


def _display_name(element: dict[str, Any]) -> str:
    web = element.get("web_name")
    if web:
        return str(web)
    return f"{element.get('first_name', '')} {element.get('second_name', '')}".strip()


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _availability(element: dict[str, Any]) -> float:
    """Probability the player features, in ``[0, 1]``.

    ``chance_of_playing_next_round`` is null when FPL has no injury news, which
    means "no doubt reported" rather than "unknown"; those become 1.0. A player
    flagged unavailable (``status`` of i/s/u/n) with no percentage becomes 0.
    """
    chance = element.get("chance_of_playing_next_round")
    if chance is not None:
        return float(chance) / 100.0
    return 0.0 if str(element.get("status", "a")) in {"i", "s", "u", "n"} else 1.0
