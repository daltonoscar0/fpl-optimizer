"""Load completed seasons from the community archive into :class:`SeasonData`.

Source: https://github.com/vaastav/Fantasy-Premier-League (credited in the
README). Per season this reads ``gws/merged_gw.csv``, ``teams.csv`` and
``fixtures.csv``.

Two things this module deliberately does *not* do:

* It does not use ``teams.csv`` strength ratings or ``fixtures.csv`` difficulty
  as model features. Those files are end-of-season snapshots, so their values at
  gameweek 1 already encode how the season turned out. Opponent strength in the
  backtest is instead built from a rolling record of results already played
  (see :mod:`fpl_optimizer.features`). The difficulty columns are still carried
  into :class:`SeasonData.fixtures` because the live path has genuine
  point-in-time values there.
* It does not provide ``average_entry_score``: the archive carries no
  per-gameweek average-manager figure, and the FPL API only serves the current
  season's. This gap is recorded in ``SeasonData.notes`` and in the README.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .config import ARCHIVE_BASE, Config
from .rules import normalize_position
from .schema import (
    CLUB_COLUMNS,
    FIXTURE_COLUMNS,
    SeasonData,
    coerce_player_gws,
)

log = logging.getLogger(__name__)

#: The archive files this module reads, per season.
ARCHIVE_FILES: tuple[str, ...] = ("gws/merged_gw.csv", "teams.csv", "fixtures.csv")

#: Position labels present in the archive that have no squad slot. 2024-25
#: carries 322 "AM" (assistant manager) rows; those are dropped.
NON_SQUAD_POSITIONS: tuple[str, ...] = ("AM", "MNG")


def available_seasons() -> tuple[str, ...]:
    """Seasons this project uses: the last three completed ones in the archive.

    Verified at build time (2026-11): the archive holds 2016-17 through
    2026-27, with 38 of 38 gameweek files for each of the three below.
    2026-27 is in progress and only partially collected, so it is not a
    backtest season here — the live path covers it instead.
    """
    return ("2023-24", "2024-25", "2025-26")


def _download(season: str, relpath: str, config: Config) -> Path:
    """Fetch one archive file to ``data/raw/<season>/``, reusing it if present."""
    dest = config.raw_dir / season / relpath.replace("/", "_")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{ARCHIVE_BASE}/{season}/{relpath}"
    log.info("GET %s", url)
    # Same politeness rule as the FPL API: at most one request per second.
    time.sleep(config.min_request_interval)
    resp = requests.get(url, timeout=120, headers={"User-Agent": config.user_agent})
    resp.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)
    return dest


def download_season(season: str, config: Config | None = None) -> dict[str, Path]:
    """Ensure every archive file for ``season`` is on disk."""
    config = config or Config()
    return {rel: _download(season, rel, config) for rel in ARCHIVE_FILES}


def load_season(
    season: str,
    config: Config | None = None,
    *,
    paths: dict[str, Path] | None = None,
) -> SeasonData:
    """Normalize one archive season.

    Parameters
    ----------
    paths:
        Pre-resolved file paths, keyed as in :data:`ARCHIVE_FILES`. Tests pass
        small saved fixtures here so no network call is made.
    """
    config = config or Config()
    paths = paths or download_season(season, config)
    merged = pd.read_csv(paths["gws/merged_gw.csv"], low_memory=False)
    teams = pd.read_csv(paths["teams.csv"])
    fixtures = pd.read_csv(paths["fixtures.csv"], low_memory=False)
    return normalize_archive_season(season, merged, teams, fixtures)


def normalize_archive_season(
    season: str,
    merged: pd.DataFrame,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> SeasonData:
    """Pure transformation from raw archive frames to :class:`SeasonData`."""
    notes: list[str] = []

    clubs = pd.DataFrame(
        {
            "season": season,
            "club_id": teams["id"].astype(int),
            "club": teams["name"].astype(str),
            "club_short": teams["short_name"].astype(str),
        }
    )[list(CLUB_COLUMNS)].reset_index(drop=True)

    id_to_name = dict(zip(clubs["club_id"], clubs["club"]))
    id_to_short = dict(zip(clubs["club_id"], clubs["club_short"]))
    name_to_short = dict(zip(clubs["club"], clubs["club_short"]))

    df = merged.copy()

    # --- drop rows with no squad slot -------------------------------------
    raw_positions = set(df["position"].astype(str).unique())
    dropped = raw_positions & set(NON_SQUAD_POSITIONS)
    if dropped:
        n = int(df["position"].astype(str).isin(NON_SQUAD_POSITIONS).sum())
        df = df[~df["position"].astype(str).isin(NON_SQUAD_POSITIONS)]
        notes.append(
            f"dropped {n} rows with non-squad position(s) {sorted(dropped)} "
            "(assistant-manager slot has no place in a 15-player squad)"
        )
    df["position"] = df["position"].map(normalize_position)
    unmapped = int(df["position"].isna().sum())
    if unmapped:
        notes.append(f"dropped {unmapped} rows with an unrecognised position")
        df = df[df["position"].notna()]

    # --- club identity ----------------------------------------------------
    df["club"] = df["team"].astype(str)
    unknown_clubs = sorted(set(df["club"]) - set(clubs["club"]))
    if unknown_clubs:
        raise ValueError(
            f"{season}: merged_gw club names not in teams.csv: {unknown_clubs}"
        )
    df["club_short"] = df["club"].map(name_to_short)
    opp = pd.to_numeric(df["opponent_team"], errors="coerce").astype("Int64")
    df["opponent"] = opp.map(id_to_name).astype("string")
    df["opponent_short"] = opp.map(id_to_short).astype("string")

    # --- the canonical columns --------------------------------------------
    gw = pd.to_numeric(df["GW"] if "GW" in df else df["round"], errors="coerce")
    out = pd.DataFrame(
        {
            "season": season,
            "gw": gw,
            "player_id": pd.to_numeric(df["element"], errors="coerce"),
            "name": df["name"].astype(str),
            "position": df["position"],
            "club": df["club"],
            "club_short": df["club_short"],
            "opponent": df["opponent"],
            "opponent_short": df["opponent_short"],
            # `value` is the player's price for that gameweek, i.e. the price as
            # it stood at the deadline. Known pre-deadline, so legal to use.
            "price": pd.to_numeric(df["value"], errors="coerce"),
            "fixture_id": pd.to_numeric(df["fixture"], errors="coerce"),
            "kickoff_time": df["kickoff_time"].astype(str),
            "was_home": df["was_home"].astype(str).str.lower().isin(["true", "1"]),
            # FPL's own published expected points for that gameweek.
            "ep": pd.to_numeric(df.get("xP"), errors="coerce"),
            # No per-gameweek injury flag in the archive.
            "availability": pd.NA,
        }
    )
    for col in (
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
    ):
        out[col] = pd.to_numeric(df.get(col), errors="coerce")
    out["points"] = pd.to_numeric(df["total_points"], errors="coerce")

    out = out[out["gw"].notna() & out["player_id"].notna()]

    # The archive's `xP` collection is patchy: in some gameweeks every player is
    # recorded as 0.0, which is a failed snapshot rather than a forecast that
    # nobody will score. Those gameweeks become null so the EP baseline reports
    # itself unavailable instead of optimizing against a flat zero.
    ep_zero_gws: list[int] = []
    if "xP" in merged.columns:
        for gw_val, grp in out.groupby("gw"):
            vals = pd.to_numeric(grp["ep"], errors="coerce")
            if vals.notna().any() and float(vals.abs().max()) == 0.0:
                ep_zero_gws.append(int(gw_val))
        if ep_zero_gws:
            out.loc[out["gw"].isin(ep_zero_gws), "ep"] = float("nan")

    player_gws = coerce_player_gws(out).sort_values(
        ["gw", "player_id", "fixture_id"], kind="stable"
    )
    player_gws = player_gws.reset_index(drop=True)

    if "xP" not in merged.columns:
        notes.append("no `xP` column: the FPL expected-points baseline is unavailable")
    elif ep_zero_gws:
        notes.append(
            f"archive `xP` is all-zero (not collected) for {len(ep_zero_gws)} of "
            f"{player_gws['gw'].nunique()} gameweeks: {ep_zero_gws} — treated as "
            "missing, so the FPL expected-points baseline cannot decide those"
        )
    if "defensive_contribution" not in merged.columns:
        notes.append(
            "no `defensive_contribution` column (scoring rule introduced in 2025-26)"
        )

    fx = pd.DataFrame(
        {
            "season": season,
            "gw": pd.to_numeric(fixtures.get("event"), errors="coerce"),
            "fixture_id": pd.to_numeric(fixtures["id"], errors="coerce"),
            "home_club": pd.to_numeric(fixtures["team_h"], errors="coerce")
            .astype("Int64")
            .map(id_to_name),
            "away_club": pd.to_numeric(fixtures["team_a"], errors="coerce")
            .astype("Int64")
            .map(id_to_name),
            "kickoff_time": fixtures.get("kickoff_time", pd.NA),
            "home_difficulty": pd.to_numeric(
                fixtures.get("team_h_difficulty"), errors="coerce"
            ),
            "away_difficulty": pd.to_numeric(
                fixtures.get("team_a_difficulty"), errors="coerce"
            ),
            "finished": fixtures.get("finished", pd.NA),
        }
    )[list(FIXTURE_COLUMNS)].reset_index(drop=True)

    gws = sorted(int(g) for g in player_gws["gw"].unique())
    events = pd.DataFrame(
        {
            "gw": gws,
            "deadline_time": [
                _first_kickoff(player_gws, fx, g) for g in gws
            ],
            # Not available: see module docstring.
            "average_entry_score": [float("nan")] * len(gws),
            "finished": [True] * len(gws),
        }
    )
    notes.append(
        "no per-gameweek average-manager score: the archive carries none and the "
        "FPL API serves only the current season's `events`, so the "
        "average-manager benchmark is unavailable for this season"
    )
    missing_gws = [g for g in range(1, 39) if g not in gws]
    if missing_gws:
        notes.append(f"gameweeks absent from merged_gw.csv: {missing_gws}")

    return SeasonData(
        season=season,
        player_gws=player_gws,
        clubs=clubs,
        fixtures=fx,
        events=events,
        source="archive",
        notes=notes,
    )


def _first_kickoff(
    player_gws: pd.DataFrame, fixtures: pd.DataFrame, gw: int
) -> str | float:
    """Earliest kickoff in a gameweek, used as a stand-in for the deadline.

    The archive has no deadline timestamps. The first kickoff is always after
    the deadline and before every result, so it orders gameweeks correctly —
    which is all the backtester needs it for.
    """
    rows = fixtures[fixtures["gw"] == gw]["kickoff_time"].dropna()
    if len(rows):
        return str(sorted(rows.astype(str))[0])
    rows = player_gws[player_gws["gw"] == gw]["kickoff_time"].dropna()
    return str(sorted(rows.astype(str))[0]) if len(rows) else float("nan")
