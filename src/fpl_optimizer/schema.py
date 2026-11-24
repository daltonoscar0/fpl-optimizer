"""The one internal schema both data sources normalize into.

Everything downstream — features, projections, optimizer, backtester — reads
:class:`SeasonData` and never touches the live API or the archive CSVs directly.
That is what makes the leakage test meaningful: there is exactly one gate
through which gameweek data reaches a decision.

Column classification
---------------------
Each player-gameweek column is either **pre-deadline** (its value is published
and knowable before the gameweek's deadline) or an **outcome** (it only exists
once the gameweek has been played). The backtester is only ever allowed to read
outcome columns for gameweeks *strictly before* the one being decided, and
:data:`OUTCOME_COLUMNS` is the list the leakage test mutates.

``price`` and ``ep`` are deliberately pre-deadline: FPL sets a player's price
before the deadline and publishes its own expected-points figure before it too.
Fixture facts (``opponent``, ``was_home``, ``kickoff_time``) are likewise known
in advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

#: Identity and fixture facts known before the deadline.
PRE_DEADLINE_COLUMNS: tuple[str, ...] = (
    "season",
    "gw",
    "player_id",
    "name",
    "position",
    "club",
    "club_short",
    "opponent",
    "opponent_short",
    "was_home",
    "price",
    "fixture_id",
    "kickoff_time",
    "ep",
    "availability",
)

#: Realized results. Reading any of these for gameweek >= g when deciding for
#: gameweek g is leakage. The leakage test mutates every one of them.
OUTCOME_COLUMNS: tuple[str, ...] = (
    "minutes",
    "points",
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

PLAYER_GW_COLUMNS: tuple[str, ...] = PRE_DEADLINE_COLUMNS + OUTCOME_COLUMNS

CLUB_COLUMNS: tuple[str, ...] = ("season", "club_id", "club", "club_short")

FIXTURE_COLUMNS: tuple[str, ...] = (
    "season",
    "gw",
    "fixture_id",
    "home_club",
    "away_club",
    "kickoff_time",
    "home_difficulty",
    "away_difficulty",
    "finished",
)

#: dtypes enforced on the player-gameweek frame so both sources are identical.
#:
#: Outcome columns use pandas' nullable integer dtypes. A gameweek that has not
#: been played has no result, and ``Int16`` can say so; ``int16`` would have to
#: write 0, making "scored nothing" and "has not played yet" the same value. The
#: live source always carries such rows, so this is what keeps its frame
#: dtype-identical to the archive's rather than silently upcast to float.
PLAYER_GW_DTYPES: dict[str, str] = {
    "season": "string",
    "gw": "int16",
    "player_id": "int32",
    "name": "string",
    "position": "string",
    "club": "string",
    "club_short": "string",
    "opponent": "string",
    "opponent_short": "string",
    "was_home": "boolean",
    "price": "int16",
    "fixture_id": "int32",
    "kickoff_time": "string",
    "ep": "float32",
    "availability": "float32",
    "minutes": "Int16",
    "points": "Int16",
    "bonus": "Int16",
    "bps": "Int16",
    "goals_scored": "Int16",
    "assists": "Int16",
    "clean_sheets": "Int16",
    "goals_conceded": "Int16",
    "own_goals": "Int16",
    "penalties_saved": "Int16",
    "penalties_missed": "Int16",
    "yellow_cards": "Int16",
    "red_cards": "Int16",
    "saves": "Int16",
    "starts": "Int16",
    "defensive_contribution": "Int16",
    "team_h_score": "Int16",
    "team_a_score": "Int16",
    "selected": "Int64",
    "transfers_in": "Int64",
    "transfers_out": "Int64",
}


@dataclass
class SeasonData:
    """One season, normalized, from either source.

    Parameters
    ----------
    player_gws:
        One row per (player, fixture). A double gameweek yields two rows with
        the same ``gw``; a blank gameweek yields none.
    clubs:
        Club identity table for the season.
    fixtures:
        Fixture list with FPL's difficulty ratings where the source has them.
    events:
        Per-gameweek metadata: ``gw``, ``deadline_time``, ``average_entry_score``,
        ``finished``. ``average_entry_score`` is the benchmark the backtest
        compares against.
    source:
        ``"live"`` or ``"archive"``, for provenance in reports.
    notes:
        Anything the reader should know — missing columns, dropped rows,
        coverage gaps. Surfaced in the README and by ``fpl backtest``.
    """

    season: str
    player_gws: pd.DataFrame
    clubs: pd.DataFrame
    fixtures: pd.DataFrame
    events: pd.DataFrame
    source: str
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_player_gws(self.player_gws)

    @property
    def gameweeks(self) -> list[int]:
        return sorted(int(g) for g in self.player_gws["gw"].unique())

    def before(self, gw: int) -> pd.DataFrame:
        """Player-gameweek rows strictly before ``gw``. The only legal source of
        outcome data when deciding for ``gw``."""
        return self.player_gws[self.player_gws["gw"] < gw]

    def at(self, gw: int) -> pd.DataFrame:
        """Rows for ``gw`` itself. Callers deciding for ``gw`` must restrict
        themselves to :data:`PRE_DEADLINE_COLUMNS` of this frame."""
        return self.player_gws[self.player_gws["gw"] == gw]

    def pre_deadline_at(self, gw: int) -> pd.DataFrame:
        """Rows for ``gw`` with every outcome column dropped.

        This is the frame the optimizer is handed. Because the outcome columns
        are physically absent, a leak cannot be written by accident.
        """
        return self.at(gw).drop(columns=list(OUTCOME_COLUMNS), errors="ignore")

    def average_score(self, gw: int) -> float | None:
        row = self.events[self.events["gw"] == gw]
        if row.empty:
            return None
        val = row.iloc[0]["average_entry_score"]
        return None if pd.isna(val) else float(val)

    def club_names(self) -> list[str]:
        return sorted(self.clubs["club"].astype(str).tolist())


def empty_player_gws() -> pd.DataFrame:
    """An empty, correctly typed player-gameweek frame."""
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in PLAYER_GW_COLUMNS})
    return coerce_player_gws(df)


def coerce_player_gws(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to the canonical columns and apply the canonical dtypes."""
    out = df.copy()
    for col in PLAYER_GW_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[list(PLAYER_GW_COLUMNS)]
    for col, dtype in PLAYER_GW_DTYPES.items():
        if dtype.startswith(("int", "float")) and dtype[0].islower():
            filler = 0 if dtype.startswith("int") else float("nan")
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if dtype.startswith("int"):
                out[col] = out[col].fillna(filler)
        try:
            out[col] = out[col].astype(dtype)
        except (TypeError, ValueError):
            # Leave the column as-is rather than lose data; validate() reports it.
            pass
    return out.reset_index(drop=True)


def validate_player_gws(df: pd.DataFrame) -> None:
    """Raise if the frame is not in canonical form.

    Called on every :class:`SeasonData`, so a source that drifts fails fast
    rather than silently producing a different schema for one input.
    """
    missing = [c for c in PLAYER_GW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"player_gws is missing columns: {missing}")
    extra = [c for c in df.columns if c not in PLAYER_GW_COLUMNS]
    if extra:
        raise ValueError(f"player_gws has unexpected columns: {extra}")
    if list(df.columns) != list(PLAYER_GW_COLUMNS):
        raise ValueError("player_gws columns are not in canonical order")
    bad_pos = set(df["position"].dropna().unique()) - {"GKP", "DEF", "MID", "FWD"}
    if bad_pos:
        raise ValueError(f"player_gws has non-canonical positions: {sorted(bad_pos)}")


def schema_signature(df: pd.DataFrame) -> list[tuple[str, str]]:
    """``(column, dtype)`` pairs — compared in the normalization test to prove
    the live and archive paths produce the identical schema."""
    return [(str(c), str(df[c].dtype)) for c in df.columns]


def describe_frame(df: pd.DataFrame) -> dict[str, Any]:
    """Small summary used in reports and test failure messages."""
    return {
        "rows": int(len(df)),
        "players": int(df["player_id"].nunique()) if len(df) else 0,
        "gameweeks": sorted(int(g) for g in df["gw"].unique()) if len(df) else [],
        "clubs": int(df["club"].nunique()) if len(df) else 0,
    }
