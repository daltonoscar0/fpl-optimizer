"""Configuration, including the values FPL's API does not expose.

Everything the API *does* expose is read at runtime in :mod:`fpl_optimizer.rules`.
This module holds (a) user preferences and (b) the short list of rule constants
that have no field in ``bootstrap-static``. Those are collected under
:class:`HardCodedRules` so the README can state exactly what is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

API_BASE = "https://fantasy.premierleague.com/api/"
ARCHIVE_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

#: Clubs Oscar will not select from, under any strategy, live or in backtests.
#: Matched case-insensitively against both full names and short names, so
#: "Spurs"/"Tottenham Hotspur"/"TOT" all resolve to the same club.
DEFAULT_EXCLUDED_CLUBS: tuple[str, ...] = (
    "Arsenal",
    "Tottenham Hotspur",
    "Man Utd",
)

#: Extra spellings so club exclusion survives the archive's naming differences
#: (the archive uses "Man Utd"/"Spurs", the live API uses "Man Utd"/"Spurs" in
#: ``short_name`` but full names differ by season).
CLUB_ALIASES: dict[str, tuple[str, ...]] = {
    "arsenal": ("arsenal", "ars"),
    "tottenham hotspur": ("tottenham hotspur", "tottenham", "spurs", "tot"),
    "man utd": (
        "man utd",
        "man united",
        "manchester united",
        "manchester utd",
        "mun",
        "man u",
    ),
}


@dataclass(frozen=True)
class HardCodedRules:
    """Rules with no ``bootstrap-static`` field, so they must be asserted here.

    Each field documents why it is hard-coded. These are surfaced in the README
    under "known limitations" and printed by ``fpl rules``.
    """

    #: Points deducted per transfer beyond the free allowance. ``game_settings``
    #: exposes ``transfers_cap`` and ``max_extra_free_transfers`` but no cost.
    transfer_hit_cost: int = 4

    #: Free transfers accrued per gameweek. ``max_extra_free_transfers`` caps the
    #: bank but the accrual rate itself is not exposed.
    free_transfers_per_gw: int = 1

    #: Captain's points multiplier. ``chips`` exposes the Triple Captain chip but
    #: no ``pick_multiplier`` for the ordinary armband.
    captain_multiplier: int = 2

    #: Free transfers available before the first gameweek of a season.
    initial_free_transfers: int = 1

    def as_rows(self) -> list[tuple[str, object, str]]:
        return [
            ("transfer_hit_cost", self.transfer_hit_cost, "no field in game_settings"),
            (
                "free_transfers_per_gw",
                self.free_transfers_per_gw,
                "only the bank cap (max_extra_free_transfers) is exposed",
            ),
            (
                "captain_multiplier",
                self.captain_multiplier,
                "only the Triple Captain chip is described in `chips`",
            ),
            (
                "initial_free_transfers",
                self.initial_free_transfers,
                "not exposed; GW1 has no prior week to accrue from",
            ),
        ]


@dataclass
class Config:
    """Runtime configuration for every command."""

    excluded_clubs: tuple[str, ...] = DEFAULT_EXCLUDED_CLUBS
    hard_coded: HardCodedRules = field(default_factory=HardCodedRules)

    # --- paths -------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _project_root() / "data")
    results_dir: Path = field(default_factory=lambda: _project_root() / "results")

    # --- HTTP politeness ---------------------------------------------------
    #: Minimum seconds between two outbound requests to the same host.
    min_request_interval: float = 1.0
    #: Cached responses older than this are refetched. Live prices move daily.
    cache_ttl_hours: float = 6.0
    user_agent: str = (
        "fpl-optimizer/0.1 (+https://github.com/daltonoscar0/fpl-optimizer) "
        "read-only portfolio project"
    )

    # --- optimizer knobs ---------------------------------------------------
    #: Weight on bench expected points in the objective. Bench cover is worth
    #: something (autosubs) but much less than a starter's points.
    bench_weight: float = 0.12
    #: Weight on the vice-captain's expected points; breaks ties toward naming
    #: the second-best starter as vice without distorting the squad choice.
    vice_weight: float = 0.02
    #: Default transfer-planning horizon in gameweeks.
    horizon: int = 4
    #: Per-gameweek discount applied to later gameweeks in the horizon.
    horizon_decay: float = 0.88
    #: Players kept in the ILP pool per position, ranked by projected points
    #: over the horizon. The current squad is always retained on top of this.
    #: Pruning is what keeps a 38-gameweek backtest to minutes rather than hours.
    pool_per_position: dict[str, int] = field(
        default_factory=lambda: {"GKP": 18, "DEF": 50, "MID": 60, "FWD": 35}
    )
    #: Solver time limit per ILP solve, in seconds.
    solver_time_limit: float = 60.0

    # --- projection knobs --------------------------------------------------
    #: Rolling window (gameweeks) for form features.
    form_window: int = 6
    #: Minimum gameweeks of history before the JAX model is trusted; before this
    #: the model falls back to a prior (see projections.model).
    min_train_gws: int = 3

    def excluded_club_keys(self) -> set[str]:
        """All lowercase spellings that count as an excluded club."""
        keys: set[str] = set()
        for club in self.excluded_clubs:
            low = club.strip().lower()
            keys.add(low)
            for canonical, aliases in CLUB_ALIASES.items():
                if low == canonical or low in aliases:
                    keys.update(aliases)
                    keys.add(canonical)
        return keys

    def is_excluded(self, *names: str | None) -> bool:
        """True if any of ``names`` (full name, short name, ...) is excluded."""
        keys = self.excluded_club_keys()
        return any(n is not None and n.strip().lower() in keys for n in names)

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def _project_root() -> Path:
    """Repo root, so ``data/`` and ``results/`` land next to ``src/``."""
    return Path(__file__).resolve().parents[2]
