"""Game rules, read from the API rather than from memory.

FPL has changed squad rules, chip rules and transfer-banking rules several times
in recent seasons, so nothing here is assumed: every value that
``bootstrap-static`` exposes is parsed out of it. The handful of values it does
not expose come from :class:`fpl_optimizer.config.HardCodedRules` and are
reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Config, HardCodedRules

#: Canonical position codes used everywhere downstream.
POSITIONS: tuple[str, ...] = ("GKP", "DEF", "MID", "FWD")

#: ``element_types.singular_name_short`` differs by season ("GK" vs "GKP");
#: archive CSVs use "GK". Everything is normalized to the codes above.
POSITION_ALIASES: dict[str, str] = {
    "gk": "GKP",
    "gkp": "GKP",
    "goalkeeper": "GKP",
    "def": "DEF",
    "defender": "DEF",
    "mid": "MID",
    "midfielder": "MID",
    "fwd": "FWD",
    "forward": "FWD",
}


def normalize_position(raw: str | int) -> str | None:
    """Map a position label to a canonical code, or ``None`` if unsupported.

    Returns ``None`` for labels outside the four squad positions — notably the
    manager slot ("AM"/"MNG") that the 2024-25 archive season contains. Those
    rows are dropped during normalization because the squad model has no slot
    for them.
    """
    if isinstance(raw, int):
        return {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(raw)
    return POSITION_ALIASES.get(str(raw).strip().lower())


@dataclass(frozen=True)
class PositionRule:
    """Quota and formation limits for one position."""

    code: str
    squad_select: int  # how many must be in the 15
    min_play: int  # minimum in the starting XI
    max_play: int  # maximum in the starting XI


@dataclass(frozen=True)
class Rules:
    """A complete, solver-ready rule set.

    Attributes mirror ``bootstrap-static`` field names where one exists so the
    provenance of each number is obvious.
    """

    squad_size: int  # game_settings.squad_squadsize
    squad_play: int  # game_settings.squad_squadplay
    team_limit: int  # game_settings.squad_team_limit
    budget: int  # game_settings.squad_total_spend, in 0.1m units
    sell_on_fee: float  # game_settings.transfers_sell_on_fee
    max_free_transfers: int  # 1 + game_settings.max_extra_free_transfers
    transfers_cap: int  # game_settings.transfers_cap
    sell_at_purchase_price: bool  # game_settings.element_sell_at_purchase_price
    positions: dict[str, PositionRule]
    chips: tuple[str, ...] = ()
    hard_coded: HardCodedRules = field(default_factory=HardCodedRules)
    source: str = "bootstrap-static"

    # --- values that come from HardCodedRules ------------------------------
    @property
    def transfer_hit_cost(self) -> int:
        return self.hard_coded.transfer_hit_cost

    @property
    def free_transfers_per_gw(self) -> int:
        return self.hard_coded.free_transfers_per_gw

    @property
    def captain_multiplier(self) -> int:
        return self.hard_coded.captain_multiplier

    def squad_select(self, position: str) -> int:
        return self.positions[position].squad_select

    def min_play(self, position: str) -> int:
        return self.positions[position].min_play

    def max_play(self, position: str) -> int:
        return self.positions[position].max_play

    def selling_price(self, purchase_price: int, current_price: int) -> int:
        """Price received for selling a player, in 0.1m units.

        FPL keeps ``sell_on_fee`` of any *profit*, rounded down to the nearest
        0.1m; losses are taken in full. When
        ``element_sell_at_purchase_price`` is set, the purchase price is
        returned regardless.
        """
        if self.sell_at_purchase_price:
            return purchase_price
        profit = current_price - purchase_price
        if profit <= 0:
            return current_price
        return purchase_price + int(profit * (1.0 - self.sell_on_fee))

    def validate(self) -> None:
        """Fail loudly if the API returned a rule set we cannot model."""
        total_select = sum(p.squad_select for p in self.positions.values())
        if total_select != self.squad_size:
            raise ValueError(
                f"position quotas sum to {total_select} but squad size is "
                f"{self.squad_size}"
            )
        min_play = sum(p.min_play for p in self.positions.values())
        max_play = sum(p.max_play for p in self.positions.values())
        if not (min_play <= self.squad_play <= max_play):
            raise ValueError(
                f"starting XI size {self.squad_play} is outside the formation "
                f"range [{min_play}, {max_play}]"
            )
        if self.budget <= 0 or self.team_limit <= 0:
            raise ValueError("budget and team limit must be positive")


def rules_from_bootstrap(
    bootstrap: dict[str, Any], config: Config | None = None
) -> Rules:
    """Build :class:`Rules` from a ``bootstrap-static`` payload."""
    config = config or Config()
    settings = bootstrap.get("game_settings") or {}
    # 2026/27 duplicates the rule block under game_config.rules; prefer whichever
    # actually carries the squad fields so this survives either shape.
    if "squad_squadsize" not in settings:
        settings = (bootstrap.get("game_config") or {}).get("rules") or settings
    if "squad_squadsize" not in settings:
        raise ValueError("bootstrap-static has no squad rules in game_settings")

    positions: dict[str, PositionRule] = {}
    for et in bootstrap.get("element_types") or []:
        code = normalize_position(
            et.get("singular_name_short") or et.get("singular_name") or et.get("id")
        )
        if code is None:
            continue  # manager slot or any future non-squad position
        positions[code] = PositionRule(
            code=code,
            squad_select=int(et["squad_select"]),
            min_play=int(et["squad_min_play"]),
            max_play=int(et["squad_max_play"]),
        )
    missing = [p for p in POSITIONS if p not in positions]
    if missing:
        raise ValueError(f"bootstrap-static is missing element_types for {missing}")

    chips = tuple(
        sorted({c["name"] for c in (bootstrap.get("chips") or []) if c.get("name")})
    )

    rules = Rules(
        squad_size=int(settings["squad_squadsize"]),
        squad_play=int(settings["squad_squadplay"]),
        team_limit=int(settings["squad_team_limit"]),
        budget=int(settings["squad_total_spend"]),
        sell_on_fee=float(settings.get("transfers_sell_on_fee", 0.5)),
        max_free_transfers=1 + int(settings.get("max_extra_free_transfers", 0)),
        transfers_cap=int(settings.get("transfers_cap", 20)),
        sell_at_purchase_price=bool(
            settings.get("element_sell_at_purchase_price", False)
        ),
        positions=positions,
        chips=chips,
        hard_coded=config.hard_coded,
    )
    rules.validate()
    return rules


def archive_season_rules(season: str, config: Config | None = None) -> Rules:
    """Rules for a completed season from the community archive.

    The archive carries no ``game_settings`` snapshot, so these are the rules as
    they stood in each season. Flagged in the README: this is the one place
    where a past season's rules are asserted rather than read.

    Free-transfer banking changed in 2024-25 (FPL began allowing up to 5 banked
    free transfers); before that the bank was capped at 1.
    """
    config = config or Config()
    year = int(season.split("-")[0])
    positions = {
        "GKP": PositionRule("GKP", 2, 1, 1),
        "DEF": PositionRule("DEF", 5, 3, 5),
        "MID": PositionRule("MID", 5, 2, 5),
        "FWD": PositionRule("FWD", 3, 1, 3),
    }
    rules = Rules(
        squad_size=15,
        squad_play=11,
        team_limit=3,
        budget=1000,
        sell_on_fee=0.5,
        max_free_transfers=5 if year >= 2024 else 1,
        transfers_cap=20,
        sell_at_purchase_price=False,
        positions=positions,
        chips=("wildcard", "freehit", "bboost", "3xc"),
        hard_coded=config.hard_coded,
        source=f"archive_season_rules({season}) — asserted, not read from API",
    )
    rules.validate()
    return rules
