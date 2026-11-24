"""Reading a real manager's team from the public endpoints.

Everything here is read-only and public: the squad as of the last deadline, the
money in the bank, the transfers already made and what they cost. There is no
login, and no endpoint outside the allowlist in :mod:`fpl_optimizer.api` is
touched — advising on someone's team does not require write access to it.

Two things have to be reconstructed, because the public API serves the squad
but not its history of prices.

**Selling price.** FPL returns the purchase price plus half of any profit,
rounded down, so what a player is worth to *you* depends on what you paid. The
``picks`` endpoint omits that (only the authenticated ``my-team`` carries it),
so purchase prices come from two places: a player transferred in has their cost
in ``entry/{id}/transfers``, and an original pick's purchase price is their
price in the gameweek the team was created, which ``element-summary`` retains
per gameweek. Both are exact; neither needs an account.

**Free transfers.** Also absent from the public API, and recoverable by
replaying the season: one a gameweek, banked up to the rules' cap, spent when
transfers are made, and untouched by a wildcard or free hit. The replay is
checked against the hits FPL actually charged — if the model says a gameweek
was free and FPL charged four points, the model is wrong about something and
says so rather than quietly advising on a bad number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from .rules import Rules
from .schema import SeasonData

log = logging.getLogger(__name__)

#: Chips that let a manager transfer without spending a free transfer.
TRANSFER_CHIPS: frozenset[str] = frozenset({"wildcard", "freehit"})


class EntryUnavailable(RuntimeError):
    """The team could not be read from the public endpoints."""


@dataclass
class EntryState:
    """A manager's team as the public API describes it, plus what it implies."""

    team_id: int
    name: str
    manager: str
    started_event: int
    #: The gameweek these picks are from — the last deadline that has passed.
    gw: int
    squad: tuple[int, ...]
    captain: int
    vice: int
    #: Money in the bank, in 0.1m units.
    bank: int
    #: What FPL calls the team's value at that deadline, in 0.1m units.
    #: It *includes* the bank and uses current prices rather than selling
    #: prices — verified against this entry's ten gameweeks, where
    #: ``value == sum(prices) + bank`` held exactly every time. Worth knowing
    #: before using it as a budget, which would double-count the bank.
    value: int
    free_transfers: int
    #: What each held player cost, in 0.1m units.
    purchase: dict[int, int] = field(default_factory=dict)
    #: What each held player would fetch, in 0.1m units.
    selling: dict[int, int] = field(default_factory=dict)
    #: Anything the reader should know: estimates, inconsistencies, chips.
    notes: list[str] = field(default_factory=list)

    @property
    def budget(self) -> int:
        """What could be spent if the whole squad were sold, in 0.1m units."""
        return self.bank + sum(self.selling.values())


def banked_free_transfers(
    history: Mapping[str, Any],
    rules: Rules,
    *,
    started_event: int = 1,
) -> tuple[int, list[str]]:
    """Replay the season's transfers to find how many are free now.

    One free transfer accrues a gameweek, banked up to
    :attr:`Rules.max_free_transfers`, and spent before any hit is charged. A
    wildcard or free hit makes that gameweek's transfers cost nothing and
    leaves the bank alone.

    The replay is verified against ``event_transfers_cost``, the hits FPL
    actually charged. A disagreement means this reconstruction has the rules
    wrong for that season, which is worth saying out loud rather than acting on.
    """
    notes: list[str] = []
    chip_gws = {
        int(chip["event"]): str(chip["name"]).lower()
        for chip in history.get("chips") or []
    }
    free = rules.hard_coded.initial_free_transfers

    for row in sorted(history.get("current") or [], key=lambda r: int(r["event"])):
        gw = int(row["event"])
        if gw < started_event:
            continue
        used = int(row.get("event_transfers") or 0)
        charged = int(row.get("event_transfers_cost") or 0)
        chip = chip_gws.get(gw)

        if chip in TRANSFER_CHIPS:
            notes.append(f"GW{gw}: {chip} played, so its transfers were free")
            free = min(rules.max_free_transfers, free + rules.free_transfers_per_gw)
            continue

        expected = max(0, used - free) * rules.transfer_hit_cost
        if expected != charged:
            notes.append(
                f"GW{gw}: {used} transfers cost {charged} points, but replaying "
                f"the rules from GW{started_event} expected {expected} — the "
                f"free-transfer count below may be wrong"
            )
        free = min(
            rules.max_free_transfers,
            free - min(used, free) + rules.free_transfers_per_gw,
        )

    return free, notes


def purchase_prices(
    squad: Sequence[int],
    transfers: Sequence[Mapping[str, Any]],
    data: SeasonData,
    *,
    started_event: int = 1,
) -> tuple[dict[int, int], list[str]]:
    """What each held player cost, in 0.1m units.

    A player transferred in carries their cost in the transfer record; the most
    recent one wins, since a player can be sold and bought back at a different
    price. Anyone never transferred in was in the original squad, so they cost
    what they were priced at in the gameweek the team was created.
    """
    notes: list[str] = []
    bought: dict[int, int] = {}
    for transfer in sorted(transfers, key=lambda t: (int(t["event"]), str(t["time"]))):
        bought[int(transfer["element_in"])] = int(transfer["element_in_cost"])

    frame = data.player_gws[["player_id", "gw", "price"]]
    # One row per player, not per fixture: a double gameweek gives a player two
    # rows, and indexing by id would then hand back a Series instead of a price.
    at_start = (
        frame[frame["gw"] == started_event]
        .drop_duplicates("player_id")
        .set_index("player_id")["price"]
    )
    latest = (
        frame.sort_values(["player_id", "gw"])
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")["price"]
    )

    prices: dict[int, int] = {}
    estimated: list[int] = []
    for pid in (int(p) for p in squad):
        if pid in bought:
            prices[pid] = bought[pid]
        elif pid in at_start.index:
            prices[pid] = int(at_start.loc[pid])
        elif pid in latest.index:
            prices[pid] = int(latest.loc[pid])
            estimated.append(pid)
        else:
            prices[pid] = 0
            estimated.append(pid)

    if estimated:
        notes.append(
            f"no purchase price could be established for {len(estimated)} "
            f"player(s) {estimated}: their current price is used instead, so "
            f"their selling price may be overstated"
        )
    return prices, notes


def latest_prices(data: SeasonData, gw: int) -> pd.Series:
    """Each player's most recent known price at or before ``gw``, by player id.

    One row per player, because the frame holds one row per *fixture* and a
    double gameweek would otherwise index two prices under one id. A player
    whose club has a blank still has a price, which is why this looks back
    rather than only at ``gw`` itself.
    """
    frame = data.player_gws[["player_id", "gw", "price"]]
    frame = frame[frame["gw"] <= gw]
    return (
        frame.sort_values(["player_id", "gw"])
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")["price"]
    )


def selling_prices(
    purchase: Mapping[int, int], data: SeasonData, rules: Rules, *, gw: int
) -> dict[int, int]:
    """What each held player would fetch: cost plus half the rise, rounded down."""
    current = latest_prices(data, gw)
    out: dict[int, int] = {}
    for pid, paid in purchase.items():
        now = int(current.loc[pid]) if pid in current.index else int(paid)
        out[int(pid)] = rules.selling_price(int(paid), now)
    return out


def squad_from_picks(picks: Mapping[str, Any]) -> tuple[tuple[int, ...], int, int]:
    """``(squad, captain, vice)`` from a ``picks`` payload."""
    entries = picks.get("picks") or []
    if not entries:
        raise EntryUnavailable("the picks payload contains no players")
    squad = tuple(int(p["element"]) for p in sorted(entries, key=lambda p: p["position"]))
    captain = next(
        (int(p["element"]) for p in entries if p.get("is_captain")), squad[0]
    )
    vice = next(
        (int(p["element"]) for p in entries if p.get("is_vice_captain")), squad[1]
    )
    return squad, captain, vice


def load_entry(
    client,
    team_id: int,
    data: SeasonData,
    rules: Rules,
    *,
    gw: int | None = None,
) -> EntryState:
    """Read a manager's team, bank, selling prices and free transfers.

    Parameters
    ----------
    gw:
        Which gameweek's picks to read. Defaults to the entry's current
        gameweek, which is the last deadline that has passed — and therefore
        the squad as it stood then. A squad changed since that deadline is not
        visible to any public endpoint; pass it in with ``--squad`` instead.
    """
    entry = client.entry(team_id)
    history = client.entry_history(team_id)
    transfers = client.entry_transfers(team_id)

    started = int(entry.get("started_event") or 1)
    target = int(gw if gw is not None else entry.get("current_event") or started)
    picks = client.entry_picks(team_id, target)

    squad, captain, vice = squad_from_picks(picks)
    entry_history = picks.get("entry_history") or {}
    bank = int(entry_history.get("bank") or 0)
    value = int(entry_history.get("value") or 0)

    free, free_notes = banked_free_transfers(history, rules, started_event=started)
    purchase, price_notes = purchase_prices(
        squad, transfers, data, started_event=started
    )
    selling = selling_prices(purchase, data, rules, gw=target)

    notes = [
        f"squad read from entry/{team_id}/event/{target}/picks — the last "
        f"deadline that has passed, not necessarily the squad now",
        *free_notes,
        *price_notes,
    ]

    # FPL's `value` is the squad at current prices plus the bank, so it
    # reconciles exactly against our own price data. When it does not, either
    # the squad or the prices are wrong, and the selling prices built on them
    # are not to be trusted.
    priced = latest_prices(data, target)
    if len(priced) and all(pid in priced.index for pid in squad):
        reconstructed = int(sum(int(priced.loc[pid]) for pid in squad)) + bank
        if value and reconstructed != value:
            notes.append(
                f"FPL reports a team value of {value / 10:.1f}m but the squad at "
                f"GW{target} prices plus {bank / 10:.1f}m in the bank comes to "
                f"{reconstructed / 10:.1f}m — the prices or the squad disagree, "
                f"so the selling prices below may be wrong"
            )
    chips = [str(c.get("name")) for c in history.get("chips") or []]
    if chips:
        notes.append(f"chips played this season: {', '.join(chips)}")

    return EntryState(
        team_id=int(team_id),
        name=str(entry.get("name") or ""),
        manager=" ".join(
            str(entry.get(k) or "")
            for k in ("player_first_name", "player_last_name")
        ).strip(),
        started_event=started,
        gw=target,
        squad=squad,
        captain=captain,
        vice=vice,
        bank=bank,
        value=value,
        free_transfers=free,
        purchase=purchase,
        selling=selling,
        notes=notes,
    )


def resolve_players(
    wanted: Sequence[str], data: SeasonData, *, gw: int | None = None
) -> tuple[tuple[int, ...], list[str]]:
    """Turn a list of names or ids into player ids.

    For ``--squad``, where a manager types what they actually own because the
    public endpoint cannot know. Names match FPL's display name, case- and
    accent-insensitively; an ambiguous name is an error rather than a guess,
    since picking the wrong Silva would quietly advise on someone else's squad.
    """
    import unicodedata

    def fold(text: str) -> str:
        stripped = unicodedata.normalize("NFKD", str(text))
        return "".join(c for c in stripped if not unicodedata.combining(c)).lower().strip()

    frame = data.player_gws[["player_id", "name", "gw"]]
    if gw is not None:
        upto = frame[frame["gw"] <= gw]
        frame = upto if len(upto) else frame
    roster = frame.drop_duplicates("player_id")[["player_id", "name"]]
    by_name: dict[str, list[int]] = {}
    for row in roster.to_dict("records"):
        by_name.setdefault(fold(row["name"]), []).append(int(row["player_id"]))
    known = set(roster["player_id"].astype(int))

    ids: list[int] = []
    notes: list[str] = []
    for token in wanted:
        text = str(token).strip()
        if not text:
            continue
        if text.isdigit():
            pid = int(text)
            if pid not in known:
                raise EntryUnavailable(f"no player with id {pid} this season")
            ids.append(pid)
            continue
        matches = by_name.get(fold(text), [])
        if not matches:
            partial = [
                pid
                for name, pids in by_name.items()
                if fold(text) in name
                for pid in pids
            ]
            matches = partial
        if not matches:
            raise EntryUnavailable(f"no player matching {text!r}")
        if len(matches) > 1:
            names = ", ".join(
                f"{pid} ({roster.set_index('player_id').loc[pid, 'name']})"
                for pid in sorted(matches)
            )
            raise EntryUnavailable(
                f"{text!r} matches more than one player: {names}. Use the id."
            )
        ids.append(matches[0])

    if len(set(ids)) != len(ids):
        raise EntryUnavailable("the same player was named twice")
    return tuple(ids), notes
