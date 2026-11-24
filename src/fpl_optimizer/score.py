"""What a squad actually scored.

This is the other half of a backtest, and the half that is easy to get subtly
wrong. The optimizer's job is to pick; this module's job is to settle up
afterwards using only realized results, applying the three rules that decide
how a set of player scores becomes a gameweek score:

**Both legs of a double count.** A player with two fixtures in a gameweek has
two chances to score and both are theirs. Points are therefore summed per
(player, gameweek) across fixtures — never per fixture, and never averaged.

**The armband doubles the gameweek, not the fixture.** The captain's multiplier
applies to their summed total. If the captain does not play at all, the
vice-captain inherits the armband; if neither plays, nobody is doubled.

**Auto-subs replace starters who did not play.** Bench players come on in bench
order, a goalkeeper can only be replaced by a goalkeeper, and a substitution
only happens if the starting XI it produces is still a legal formation. A
player with no fixture — a blank gameweek — did not play, and is substituted
like anyone else.

Nothing here reads a projection. The scorer is handed realized results and the
squad that was chosen before them, which is what keeps the two sides of a
backtest separable: a bug in the optimizer cannot flatter the score, and a bug
in the scorer cannot flatter the optimizer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from .rules import POSITIONS, Rules
from .schema import SeasonData

#: Order positions are considered in when more than one starter could be
#: substituted, so the result does not depend on dictionary iteration order.
_POSITION_ORDER: dict[str, int] = {code: i for i, code in enumerate(POSITIONS)}


class IllegalSquad(ValueError):
    """Raised when a squad could not have existed under the rules."""


@dataclass(frozen=True)
class Squad:
    """A chosen squad: a starting XI, a bench in order, a captain and a vice.

    ``bench`` is in auto-sub priority order, which in FPL is a decision the
    manager makes before the deadline and not a consequence of anything else.
    """

    starting: tuple[int, ...]
    bench: tuple[int, ...]
    captain: int
    vice: int

    @property
    def players(self) -> tuple[int, ...]:
        return tuple(self.starting) + tuple(self.bench)

    def validate(self, positions: Mapping[int, str], rules: Rules) -> None:
        """Raise :class:`IllegalSquad` if this squad breaks a rule.

        Checked here rather than trusted, because a backtest that scores an
        illegal squad reports points nobody could have scored.
        """
        players = self.players
        if len(set(players)) != len(players):
            raise IllegalSquad("a player appears twice in the squad")
        if len(players) != rules.squad_size:
            raise IllegalSquad(
                f"squad has {len(players)} players, rules require {rules.squad_size}"
            )
        if len(self.starting) != rules.squad_play:
            raise IllegalSquad(
                f"starting XI has {len(self.starting)}, rules require "
                f"{rules.squad_play}"
            )
        missing = [p for p in players if p not in positions]
        if missing:
            raise IllegalSquad(f"no position known for players {missing}")

        squad_counts = Counter(positions[p] for p in players)
        for code, rule in rules.positions.items():
            if squad_counts.get(code, 0) != rule.squad_select:
                raise IllegalSquad(
                    f"squad has {squad_counts.get(code, 0)} {code}, rules require "
                    f"{rule.squad_select}"
                )
        if not formation_is_legal(
            Counter(positions[p] for p in self.starting), rules
        ):
            raise IllegalSquad(
                "starting XI is not a legal formation: "
                + ", ".join(
                    f"{c}={Counter(positions[p] for p in self.starting).get(c, 0)}"
                    for c in POSITIONS
                )
            )
        if self.captain not in self.starting:
            raise IllegalSquad("the captain is not in the starting XI")
        if self.vice not in self.starting:
            raise IllegalSquad("the vice-captain is not in the starting XI")
        if self.captain == self.vice:
            raise IllegalSquad("the captain and vice-captain are the same player")


@dataclass(frozen=True)
class GameweekScore:
    """What a squad scored in one gameweek, and how it got there.

    ``points`` is the figure a league table would show: the starting XI as it
    ended up after auto-subs, plus the armband bonus, minus any transfer hit.
    """

    gw: int
    points: int
    starters_points: int
    captain_bonus: int
    bench_points: int
    transfer_hit: int
    #: Who actually wore the armband, or ``None`` if neither captain nor vice
    #: played and so nobody was doubled.
    armband: int | None
    #: True when the captain did not play and the vice inherited the armband.
    vice_used: bool
    #: ``(off, on)`` pairs, in the order the substitutions were made.
    autosubs: tuple[tuple[int, int], ...]
    #: The XI after auto-subs.
    final_xi: tuple[int, ...]
    #: Realized points per squad player, summed over their fixtures.
    player_points: Mapping[int, int] = field(default_factory=dict)
    #: Realized minutes per squad player, summed over their fixtures.
    player_minutes: Mapping[int, int] = field(default_factory=dict)

    @property
    def points_on_bench(self) -> int:
        """Points left unused. Not a cost, but worth watching in a backtest."""
        return self.bench_points


def formation_is_legal(counts: Mapping[str, int], rules: Rules) -> bool:
    """Whether a starting XI with these position counts is playable."""
    if sum(counts.values()) != rules.squad_play:
        return False
    return all(
        rule.min_play <= counts.get(code, 0) <= rule.max_play
        for code, rule in rules.positions.items()
    )


def transfer_cost(n_transfers: int, free_transfers: int, rules: Rules) -> int:
    """Points deducted for making ``n_transfers`` with ``free_transfers`` free."""
    chargeable = max(0, n_transfers - max(0, free_transfers))
    return chargeable * rules.transfer_hit_cost


def realized(
    data: SeasonData, gw: int, players: tuple[int, ...] | list[int]
) -> tuple[dict[int, int], dict[int, int]]:
    """``(points, minutes)`` per player for ``gw``, summed over their fixtures.

    A player with no row in ``gw`` — a blank gameweek, or a player who left the
    league — gets zero of both, which is what makes them substitutable rather
    than missing.
    """
    rows = data.at(gw)
    rows = rows[rows["player_id"].isin(list(players))]
    points = {int(p): 0 for p in players}
    minutes = {int(p): 0 for p in players}
    if not rows.empty:
        grouped = rows.groupby("player_id")[["points", "minutes"]].sum()
        for pid, row in grouped.iterrows():
            points[int(pid)] = int(pd.to_numeric(row["points"], errors="coerce") or 0)
            minutes[int(pid)] = int(pd.to_numeric(row["minutes"], errors="coerce") or 0)
    return points, minutes


def positions_from(data: SeasonData) -> dict[int, str]:
    """``player_id -> position`` over the whole season.

    Read from every gameweek rather than from the one being scored, so a player
    whose club has a blank still has a known position.
    """
    frame = data.player_gws[["player_id", "position"]].dropna().drop_duplicates(
        "player_id"
    )
    return {int(r.player_id): str(r.position) for r in frame.itertuples()}


def apply_autosubs(
    squad: Squad,
    minutes: Mapping[int, int],
    positions: Mapping[int, str],
    rules: Rules,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """``(final_xi, substitutions)`` after replacing starters who did not play.

    Bench players are considered in bench order — that is the manager's stated
    priority and FPL honours it. A bench player comes on for the first
    non-playing starter they can legally replace; a goalkeeper swaps only with a
    goalkeeper; and the swap is made only if the resulting XI is a legal
    formation, which is why a side already down to the minimum three defenders
    cannot bring on a midfielder for a defender who was dropped.

    Where more than one non-playing starter could be replaced, the one earliest
    in :data:`_POSITION_ORDER` (then by id) goes, so the outcome does not depend
    on iteration order.
    """
    xi = list(squad.starting)
    subs: list[tuple[int, int]] = []

    for incoming in squad.bench:
        if minutes.get(incoming, 0) <= 0:
            continue
        candidates = sorted(
            (p for p in xi if minutes.get(p, 0) <= 0),
            key=lambda p: (_POSITION_ORDER.get(positions[p], 99), p),
        )
        for outgoing in candidates:
            is_keeper_swap = "GKP" in (positions[incoming], positions[outgoing])
            if is_keeper_swap and positions[incoming] != positions[outgoing]:
                continue
            trial = [incoming if p == outgoing else p for p in xi]
            if formation_is_legal(Counter(positions[p] for p in trial), rules):
                xi = trial
                subs.append((outgoing, incoming))
                break

    return tuple(xi), tuple(subs)


def score_gameweek(
    squad: Squad,
    data: SeasonData,
    gw: int,
    rules: Rules,
    *,
    transfer_hit: int = 0,
    positions: Mapping[int, str] | None = None,
    validate: bool = True,
) -> GameweekScore:
    """Score ``squad`` over ``gw`` using realized results only.

    Parameters
    ----------
    transfer_hit:
        Points already deducted for transfers made before this gameweek — see
        :func:`transfer_cost`. Passed in rather than computed here because the
        scorer does not know what the previous squad was.
    """
    positions = positions or positions_from(data)
    if validate:
        squad.validate(positions, rules)

    points, minutes = realized(data, gw, squad.players)
    final_xi, subs = apply_autosubs(squad, minutes, positions, rules)

    starters_points = sum(points[p] for p in final_xi)
    bench_points = sum(points[p] for p in squad.players if p not in set(final_xi))

    # The armband: the captain if they played at all, otherwise the vice.
    armband: int | None = None
    vice_used = False
    if minutes.get(squad.captain, 0) > 0:
        armband = squad.captain
    elif minutes.get(squad.vice, 0) > 0:
        armband = squad.vice
        vice_used = True
    captain_bonus = (
        points[armband] * (rules.captain_multiplier - 1) if armband is not None else 0
    )

    return GameweekScore(
        gw=gw,
        points=starters_points + captain_bonus - transfer_hit,
        starters_points=starters_points,
        captain_bonus=captain_bonus,
        bench_points=bench_points,
        transfer_hit=transfer_hit,
        armband=armband,
        vice_used=vice_used,
        autosubs=subs,
        final_xi=final_xi,
        player_points=points,
        player_minutes=minutes,
    )
