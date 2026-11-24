"""Picking the squad: a mixed-integer program, not a ranked list.

Taking the fifteen highest-projected players is wrong, and not by a little. The
squad has to satisfy a budget, a quota per position, a cap of three players per
club, *and* field a legal starting XI — and those constraints interact, so the
best legal squad routinely contains players who are nowhere near the top of the
projection. That is the whole reason this is an integer program.

What the program decides, in one solve:

``squad``
    Which fifteen to own.
``start``
    Which eleven of them play. The bench still has value, because auto-subs
    sometimes bring it on, but much less than a starter's — hence
    :attr:`~fpl_optimizer.config.Config.bench_weight`.
``captain`` / ``vice``
    Who wears the armband, worth ``captain_multiplier - 1`` extra expected
    points, and who inherits it, worth a tie-breaking fraction.
``transfers``
    When a current squad is supplied, which players to bring in — paid for out
    of the bank plus the proceeds of whoever leaves, and charged four points
    each beyond the free ones. The hit is part of the objective rather than a
    post-hoc deduction, so the program will decline a transfer that does not
    pay for itself.

The objective is expected points, and everything above is expressed in those
units so the trade-offs are commensurable: a transfer happens only if it buys
back more than the four points it costs.

The returned :class:`~fpl_optimizer.score.Squad` is the same type the scorer
consumes and validates, so an optimizer that produced an illegal squad would be
caught by the scorer rather than quietly scored.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd
import pulp

from ..config import Config
from ..rules import POSITIONS, Rules
from ..score import Squad

log = logging.getLogger(__name__)

#: Columns a candidate frame must carry.
CANDIDATE_COLUMNS: tuple[str, ...] = ("player_id", "position", "club", "price", "xpts")

#: Optional columns. ``xpts_future`` is expected points over the gameweeks after
#: this one, already discounted — it is what stops the program selling a player
#: who is about to have a double. ``selling_price`` is what an owned player
#: fetches, which is not the same as what he costs to buy.
OPTIONAL_COLUMNS: tuple[str, ...] = ("xpts_future", "selling_price")


#: Weight of the tie-break nudge on each binary, per place in player-id order.
#: With ~700 candidates and 26 selected binaries the whole term stays under
#: 1e-3 points, so it can only decide between options the objective already
#: considers equal — and it makes the solve reproducible, which an arbitrary
#: choice among tied optima is not.
TIE_BREAK_SCALE = 1e-7


class InfeasibleSquad(RuntimeError):
    """No legal squad exists for the inputs given.

    The message names the constraint most likely to be responsible, because
    "infeasible" on its own is the least useful thing a solver can say.
    """


@dataclass
class SquadPlan:
    """The chosen squad and the arithmetic behind it."""

    squad: Squad
    #: Objective value: expected points for the gameweek, net of any hit.
    expected_points: float
    transfers_in: tuple[int, ...] = ()
    transfers_out: tuple[int, ...] = ()
    transfer_hit: int = 0
    #: Total price of the fifteen, in 0.1m units.
    spend: int = 0
    #: Money left over, in 0.1m units.
    bank: int = 0
    solver_status: str = "Optimal"
    #: Expected points per chosen player, for reporting.
    detail: dict[int, float] = field(default_factory=dict)

    @property
    def n_transfers(self) -> int:
        return len(self.transfers_in)


def _solver(config: Config) -> pulp.LpSolver:
    """HiGHS if it is importable, CBC otherwise. Both ship with the deps."""
    try:
        # Single-threaded with a fixed seed and no optimality gap: a MIP gap
        # lets the solver stop at a near-optimum, which is another way for two
        # runs of the same problem to disagree.
        return pulp.HiGHS(
            msg=False,
            timeLimit=config.solver_time_limit,
            threads=1,
            random_seed=0,
            mip_rel_gap=0.0,
        )
    except Exception:  # pragma: no cover - depends on the local build
        log.info("HiGHS unavailable, falling back to CBC")
        return pulp.PULP_CBC_CMD(msg=False, timeLimit=int(config.solver_time_limit))


@contextlib.contextmanager
def _quiet_stdout():
    """Silence writes to file descriptor 1 for the duration of the block.

    HiGHS prints a two-line banner from C when it is constructed, which neither
    ``msg=False`` nor its own ``output_flag`` option suppresses — it is written
    straight to the file descriptor, so Python-level redirection does not see
    it. A solver banner in the middle of a results table, or thirty-eight of
    them in a backtest, is noise. The solve's status is inspected afterwards, so
    nothing diagnostic is being discarded.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def validate_candidates(candidates: pd.DataFrame) -> None:
    missing = [c for c in CANDIDATE_COLUMNS if c not in candidates.columns]
    if missing:
        raise ValueError(f"candidate frame is missing columns: {missing}")
    if candidates["player_id"].duplicated().any():
        dupes = candidates.loc[
            candidates["player_id"].duplicated(), "player_id"
        ].tolist()
        raise ValueError(f"candidate frame has duplicate players: {dupes}")
    bad = set(candidates["position"].unique()) - set(POSITIONS)
    if bad:
        raise ValueError(f"candidate frame has unknown positions: {sorted(bad)}")


def prune_pool(
    candidates: pd.DataFrame,
    config: Config,
    *,
    keep: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Keep the top ``pool_per_position`` by expected points, plus ``keep``.

    A 38-gameweek backtest solves this program 38 times per strategy. Pruning
    the pool is what keeps that in minutes rather than hours, and it is close to
    free: a player outside the top 50 defenders by projection is not going to
    appear in the optimal squad.
    """
    frames = []
    value = (
        candidates["xpts"] + candidates.get("xpts_future", 0.0)
        if "xpts_future" in candidates.columns
        else candidates["xpts"]
    )
    ranked = candidates.assign(_value=value)
    for position, limit in config.pool_per_position.items():
        frames.append(
            ranked[ranked["position"] == position].nlargest(limit, "_value")
        )
    pool = pd.concat(frames) if frames else ranked
    if keep:
        held = ranked[ranked["player_id"].isin(keep)]
        pool = pd.concat([pool, held]).drop_duplicates("player_id")
    return pool.drop(columns="_value").reset_index(drop=True)


def _check_feasible_counts(pool: pd.DataFrame, rules: Rules) -> None:
    """Fail with a useful message before handing an impossible job to a solver."""
    counts = pool["position"].value_counts()
    for code, rule in rules.positions.items():
        have = int(counts.get(code, 0))
        if have < rule.squad_select:
            raise InfeasibleSquad(
                f"only {have} {code} candidates but the squad needs "
                f"{rule.squad_select}"
            )
    by_club = pool["club"].value_counts()
    if int((by_club.clip(upper=rules.team_limit)).sum()) < rules.squad_size:
        raise InfeasibleSquad(
            f"at most {rules.team_limit} players per club, and the candidates "
            f"span only {len(by_club)} clubs — not enough for a squad of "
            f"{rules.squad_size}"
        )


def optimize_squad(
    candidates: pd.DataFrame,
    rules: Rules,
    *,
    config: Config | None = None,
    current: tuple[int, ...] | None = None,
    free_transfers: int = 0,
    bank: int = 0,
    budget: int | None = None,
    max_hits: int | None = None,
    max_transfers: int | None = None,
    min_transfers: int | None = None,
    must_include: tuple[int, ...] = (),
    must_exclude: tuple[int, ...] = (),
    allow_excluded_clubs: bool = False,
) -> SquadPlan:
    """Solve for the best legal squad, XI, captain and transfers.

    Parameters
    ----------
    current:
        The fifteen currently owned. ``None`` means a fresh pick against the
        full budget; otherwise the program buys and sells, and pays for hits.
    free_transfers:
        Transfers available at no cost. Beyond these, each costs
        ``rules.transfer_hit_cost`` points *in the objective*.
    bank:
        Money not spent on the current squad, in 0.1m units.
    budget:
        Overrides ``rules.budget`` for a fresh pick. Ignored when ``current`` is
        given, where the budget is the bank plus sale proceeds.
    max_transfers:
        Cap on transfers of any kind, free ones included. Solving for one, then
        two, then three and differencing the objectives is how the marginal
        value of each additional transfer is measured.
    min_transfers:
        Floor on the same count. Setting both to ``n`` asks for the best plan
        of exactly that size, which is the only honest way to compare plan
        sizes: the best four-transfer plan is not the best five with a move
        removed.
    max_hits:
        Cap on hit-taking transfers. ``0`` forbids them outright, which is how
        "what do my free transfers alone buy" is asked; ``None`` leaves the
        objective to decide, which it does by weighing each hit's four points
        against what the transfer brings in.
    allow_excluded_clubs:
        :attr:`Config.excluded_clubs` is honoured by default — those clubs are
        never bought. A player already owned is never *forced* out by it.
    """
    config = config or Config()
    rules.validate()
    validate_candidates(candidates)
    current = tuple(int(p) for p in (current or ()))

    frame = candidates.copy()
    frame["player_id"] = frame["player_id"].astype(int)
    if not allow_excluded_clubs:
        excluded = frame["club"].map(lambda c: config.is_excluded(str(c)))
        # Owned players stay in the pool so they can be sold rather than
        # making the problem infeasible.
        frame = frame[~excluded | frame["player_id"].isin(current)]
    if must_exclude:
        frame = frame[~frame["player_id"].isin([int(p) for p in must_exclude])]

    pool = prune_pool(frame, config, keep=current + tuple(must_include))
    _check_feasible_counts(pool, rules)

    missing_required = set(int(p) for p in must_include) - set(pool["player_id"])
    if missing_required:
        raise InfeasibleSquad(
            f"must_include players are not among the candidates: "
            f"{sorted(missing_required)}"
        )

    ids = pool["player_id"].tolist()
    position = dict(zip(ids, pool["position"]))
    club = dict(zip(ids, pool["club"].astype(str)))
    price = dict(zip(ids, pool["price"].astype(int)))
    xpts = dict(zip(ids, pool["xpts"].astype(float)))
    future = (
        dict(zip(ids, pool["xpts_future"].astype(float)))
        if "xpts_future" in pool.columns
        else {i: 0.0 for i in ids}
    )
    selling = (
        dict(zip(ids, pool["selling_price"].astype(int)))
        if "selling_price" in pool.columns
        else dict(price)
    )

    problem = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    in_squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    starting = pulp.LpVariable.dicts("start", ids, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", ids, cat="Binary")
    vice = pulp.LpVariable.dicts("vice", ids, cat="Binary")
    hits = pulp.LpVariable("hits", lowBound=0, cat="Integer")

    bonus = rules.captain_multiplier - 1
    # Ties are common — two defenders on identical projections are not unusual —
    # and a solver is free to return any one of several optima. That showed up
    # as a backtest whose total moved by six points between identical runs,
    # because the two tied players went on to score differently. The decision
    # is genuinely arbitrary; which arbitrary answer it is should not be.
    #
    # So the objective gets a deterministic nudge, ordered by player id. The
    # whole term is bounded below a thousandth of a point, far under any
    # difference that could matter to a score, and it makes the optimum unique
    # and the run reproducible.
    order = {pid: rank for rank, pid in enumerate(sorted(ids))}
    tie_break = {i: -TIE_BREAK_SCALE * order[i] for i in ids}

    problem += (
        pulp.lpSum(starting[i] * xpts[i] for i in ids)
        + pulp.lpSum(
            (in_squad[i] - starting[i]) * xpts[i] * config.bench_weight for i in ids
        )
        + pulp.lpSum(captain[i] * xpts[i] * bonus for i in ids)
        + pulp.lpSum(vice[i] * xpts[i] * config.vice_weight for i in ids)
        + pulp.lpSum(in_squad[i] * future[i] for i in ids)
        + pulp.lpSum((in_squad[i] + starting[i]) * tie_break[i] for i in ids)
        - hits * rules.transfer_hit_cost
    )

    # --- the squad ---------------------------------------------------------
    problem += pulp.lpSum(in_squad[i] for i in ids) == rules.squad_size
    for code, rule in rules.positions.items():
        problem += (
            pulp.lpSum(in_squad[i] for i in ids if position[i] == code)
            == rule.squad_select
        )
    for name in set(club.values()):
        problem += (
            pulp.lpSum(in_squad[i] for i in ids if club[i] == name)
            <= rules.team_limit
        )

    # --- the starting XI ---------------------------------------------------
    problem += pulp.lpSum(starting[i] for i in ids) == rules.squad_play
    for code, rule in rules.positions.items():
        in_position = [starting[i] for i in ids if position[i] == code]
        problem += pulp.lpSum(in_position) >= rule.min_play
        problem += pulp.lpSum(in_position) <= rule.max_play
    for i in ids:
        problem += starting[i] <= in_squad[i]

    # --- the armband -------------------------------------------------------
    problem += pulp.lpSum(captain[i] for i in ids) == 1
    problem += pulp.lpSum(vice[i] for i in ids) == 1
    for i in ids:
        problem += captain[i] <= starting[i]
        problem += vice[i] <= starting[i]
        problem += captain[i] + vice[i] <= 1

    # --- money and transfers ----------------------------------------------
    if current:
        owned = [i for i in ids if i in set(current)]
        incoming = [i for i in ids if i not in set(current)]
        problem += (
            pulp.lpSum(price[i] * in_squad[i] for i in incoming)
            <= bank
            + pulp.lpSum(selling[i] * (1 - in_squad[i]) for i in owned)
        )
        problem += hits >= pulp.lpSum(in_squad[i] for i in incoming) - free_transfers
        if max_hits is not None:
            problem += hits <= int(max_hits)
        if max_transfers is not None:
            problem += (
                pulp.lpSum(in_squad[i] for i in incoming) <= int(max_transfers)
            )
        if min_transfers is not None:
            problem += (
                pulp.lpSum(in_squad[i] for i in incoming) >= int(min_transfers)
            )
    else:
        problem += (
            pulp.lpSum(price[i] * in_squad[i] for i in ids)
            <= (rules.budget if budget is None else budget)
        )
        problem += hits == 0

    for i in (int(p) for p in must_include):
        problem += in_squad[i] == 1

    with _quiet_stdout():
        status = problem.solve(_solver(config))
    label = pulp.LpStatus[status]
    if label != "Optimal":
        raise InfeasibleSquad(
            f"solver returned {label!r}: no legal squad exists under a budget of "
            f"{(rules.budget if budget is None else budget) / 10:.1f}m with "
            f"{len(ids)} candidates"
        )

    chosen = [i for i in ids if in_squad[i].value() > 0.5]
    xi = [i for i in chosen if starting[i].value() > 0.5]
    bench = [i for i in chosen if i not in set(xi)]
    cap = next(i for i in ids if captain[i].value() > 0.5)
    vcap = next(i for i in ids if vice[i].value() > 0.5)

    # Bench order is a manager's pre-deadline choice; the best projected sub
    # goes first. Where the reserve goalkeeper lands in the order does not
    # matter — substitutions match positions, so he can only replace a keeper.
    bench_order = tuple(sorted(bench, key=lambda i: -xpts[i]))
    xi_order = tuple(
        sorted(xi, key=lambda i: (POSITIONS.index(position[i]), -xpts[i]))
    )

    bought = tuple(sorted(i for i in chosen if current and i not in set(current)))
    sold = tuple(sorted(i for i in current if i not in set(chosen)))
    spend = sum(price[i] for i in chosen)

    if current:
        proceeds = sum(selling[i] for i in sold)
        bank_after = bank + proceeds - sum(price[i] for i in bought)
    else:
        bank_after = (rules.budget if budget is None else budget) - spend

    return SquadPlan(
        squad=Squad(
            starting=xi_order, bench=bench_order, captain=cap, vice=vcap
        ),
        expected_points=float(pulp.value(problem.objective)),
        transfers_in=bought,
        transfers_out=sold,
        transfer_hit=int(round(hits.value() or 0)) * rules.transfer_hit_cost,
        spend=spend,
        bank=int(bank_after),
        solver_status=label,
        detail={i: xpts[i] for i in chosen},
    )


def pair_transfers(
    transfers_out: Sequence[int],
    transfers_in: Sequence[int],
    positions: Mapping[int, str],
    prices: Mapping[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Pair players leaving with players arriving, position by position.

    An FPL transfer is like-for-like: a midfielder out and a midfielder in. The
    squad quotas are fixed per position, so any plan sells exactly as many of a
    position as it buys and a valid pairing always exists — but it has to be
    *constructed*, not assumed. Zipping two sorted lists produced
    "a midfielder out, a defender in", which the game would refuse and a
    manager could not enter.

    Within a position the costliest departure is matched to the costliest
    arrival, so each row's money roughly balances. FPL only requires the total
    to balance, but a row that reads sensibly is easier to trust.

    Raises
    ------
    ValueError
        If the positions do not match up, which would mean the plan itself is
        not enterable rather than merely mis-displayed.
    """
    out_by_position: dict[str, list[int]] = {}
    in_by_position: dict[str, list[int]] = {}
    for pid in transfers_out:
        out_by_position.setdefault(positions[pid], []).append(pid)
    for pid in transfers_in:
        in_by_position.setdefault(positions[pid], []).append(pid)

    if {k: len(v) for k, v in out_by_position.items()} != {
        k: len(v) for k, v in in_by_position.items()
    }:
        raise ValueError(
            f"a transfer plan must be like-for-like, but "
            f"{ {k: len(v) for k, v in out_by_position.items()} } leave and "
            f"{ {k: len(v) for k, v in in_by_position.items()} } arrive"
        )

    cost = prices or {}
    pairs: list[tuple[int, int]] = []
    for position in POSITIONS:
        leaving = sorted(
            out_by_position.get(position, []), key=lambda p: -cost.get(p, 0)
        )
        arriving = sorted(
            in_by_position.get(position, []), key=lambda p: -cost.get(p, 0)
        )
        pairs.extend(zip(leaving, arriving))
    return pairs


def pick_squad(
    candidates: pd.DataFrame,
    rules: Rules,
    *,
    config: Config | None = None,
    budget: int | None = None,
    must_include: tuple[int, ...] = (),
    must_exclude: tuple[int, ...] = (),
    allow_excluded_clubs: bool = False,
) -> SquadPlan:
    """A fresh squad against the full budget. See :func:`optimize_squad`."""
    return optimize_squad(
        candidates,
        rules,
        config=config,
        budget=budget,
        must_include=must_include,
        must_exclude=must_exclude,
        allow_excluded_clubs=allow_excluded_clubs,
    )
