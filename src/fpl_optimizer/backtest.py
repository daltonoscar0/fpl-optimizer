"""The walk-forward backtest: what a strategy would actually have scored.

One loop, and the order of its four steps is the whole point:

1. **Fit** on gameweeks strictly before ``gw``. Nothing else is visible.
2. **Project** ``gw`` — and the few after it, discounted — from pre-deadline
   facts only.
3. **Decide**: solve the integer program for a squad, an XI, an armband and
   whatever transfers pay for themselves.
4. **Score** with realized results, which are read for the first time here,
   after the decision is already fixed.

Steps 1 to 3 cannot see what step 4 reads. That is enforced by
:class:`~fpl_optimizer.schema.SeasonData` — the frames handed to the first
three steps have the outcome columns physically removed — and tested by
rewriting every result from ``gw`` onwards and checking the decisions do not
move.

Carried between gameweeks, because FPL carries them: the squad, the bank, the
purchase price of every player held (which sets what they sell for), and the
free-transfer count, which accrues one per gameweek up to the rules' cap.

When a strategy has no view
---------------------------
Two ways that happens. FPL's expected points are missing from the archive for
whole gameweeks, so the ``ep`` strategy has nothing to say in them. And before
any football has been played the ``naive`` rate is zero for everyone, which is
not an opinion about footballers — it is the absence of one.

Either way the strategy does not get to decide: the squad is held unchanged, no
transfers are made, and the gameweek is recorded in
:attr:`BacktestResult.held_gws`. It still scores, because the players still
played.

Letting a flat projection decide anyway is worse than it sounds. An earlier
version did, and the arbitrary legal squad it produced for gameweek 1 — 74.5m
of a 100.0m budget, chosen with no preference between any two players — cost 40
points in hits over the next two gameweeks to unwind. The squad a strategy
cannot justify is not a cheap mistake.

Because strategies therefore start in different gameweeks,
:func:`compare` reports totals over the window every strategy decided, as well
as each strategy's own: a total over 38 gameweeks and a total over 37 are not
the same number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .config import Config
from .features import FeatureCache
from .optimize import build_candidates, horizon_predictions, optimize_squad, pick_squad
from .projections import STRATEGIES, JaxPointsModel
from .projections.baselines import EP_COVERAGE_THRESHOLD, ep_coverage
from .rules import Rules
from .schema import SeasonData
from .score import GameweekScore, Squad, positions_from, score_gameweek

log = logging.getLogger(__name__)


@dataclass
class GameweekRecord:
    """One gameweek of a backtest: the decision, and what it scored."""

    gw: int
    points: int
    cumulative: int
    starters_points: int
    captain_bonus: int
    bench_points: int
    transfer_hit: int
    n_transfers: int
    transfers_in: tuple[int, ...]
    transfers_out: tuple[int, ...]
    captain: int
    armband: int | None
    vice_used: bool
    autosubs: tuple[tuple[int, int], ...]
    #: The decision itself, kept so it can be inspected and compared. The
    #: leakage test works by checking these do not move when results do.
    squad: tuple[int, ...] = ()
    starting: tuple[int, ...] = ()
    squad_value: int = 0
    bank: int = 0
    free_transfers_after: int = 0
    #: True when the strategy had no projection and the squad was held.
    held: bool = False
    #: True when the model deferred to the expected-points baseline.
    fallback: bool = False


@dataclass
class BacktestResult:
    """A strategy's season."""

    season: str
    strategy: str
    records: list[GameweekRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total_points(self) -> int:
        return sum(r.points for r in self.records)

    @property
    def gameweeks(self) -> int:
        return len(self.records)

    @property
    def decided_gws(self) -> list[int]:
        """Gameweeks where the strategy had a projection and chose."""
        return [r.gw for r in self.records if not r.held]

    @property
    def held_gws(self) -> list[int]:
        """Gameweeks where it had no view and the squad was held unchanged."""
        return [r.gw for r in self.records if r.held]

    @property
    def fallback_gws(self) -> list[int]:
        return [r.gw for r in self.records if r.fallback]

    @property
    def total_hits(self) -> int:
        return sum(r.transfer_hit for r in self.records)

    @property
    def total_transfers(self) -> int:
        return sum(r.n_transfers for r in self.records)

    @property
    def points_per_gameweek(self) -> float:
        return self.total_points / self.gameweeks if self.records else 0.0

    def frame(self) -> pd.DataFrame:
        """Per-gameweek records, for writing to disk or plotting."""
        return pd.DataFrame(
            [
                {
                    "season": self.season,
                    "strategy": self.strategy,
                    "gw": r.gw,
                    "points": r.points,
                    "cumulative": r.cumulative,
                    "starters_points": r.starters_points,
                    "captain_bonus": r.captain_bonus,
                    "bench_points": r.bench_points,
                    "transfer_hit": r.transfer_hit,
                    "n_transfers": r.n_transfers,
                    "captain": r.captain,
                    "armband": r.armband,
                    "vice_used": r.vice_used,
                    "autosubs": len(r.autosubs),
                    "squad": " ".join(str(p) for p in r.squad),
                    "starting": " ".join(str(p) for p in r.starting),
                    "squad_value": r.squad_value,
                    "bank": r.bank,
                    "free_transfers_after": r.free_transfers_after,
                    "held": r.held,
                    "fallback": r.fallback,
                }
                for r in self.records
            ]
        )

    def summary(self) -> dict[str, object]:
        return {
            "season": self.season,
            "strategy": self.strategy,
            "gameweeks": self.gameweeks,
            "decided_gameweeks": len(self.decided_gws),
            "held_gameweeks": sorted(self.held_gws),
            "fallback_gameweeks": sorted(self.fallback_gws),
            "total_points": self.total_points,
            "scored_gameweeks": sorted(r.gw for r in self.records),
            "points_per_gameweek": round(self.points_per_gameweek, 2),
            "total_transfers": self.total_transfers,
            "total_hits": self.total_hits,
            "notes": list(self.notes),
        }


def _next_free_transfers(free: int, used: int, rules: Rules) -> int:
    """FPL's accrual: spend what you use, gain one, bank up to the cap."""
    spent = min(used, free)
    return min(rules.max_free_transfers, free - spent + rules.free_transfers_per_gw)


def backtest(
    data: SeasonData,
    strategy: str,
    rules: Rules,
    *,
    config: Config | None = None,
    start_gw: int | None = None,
    end_gw: int | None = None,
    cache: FeatureCache | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> BacktestResult:
    """Run one strategy over one season, deciding before each gameweek."""
    config = config or Config()
    rules.validate()
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}: choose from {sorted(STRATEGIES)}")

    gameweeks = [
        g
        for g in data.gameweeks
        if (start_gw is None or g >= start_gw) and (end_gw is None or g <= end_gw)
    ]
    projection = STRATEGIES[strategy]()
    if isinstance(projection, JaxPointsModel):
        projection.attach(cache or FeatureCache(data, form_window=config.form_window))

    result = BacktestResult(season=data.season, strategy=strategy)
    coverage, missing = ep_coverage(data)
    if strategy == "ep":
        result.notes.append(
            f"FPL expected points cover {coverage:.0%} of gameweeks "
            f"({len(data.gameweeks) - len(missing)} of {len(data.gameweeks)}); "
            f"the squad is held unchanged in the rest"
        )
        if coverage < EP_COVERAGE_THRESHOLD:
            result.notes.append(
                f"coverage is below the {EP_COVERAGE_THRESHOLD:.0%} threshold, so "
                f"this season's expected-points result is not comparable"
            )

    positions = positions_from(data)
    squad: Squad | None = None
    bank = 0
    free_transfers = rules.hard_coded.initial_free_transfers
    purchase: dict[int, int] = {}
    cumulative = 0

    for n, gw in enumerate(gameweeks, start=1):
        projection.fit(data, gw)
        preds = projection.predict(data, gw)
        # A flat projection is not a view. See the module docstring: letting one
        # pick a squad costs more to unwind than it ever wins.
        held = preds.empty or float(preds["xpts"].abs().max()) == 0.0

        if held and squad is None:
            # Nothing to decide on and nothing to hold: the strategy cannot
            # start a season it has no view of.
            log.info("gw%d: %s has no projection and no squad to hold", gw, strategy)
            if progress is not None:
                progress(n, len(gameweeks))
            continue

        if held:
            plan_squad, hit, bought, sold = squad, 0, (), ()
            spend = _squad_value(data, gw, squad.players, purchase)
        else:
            future = horizon_predictions(
                projection,
                data,
                gw,
                horizon=config.horizon,
                decay=config.horizon_decay,
            )
            selling = (
                {
                    pid: rules.selling_price(
                        purchase.get(pid, 0), _current_price(data, gw, pid, purchase)
                    )
                    for pid in squad.players
                }
                if squad is not None
                else None
            )
            candidates = build_candidates(
                data, gw, preds, future=future, selling_prices=selling
            )
            if squad is None:
                plan = pick_squad(candidates, rules, config=config)
            else:
                plan = optimize_squad(
                    candidates,
                    rules,
                    config=config,
                    current=squad.players,
                    free_transfers=free_transfers,
                    bank=bank,
                )
            plan_squad, hit = plan.squad, plan.transfer_hit
            bought, sold = plan.transfers_in, plan.transfers_out
            bank = plan.bank
            spend = plan.spend
            prices = candidates.set_index("player_id")["price"]
            for pid in plan_squad.players:
                purchase.setdefault(int(pid), int(prices.get(int(pid), 0)))
            for pid in sold:
                purchase.pop(int(pid), None)

        score: GameweekScore = score_gameweek(
            plan_squad,
            data,
            gw,
            rules,
            transfer_hit=hit,
            positions=positions,
            validate=squad is None or not held,
        )
        cumulative += score.points
        used = len(bought)
        free_transfers = (
            free_transfers
            if held
            else _next_free_transfers(free_transfers, used, rules)
        )

        result.records.append(
            GameweekRecord(
                gw=gw,
                points=score.points,
                cumulative=cumulative,
                starters_points=score.starters_points,
                captain_bonus=score.captain_bonus,
                bench_points=score.bench_points,
                transfer_hit=hit,
                n_transfers=used,
                transfers_in=bought,
                transfers_out=sold,
                captain=plan_squad.captain,
                armband=score.armband,
                vice_used=score.vice_used,
                autosubs=score.autosubs,
                squad=tuple(plan_squad.players),
                starting=tuple(plan_squad.starting),
                squad_value=spend,
                bank=bank,
                free_transfers_after=free_transfers,
                held=held,
                fallback=gw in getattr(projection, "fallback_gws", []),
            )
        )
        squad = plan_squad
        if progress is not None:
            progress(n, len(gameweeks))

    if isinstance(projection, JaxPointsModel):
        if projection.fallback_gws:
            result.notes.append(
                f"the model deferred to FPL's expected points for "
                f"{len(projection.fallback_gws)} gameweeks with too little "
                f"history: {sorted(projection.fallback_gws)}"
            )
        if projection.no_view_gws:
            result.notes.append(
                f"the model had too little history for "
                f"{len(projection.no_view_gws)} gameweeks and would not defer to "
                f"this season's expected-points column, which knows who played: "
                f"{sorted(projection.no_view_gws)}"
            )
    result.notes.extend(data.notes)
    return result


def _current_price(
    data: SeasonData, gw: int, pid: int, purchase: dict[int, int]
) -> int:
    """A held player's latest known price, falling back to what was paid."""
    rows = data.player_gws
    rows = rows[(rows["player_id"] == pid) & (rows["gw"] <= gw)]
    if rows.empty:
        return purchase.get(int(pid), 0)
    return int(rows.sort_values("gw").iloc[-1]["price"])


def _squad_value(
    data: SeasonData, gw: int, players: tuple[int, ...], purchase: dict[int, int]
) -> int:
    return sum(_current_price(data, gw, int(p), purchase) for p in players)


def common_window(results: list[BacktestResult]) -> dict[str, set[int]]:
    """Per season, the gameweeks every strategy in ``results`` scored.

    Strategies start in different gameweeks — ``naive`` has no view until a
    gameweek has been played — so their own totals run over different numbers
    of gameweeks. Summing within this window is what makes the comparison a
    comparison.
    """
    by_season: dict[str, list[BacktestResult]] = {}
    for r in results:
        by_season.setdefault(r.season, []).append(r)
    windows: dict[str, set[int]] = {}
    for season, group in by_season.items():
        sets = [{rec.gw for rec in r.records} for r in group]
        windows[season] = set.intersection(*sets) if sets else set()
    return windows


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    """Side-by-side totals, with the gameweek counts that qualify them.

    ``window_points`` is the total over :func:`common_window` — the gameweeks
    every strategy in the comparison decided — and ``vs_naive`` is the gap to
    the naive baseline measured inside it. That is the only comparison worth
    making here: the strategies run through identical machinery, so a gap over
    the same gameweeks is attributable to the projections and nothing else.
    """
    windows = common_window(results)
    naive_window = {}
    for r in results:
        if r.strategy == "naive":
            window = windows.get(r.season, set())
            naive_window[r.season] = sum(
                rec.points for rec in r.records if rec.gw in window
            )

    rows = []
    for r in results:
        window = windows.get(r.season, set())
        window_points = sum(rec.points for rec in r.records if rec.gw in window)
        baseline = naive_window.get(r.season)
        rows.append(
            {
                "season": r.season,
                "strategy": r.strategy,
                "gameweeks": r.gameweeks,
                "decided": len(r.decided_gws),
                "total_points": r.total_points,
                "window_gws": len(window),
                "window_points": window_points,
                "per_gw": round(r.points_per_gameweek, 2),
                "transfers": r.total_transfers,
                "hits": r.total_hits,
                "vs_naive": (
                    None
                    if baseline is None or r.strategy == "naive"
                    else window_points - baseline
                ),
            }
        )
    return pd.DataFrame(rows)
