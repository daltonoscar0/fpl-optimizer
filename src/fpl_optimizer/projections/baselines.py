"""The two baseline projections the JAX model is measured against.

:class:`EPProjection`
    FPL's own published expected points. ``ep_next`` live, the archive's ``xP``
    column for past seasons. Nothing is fitted — this is a read of a number
    someone else computed, which is exactly what makes it a fair baseline.

:class:`NaiveProjection`
    "Whoever has scored most so far will score most next." Total points to date
    divided by fixtures to date, carried forward. A strategy this crude is worth
    running because it is what a manager does by eye, and beating it is the
    minimum bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..schema import SeasonData
from .base import aggregate_to_gw, empty_predictions


@dataclass
class EPProjection:
    """FPL's published expected points, read straight from the data."""

    name: str = "ep"
    #: Gameweeks for which the source carried no expected-points value.
    missing_gws: list[int] = field(default_factory=list)

    def fit(self, data: SeasonData, upto_gw: int) -> None:  # noqa: ARG002
        """Nothing to fit."""
        return None

    def predict(self, data: SeasonData, gw: int) -> pd.DataFrame:
        rows = data.pre_deadline_at(gw)
        if rows.empty:
            return empty_predictions()
        ep = pd.to_numeric(rows["ep"], errors="coerce")
        if ep.isna().all():
            if gw not in self.missing_gws:
                self.missing_gws.append(gw)
            return empty_predictions()
        per_fixture = pd.DataFrame(
            {
                "player_id": rows["player_id"].astype("int64"),
                "gw": int(gw),
                # A missing value for one player in a gameweek that generally
                # has them is treated as zero, not dropped, so the squad is
                # still legal.
                "xpts": ep.fillna(0.0).astype(float),
            }
        )
        return aggregate_to_gw(per_fixture)

    def available_for(self, data: SeasonData, gw: int) -> bool:
        """Whether this season actually publishes expected points for ``gw``."""
        rows = data.pre_deadline_at(gw)
        if rows.empty:
            return False
        return bool(pd.to_numeric(rows["ep"], errors="coerce").notna().any())


#: Fraction of a season's gameweeks that must carry expected points before the
#: EP strategy is worth running at all. Below this the strategy is skipped for
#: the season and reported as skipped rather than run on fragments.
EP_COVERAGE_THRESHOLD = 0.80


def ep_coverage(data: SeasonData) -> tuple[float, list[int]]:
    """``(fraction_covered, gameweeks_missing)`` for the expected-points column."""
    proj = EPProjection()
    gws = data.gameweeks
    if not gws:
        return 0.0, []
    missing = [g for g in gws if not proj.available_for(data, g)]
    return 1.0 - len(missing) / len(gws), missing


#: Fixtures of exposure a player must have before their points-per-fixture rate
#: is taken at face value. Below this the denominator is floored, so a player
#: who has appeared once and hauled is rated on his total rather than on a rate
#: extrapolated from a single match.
#:
#: This is the same move as the one-full-match floor on ``points_per_90`` in
#: :mod:`fpl_optimizer.features`, but the floor has to be more than one fixture
#: to do anything. ``points_per_90`` divides by minutes, so flooring at 90
#: bounds a sub-match sample; here the unit of the denominator already *is* the
#: match, and flooring at one fixture would leave a single-appearance haul
#: reporting itself unchanged. Three fixtures is about a month of football.
MIN_RATE_FIXTURES = 3


@dataclass
class NaiveProjection:
    """Points per fixture so far, carried forward unchanged.

    The rate is per *fixture* rather than per gameweek so that a double
    gameweek is worth two matches, as it is for the other two strategies —
    FPL's ``xP`` and the model both predict a fixture at a time. A per-gameweek
    rate would hand this strategy a different view of doubles than its rivals,
    and the backtest would be comparing two things at once.

    The denominator is floored at :data:`MIN_RATE_FIXTURES` so a player with
    almost no history cannot top the ranking on one good afternoon.
    """

    name: str = "naive"
    #: player_id -> points per fixture over the gameweeks fitted on.
    _ppf: dict[int, float] = field(default_factory=dict, repr=False)

    def fit(self, data: SeasonData, upto_gw: int) -> None:
        hist = data.before(upto_gw)
        self._ppf = {}
        if hist.empty:
            return
        # One row per (player, fixture), so the count is fixtures their club
        # played — a player who sat on the bench all season rates low, which is
        # the behaviour wanted from a baseline this crude.
        agg = hist.groupby("player_id")["points"].agg(["sum", "size"])
        self._ppf = {
            int(pid): float(r["sum"]) / max(int(r["size"]), MIN_RATE_FIXTURES)
            for pid, r in agg.iterrows()
        }

    def predict(self, data: SeasonData, gw: int) -> pd.DataFrame:
        rows = data.pre_deadline_at(gw)
        if rows.empty:
            return empty_predictions()
        per_fixture = pd.DataFrame(
            {
                "player_id": rows["player_id"].astype("int64"),
                "gw": int(gw),
                "xpts": [self._ppf.get(int(pid), 0.0) for pid in rows["player_id"]],
            }
        )
        return aggregate_to_gw(per_fixture)
