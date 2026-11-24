"""The two baseline projections the JAX model is measured against.

:class:`EPProjection`
    FPL's own published expected points. ``ep_next`` live, the archive's ``xP``
    column for past seasons. Nothing is fitted — it is a read of a number
    someone else computed.

    It was meant to be the fair baseline, and on live data it is. On the
    archive it is not: see :func:`ep_looks_like_hindsight`. The column there
    already knows who played, so it cannot be compared against strategies that
    had to guess.

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
from .base import PREDICTION_COLUMNS, aggregate_to_gw, empty_predictions


@dataclass
class EPProjection:
    """FPL's published expected points, read straight from the data.

    Per gameweek, not per fixture, which is different from the other two
    strategies and not a choice made here. FPL publishes one ``ep_next`` per
    player; the archive's ``xP`` is the same figure, and in every one of the
    1,278 double-gameweek player-gameweeks across 2023-24 and 2024-25 it is
    *identical* on both fixture rows. Summing it would double a number that was
    only ever quoted once, inventing expected points FPL never published.

    So this baseline systematically undervalues a double gameweek, and there is
    no honest way to fix that from the data: the published figure simply does
    not say anything per fixture. It is recorded as a limitation of the source
    rather than patched over.
    """

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
        per_gw = pd.DataFrame(
            {
                "player_id": rows["player_id"].astype("int64"),
                "gw": int(gw),
                # A missing value for one player in a gameweek that generally
                # has them is treated as zero, not dropped, so the squad is
                # still legal.
                "xpts": ep.fillna(0.0).astype(float),
            }
        )
        # `max`, not `sum`: see the class docstring. The figure is identical on
        # both rows of a double gameweek, so taking it once is taking it as
        # published.
        return (
            per_gw.groupby(["player_id", "gw"], as_index=False)["xpts"]
            .max()
            .astype({"player_id": "int64", "gw": "int64"})[list(PREDICTION_COLUMNS)]
        )

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


def ep_looks_like_hindsight(data: SeasonData) -> tuple[float, float]:
    """``(share_of_absences_rated_zero, mean_rating_of_absences)``.

    A pre-deadline forecast cannot know who will not play. FPL publishes an
    ``ep_next`` for every registered player, including the nailed starter who
    turns out to be a surprise omission — so a forecast's ratings for players
    who went on to play no minutes should look much like its ratings for
    everyone else, minus whatever the injury flags implied.

    In the archive they do not. Measured over 2023-24 and 2024-25, two thirds
    of player-gameweeks with zero minutes carry ``xP`` of exactly 0.0, against
    1.3% of the player-gameweeks where someone played an hour or more. Narrowed
    to players who had started each of the three previous gameweeks — whom no
    forecast could know were about to be dropped — it is 31% against 0.3%.

    What that proves is weaker than it first looks, and worth stating exactly:
    pre-deadline injury and suspension news would produce much of the same
    pattern, since a starter ruled out on the Friday genuinely has an expected
    return near zero. This statistic cannot separate "recorded after the fact"
    from "a forecast with good team news".

    What it does establish is that the column tracks realized minutes closely
    enough that a strategy following it scores past any human manager, and so
    its total cannot be set beside strategies that had to guess. The question
    of mechanism is settled by a genuinely pre-deadline snapshot of the same
    field, which :mod:`fpl_optimizer.track` now collects weekly.
    """
    frame = data.player_gws[["minutes", "ep"]]
    minutes = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0)
    ep = pd.to_numeric(frame["ep"], errors="coerce")
    absent = (minutes == 0) & ep.notna()
    if not absent.any():
        return 0.0, 0.0
    return float((ep[absent] == 0).mean()), float(ep[absent].mean())


#: Share of non-appearances rated exactly zero above which the published
#: expected-points column is treated as knowing who played. A genuine forecast
#: is nowhere near this; the archive's seasons measure 0.65 and 0.66.
EP_HINDSIGHT_THRESHOLD = 0.40


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

    The rate is per *fixture* rather than per gameweek, so a double gameweek is
    worth two matches — as it is for the model, which projects a fixture at a
    time. (FPL's published figure cannot do this; see :class:`EPProjection`.) A
    per-gameweek rate would hand this strategy a different view of doubles than
    the model's, and the backtest would be comparing two things at once.

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
