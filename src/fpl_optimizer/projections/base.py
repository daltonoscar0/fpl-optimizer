"""The projection interface every strategy implements.

A projection answers one question: *how many points will this player score in
this gameweek?* It is fitted on gameweeks before the one being decided and then
asked for predictions. Keeping the interface this narrow is what lets the
backtester run three strategies through identical machinery, so differences in
the results come from the projections and nothing else.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from ..schema import SeasonData

#: Columns a projection must return from :meth:`Projection.predict`.
PREDICTION_COLUMNS: tuple[str, ...] = ("player_id", "gw", "xpts")


@runtime_checkable
class Projection(Protocol):
    """Expected points per player per gameweek."""

    name: str

    def fit(self, data: SeasonData, upto_gw: int) -> None:
        """Train using gameweeks strictly before ``upto_gw``.

        Implementations must not read outcome columns at or after ``upto_gw``.
        """

    def predict(self, data: SeasonData, gw: int) -> pd.DataFrame:
        """Expected points for every player with a fixture in ``gw``.

        Returns one row per player (fixtures within a double gameweek summed),
        with columns :data:`PREDICTION_COLUMNS`. Players with no fixture are
        absent, which the optimizer reads as zero available points.
        """


def aggregate_to_gw(per_fixture: pd.DataFrame) -> pd.DataFrame:
    """Sum per-fixture predictions into one row per player.

    A double gameweek gives a player two fixtures and therefore two chances to
    score; a blank gameweek gives none.
    """
    if per_fixture.empty:
        return pd.DataFrame(columns=list(PREDICTION_COLUMNS))
    out = (
        per_fixture.groupby(["player_id", "gw"], as_index=False)["xpts"]
        .sum()
        .astype({"player_id": "int64", "gw": "int64"})
    )
    return out[list(PREDICTION_COLUMNS)]


def empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": pd.Series(dtype="int64"),
            "gw": pd.Series(dtype="int64"),
            "xpts": pd.Series(dtype="float64"),
        }
    )
