"""Who is available to own, as opposed to who happens to be playing.

A squad is owned across gameweeks, not rented for one. In a blank gameweek a
club has no fixture, so none of its players appear in that gameweek's rows —
but they are still in your squad, still cost money, and can still be sold next
week. Building the candidate pool from the gameweek's fixtures therefore makes
blanks unsolvable: with two clubs idle there may not be five ownable defenders
in the league, and the program reports infeasible for a gameweek a real manager
played through without difficulty.

So the pool is the *roster*: every player known to exist from gameweeks already
played, carrying their most recent known position, club and price. A player
with no fixture this gameweek is in the pool with zero expected points, which
is exactly the trade-off a manager faces — own him through the blank because of
what he will do afterwards, or sell him.

Leakage
-------
Only two things are read: pre-deadline columns, and gameweeks at or before the
one being decided. Position, club and price are all pre-deadline, and
``latest_known`` takes the most recent row at or before ``gw`` — never a later
one, which would reveal that a player is still in the league.
"""

from __future__ import annotations

import pandas as pd

from ..schema import PRE_DEADLINE_COLUMNS, SeasonData

#: Columns the roster carries through to the candidate frame.
ROSTER_COLUMNS: tuple[str, ...] = ("player_id", "position", "club", "price")


def latest_known(data: SeasonData, gw: int) -> pd.DataFrame:
    """One row per player: their most recent pre-deadline facts up to ``gw``.

    Includes ``gw`` itself, whose pre-deadline columns are published before the
    deadline and so are legal to read.
    """
    frame = data.player_gws[list(PRE_DEADLINE_COLUMNS)]
    frame = frame[frame["gw"] <= gw]
    if frame.empty:
        return pd.DataFrame(columns=list(ROSTER_COLUMNS))
    latest = (
        frame.sort_values(["player_id", "gw", "fixture_id"], kind="stable")
        .drop_duplicates("player_id", keep="last")
        .reset_index(drop=True)
    )
    out = latest[["player_id", "position", "club", "price"]].copy()
    out["player_id"] = out["player_id"].astype(int)
    out["position"] = out["position"].astype(str)
    out["club"] = out["club"].astype(str)
    out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(40).astype(int)
    return out


def build_candidates(
    data: SeasonData,
    gw: int,
    predictions: pd.DataFrame,
    *,
    future: pd.DataFrame | None = None,
    selling_prices: dict[int, int] | None = None,
) -> pd.DataFrame:
    """The frame :func:`~fpl_optimizer.optimize.squad.optimize_squad` consumes.

    Parameters
    ----------
    predictions:
        ``player_id``/``gw``/``xpts`` for ``gw``, as a projection returns. A
        player absent from it — a blank gameweek, or a strategy with no view —
        is kept at zero rather than dropped, so he stays ownable.
    future:
        Optional ``player_id``/``xpts`` of discounted expected points over the
        gameweeks *after* this one. This is what keeps a player who is about to
        have a double gameweek from being sold on the strength of a blank.
    selling_prices:
        What owned players fetch, which after a price rise is not what they
        cost. Defaults to the current price for everyone.
    """
    roster = latest_known(data, gw)
    if roster.empty:
        return pd.DataFrame(
            columns=["player_id", "position", "club", "price", "xpts", "xpts_future"]
        )

    xpts = (
        predictions.set_index("player_id")["xpts"]
        if not predictions.empty
        else pd.Series(dtype="float64")
    )
    out = roster.copy()
    out["xpts"] = (
        out["player_id"].map(xpts).astype(float).fillna(0.0)
    )

    if future is not None and not future.empty:
        future_xpts = future.set_index("player_id")["xpts"]
        out["xpts_future"] = out["player_id"].map(future_xpts).astype(float).fillna(0.0)
    else:
        out["xpts_future"] = 0.0

    out["selling_price"] = out["player_id"].map(
        lambda p: (selling_prices or {}).get(int(p), 0)
    )
    # Anyone not owned has no selling price; use the market price so the column
    # is meaningful for every row.
    unowned = out["selling_price"] <= 0
    out.loc[unowned, "selling_price"] = out.loc[unowned, "price"]
    out["selling_price"] = out["selling_price"].astype(int)

    return out.reset_index(drop=True)


def horizon_predictions(
    projection,
    data: SeasonData,
    gw: int,
    *,
    horizon: int,
    decay: float,
) -> pd.DataFrame:
    """Expected points per gameweek over the ``horizon`` gameweeks after ``gw``.

    Each later gameweek is weighted ``decay ** k``, so the fixtures ahead inform
    who is worth owning without gameweek ``gw + 4`` outweighing the one being
    decided. The projection is *not* refitted here: it was fitted on history
    before ``gw``, and asking it about later gameweeks reads only their
    pre-deadline facts.

    The result is a weighted **average**, not a sum, and that choice is load
    bearing. Summed, a transfer's per-gameweek gain is counted once for this
    gameweek and again for every gameweek in the horizon — three or four times
    over — while the four-point hit is charged once. The program then buys
    almost anything: an early run of this backtest took 88 points of hits in
    ten gameweeks. Worse, the future half of that benefit is not even locked in,
    because next gameweek re-solves and may sell the player straight back.

    Averaged, ``xpts_future`` is on the same scale as ``xpts``: the two terms
    together are worth about two gameweeks, so a transfer has to find roughly
    two points a gameweek to justify a hit. That is about where a human manager
    draws the line too.
    """
    frames = []
    total_weight = 0.0
    for step in range(1, horizon + 1):
        target = gw + step
        if target not in set(data.gameweeks):
            continue
        preds = projection.predict(data, target)
        if preds.empty:
            continue
        weight = decay**step
        total_weight += weight
        frames.append(
            preds.assign(xpts=preds["xpts"] * weight)[["player_id", "xpts"]]
        )
    if not frames or total_weight <= 0:
        return pd.DataFrame(
            {
                "player_id": pd.Series(dtype="int64"),
                "xpts": pd.Series(dtype="float64"),
            }
        )
    summed = (
        pd.concat(frames, ignore_index=True)
        .groupby("player_id", as_index=False)["xpts"]
        .sum()
    )
    summed["xpts"] = summed["xpts"] / total_weight
    return summed
