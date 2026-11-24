"""Pre-deadline feature construction.

This is the single place where history is turned into numbers a model can
consume, and therefore the single place leakage could enter. The whole module
is built around one function signature:

    build_features(data, gw) -> DataFrame

Everything it reads is either (a) an outcome column from a gameweek *strictly
before* ``gw``, or (b) a pre-deadline column of ``gw`` itself. It never reads an
outcome column at or after ``gw``, and :func:`build_features` asserts that by
slicing the history frame before computing anything.

Opponent strength comes from a rolling record of goals already conceded and
scored, not from FPL's difficulty ratings — for past seasons the archive's
difficulty values are an end-of-season snapshot, so using them would leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .rules import POSITIONS
from .schema import OUTCOME_COLUMNS, SeasonData

#: Feature columns, in the order the model sees them. Keeping this explicit
#: means model coefficients can be printed against readable names.
FEATURE_COLUMNS: tuple[str, ...] = (
    "minutes_mean_3",
    "minutes_mean_6",
    "start_rate_6",
    "played_rate_6",
    "full_game_rate_6",
    "games_available",
    "points_per_90_6",
    "points_mean_3",
    "points_mean_6",
    "bps_per_90_6",
    "price",
    "is_home",
    "opp_conceded_per_game",
    "opp_scored_per_game",
    "own_scored_per_game",
    "own_conceded_per_game",
    "is_gkp",
    "is_def",
    "is_mid",
    "is_fwd",
    "availability",
)

#: Season-level priors used before a player has any history, so gameweek 1 is
#: not a division by zero. Deliberately pessimistic about minutes.
_COLD_START = {
    "minutes_mean_3": 0.0,
    "minutes_mean_6": 0.0,
    "start_rate_6": 0.0,
    "played_rate_6": 0.0,
    "full_game_rate_6": 0.0,
    "games_available": 0.0,
    "points_per_90_6": 0.0,
    "points_mean_3": 0.0,
    "points_mean_6": 0.0,
    "bps_per_90_6": 0.0,
}


@dataclass(frozen=True)
class FeatureFrame:
    """Features for one gameweek, plus the identity columns to join back on.

    ``X`` is strictly numeric and ordered as :data:`FEATURE_COLUMNS`; ``meta``
    carries ``player_id``, ``gw``, ``position``, ``club``, ``price``, ``name``
    and ``fixture_id`` so results can be reported and scored.
    """

    X: pd.DataFrame
    meta: pd.DataFrame

    def __len__(self) -> int:
        return len(self.X)


def rolling_team_form(history: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Goals scored and conceded per game by club, over the last ``window`` games.

    Derived from ``team_h_score``/``team_a_score`` on rows already played, which
    is why this is only ever called with a frame sliced to gameweeks before the
    one being decided. One row per (club, games_played).
    """
    if history.empty:
        return pd.DataFrame(
            columns=["club", "scored_per_game", "conceded_per_game", "games"]
        )

    # Collapse to one row per club-fixture: every player of a club in a fixture
    # carries the same scoreline, so take the first.
    fx = (
        history.dropna(subset=["team_h_score", "team_a_score", "fixture_id"])
        .groupby(["club", "fixture_id"], as_index=False)
        .agg(
            gw=("gw", "first"),
            was_home=("was_home", "first"),
            h=("team_h_score", "first"),
            a=("team_a_score", "first"),
        )
    )
    if fx.empty:
        return pd.DataFrame(
            columns=["club", "scored_per_game", "conceded_per_game", "games"]
        )

    home = fx["was_home"].fillna(False).astype(bool)
    fx["scored"] = np.where(home, fx["h"], fx["a"]).astype(float)
    fx["conceded"] = np.where(home, fx["a"], fx["h"]).astype(float)

    fx = fx.sort_values(["club", "gw", "fixture_id"], kind="stable")
    rows = []
    for club, grp in fx.groupby("club", sort=False):
        tail = grp.tail(window)
        rows.append(
            {
                "club": club,
                "scored_per_game": float(tail["scored"].mean()),
                "conceded_per_game": float(tail["conceded"].mean()),
                "games": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def _player_history_features(
    history: pd.DataFrame, form_window: int
) -> pd.DataFrame:
    """Per-player rolling form from already-played gameweeks."""
    if history.empty:
        return pd.DataFrame(columns=["player_id", *_COLD_START.keys()])

    # A double gameweek gives a player two rows; aggregate to one row per
    # (player, gw) first so "last 6 gameweeks" means gameweeks, not fixtures.
    per_gw = history.groupby(["player_id", "gw"], as_index=False).agg(
        minutes=("minutes", "sum"),
        points=("points", "sum"),
        bps=("bps", "sum"),
        starts=("starts", "sum"),
    )
    per_gw = per_gw.sort_values(["player_id", "gw"], kind="stable")

    rows = []
    for pid, grp in per_gw.groupby("player_id", sort=False):
        w = grp.tail(form_window)
        w3 = grp.tail(3)
        minutes = w["minutes"].astype(float)
        points = w["points"].astype(float)
        total_minutes = float(minutes.sum())
        rows.append(
            {
                "player_id": int(pid),
                "minutes_mean_3": float(w3["minutes"].astype(float).mean()),
                "minutes_mean_6": float(minutes.mean()),
                "start_rate_6": float((w["starts"].astype(float) > 0).mean()),
                "played_rate_6": float((minutes > 0).mean()),
                "full_game_rate_6": float((minutes >= 60).mean()),
                "games_available": float(len(grp)),
                # Denominator floored at one full match so a one-minute cameo
                # that happened to score cannot report 90 points per 90.
                "points_per_90_6": float(points.sum()) / max(total_minutes, 90.0) * 90.0,
                "points_mean_3": float(w3["points"].astype(float).mean()),
                "points_mean_6": float(points.mean()),
                "bps_per_90_6": float(w["bps"].astype(float).sum())
                / max(total_minutes, 90.0)
                * 90.0,
            }
        )
    return pd.DataFrame(rows)


def build_features(
    data: SeasonData,
    gw: int,
    *,
    form_window: int = 6,
    history: pd.DataFrame | None = None,
) -> FeatureFrame:
    """Features for every player with a fixture in ``gw``.

    Parameters
    ----------
    history:
        Override for the history frame, used by tests. Must contain only
        gameweeks before ``gw``; this is checked, not assumed.

    Raises
    ------
    ValueError
        If ``history`` contains a gameweek at or after ``gw``. That check is the
        programmatic statement of the no-leakage rule.
    """
    hist = data.before(gw) if history is None else history
    if len(hist) and int(hist["gw"].max()) >= gw:
        raise ValueError(
            f"history for gameweek {gw} contains gameweek "
            f"{int(hist['gw'].max())}: that would leak the result being predicted"
        )

    # Only pre-deadline columns of the target gameweek. The outcome columns are
    # not merely unused here — they are absent from the frame.
    target = data.pre_deadline_at(gw)
    if target.empty:
        return FeatureFrame(
            X=pd.DataFrame(columns=list(FEATURE_COLUMNS)),
            meta=pd.DataFrame(
                columns=[
                    "player_id",
                    "gw",
                    "name",
                    "position",
                    "club",
                    "price",
                    "fixture_id",
                ]
            ),
        )
    assert not set(target.columns) & set(OUTCOME_COLUMNS), (
        "pre_deadline_at leaked an outcome column"
    )

    player_form = _player_history_features(hist, form_window)
    team_form = rolling_team_form(hist, form_window)

    league_conceded = (
        float(team_form["conceded_per_game"].mean()) if len(team_form) else 1.4
    )
    league_scored = (
        float(team_form["scored_per_game"].mean()) if len(team_form) else 1.4
    )

    df = target.merge(player_form, on="player_id", how="left")
    for col, default in _COLD_START.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    opp = team_form.rename(
        columns={
            "club": "opponent",
            "scored_per_game": "opp_scored_per_game",
            "conceded_per_game": "opp_conceded_per_game",
        }
    )[["opponent", "opp_scored_per_game", "opp_conceded_per_game"]]
    own = team_form.rename(
        columns={
            "club": "club",
            "scored_per_game": "own_scored_per_game",
            "conceded_per_game": "own_conceded_per_game",
        }
    )[["club", "own_scored_per_game", "own_conceded_per_game"]]
    df = df.merge(opp, on="opponent", how="left").merge(own, on="club", how="left")
    df["opp_scored_per_game"] = df["opp_scored_per_game"].fillna(league_scored)
    df["opp_conceded_per_game"] = df["opp_conceded_per_game"].fillna(league_conceded)
    df["own_scored_per_game"] = df["own_scored_per_game"].fillna(league_scored)
    df["own_conceded_per_game"] = df["own_conceded_per_game"].fillna(league_conceded)

    df["is_home"] = df["was_home"].fillna(False).astype(float)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(40).astype(float)
    for pos in POSITIONS:
        df[f"is_{pos.lower()}"] = (df["position"] == pos).astype(float)
    # No injury flag in the archive; absence means "assume available".
    df["availability"] = (
        pd.to_numeric(df["availability"], errors="coerce").fillna(1.0).astype(float)
    )

    X = df[list(FEATURE_COLUMNS)].astype("float64")
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    meta = df[
        ["player_id", "gw", "name", "position", "club", "price", "fixture_id"]
    ].copy()
    return FeatureFrame(X=X.reset_index(drop=True), meta=meta.reset_index(drop=True))


class FeatureCache:
    """Features for every gameweek of a season, each built from its own past.

    A backtest refits the model before every gameweek, and each refit wants the
    features of all earlier gameweeks. Rebuilding them per refit is quadratic;
    building each gameweek once and slicing is not. The cache is what makes a
    38-gameweek walk-forward backtest cheap.

    The leakage property is preserved because entry ``g`` is built by
    :func:`build_features` from gameweeks before ``g`` only. Slicing the cache
    to ``gw < g`` therefore yields exactly the walk-forward training set.
    """

    def __init__(self, data: SeasonData, *, form_window: int = 6) -> None:
        self.data = data
        self.form_window = form_window
        self._frames: dict[int, FeatureFrame] = {}

    def at(self, gw: int) -> FeatureFrame:
        if gw not in self._frames:
            self._frames[gw] = build_features(
                self.data, gw, form_window=self.form_window
            )
        return self._frames[gw]

    def warm(self, gameweeks: list[int] | None = None) -> None:
        for g in gameweeks if gameweeks is not None else self.data.gameweeks:
            self.at(g)

    def training_set(
        self, upto_gw: int, *, min_gw: int = 2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """``(X, y)`` for every gameweek in ``[min_gw, upto_gw)``.

        ``y`` carries ``minutes`` and ``points``, the realized outcomes of
        gameweeks that have already been played by the time ``upto_gw`` is
        decided.
        """
        feats: list[pd.DataFrame] = []
        targets: list[pd.DataFrame] = []
        for g in range(min_gw, upto_gw):
            ff = self.at(g)
            if not len(ff):
                continue
            realized = self.data.at(g)[
                ["player_id", "fixture_id", "minutes", "points"]
            ]
            idx = ff.meta.reset_index()[["index", "player_id", "fixture_id"]]
            joined = idx.merge(realized, on=["player_id", "fixture_id"], how="inner")
            if joined.empty:
                continue
            feats.append(ff.X.iloc[joined["index"].to_numpy()].reset_index(drop=True))
            targets.append(
                joined[["player_id", "minutes", "points"]]
                .assign(gw=g)
                .reset_index(drop=True)
            )
        if not feats:
            return (
                pd.DataFrame(columns=list(FEATURE_COLUMNS)),
                pd.DataFrame(columns=["player_id", "gw", "minutes", "points"]),
            )
        return (
            pd.concat(feats, ignore_index=True),
            pd.concat(targets, ignore_index=True),
        )


def build_training_set(
    data: SeasonData,
    upto_gw: int,
    *,
    form_window: int = 6,
    min_gw: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-shot walk-forward training set. See :meth:`FeatureCache.training_set`."""
    return FeatureCache(data, form_window=form_window).training_set(
        upto_gw, min_gw=min_gw
    )
