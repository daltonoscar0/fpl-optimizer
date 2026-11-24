"""Tracking the current season: call it in advance, then score it.

A backtest can be run again tomorrow with a different answer. A live record
cannot, which makes it the one piece of evidence here that is impossible to
overfit — and only if it is kept honestly. One rule does that work:

    **A gameweek is scored only from a recommendation recorded before its
    deadline.**

So :func:`record` writes the squad the optimizer picked, with the timestamp it
was picked at and the deadline it was picked for, and :func:`score_recorded`
refuses anything else. There is no path that produces a score for a gameweek
nobody called in advance — not for the gameweeks already played this season,
and not for a gameweek whose deadline passed while the command was running.
That is why the table in the README starts empty and fills a row a week rather
than arriving complete.

Each scored gameweek is written next to the figure FPL publishes for the
average of every entered team, which is the only benchmark the live path has
that the archive does not (see :mod:`fpl_optimizer.archive`). Beating it is a
low bar and a real one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config
from .optimize import build_candidates, horizon_predictions, optimize_squad, pick_squad
from .projections import STRATEGIES
from .rules import Rules
from .schema import SeasonData
from .score import Squad, positions_from, score_gameweek

log = logging.getLogger(__name__)

#: Subdirectory of ``results/`` the live record lives in.
LIVE_DIR = "live"


class NotRecordedInAdvance(RuntimeError):
    """Raised when a gameweek has no recommendation from before its deadline.

    The only honest response to "what would you have scored last week" when
    nobody wrote down an answer last week.
    """


@dataclass
class Recommendation:
    """A squad called for a gameweek, and when it was called."""

    season: str
    gw: int
    strategy: str
    #: UTC, when the optimizer was run.
    recorded_at: str
    #: The gameweek's deadline, from the API. The comparison that matters.
    deadline: str | None
    squad: list[int]
    starting: list[int]
    bench: list[int]
    captain: int
    vice: int
    expected_points: float
    transfers_in: list[int] = field(default_factory=list)
    transfers_out: list[int] = field(default_factory=list)
    transfer_hit: int = 0
    #: ``player_id -> name``, so the record is readable a year later.
    names: dict[str, str] = field(default_factory=dict)

    @property
    def in_advance(self) -> bool:
        """Whether this was recorded before the deadline it was recorded for.

        ``None`` deadlines count as not in advance: an unknown deadline cannot
        be shown to be in the future.
        """
        if self.deadline is None:
            return False
        return _parse(self.recorded_at) < _parse(self.deadline)

    def as_squad(self) -> Squad:
        return Squad(
            starting=tuple(self.starting),
            bench=tuple(self.bench),
            captain=self.captain,
            vice=self.vice,
        )


@dataclass
class TrackedGameweek:
    """One live gameweek: what the recommendation scored, and the average."""

    season: str
    gw: int
    strategy: str
    recorded_at: str
    deadline: str | None
    points: int
    average_entry_score: float | None
    starters_points: int
    captain_bonus: int
    bench_points: int
    autosubs: int
    armband: int | None
    vice_used: bool
    transfer_hit: int

    @property
    def delta(self) -> float | None:
        """Points above the average entered team, or ``None`` if unpublished."""
        if self.average_entry_score is None:
            return None
        return self.points - self.average_entry_score


def _parse(stamp: str) -> datetime:
    """Parse an ISO timestamp, treating a naive one as UTC."""
    text = stamp.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass
class TrackPlan:
    """What a run of ``fpl track`` should do, decided before fetching anything.

    Deciding is cheap — it needs the gameweek deadlines, which arrive in one
    request, plus the records already on disk. Acting is expensive: recording
    needs a history for every player in the game, one request each. Separating
    them is what makes running this every few hours reasonable rather than
    thousands of requests a day to discover there is nothing to do.
    """

    season: str
    next_gw: int | None
    next_deadline: str | None
    #: The gameweek to record for, or ``None`` if there is nothing to record.
    record_gw: int | None
    #: Finished gameweeks with a recommendation on record and no score yet.
    score_gws: list[int]
    #: Gameweeks whose deadline passed with nothing on record, after tracking
    #: began. Each one is a permanent hole in the record.
    missed_gws: list[int]
    #: Why there is nothing to record, when there is nothing to record.
    reason: str = ""

    @property
    def nothing_to_do(self) -> bool:
        return self.record_gw is None and not self.score_gws

    @property
    def needs_full_season(self) -> bool:
        """Recording projects every player, so it needs every player's history."""
        return self.record_gw is not None


def plan_track(
    season: str,
    events: pd.DataFrame,
    *,
    recorded_gws: set[int],
    scored_gws: set[int],
    next_gw: int | None,
    now: datetime | str,
    strategy: str = "model",
) -> TrackPlan:
    """Decide what to do, from deadlines and what is already on record.

    Pure: no network, no disk. Every branch the scheduled workflow can take is
    therefore testable without either.
    """
    moment = _parse(now) if isinstance(now, str) else now
    deadlines: dict[int, str | None] = {}
    for row in events.to_dict("records"):
        value = row.get("deadline_time")
        deadlines[int(row["gw"])] = None if pd.isna(value) else str(value)
    finished = {
        int(row["gw"]): bool(row.get("finished"))
        for row in events.to_dict("records")
    }

    def passed(gw: int) -> bool:
        stamp = deadlines.get(gw)
        return stamp is not None and _parse(stamp) <= moment

    # --- what to record ----------------------------------------------------
    record_gw: int | None = None
    reason = ""
    next_deadline = deadlines.get(next_gw) if next_gw is not None else None
    if next_gw is None:
        reason = "no upcoming gameweek"
    elif next_gw in recorded_gws:
        reason = f"GW{next_gw} is already on record"
    elif next_deadline is None:
        reason = f"GW{next_gw} has no published deadline"
    elif passed(next_gw):
        # The record is only worth anything if it predates the deadline, so a
        # late run records nothing rather than something worthless.
        reason = f"GW{next_gw}'s deadline has passed"
    else:
        record_gw = next_gw

    # --- what to score -----------------------------------------------------
    score_gws = sorted(
        gw
        for gw in recorded_gws
        if finished.get(gw) and gw not in scored_gws
    )

    # --- what was missed ---------------------------------------------------
    # Only after tracking began: the gameweeks played before the first
    # recommendation was ever written are not missed, they are prehistory.
    missed_gws: list[int] = []
    if recorded_gws:
        began = min(recorded_gws)
        missed_gws = sorted(
            gw
            for gw in deadlines
            if gw > began and passed(gw) and gw not in recorded_gws
        )

    return TrackPlan(
        season=season,
        next_gw=next_gw,
        next_deadline=next_deadline,
        record_gw=record_gw,
        score_gws=score_gws,
        missed_gws=missed_gws,
        reason=reason,
    )


def live_dir(config: Config | None = None) -> Path:
    directory = (config or Config()).results_dir / LIVE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def recommendation_path(
    season: str, gw: int, strategy: str, config: Config | None = None
) -> Path:
    return live_dir(config) / f"{season}-gw{gw:02d}-{strategy}.json"


def tracking_path(season: str, config: Config | None = None) -> Path:
    return live_dir(config) / f"{season}.csv"


def deadline_for(data: SeasonData, gw: int) -> str | None:
    row = data.events[data.events["gw"] == gw]
    if row.empty:
        return None
    value = row.iloc[0].get("deadline_time")
    return None if pd.isna(value) else str(value)


def is_finished(data: SeasonData, gw: int) -> bool:
    row = data.events[data.events["gw"] == gw]
    if row.empty:
        return False
    return bool(row.iloc[0].get("finished"))


def recommend(
    data: SeasonData,
    gw: int,
    strategy: str,
    rules: Rules,
    *,
    config: Config | None = None,
    current: Squad | None = None,
    free_transfers: int = 1,
    bank: int = 0,
    now: str | None = None,
) -> Recommendation:
    """Run the optimizer for ``gw`` and return the squad it recommends.

    Reads only pre-deadline facts, exactly as the backtest does — this is the
    same code path, which is the point of having one.
    """
    config = config or Config()
    projection = STRATEGIES[strategy]()
    projection.fit(data, gw)
    preds = projection.predict(data, gw)
    if preds.empty or float(preds["xpts"].abs().max()) == 0.0:
        raise NotRecordedInAdvance(
            f"{strategy} has no projection for GW{gw}, so there is nothing to "
            f"recommend. Nothing is written."
        )

    future = horizon_predictions(
        projection, data, gw, horizon=config.horizon, decay=config.horizon_decay
    )
    candidates = build_candidates(data, gw, preds, future=future)
    if current is None:
        plan = pick_squad(candidates, rules, config=config)
    else:
        plan = optimize_squad(
            candidates,
            rules,
            config=config,
            current=current.players,
            free_transfers=free_transfers,
            bank=bank,
        )

    names = (
        data.player_gws.drop_duplicates("player_id")
        .set_index("player_id")["name"]
        .astype(str)
    )
    return Recommendation(
        season=data.season,
        gw=gw,
        strategy=strategy,
        recorded_at=now or datetime.now(timezone.utc).isoformat(),
        deadline=deadline_for(data, gw),
        squad=[int(p) for p in plan.squad.players],
        starting=[int(p) for p in plan.squad.starting],
        bench=[int(p) for p in plan.squad.bench],
        captain=int(plan.squad.captain),
        vice=int(plan.squad.vice),
        expected_points=round(float(plan.expected_points), 3),
        transfers_in=[int(p) for p in plan.transfers_in],
        transfers_out=[int(p) for p in plan.transfers_out],
        transfer_hit=int(plan.transfer_hit),
        names={str(p): str(names.get(p, "?")) for p in plan.squad.players},
    )


def record(rec: Recommendation, config: Config | None = None) -> Path:
    """Write a recommendation to disk, once, and only ahead of its deadline.

    Two refusals, and they are the whole contract of the live record:

    *Already on record* — overwriting would destroy the only thing that makes
    the file worth anything, that it was written before the result was known.
    This is also what makes the command idempotent: run it six times a day and
    at most one file per gameweek is ever written.

    *Not ahead of the deadline* — a recommendation written at or after kick-off
    is not a prediction, and storing one would put a row in the table that
    cannot be distinguished later from a real one.
    """
    path = recommendation_path(rec.season, rec.gw, rec.strategy, config)
    if path.exists():
        raise FileExistsError(
            f"a recommendation for GW{rec.gw} is already on record at {path}; "
            f"it will not be overwritten"
        )
    if not rec.in_advance:
        raise NotRecordedInAdvance(
            f"the GW{rec.gw} recommendation was made at {rec.recorded_at}, which "
            f"is not before its deadline ({rec.deadline}): it will not be "
            f"written, because a record that late is worth nothing"
        )
    path.write_text(json.dumps(asdict(rec), indent=2) + "\n")
    log.info("recorded GW%d recommendation to %s", rec.gw, path)
    return path


def load_recommendation(
    season: str, gw: int, strategy: str, config: Config | None = None
) -> Recommendation | None:
    path = recommendation_path(season, gw, strategy, config)
    if not path.exists():
        return None
    return Recommendation(**json.loads(path.read_text()))


def score_recorded(
    data: SeasonData,
    rec: Recommendation,
    rules: Rules,
    *,
    positions: dict[int, str] | None = None,
) -> TrackedGameweek:
    """Score a recorded recommendation against realized results.

    Raises
    ------
    NotRecordedInAdvance
        If the recommendation was written at or after its own deadline. Scoring
        it would report a result the optimizer did not actually call.
    """
    if not rec.in_advance:
        raise NotRecordedInAdvance(
            f"the GW{rec.gw} recommendation was recorded at {rec.recorded_at}, "
            f"which is not before its deadline ({rec.deadline}): it cannot be "
            f"scored"
        )
    if not is_finished(data, rec.gw):
        raise NotRecordedInAdvance(
            f"GW{rec.gw} is not finished yet, so there is nothing to score"
        )

    score = score_gameweek(
        rec.as_squad(),
        data,
        rec.gw,
        rules,
        transfer_hit=rec.transfer_hit,
        positions=positions or positions_from(data),
    )
    return TrackedGameweek(
        season=rec.season,
        gw=rec.gw,
        strategy=rec.strategy,
        recorded_at=rec.recorded_at,
        deadline=rec.deadline,
        points=score.points,
        average_entry_score=data.average_score(rec.gw),
        starters_points=score.starters_points,
        captain_bonus=score.captain_bonus,
        bench_points=score.bench_points,
        autosubs=len(score.autosubs),
        armband=score.armband,
        vice_used=score.vice_used,
        transfer_hit=score.transfer_hit,
    )


def append_tracked(
    tracked: TrackedGameweek, config: Config | None = None
) -> Path:
    """Add a scored gameweek to the season's record, replacing any prior row."""
    path = tracking_path(tracked.season, config)
    row = {**asdict(tracked), "delta": tracked.delta}
    frame = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[
            ~(
                (existing["gw"] == tracked.gw)
                & (existing["strategy"] == tracked.strategy)
            )
        ]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = frame.sort_values(["gw", "strategy"], kind="stable")
    frame.to_csv(path, index=False)
    return path


def tracked_frame(season: str, config: Config | None = None) -> pd.DataFrame:
    path = tracking_path(season, config)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "season",
                "gw",
                "strategy",
                "points",
                "average_entry_score",
                "delta",
            ]
        )
    return pd.read_csv(path)


def markdown_table(frame: pd.DataFrame) -> str:
    """The live record as Markdown, for the README.

    Empty until a gameweek has been both called in advance and played, which is
    the honest starting state rather than a gap to be filled in.
    """
    lines = [
        "| GW | Recommended XI scored | Average manager | Difference |",
        "|---|---|---|---|",
    ]
    if frame.empty:
        lines.append("| — | *no gameweek called in advance yet* | — | — |")
        return "\n".join(lines)

    for row in frame.sort_values("gw").to_dict("records"):
        average = row.get("average_entry_score")
        delta = row.get("delta")
        lines.append(
            f"| {int(row['gw'])} | {int(row['points'])} | "
            f"{'—' if pd.isna(average) else int(average)} | "
            f"{'—' if pd.isna(delta) else format(int(delta), '+d')} |"
        )
    scored = frame.dropna(subset=["average_entry_score"])
    if len(scored):
        total = int(scored["points"].sum())
        average_total = int(scored["average_entry_score"].sum())
        lines.append(
            f"| **{len(scored)} GWs** | **{total}** | **{average_total}** | "
            f"**{total - average_total:+d}** |"
        )
    return "\n".join(lines)
