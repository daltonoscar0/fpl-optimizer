"""The ``fpl`` command line.

Commands are added as the layers behind them land, so what the CLI exposes is
an honest inventory of what the project can actually do:

``fpl seasons``
    Which archive seasons this project backtests, and which are on disk.
``fpl rules``
    The rule set in force for a season, and where each number came from.
``fpl data``
    Load a season and print its shape — and its notes, which are where every
    known gap in the data is recorded.
``fpl project``
    Run one projection strategy for one gameweek and show the top of its board.
``fpl backtest``
    Walk a season forward, deciding before each gameweek and scoring after it.
``fpl table``
    Turn the backtest summary into the Markdown table in the README.
``fpl ingest``
    Build finished gameweeks of the current season into the archive's format,
    so a model can train on this season walk-forward.
``fpl transfers``
    Recommend transfers for a real team, read from the public endpoints.
``fpl track``
    Before the deadline: record a squad per strategy and FPL's published
    expected points for every player. After the gameweek: every player's
    realized points, and each squad's score against the average manager.

Every command prints the source and the notes rather than only the numbers. A
projection that is quietly running on a fallback, or a season whose expected
points were never collected, is the kind of thing that turns into a wrong
conclusion three steps later if it is not said out loud.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .api import FPLClient
from .archive import available_seasons, load_season
from .backtest import BacktestResult, compare
from .backtest import backtest as run_backtest
from .config import Config
from .entry import EntryUnavailable, load_entry, resolve_players
from .ingest import (
    IngestFailed,
    season_with_upcoming,
    append_gameweek,
    ingest_gameweek,
    ingested_gameweeks,
    load_ingested_season,
    write_provenance,
    write_reference,
)
from .features import FeatureCache
from .live import (
    current_season,
    events_from_bootstrap,
    load_live_season,
    next_gw,
    season_from_event_live,
)
from .optimize import (
    InfeasibleSquad,
    SquadPlan,
    build_candidates,
    horizon_predictions,
    optimize_squad,
)
from .projections import STRATEGIES
from .projections.baselines import EP_COVERAGE_THRESHOLD, ep_coverage
from .rules import archive_season_rules, rules_from_bootstrap
from .schema import SeasonData, describe_frame
from .score import positions_from
from .track import (
    NotRecordedInAdvance,
    append_tracked,
    load_recommendation,
    load_snapshot,
    markdown_table,
    outcomes_from,
    outcomes_path,
    plan_track,
    record,
    recommend,
    score_recorded,
    snapshot_from_bootstrap,
    tracked_frame,
    write_outcomes,
    write_snapshot,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Squad and transfer optimizer for Fantasy Premier League.",
)
console = Console()

#: Expected-points margin below which a recommendation is not worth making.
#: The projections carry far more error than this, so a tenth of a point of
#: advantage is noise with a decimal point on it.
THIN_MARGIN = 1.0


def _verbose(flag: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if flag else logging.WARNING,
        format="%(message)s",
    )


def _print_notes(data: SeasonData) -> None:
    if not data.notes:
        return
    console.print("\n[bold]notes[/bold]")
    for note in data.notes:
        console.print(f"  • {note}")


@app.command()
def seasons() -> None:
    """List the archive seasons this project backtests."""
    config = Config()
    table = Table(title="archive seasons")
    table.add_column("season")
    table.add_column("on disk")
    table.add_column("path")
    for season in available_seasons():
        directory = config.raw_dir / season
        present = directory.is_dir() and any(directory.glob("*.csv"))
        table.add_row(
            season,
            "[green]yes[/green]" if present else "[yellow]no[/yellow]",
            str(directory),
        )
    console.print(table)
    console.print(
        "\nRun [bold]fpl data <season>[/bold] to fetch and summarize one; "
        "files already present are never refetched."
    )


@app.command()
def rules(season: str = typer.Argument(..., help="e.g. 2024-25")) -> None:
    """Show the rule set in force for a season."""
    rule_set = archive_season_rules(season)
    rule_set.validate()

    table = Table(title=f"{season} rules ({rule_set.source})")
    table.add_column("rule")
    table.add_column("value", justify="right")
    table.add_row("squad size", str(rule_set.squad_size))
    table.add_row("starting XI", str(rule_set.squad_play))
    table.add_row("budget", f"{rule_set.budget / 10:.1f}m")
    table.add_row("players per club", str(rule_set.team_limit))
    table.add_row("free transfers banked", str(rule_set.max_free_transfers))
    table.add_row("transfer hit", f"-{rule_set.transfer_hit_cost}")
    table.add_row("captain multiplier", f"x{rule_set.captain_multiplier}")
    console.print(table)

    formation = Table(title="formation")
    formation.add_column("position")
    formation.add_column("in squad", justify="right")
    formation.add_column("XI min", justify="right")
    formation.add_column("XI max", justify="right")
    for code, rule in rule_set.positions.items():
        formation.add_row(
            code, str(rule.squad_select), str(rule.min_play), str(rule.max_play)
        )
    console.print(formation)


@app.command()
def data(
    season: str = typer.Argument(..., help="e.g. 2024-25"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Load a season from the archive and summarize it."""
    _verbose(verbose)
    loaded = load_season(season)
    summary = describe_frame(loaded.player_gws)

    table = Table(title=f"{season} ({loaded.source})")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("player-gameweek rows", f"{summary['rows']:,}")
    table.add_row("players", str(summary["players"]))
    table.add_row("clubs", str(summary["clubs"]))
    table.add_row("gameweeks", str(len(summary["gameweeks"])))
    covered, missing = ep_coverage(loaded)
    table.add_row("FPL expected points coverage", f"{covered:.0%}")
    console.print(table)

    if covered < EP_COVERAGE_THRESHOLD:
        console.print(
            f"\n[yellow]The FPL expected-points baseline covers {covered:.0%} of "
            f"gameweeks, below the {EP_COVERAGE_THRESHOLD:.0%} threshold, so it "
            f"is not comparable over this season as a whole.[/yellow]\n"
            f"Missing: {missing}"
        )
    _print_notes(loaded)


@app.command()
def project(
    season: str = typer.Argument(..., help="e.g. 2024-25"),
    gw: int = typer.Argument(..., help="gameweek to project"),
    strategy: str = typer.Option("model", "--strategy", "-s"),
    top: int = typer.Option(15, "--top", "-n", help="how many players to show"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Project expected points for one gameweek."""
    _verbose(verbose)
    if strategy not in STRATEGIES:
        raise typer.BadParameter(
            f"unknown strategy {strategy!r}: choose from {sorted(STRATEGIES)}"
        )
    loaded = load_season(season)
    projection = STRATEGIES[strategy]()
    projection.fit(loaded, gw)
    preds = projection.predict(loaded, gw)

    if preds.empty:
        console.print(
            f"[yellow]No projection for gameweek {gw}: the strategy reported "
            f"itself unavailable, or no fixtures exist.[/yellow]"
        )
        _print_notes(loaded)
        raise typer.Exit(code=0)

    meta = (
        loaded.pre_deadline_at(gw)
        .drop_duplicates("player_id")
        .set_index("player_id")[["name", "position", "club", "price"]]
    )
    board = preds.join(meta, on="player_id").nlargest(top, "xpts")

    table = Table(title=f"{season} GW{gw} — {strategy}")
    table.add_column("player")
    table.add_column("pos")
    table.add_column("club")
    table.add_column("price", justify="right")
    table.add_column("xPts", justify="right")
    # Iterate over records rather than itertuples: one of the columns is
    # literally called `name`, which itertuples cannot expose as an attribute.
    for row in board.to_dict("records"):
        table.add_row(
            str(row["name"]),
            str(row["position"]),
            str(row["club"]),
            f"{row['price'] / 10:.1f}",
            f"{row['xpts']:.2f}",
        )
    console.print(table)

    fallback = getattr(projection, "fallback_gws", [])
    if gw in fallback:
        console.print(
            "\n[yellow]The model had too little history for this gameweek and "
            "deferred to FPL's published expected points.[/yellow]"
        )
    _print_notes(loaded)


@app.command()
def backtest(
    season: str = typer.Argument(..., help="e.g. 2024-25, or 'all'"),
    strategy: str = typer.Option("all", "--strategy", "-s", help="or 'all'"),
    gw_from: int | None = typer.Option(None, "--from", help="first gameweek"),
    gw_to: int | None = typer.Option(None, "--to", help="last gameweek"),
    out: Path | None = typer.Option(None, "--out", help="where to write results"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Walk forward through a season, deciding before each gameweek.

    Writes one CSV per season and strategy, plus a summary JSON that
    ``fpl table`` turns into the README's results table.
    """
    _verbose(verbose)
    config = Config()
    results_dir = out or config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    seasons = list(available_seasons()) if season == "all" else [season]
    strategies = sorted(STRATEGIES) if strategy == "all" else [strategy]
    for name in strategies:
        if name not in STRATEGIES:
            raise typer.BadParameter(
                f"unknown strategy {name!r}: choose from {sorted(STRATEGIES)}"
            )

    results: list[BacktestResult] = []
    for season_name in seasons:
        loaded = load_season(season_name)
        rule_set = archive_season_rules(season_name)
        # One feature cache per season, shared by every strategy that wants it.
        cache = FeatureCache(loaded, form_window=config.form_window)
        coverage, _ = ep_coverage(loaded)

        for name in strategies:
            if name == "ep" and coverage < EP_COVERAGE_THRESHOLD:
                console.print(
                    f"[yellow]{season_name} {name}: skipped — FPL expected points "
                    f"cover {coverage:.0%} of gameweeks, below the "
                    f"{EP_COVERAGE_THRESHOLD:.0%} threshold.[/yellow]"
                )
                continue
            with console.status(f"{season_name} {name}…"):
                result = run_backtest(
                    loaded,
                    name,
                    rule_set,
                    config=config,
                    start_gw=gw_from,
                    end_gw=gw_to,
                    cache=cache,
                )
            if not result.records:
                console.print(
                    f"[yellow]{season_name} {name}: no gameweeks decided.[/yellow]"
                )
                continue
            results.append(result)
            path = results_dir / f"{season_name}-{name}.csv"
            result.frame().to_csv(path, index=False)
            console.print(
                f"{season_name} [bold]{name}[/bold]: {result.total_points} points "
                f"over {result.gameweeks} gameweeks "
                f"({len(result.decided_gws)} decided) → {path.name}"
            )

    if not results:
        console.print("[yellow]Nothing to report.[/yellow]")
        raise typer.Exit(code=0)

    table = compare(results)
    summary_path = results_dir / "summary.json"
    summary_path.write_text(
        json.dumps([r.summary() for r in results], indent=2) + "\n"
    )

    columns = (
        "season",
        "strategy",
        "gameweeks",
        "decided",
        "total_points",
        "window_gws",
        "window_points",
        "per_gw",
        "transfers",
        "hits",
        "comparable",
        "vs_naive",
    )
    display = Table(title="backtest")
    for column in columns:
        display.add_column(
            column, justify="right" if column != "strategy" else "left"
        )
    for row in table.to_dict("records"):
        display.add_row(
            *["—" if row[c] is None else str(row[c]) for c in columns]
        )
    console.print(display)
    console.print(f"\nSummary written to {summary_path}")


@app.command()
def table(
    results_dir: Path | None = typer.Option(None, "--results", help="results dir"),
) -> None:
    """Print the results table as Markdown, for pasting into the README."""
    config = Config()
    path = (results_dir or config.results_dir) / "summary.json"
    if not path.exists():
        raise typer.BadParameter(
            f"no summary at {path}: run `fpl backtest all` first"
        )
    summaries = json.loads(path.read_text())

    # The window is the gameweeks every strategy in a season scored, so the
    # comparison has one denominator. Strategies start in different gameweeks.
    windows: dict[str, set[int]] = {}
    for entry in summaries:
        scored = set(entry["scored_gameweeks"])
        windows[entry["season"]] = (
            scored
            if entry["season"] not in windows
            else windows[entry["season"]] & scored
        )

    # A strategy reading a column that already knows who played is not a
    # baseline, and putting it in a comparison table implies it is one. It is
    # omitted here and the reason is printed, the same treatment the
    # average-manager row gets for being unavailable.
    omitted = [entry for entry in summaries if not entry.get("comparable", True)]
    omitted_keys = {(e["season"], e["strategy"]) for e in omitted}
    summaries = [
        e for e in summaries if (e["season"], e["strategy"]) not in omitted_keys
    ]

    lines = [
        "| Season | Strategy | Gameweeks decided | Points in common window | "
        "vs `naive` |",
        "|---|---|---|---|---|",
    ]
    rows = []
    for entry in summaries:
        window = windows[entry["season"]]
        path_csv = (results_dir or config.results_dir) / (
            f"{entry['season']}-{entry['strategy']}.csv"
        )
        frame = pd.read_csv(path_csv)
        window_points = int(frame.loc[frame["gw"].isin(window), "points"].sum())
        rows.append((entry, window, window_points))

    baselines = {
        entry["season"]: points
        for entry, _, points in rows
        if entry["strategy"] == "naive"
    }
    for entry, window, window_points in rows:
        baseline = baselines.get(entry["season"])
        if entry["strategy"] == "naive" or baseline is None:
            delta = "—"
        else:
            delta = f"{window_points - baseline:+d}"
        decided = f"{entry['decided_gameweeks']} of {entry['gameweeks']}"
        lines.append(
            f"| {entry['season']} | `{entry['strategy']}` | {decided} | "
            f"{window_points} over {len(window)} GWs | {delta} |"
        )

    if omitted:
        lines.append("")
        for entry in sorted(omitted, key=lambda e: (e["season"], e["strategy"])):
            lines.append(
                f"`{entry['strategy']}` is omitted for {entry['season']} — "
                f"{entry['not_comparable_because']}. It scored "
                f"{entry['total_points']} "
                f"({entry['points_per_gameweek']} a gameweek), against a human "
                f"record nearer 71."
            )
    print("\n".join(lines))


@app.command()
def track(
    strategies: str = typer.Option(
        "model,naive", "--strategies", "-s", help="comma-separated"
    ),
    gw: int | None = typer.Option(
        None, "--gw", help="gameweek to record for; default is the next one"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="decide, write nothing"),
    status_file: Path | None = typer.Option(
        None, "--status-file", help="write a JSON summary here, for automation"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Record this gameweek before its deadline, and score the ones played.

    Safe to run on a schedule. Deciding what to do costs one request: the
    gameweek deadlines, against the records already on disk. Only a run with
    something to do fetches anything more.

    What a gameweek's record contains, all captured at the same pre-deadline
    moment: a squad per strategy, and FPL's own published expected points for
    every player. Afterwards: every player's realized points, and the average
    score of every team entered worldwide.

    Idempotent by construction. Each piece is written at most once, only while
    the deadline is still ahead; everything else is a no-op.
    """
    _verbose(verbose)
    config = Config()
    wanted = [name.strip() for name in strategies.split(",") if name.strip()]
    for name in wanted:
        if name not in STRATEGIES:
            raise typer.BadParameter(
                f"unknown strategy {name!r}: choose from {sorted(STRATEGIES)}"
            )

    # --- decide: one request ----------------------------------------------
    client = FPLClient(config=config)
    bootstrap = client.bootstrap_static()
    season = current_season(bootstrap)
    events = events_from_bootstrap(bootstrap)
    console.print(
        f"FPL API reachable: {len(bootstrap['elements'])} players, "
        f"{len(events)} gameweeks, season {season}"
    )

    all_gws = [int(row["gw"]) for row in events.to_dict("records")]
    recorded = {
        name: {
            g
            for g in all_gws
            if load_recommendation(season, g, name, config) is not None
        }
        for name in wanted
    }
    existing = tracked_frame(season, config)
    scored = (
        set()
        if existing.empty
        else {(int(r["gw"]), str(r["strategy"])) for r in existing.to_dict("records")}
    )
    plan = plan_track(
        season,
        events,
        recorded=recorded,
        scored=scored,
        snapshot_gws={g for g in all_gws if load_snapshot(season, g, config) is not None},
        outcome_gws={g for g in all_gws if outcomes_path(season, g, config).exists()},
        next_gw=gw if gw is not None else next_gw(bootstrap),
        now=datetime.now(timezone.utc),
        strategies=tuple(wanted),
    )
    if plan.next_gw is not None:
        console.print(
            f"next gameweek GW{plan.next_gw}, deadline {plan.next_deadline}"
        )

    status: dict[str, object] = {
        "season": season,
        "strategies": wanted,
        "next_gw": plan.next_gw,
        "next_deadline": plan.next_deadline,
        "recorded": [],
        "snapshot": None,
        "scored": [],
        "outcomes": [],
        "missed": [f"{g}:{label}" for g, label in plan.missed],
        "api_players": len(bootstrap["elements"]),
        "reason": plan.reason,
    }

    def finish() -> None:
        if status_file is not None:
            status_file.write_text(json.dumps(status, indent=2) + "\n")
        console.print("\n" + markdown_table(tracked_frame(season, config)))
        if plan.missed:
            console.print(
                f"\n[red]Nothing was recorded in advance for "
                f"{', '.join(f'GW{g} ({label})' for g, label in plan.missed)}, "
                f"whose deadlines have passed. Those cannot be scored and are a "
                f"permanent gap.[/red]"
            )

    if plan.nothing_to_do:
        console.print(f"Nothing to do: {plan.reason or 'everything is up to date'}.")
        finish()
        raise typer.Exit(code=0)

    rule_set = rules_from_bootstrap(bootstrap, config)
    rule_set.validate()
    now_stamp = datetime.now(timezone.utc).isoformat()

    # --- the snapshot: free, since bootstrap is already in hand ------------
    if plan.record_snapshot and plan.next_gw is not None:
        frame = snapshot_from_bootstrap(
            bootstrap, plan.next_gw, recorded_at=now_stamp, deadline=plan.next_deadline
        )
        if dry_run:
            console.print(
                f"[dim]--dry-run: not writing the GW{plan.next_gw} "
                f"published-expected-points snapshot ({len(frame)} players)[/dim]"
            )
        else:
            path = write_snapshot(frame, season, plan.next_gw, config)
            console.print(
                f"Recorded FPL's published expected points for {len(frame)} "
                f"players → {path.name}"
            )
        status["snapshot"] = plan.next_gw

    # --- results: one request for the whole gameweek -----------------------
    for target in plan.result_gws:
        results = season_from_event_live(bootstrap, client.event_live(target), target)
        positions = positions_from(results)

        if target in plan.outcomes_due:
            outcomes = outcomes_from(results, target)
            if not dry_run:
                path = write_outcomes(outcomes, season, target, config)
                console.print(
                    f"GW{target}: realized points for {len(outcomes)} players "
                    f"→ {path.name}"
                )
            status["outcomes"] = list(status["outcomes"]) + [target]

        for scored_gw, name in plan.score:
            if scored_gw != target:
                continue
            rec = load_recommendation(season, target, name, config)
            if rec is None:  # pragma: no cover - the plan guarantees one
                continue
            try:
                tracked = score_recorded(
                    results, rec, rule_set, positions=positions
                )
            except NotRecordedInAdvance as exc:
                console.print(f"[yellow]GW{target} ({name}): {exc}[/yellow]")
                continue
            if not dry_run:
                append_tracked(tracked, config)
            status["scored"] = list(status["scored"]) + [f"{target}:{name}"]
            average = tracked.average_entry_score
            console.print(
                f"GW{target} [bold]{name}[/bold]: scored "
                f"[bold]{tracked.points}[/bold], average manager "
                f"{'unpublished' if average is None else int(average)}"
                + (
                    ""
                    if tracked.delta is None
                    else f" ([bold]{tracked.delta:+.0f}[/bold])"
                )
            )

    # --- record the next gameweek: the one expensive path ------------------
    if plan.record_strategies:
        console.print(
            f"Recording {', '.join(plan.record_strategies)} needs a projection "
            f"for every player, so every player's history (one request each)…"
        )
        with console.status("loading") as spinner:
            loaded = load_live_season(
                client=client,
                config=config,
                bootstrap=bootstrap,
                fixtures=client.fixtures(),
                progress=lambda done, total: spinner.update(
                    f"player histories {done}/{total}"
                ),
            )
        for name in plan.record_strategies:
            previous = _latest_recommendation(season, name, plan.next_gw, config)
            try:
                rec = recommend(
                    loaded,
                    plan.next_gw,
                    name,
                    rule_set,
                    config=config,
                    current=previous.as_squad() if previous else None,
                )
                if dry_run:
                    console.print(
                        f"[dim]--dry-run: not writing GW{plan.next_gw} "
                        f"({name})[/dim]"
                    )
                else:
                    path = record(rec, config)
                    console.print(f"Recorded GW{plan.next_gw} ({name}) → {path.name}")
                status["recorded"] = list(status["recorded"]) + [name]
            except (NotRecordedInAdvance, InfeasibleSquad, FileExistsError) as exc:
                console.print(f"[yellow]GW{plan.next_gw} ({name}): {exc}[/yellow]")
                continue
            if name == plan.record_strategies[0]:
                board = Table(title=f"GW{plan.next_gw} recommendation ({name})")
                board.add_column("role")
                board.add_column("player")
                for pid in rec.starting:
                    role = (
                        "captain"
                        if pid == rec.captain
                        else "vice" if pid == rec.vice else "starter"
                    )
                    board.add_row(role, rec.names.get(str(pid), str(pid)))
                for n, pid in enumerate(rec.bench, start=1):
                    board.add_row(f"bench {n}", rec.names.get(str(pid), str(pid)))
                console.print(board)
        _print_notes(loaded)

    finish()


@app.command()
def ingest(
    season: str | None = typer.Option(
        None, "--season", help="defaults to the season now in progress"
    ),
    gw: int | None = typer.Option(None, "--gw", help="one gameweek, not all"),
    with_prices: bool = typer.Option(
        True,
        "--prices/--no-prices",
        help="fetch the per-gameweek price series (one request per player)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="build, write nothing"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build finished gameweeks of the current season into the archive's format.

    The archive publishes a season months after it ends, so this builds one as
    the season runs: each gameweek whose data FPL has marked *checked* is
    written into the same CSV layout the archive ships, which means the same
    loader reads it and a model can train on this season walk-forward exactly
    as the backtest trains on a completed one.

    A gameweek is ingested once and never rewritten — FPL revises results for a
    day or two, and a rewrite would silently change what a model was trained
    on. Re-running is therefore safe and usually does nothing.
    """
    _verbose(verbose)
    config = Config()
    client = FPLClient(config=config)
    bootstrap = client.bootstrap_static()
    fixtures = client.fixtures()
    season_label = season or current_season(bootstrap)

    # `finished` goes true when the last whistle blows; `data_checked` goes
    # true when FPL is done revising bonus points and appearances. Ingesting on
    # the former would capture figures that are still moving.
    checked = [
        int(event["id"])
        for event in bootstrap.get("events") or []
        if event.get("data_checked")
    ]
    already = ingested_gameweeks(season_label, config)
    wanted = [gw] if gw is not None else checked
    todo = [g for g in wanted if g not in already]

    console.print(
        f"{season_label}: {len(checked)} gameweek(s) with checked data, "
        f"{len(already)} already ingested, {len(todo)} to do"
    )
    # The reference files are refreshed whether or not a gameweek is new:
    # fixtures get rescheduled and average-manager scores only exist while the
    # API is still serving them, so there is no sense in letting them go stale
    # behind an early exit.
    if not dry_run:
        write_reference(season_label, bootstrap, fixtures, config)
        write_provenance(season_label, config)

    if not todo:
        console.print("Nothing to ingest; reference files refreshed.")
        raise typer.Exit(code=0)

    # Prices first, because they are the one field the cheap endpoints do not
    # carry. The weekly pre-deadline snapshot is the same figure and costs
    # nothing, so it is preferred; only gameweeks with no snapshot need the
    # per-player series, which is a request each.
    prices: dict[int, dict[int, int]] = {}
    sources: dict[int, str] = {}
    for target in todo:
        snapshot = load_snapshot(season_label, target, config)
        if snapshot is not None:
            prices[target] = dict(
                zip(snapshot["player_id"].astype(int), snapshot["price"].astype(int))
            )
            sources[target] = "the pre-deadline snapshot"

    uncovered = [g for g in todo if g not in prices]
    if uncovered and with_prices:
        console.print(
            f"GW{uncovered} have no pre-deadline snapshot, so their prices come "
            f"from the per-gameweek series (one request per player)…"
        )
        ids = [int(e["id"]) for e in bootstrap["elements"]]
        with console.status("loading") as spinner:
            for n, pid in enumerate(ids, start=1):
                spinner.update(f"price history {n}/{len(ids)}")
                try:
                    history = client.element_summary(pid).get("history") or []
                except FileNotFoundError:  # pragma: no cover - network shape
                    continue
                for row in history:
                    gw_seen = int(row["round"])
                    if gw_seen in uncovered:
                        prices.setdefault(gw_seen, {})[pid] = int(row["value"])
        for g in uncovered:
            sources.setdefault(g, "the per-gameweek price series")
    elif uncovered:
        for g in uncovered:
            sources[g] = "today's price (approximated)"

    failures: list[tuple[int, str]] = []
    for target in todo:
        try:
            frame, notes = ingest_gameweek(
                target,
                bootstrap=bootstrap,
                fixtures=fixtures,
                event_live=client.event_live(target),
                prices=prices.get(target),
            )
            if not dry_run:
                append_gameweek(frame, season_label, target, config)
            console.print(
                f"GW{target}: {len(frame)} rows, "
                f"{frame['element'].nunique()} players, prices from "
                f"{sources.get(target, 'today (approximated)')}"
                + ("  [dim](dry run, not written)[/dim]" if dry_run else "")
            )
            for note in notes[:3]:
                console.print(f"    [yellow]{note}[/yellow]")
        except IngestFailed as exc:
            failures.append((target, str(exc)))
            console.print(f"[red]GW{target}: {exc}[/red]")

    if not dry_run:
        write_reference(season_label, bootstrap, fixtures, config)
        path = write_provenance(season_label, config)
        console.print(f"Wrote {path.parent}/ (teams, fixtures, {path.name})")

    if failures:
        console.print(
            f"\n[red]{len(failures)} gameweek(s) failed to ingest: "
            f"{[g for g, _ in failures]}[/red]"
        )
        raise typer.Exit(code=1)


@app.command()
def transfers(
    team_id: int | None = typer.Option(
        None, "--team-id", help="your FPL entry id, from the URL of your team page"
    ),
    horizon: int = typer.Option(4, "--horizon", help="gameweeks to look ahead"),
    max_hits: int = typer.Option(
        1, "--max-hits", help="most hit-taking transfers to consider"
    ),
    squad: str | None = typer.Option(
        None,
        "--squad",
        help="15 names or ids, comma-separated, if your squad has changed "
        "since the last deadline",
    ),
    bank: float | None = typer.Option(
        None, "--bank", help="money in the bank, in millions, with --squad"
    ),
    free_transfers: int | None = typer.Option(
        None, "--free-transfers", help="override the replayed free-transfer count"
    ),
    strategy: str = typer.Option("model", "--strategy", "-s"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Recommend transfers for a real team, read-only.

    Reads the squad, bank and transfer history from the public endpoints — no
    login, nothing outside the read-only allowlist — and runs the same
    projections, the same integer program and the same club exclusions the
    backtest uses. Selling prices are reconstructed from what you paid, because
    FPL returns the purchase price plus half of any profit.

    Three solves, so the answer comes with its alternatives: hold and roll the
    free transfer, spend free transfers only, or take a hit. The expected-points
    gain of each is reported against holding, and a hit is recommended only when
    it beats the free-transfer-only plan after paying for itself.

    The public endpoint serves the squad as of the last deadline. If you have
    transferred since, pass ``--squad`` with what you actually own.
    """
    _verbose(verbose)
    config = Config()
    if strategy not in STRATEGIES:
        raise typer.BadParameter(
            f"unknown strategy {strategy!r}: choose from {sorted(STRATEGIES)}"
        )
    if team_id is None and squad is None:
        raise typer.BadParameter("give --team-id, or --squad with --bank")
    if squad is not None and bank is None and team_id is None:
        raise typer.BadParameter("--squad needs --bank (in millions)")

    client = FPLClient(config=config)
    bootstrap = client.bootstrap_static()
    rule_set = rules_from_bootstrap(bootstrap, config)
    rule_set.validate()
    target = next_gw(bootstrap)
    if target is None:
        console.print("[yellow]No upcoming gameweek to plan for.[/yellow]")
        raise typer.Exit(code=0)

    # The ingested season is the same walk-forward history the backtest trains
    # on, already on disk, so this costs no per-player requests. Falling back
    # to `element-summary` only if a gameweek has not been ingested yet.
    fixtures_payload = client.fixtures()
    try:
        data = season_with_upcoming(
            current_season(bootstrap), bootstrap, fixtures_payload, target, config
        )
        console.print(
            f"Training on the ingested season: gameweeks "
            f"{min(data.before(target)['gw'])}–{max(data.before(target)['gw'])}, "
            f"all final before GW{target}'s deadline."
        )
    except IngestFailed as exc:
        console.print(
            f"[yellow]{exc}\nFalling back to the per-player history "
            f"(one request per player).[/yellow]"
        )
        with console.status("loading") as spinner:
            data = load_live_season(
                client=client,
                config=config,
                bootstrap=bootstrap,
                fixtures=fixtures_payload,
                progress=lambda done, total: spinner.update(
                    f"player histories {done}/{total}"
                ),
            )

    # --- what do we own, and what is it worth to us? ----------------------
    notes: list[str] = []
    state = None
    if team_id is not None:
        state = load_entry(client, team_id, data, rule_set)
        held = state.squad
        money = state.bank
        banked = state.free_transfers
        purchase = dict(state.purchase)
        notes.extend(state.notes)
        console.print(
            f"\n[bold]{state.name}[/bold] ({state.manager}) — entry {state.team_id}, "
            f"started GW{state.started_event}"
        )
    else:
        held, money, banked, purchase = (), 0, 1, {}

    if squad is not None:
        named = [token for token in squad.replace(";", ",").split(",") if token.strip()]
        try:
            held, _ = resolve_players(named, data, gw=target)
        except EntryUnavailable as exc:
            # A misread name would advise on someone else's squad, so this is
            # a refusal rather than a guess — but it should read like one.
            raise typer.BadParameter(str(exc)) from None
        if len(held) != rule_set.squad_size:
            raise typer.BadParameter(
                f"--squad needs {rule_set.squad_size} players, got {len(held)}"
            )
        if bank is not None:
            money = int(round(bank * 10))
        notes.append(
            f"squad taken from --squad rather than from the last deadline's picks"
        )
        if state is not None:
            # Purchase prices still come from the transfer history where the
            # player appears in it; the rest fall back to today's price.
            purchase = {
                pid: state.purchase.get(pid, 0) for pid in held
            }
            missing = [pid for pid, paid in purchase.items() if not paid]
            if missing:
                notes.append(
                    f"{len(missing)} player(s) in --squad were never transferred "
                    f"in by this entry, so their purchase price is unknown and "
                    f"today's price is used: selling prices may be overstated"
                )
        else:
            notes.append(
                "no --team-id, so purchase prices are unknown: selling prices "
                "are assumed equal to current prices"
            )

    if free_transfers is not None:
        banked = int(free_transfers)
        notes.append(f"free transfers overridden to {banked}")

    prices = (
        data.player_gws[data.player_gws["gw"] <= target]
        .sort_values(["player_id", "gw"])
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")["price"]
    )
    if not purchase:
        purchase = {pid: int(prices.get(pid, 0)) for pid in held}
    selling = {
        pid: rule_set.selling_price(int(paid), int(prices.get(pid, paid)))
        for pid, paid in purchase.items()
    }

    # --- project ----------------------------------------------------------
    projection = STRATEGIES[strategy]()
    projection.fit(data, target)
    preds = projection.predict(data, target)
    if preds.empty or float(preds["xpts"].abs().max()) == 0.0:
        console.print(
            f"[yellow]{strategy} has no projection for GW{target}, so there is "
            f"nothing to recommend.[/yellow]"
        )
        raise typer.Exit(code=0)
    future = horizon_predictions(
        projection, data, target, horizon=horizon, decay=config.horizon_decay
    )
    candidates = build_candidates(
        data, target, preds, future=future, selling_prices=selling
    )

    # --- three solves: hold, free only, with hits -------------------------
    def solve(limit: int | None) -> SquadPlan:
        return optimize_squad(
            candidates,
            rule_set,
            config=config,
            current=held,
            free_transfers=banked,
            bank=money,
            max_hits=limit,
        )

    hold = optimize_squad(
        candidates,
        rule_set,
        config=config,
        current=held,
        free_transfers=banked,
        bank=money,
        max_hits=0,
        must_include=tuple(held),
    )
    free_only = solve(0)
    with_hits = solve(max_hits) if max_hits > 0 else free_only

    best = with_hits if with_hits.expected_points > free_only.expected_points else free_only
    hit_gain = with_hits.expected_points - free_only.expected_points
    gain = best.expected_points - hold.expected_points

    # What is each individual transfer worth? Solve for one transfer, then two,
    # then three, and difference the objectives: the k-th figure is the gain
    # from being allowed one more move. Attributing a figure to a *named*
    # transfer is only honest when the smaller plan is a subset of the larger —
    # the optimizer is free to change its mind about which players to move when
    # it gets another go, and often does.
    chain: list[tuple[int, SquadPlan, float]] = []
    previous = hold
    nested = True
    for k in range(1, best.n_transfers + 1):
        step = optimize_squad(
            candidates,
            rule_set,
            config=config,
            current=held,
            free_transfers=banked,
            bank=money,
            max_hits=max_hits,
            max_transfers=k,
        )
        if not set(previous.transfers_in) <= set(step.transfers_in):
            nested = False
        chain.append((k, step, step.expected_points - previous.expected_points))
        previous = step

    names = (
        data.player_gws.drop_duplicates("player_id")
        .set_index("player_id")["name"]
        .astype(str)
    )
    positions = positions_from(data)
    # Never print a squad that could not be fielded.
    best.squad.validate(positions, rule_set)

    def label(pid: int) -> str:
        return f"{names.get(pid, pid)} ({positions.get(pid, '?')})"

    console.print(
        f"GW{target} — {strategy}, horizon {horizon}, "
        f"{banked} free transfer(s), {money / 10:.1f}m in the bank"
    )

    # --- the recommendation -----------------------------------------------
    if not best.transfers_in:
        console.print(
            "\n[bold]Recommendation: no transfer.[/bold] Nothing available "
            "beats holding, so roll the free transfer."
        )
    else:
        table = Table(title=f"GW{target} recommended transfers")
        table.add_column("out")
        table.add_column("in")
        table.add_column("sell", justify="right")
        table.add_column("buy", justify="right")
        for out_id, in_id in zip(best.transfers_out, best.transfers_in):
            table.add_row(
                label(out_id),
                label(in_id),
                f"{selling.get(out_id, 0) / 10:.1f}",
                f"{int(prices.get(in_id, 0)) / 10:.1f}",
            )
        console.print(table)
        console.print(
            f"Hit: [bold]{'-' + str(best.transfer_hit) if best.transfer_hit else 'none'}"
            f"[/bold]  ·  bank after: {best.bank / 10:.1f}m"
        )

        # --- what each move is worth on its own ---------------------------
        margins = Table(title="expected-points margin behind each move")
        margins.add_column("move")
        margins.add_column("transfer" if nested else "plan")
        margins.add_column("margin", justify="right")
        margins.add_column("confidence")
        low_confidence: list[str] = []
        for k, step, margin in chain:
            if nested and step.transfers_in:
                added_in = set(step.transfers_in) - set(
                    chain[k - 2][1].transfers_in if k > 1 else ()
                )
                added_out = set(step.transfers_out) - set(
                    chain[k - 2][1].transfers_out if k > 1 else ()
                )
                description = " · ".join(
                    f"{label(o)} → {label(i)}"
                    for o, i in zip(sorted(added_out), sorted(added_in))
                ) or f"{k} transfers"
            else:
                description = f"{k} transfer(s) allowed"
            thin = margin < THIN_MARGIN
            if thin:
                low_confidence.append(description)
            margins.add_row(
                f"#{k}",
                description,
                f"{margin:+.2f}",
                "[yellow]low — under a point[/yellow]" if thin else "ok",
            )
        console.print(margins)
        if not nested:
            console.print(
                "[dim]The margins are per additional transfer allowed, not per "
                "named move: the optimizer changed which players it wanted when "
                "given another go, so attributing a figure to one transfer "
                "would misrepresent it.[/dim]"
            )
        if low_confidence:
            console.print(
                f"[yellow]Low confidence on {len(low_confidence)} move(s): a "
                f"margin under {THIN_MARGIN:.0f} point is inside the "
                f"projections' own error, so treat those as optional.[/yellow]"
            )

    verdict = Table(title="is it worth it?")
    verdict.add_column("plan")
    verdict.add_column("transfers", justify="right")
    verdict.add_column("hit", justify="right")
    verdict.add_column("expected points", justify="right")
    verdict.add_column("vs holding", justify="right")
    for name, plan in (
        ("hold (roll the transfer)", hold),
        ("free transfers only", free_only),
        (f"up to {max_hits} hit(s)", with_hits),
    ):
        verdict.add_row(
            name,
            str(plan.n_transfers),
            f"-{plan.transfer_hit}" if plan.transfer_hit else "0",
            f"{plan.expected_points:.2f}",
            f"{plan.expected_points - hold.expected_points:+.2f}",
        )
    console.print(verdict)
    console.print(
        f"Expected gain over the horizon versus rolling: "
        f"[bold]{gain:+.2f}[/bold] points."
        + (
            f"  [yellow](under {THIN_MARGIN:.0f} point — low confidence)[/yellow]"
            if 0 < gain < THIN_MARGIN
            else ""
        )
    )
    if max_hits <= 0:
        console.print("Hits were not considered (--max-hits 0).")
    elif best.transfer_hit == 0 and best.n_transfers <= banked:
        console.print(
            f"No hit needed: {banked} free transfer(s) cover the "
            f"{best.n_transfers} worth making."
        )
    elif hit_gain >= THIN_MARGIN:
        console.print(
            f"Taking a hit is [bold green]worth it[/bold green]: "
            f"{hit_gain:+.2f} points beyond what the free transfers manage, "
            f"after paying the {with_hits.transfer_hit}-point charge."
        )
    elif hit_gain > 0:
        console.print(
            f"Taking a hit comes out {hit_gain:+.2f} points ahead, which is "
            f"[bold]too thin to act on[/bold]: the projections are not accurate "
            f"to a fraction of a point, so this is a coin toss dressed as a "
            f"recommendation. The free-transfer plan is the safer read."
        )
    else:
        console.print(
            f"Taking a hit is [bold]not worth it[/bold]: it would bring in "
            f"{abs(hit_gain):.2f} points less than it costs."
        )

    xi = Table(title=f"GW{target} starting XI")
    xi.add_column("role")
    xi.add_column("player")
    xi.add_column("xPts", justify="right")
    detail = best.detail
    for pid in best.squad.starting:
        role = (
            "captain"
            if pid == best.squad.captain
            else "vice" if pid == best.squad.vice else positions.get(pid, "")
        )
        xi.add_row(role, str(names.get(pid, pid)), f"{detail.get(pid, 0.0):.2f}")
    console.print(xi)

    bench = Table(title="bench, in order")
    bench.add_column("#")
    bench.add_column("player")
    bench.add_column("xPts", justify="right")
    for n, pid in enumerate(best.squad.bench, start=1):
        bench.add_row(str(n), label(pid), f"{detail.get(pid, 0.0):.2f}")
    console.print(bench)

    if notes:
        console.print("\n[bold]notes on your team[/bold]")
        for note in notes:
            console.print(f"  • {note}")
    _print_notes(data)


def _latest_recommendation(
    season: str, strategy: str, before_gw: int, config: Config
):
    """The most recent recommendation before ``before_gw``, if there is one.

    The squad carries between gameweeks, so a new recommendation is a transfer
    decision from the last one rather than a fresh fifteen.
    """
    for candidate in range(before_gw - 1, 0, -1):
        rec = load_recommendation(season, candidate, strategy, config)
        if rec is not None:
            return rec
    return None


def main() -> None:
    """Entry point for the ``fpl`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
