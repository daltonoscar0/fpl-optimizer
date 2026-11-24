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
``fpl track``
    Record the current season's recommendation before the deadline, and score
    the ones already on record against the average manager.

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
from .features import FeatureCache
from .live import current_season, events_from_bootstrap, load_live_season, next_gw
from .optimize import InfeasibleSquad
from .projections import STRATEGIES
from .projections.baselines import EP_COVERAGE_THRESHOLD, ep_coverage
from .rules import archive_season_rules, rules_from_bootstrap
from .schema import SeasonData, describe_frame
from .score import positions_from
from .track import (
    NotRecordedInAdvance,
    append_tracked,
    load_recommendation,
    markdown_table,
    plan_track,
    record,
    recommend,
    score_recorded,
    tracked_frame,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Squad and transfer optimizer for Fantasy Premier League.",
)
console = Console()


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
    strategy: str = typer.Option("model", "--strategy", "-s"),
    gw: int | None = typer.Option(
        None, "--gw", help="gameweek to record for; default is the next one"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="decide, write nothing"),
    status_file: Path | None = typer.Option(
        None, "--status-file", help="write a JSON summary here, for automation"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Record this gameweek's recommendation, and score the ones already played.

    Safe to run on a schedule. Deciding what to do costs one request: the
    gameweek deadlines, against the records already on disk. Only a run that
    has something to do fetches player histories, so most runs are nearly free
    and finish in seconds.

    Idempotent by construction. A gameweek is recorded at most once, only while
    its deadline is still in the future, and a gameweek is scored only from a
    recommendation recorded before that deadline. Everything else is a no-op.
    """
    _verbose(verbose)
    config = Config()
    if strategy not in STRATEGIES:
        raise typer.BadParameter(
            f"unknown strategy {strategy!r}: choose from {sorted(STRATEGIES)}"
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

    recorded = {
        int(row["gw"])
        for row in events.to_dict("records")
        if load_recommendation(season, int(row["gw"]), strategy, config) is not None
    }
    existing = tracked_frame(season, config)
    scored = set() if existing.empty else {int(g) for g in existing["gw"]}
    plan = plan_track(
        season,
        events,
        recorded_gws=recorded,
        scored_gws=scored,
        next_gw=gw if gw is not None else next_gw(bootstrap),
        now=datetime.now(timezone.utc),
        strategy=strategy,
    )
    if plan.next_gw is not None:
        console.print(
            f"next gameweek GW{plan.next_gw}, deadline {plan.next_deadline}"
        )

    status: dict[str, object] = {
        "season": season,
        "strategy": strategy,
        "next_gw": plan.next_gw,
        "next_deadline": plan.next_deadline,
        "recorded_gw": None,
        "scored_gws": [],
        "missed_gws": plan.missed_gws,
        "api_players": len(bootstrap["elements"]),
        "reason": plan.reason,
    }

    def finish() -> None:
        if status_file is not None:
            status_file.write_text(json.dumps(status, indent=2) + "\n")
        frame = tracked_frame(season, config)
        console.print("\n" + markdown_table(frame))
        if plan.missed_gws:
            console.print(
                f"\n[red]No recommendation was ever recorded for "
                f"GW{plan.missed_gws}, whose deadline has passed. Those "
                f"gameweeks cannot be scored and are a permanent gap.[/red]"
            )

    if plan.nothing_to_do:
        console.print(f"Nothing to do: {plan.reason or 'everything is up to date'}.")
        finish()
        raise typer.Exit(code=0)

    # --- act: fetch only what the plan needs ------------------------------
    if plan.needs_full_season:
        console.print(
            "Recording needs a projection for every player, so every player's "
            "history (one request each)…"
        )
        wanted_ids = None
    else:
        # Scoring needs only the squads on record: fifteen players, not the
        # whole game. The plan only lists gameweeks that have a record, so
        # every lookup here resolves.
        squads = [
            load_recommendation(season, target, strategy, config)
            for target in plan.score_gws
        ]
        wanted_ids = sorted({pid for rec in squads if rec for pid in rec.squad})
        console.print(f"Scoring needs {len(wanted_ids)} player histories…")

    with console.status("loading") as spinner:
        loaded = load_live_season(
            client=client,
            config=config,
            bootstrap=bootstrap,
            fixtures=client.fixtures(),
            player_ids=wanted_ids,
            progress=lambda done, total: spinner.update(
                f"player histories {done}/{total}"
            ),
        )
    rule_set = rules_from_bootstrap(bootstrap, config)
    rule_set.validate()
    positions = positions_from(loaded)

    # --- score what is on record ------------------------------------------
    for target in plan.score_gws:
        rec = load_recommendation(season, target, strategy, config)
        if rec is None:  # pragma: no cover - plan guarantees one exists
            continue
        try:
            tracked = score_recorded(loaded, rec, rule_set, positions=positions)
        except NotRecordedInAdvance as exc:
            console.print(f"[yellow]GW{target}: {exc}[/yellow]")
            continue
        if not dry_run:
            append_tracked(tracked, config)
        status["scored_gws"] = list(status["scored_gws"]) + [target]
        average = tracked.average_entry_score
        console.print(
            f"GW{target}: recommended XI scored [bold]{tracked.points}[/bold], "
            f"average manager "
            f"{'unpublished' if average is None else int(average)}"
            + ("" if tracked.delta is None else f" ([bold]{tracked.delta:+.0f}[/bold])")
        )

    # --- record the next gameweek -----------------------------------------
    if plan.record_gw is not None:
        previous = _latest_recommendation(season, strategy, plan.record_gw, config)
        try:
            rec = recommend(
                loaded,
                plan.record_gw,
                strategy,
                rule_set,
                config=config,
                current=previous.as_squad() if previous else None,
            )
            if dry_run:
                console.print(f"[dim]--dry-run: not writing GW{plan.record_gw}[/dim]")
            else:
                path = record(rec, config)
                console.print(f"Recorded GW{plan.record_gw} → {path.name}")
            status["recorded_gw"] = plan.record_gw
        except (NotRecordedInAdvance, InfeasibleSquad, FileExistsError) as exc:
            console.print(f"[yellow]GW{plan.record_gw}: {exc}[/yellow]")
            status["reason"] = str(exc)
        else:
            board = Table(title=f"GW{plan.record_gw} recommendation ({strategy})")
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

    finish()
    _print_notes(loaded)


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
