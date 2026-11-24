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

Every command prints the source and the notes rather than only the numbers. A
projection that is quietly running on a fallback, or a season whose expected
points were never collected, is the kind of thing that turns into a wrong
conclusion three steps later if it is not said out loud.
"""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from .archive import available_seasons, load_season
from .config import Config
from .projections import STRATEGIES
from .projections.baselines import EP_COVERAGE_THRESHOLD, ep_coverage
from .rules import archive_season_rules
from .schema import SeasonData, describe_frame

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


def main() -> None:
    """Entry point for the ``fpl`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
