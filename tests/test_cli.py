"""The ``fpl`` command line.

These tests run the commands the way a user does — through the typer app, by
name, checking exit codes and what lands on stdout. The season loader is
pointed at the committed archive slice so no command reaches the network.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from fpl_optimizer import cli
from fpl_optimizer.archive import available_seasons
from fpl_optimizer.config import Config

from .fixtures import ARCHIVE_SEASON, load_archive_fixture

runner = CliRunner()


@pytest.fixture(autouse=True)
def _archive_from_fixtures(monkeypatch):
    """Make ``load_season`` read the committed slice whatever season is asked for.

    The CLI calls ``load_season(season)``, which would otherwise download. The
    test asserts on CLI behaviour, not on the archive, so the loader is pinned
    to the fixture and the requested season name is ignored.
    """
    def fake_load_season(season, config=None, **kwargs):
        return load_archive_fixture()

    monkeypatch.setattr(cli, "load_season", fake_load_season)


def test_entry_point_is_importable_and_callable():
    """What `pyproject`'s console script resolves to."""
    assert callable(cli.main)


def test_no_arguments_prints_the_help():
    """Exit code 2 is click's convention for "no command given", not a crash."""
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 2
    assert "Fantasy Premier League" in result.stdout
    for command in ("seasons", "rules", "data", "project"):
        assert command in result.stdout


def test_seasons_lists_every_backtest_season():
    result = runner.invoke(cli.app, ["seasons"])
    assert result.exit_code == 0
    for season in available_seasons():
        assert season in result.stdout


def test_rules_prints_the_rule_set_and_its_provenance():
    result = runner.invoke(cli.app, ["rules", "2024-25"])
    assert result.exit_code == 0
    assert "squad size" in result.stdout
    assert "100.0m" in result.stdout
    # The archive has no rules snapshot, so the source must say it is asserted.
    assert "asserted" in result.stdout


def test_rules_formation_covers_all_four_positions():
    result = runner.invoke(cli.app, ["rules", "2023-24"])
    assert result.exit_code == 0
    for position in ("GKP", "DEF", "MID", "FWD"):
        assert position in result.stdout


def test_data_summarizes_the_season_and_prints_its_notes():
    result = runner.invoke(cli.app, ["data", ARCHIVE_SEASON])
    assert result.exit_code == 0
    assert "player-gameweek rows" in result.stdout
    assert "notes" in result.stdout
    # The slice's known gap is reported rather than hidden.
    assert "all-zero" in result.stdout


def test_data_warns_when_expected_points_coverage_is_too_low():
    """The fixture has xP for 7 of 8 gameweeks, so coverage is 88% — no warning.

    The warning path is covered by the threshold test in test_projections; what
    matters here is that the figure is printed either way.
    """
    result = runner.invoke(cli.app, ["data", ARCHIVE_SEASON])
    assert result.exit_code == 0
    assert "coverage" in result.stdout


@pytest.mark.parametrize("strategy", ["naive", "ep"])
def test_project_prints_a_board_for_each_baseline(strategy):
    result = runner.invoke(
        cli.app, ["project", ARCHIVE_SEASON, "8", "--strategy", strategy, "--top", "5"]
    )
    assert result.exit_code == 0
    assert "xPts" in result.stdout
    assert "GW8" in result.stdout


def test_project_rejects_an_unknown_strategy():
    result = runner.invoke(
        cli.app, ["project", ARCHIVE_SEASON, "8", "--strategy", "vibes"]
    )
    assert result.exit_code != 0
    # typer writes parameter errors to stderr.
    assert "unknown strategy" in result.stderr


def test_project_says_so_when_a_strategy_has_nothing_to_say():
    """Gameweek 7 of the slice has no expected points at all."""
    result = runner.invoke(
        cli.app, ["project", ARCHIVE_SEASON, "7", "--strategy", "ep"]
    )
    assert result.exit_code == 0
    assert "reported" in result.stdout or "unavailable" in result.stdout


def test_project_on_a_gameweek_with_no_fixtures_is_not_an_error():
    result = runner.invoke(
        cli.app, ["project", ARCHIVE_SEASON, "99", "--strategy", "naive"]
    )
    assert result.exit_code == 0
    assert "No projection" in result.stdout


# --- backtest and table -----------------------------------------------------


@pytest.fixture
def hand_built(monkeypatch):
    """Point the loader at the hand-built season.

    The committed archive slice has two clubs, and three players per club caps a
    squad at six — not fifteen — so it cannot be backtested. The hand-built
    season has six clubs, which is the smallest thing that can.
    """
    from .fixtures.handbuilt import hand_built_season

    monkeypatch.setattr(cli, "load_season", lambda *a, **k: hand_built_season())
    monkeypatch.setattr(cli, "available_seasons", lambda: ("2024-25",))
    return hand_built_season()


def test_backtest_writes_a_csv_and_a_summary(hand_built, tmp_path):
    result = runner.invoke(
        cli.app,
        ["backtest", "2024-25", "--strategy", "ep", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "2024-25-ep.csv").exists()
    assert (tmp_path / "summary.json").exists()

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary[0]["strategy"] == "ep"
    assert summary[0]["total_points"] > 0
    assert "points" in (tmp_path / "2024-25-ep.csv").read_text()


def test_backtest_runs_every_strategy_when_asked(hand_built, tmp_path):
    result = runner.invoke(
        cli.app, ["backtest", "all", "--strategy", "all", "--out", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert {s["strategy"] for s in summary} == {"model", "ep", "naive"}


def test_backtest_skips_expected_points_below_the_coverage_threshold(
    monkeypatch, tmp_path
):
    """A season whose xP was never collected is reported as skipped, not run."""
    from .fixtures.handbuilt import hand_built_season

    data = hand_built_season()
    df = data.player_gws.copy()
    df["ep"] = float("nan")
    data.player_gws = df
    monkeypatch.setattr(cli, "load_season", lambda *a, **k: data)

    result = runner.invoke(
        cli.app,
        ["backtest", "2024-25", "--strategy", "ep", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "skipped" in result.stdout
    assert not (tmp_path / "2024-25-ep.csv").exists()


def test_backtest_rejects_an_unknown_strategy(hand_built, tmp_path):
    result = runner.invoke(
        cli.app,
        ["backtest", "2024-25", "--strategy", "vibes", "--out", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "unknown strategy" in result.stderr


def test_table_prints_markdown_with_the_window_stated(hand_built, tmp_path):
    runner.invoke(
        cli.app, ["backtest", "2024-25", "--strategy", "all", "--out", str(tmp_path)]
    )
    result = runner.invoke(cli.app, ["table", "--results", str(tmp_path)])

    assert result.exit_code == 0, result.output
    lines = [l for l in result.stdout.splitlines() if l.startswith("|")]
    assert lines[0].startswith("| Season | Strategy |")
    assert "common window" in lines[0]
    # One header, one separator, one row per strategy.
    assert len(lines) >= 4
    assert any("`naive`" in l for l in lines)
    assert any("GWs" in l for l in lines)


def test_table_omits_a_strategy_that_knows_who_played(monkeypatch, tmp_path):
    """`ep` is left out of the comparison, with the reason printed.

    Putting it in a table of results implies it is a baseline. It is not: the
    archive's column rates most non-appearances at exactly zero.
    """
    from .fixtures.handbuilt import hand_built_season

    data = hand_built_season()
    df = data.player_gws.copy()
    absent = df["minutes"].fillna(0) == 0
    df.loc[absent, "ep"] = 0.0
    df.loc[~absent, "ep"] = 4.0
    data.player_gws = df
    monkeypatch.setattr(cli, "load_season", lambda *a, **k: data)
    monkeypatch.setattr(cli, "available_seasons", lambda: ("2024-25",))

    runner.invoke(
        cli.app, ["backtest", "2024-25", "--strategy", "all", "--out", str(tmp_path)]
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert any(s["strategy"] == "ep" for s in summary), "it is still run and kept"

    result = runner.invoke(cli.app, ["table", "--results", str(tmp_path)])
    assert result.exit_code == 0, result.output
    table_rows = [l for l in result.stdout.splitlines() if l.startswith("| 2")]
    assert not any("`ep`" in row for row in table_rows)
    assert "omitted" in result.stdout
    assert "knows who played" in result.stdout


def test_table_without_a_backtest_says_what_to_run(tmp_path):
    result = runner.invoke(cli.app, ["table", "--results", str(tmp_path)])
    assert result.exit_code != 0
    assert "fpl backtest" in result.stderr


# --- track ------------------------------------------------------------------


@pytest.fixture
def live(monkeypatch, tmp_path):
    """A fake current season: the hand-built one, with GW3 still to come.

    `track` reaches the API for the bootstrap payload, the fixture list and a
    history per player. All three are stubbed, so the test exercises the
    command's logic — record, score, refuse — without a request.
    """
    from fpl_optimizer.rules import archive_season_rules
    from .fixtures.handbuilt import hand_built_season

    data = hand_built_season()
    events = data.events.copy()
    events.loc[events["gw"] == 3, "finished"] = False
    data.events = events
    data.source = "live"

    class StubClient:
        def __init__(self, *a, **k):
            pass

        def bootstrap_static(self):
            return {}

        def fixtures(self):
            return []

    monkeypatch.setattr(cli, "FPLClient", StubClient)
    monkeypatch.setattr(cli, "load_live_season", lambda *a, **k: data)
    monkeypatch.setattr(
        cli, "rules_from_bootstrap", lambda *a, **k: archive_season_rules("2024-25")
    )
    monkeypatch.setattr(cli, "next_gw", lambda *a, **k: 3)
    monkeypatch.setattr(
        cli, "Config", lambda *a, **k: Config(excluded_clubs=(), horizon=2, results_dir=tmp_path)
    )
    return data, tmp_path


def test_track_records_the_upcoming_gameweek(live):
    _, results_dir = live
    result = runner.invoke(cli.app, ["track", "--strategy", "ep"])

    assert result.exit_code == 0, result.output
    written = list((results_dir / "live").glob("*-gw03-ep.json"))
    assert len(written) == 1
    rec = json.loads(written[0].read_text())
    assert rec["gw"] == 3
    assert len(rec["squad"]) == 15
    assert "Recorded GW3" in result.stdout


def test_track_scores_nothing_on_its_first_run(live):
    """There is nothing on record yet, and it says so instead of inventing one."""
    result = runner.invoke(cli.app, ["track", "--strategy", "ep"])

    assert result.exit_code == 0
    assert "Nothing to score yet" in result.stdout
    assert "no gameweek called in advance yet" in result.stdout


def test_track_will_not_overwrite_a_recommendation_already_on_record(live):
    first = runner.invoke(cli.app, ["track", "--strategy", "ep"])
    assert "Recorded GW3" in first.stdout

    second = runner.invoke(cli.app, ["track", "--strategy", "ep"])
    assert second.exit_code == 0
    assert "already on record" in second.stdout


def test_track_dry_run_decides_but_writes_nothing(live):
    _, results_dir = live
    result = runner.invoke(cli.app, ["track", "--strategy", "ep", "--dry-run"])

    assert result.exit_code == 0
    assert "GW3 recommendation" in result.stdout
    assert not list((results_dir / "live").glob("*.json"))


def test_track_scores_a_gameweek_that_was_called_in_advance(live):
    """Record GW2 in advance, then let `track` find it finished and score it."""
    from fpl_optimizer.track import record, recommend
    from fpl_optimizer.rules import archive_season_rules

    data, results_dir = live
    config = Config(excluded_clubs=(), horizon=2, results_dir=results_dir)
    rec = recommend(
        data,
        2,
        "ep",
        archive_season_rules("2024-25"),
        config=config,
        now="2024-09-01T09:00:00+00:00",
    )
    record(rec, config)

    result = runner.invoke(cli.app, ["track", "--strategy", "ep"])
    assert result.exit_code == 0, result.output
    assert "recommended XI scored" in result.stdout
    assert "average manager 70" in result.stdout
    assert (results_dir / "live" / "2024-25.csv").exists()


def test_track_rejects_an_unknown_strategy(live):
    result = runner.invoke(cli.app, ["track", "--strategy", "vibes"])
    assert result.exit_code != 0
    assert "unknown strategy" in result.stderr
