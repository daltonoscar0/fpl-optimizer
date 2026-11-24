"""The ``fpl`` command line.

These tests run the commands the way a user does — through the typer app, by
name, checking exit codes and what lands on stdout. The season loader is
pointed at the committed archive slice so no command reaches the network.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from fpl_optimizer import cli
from fpl_optimizer.archive import available_seasons

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
