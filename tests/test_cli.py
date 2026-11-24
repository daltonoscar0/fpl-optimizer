"""The ``fpl`` command line.

These tests run the commands the way a user does — through the typer app, by
name, checking exit codes and what lands on stdout. The season loader is
pointed at the committed archive slice so no command reaches the network.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

import pytest
from typer.testing import CliRunner

from fpl_optimizer import cli
from fpl_optimizer.archive import available_seasons
from fpl_optimizer.config import Config

from .fixtures import ARCHIVE_SEASON, load_archive_fixture

runner = CliRunner()

#: Rich styles its error output, and whether it does depends on the terminal:
#: on CI a flag like `--team-id` arrives as several separately-coloured spans,
#: so `"--team-id" in stderr` passes locally and fails there. Assertions go
#: through this instead of depending on colour detection.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """CLI output with the styling stripped."""
    return _ANSI.sub("", text)


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
    assert "Fantasy Premier League" in plain(result.stdout)
    for command in ("seasons", "rules", "data", "project"):
        assert command in plain(result.stdout)


def test_seasons_lists_every_backtest_season():
    result = runner.invoke(cli.app, ["seasons"])
    assert result.exit_code == 0
    for season in available_seasons():
        assert season in plain(result.stdout)


def test_rules_prints_the_rule_set_and_its_provenance():
    result = runner.invoke(cli.app, ["rules", "2024-25"])
    assert result.exit_code == 0
    assert "squad size" in plain(result.stdout)
    assert "100.0m" in plain(result.stdout)
    # The archive has no rules snapshot, so the source must say it is asserted.
    assert "asserted" in plain(result.stdout)


def test_rules_formation_covers_all_four_positions():
    result = runner.invoke(cli.app, ["rules", "2023-24"])
    assert result.exit_code == 0
    for position in ("GKP", "DEF", "MID", "FWD"):
        assert position in plain(result.stdout)


def test_data_summarizes_the_season_and_prints_its_notes():
    result = runner.invoke(cli.app, ["data", ARCHIVE_SEASON])
    assert result.exit_code == 0
    assert "player-gameweek rows" in plain(result.stdout)
    assert "notes" in plain(result.stdout)
    # The slice's known gap is reported rather than hidden.
    assert "all-zero" in plain(result.stdout)


def test_data_warns_when_expected_points_coverage_is_too_low():
    """The fixture has xP for 7 of 8 gameweeks, so coverage is 88% — no warning.

    The warning path is covered by the threshold test in test_projections; what
    matters here is that the figure is printed either way.
    """
    result = runner.invoke(cli.app, ["data", ARCHIVE_SEASON])
    assert result.exit_code == 0
    assert "coverage" in plain(result.stdout)


@pytest.mark.parametrize("strategy", ["naive", "ep"])
def test_project_prints_a_board_for_each_baseline(strategy):
    result = runner.invoke(
        cli.app, ["project", ARCHIVE_SEASON, "8", "--strategy", strategy, "--top", "5"]
    )
    assert result.exit_code == 0
    assert "xPts" in plain(result.stdout)
    assert "GW8" in plain(result.stdout)


def test_project_rejects_an_unknown_strategy():
    result = runner.invoke(
        cli.app, ["project", ARCHIVE_SEASON, "8", "--strategy", "vibes"]
    )
    assert result.exit_code != 0
    # typer writes parameter errors to stderr.
    assert "unknown strategy" in plain(result.stderr)


def test_project_says_so_when_a_strategy_has_nothing_to_say():
    """Gameweek 7 of the slice has no expected points at all."""
    result = runner.invoke(
        cli.app, ["project", ARCHIVE_SEASON, "7", "--strategy", "ep"]
    )
    assert result.exit_code == 0
    assert "reported" in plain(result.stdout) or "unavailable" in plain(result.stdout)


def test_project_on_a_gameweek_with_no_fixtures_is_not_an_error():
    result = runner.invoke(
        cli.app, ["project", ARCHIVE_SEASON, "99", "--strategy", "naive"]
    )
    assert result.exit_code == 0
    assert "No projection" in plain(result.stdout)


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
    assert "skipped" in plain(result.stdout)
    assert not (tmp_path / "2024-25-ep.csv").exists()


def test_backtest_rejects_an_unknown_strategy(hand_built, tmp_path):
    result = runner.invoke(
        cli.app,
        ["backtest", "2024-25", "--strategy", "vibes", "--out", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "unknown strategy" in plain(result.stderr)


def test_table_prints_markdown_with_the_window_stated(hand_built, tmp_path):
    runner.invoke(
        cli.app, ["backtest", "2024-25", "--strategy", "all", "--out", str(tmp_path)]
    )
    result = runner.invoke(cli.app, ["table", "--results", str(tmp_path)])

    assert result.exit_code == 0, result.output
    lines = [l for l in plain(result.stdout).splitlines() if l.startswith("|")]
    assert lines[0].startswith("| Season | Strategy |")
    assert "common window" in lines[0]
    # One header, one separator, one row per strategy.
    assert len(lines) >= 4
    assert any("`naive`" in l for l in lines)
    assert any("GWs" in l for l in lines)


def test_table_omits_a_strategy_whose_total_is_not_comparable(monkeypatch, tmp_path):
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
    table_rows = [l for l in plain(result.stdout).splitlines() if l.startswith("| 2")]
    assert not any("`ep`" in row for row in table_rows)
    assert "omitted" in plain(result.stdout)
    assert "not a comparable result" in plain(result.stdout)


def test_table_without_a_backtest_says_what_to_run(tmp_path):
    result = runner.invoke(cli.app, ["table", "--results", str(tmp_path)])
    assert result.exit_code != 0
    assert "fpl backtest" in plain(result.stderr)


# --- track ------------------------------------------------------------------


@pytest.fixture
def live(monkeypatch, tmp_path):
    """A fake current season: the hand-built one, with GW3 still to come.

    `track` reaches the API for the bootstrap payload, the fixture list and a
    history per player. All three are stubbed, so the test exercises the
    command's logic — record, score, refuse — without a request.
    """
    from fpl_optimizer.rules import archive_season_rules
    from .fixtures.handbuilt import CLUBS, PLAYERS, hand_built_season

    data = hand_built_season()
    events = data.events.copy()
    events.loc[events["gw"] == 3, "finished"] = False
    # GW3 is the upcoming one, so its deadline has to actually be upcoming:
    # `track` refuses to record a gameweek whose deadline has passed, which is
    # the behaviour under test everywhere else in this file.
    upcoming = (datetime.now(timezone.utc) + timedelta(days=3)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    events.loc[events["gw"] == 3, "deadline_time"] = upcoming
    data.events = events
    data.source = "live"

    # A bootstrap payload consistent with the season, because deciding what to
    # do reads it: the deadlines, the finished flags and which gameweek is next.
    bootstrap = {
        "events": [
            {
                "id": int(row["gw"]),
                "deadline_time": row["deadline_time"],
                "finished": bool(row["finished"]),
                "is_current": int(row["gw"]) == 2,
                "is_next": int(row["gw"]) == 3,
                "average_entry_score": row["average_entry_score"],
            }
            for row in data.events.to_dict("records")
        ],
        # Elements as the API serves them, because the snapshot reads the
        # published expected points and prices straight out of this payload.
        "element_types": [
            {"id": 1, "singular_name_short": "GKP"},
            {"id": 2, "singular_name_short": "DEF"},
            {"id": 3, "singular_name_short": "MID"},
            {"id": 4, "singular_name_short": "FWD"},
        ],
        "teams": [
            {"id": i + 1, "name": club, "short_name": short}
            for i, (club, short) in enumerate(CLUBS.items())
        ],
        "elements": [
            {
                "id": pid,
                "web_name": name,
                "element_type": {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}[position],
                "team": list(CLUBS).index(club) + 1,
                "now_cost": price,
                "ep_next": "4.0",
                "status": "a",
                "chance_of_playing_next_round": 100,
            }
            for pid, (name, position, club, price) in PLAYERS.items()
        ],
    }

    def event_live_for(gw):
        """`event/{gw}/live` as the API serves it, from the season's results.

        One request carries every player's gameweek totals, already summed
        across both fixtures of a double — which is what the scorer wants.
        """
        totals = data.at(gw).groupby("player_id")[["points", "minutes", "starts"]].sum()
        return {
            "elements": [
                {
                    "id": int(pid),
                    "stats": {
                        "total_points": int(row["points"]),
                        "minutes": int(row["minutes"]),
                        "starts": int(row["starts"]),
                    },
                }
                for pid, row in totals.iterrows()
            ]
        }

    class StubClient:
        def __init__(self, *a, **k):
            pass

        def bootstrap_static(self):
            return bootstrap

        def fixtures(self):
            return []

        def event_live(self, gw):
            return event_live_for(int(gw))

    monkeypatch.setattr(cli, "FPLClient", StubClient)
    monkeypatch.setattr(cli, "load_live_season", lambda *a, **k: data)
    monkeypatch.setattr(
        cli, "rules_from_bootstrap", lambda *a, **k: archive_season_rules("2024-25")
    )
    monkeypatch.setattr(
        cli,
        "Config",
        lambda *a, **k: Config(
            excluded_clubs=(), horizon=2, results_dir=tmp_path
        ),
    )
    # Deadlines are read from the bootstrap payload, not from the loaded season
    # — that is the point of the cheap decide step — so a test that moves a
    # deadline has to move it there.
    return SimpleNamespace(data=data, results_dir=tmp_path, bootstrap=bootstrap)


def test_track_records_the_upcoming_gameweek(live):
    results_dir = live.results_dir
    result = runner.invoke(cli.app, ["track", "--strategies", "ep"])

    assert result.exit_code == 0, result.output
    written = list((results_dir / "live").glob("*-gw03-ep.json"))
    assert len(written) == 1
    rec = json.loads(written[0].read_text())
    assert rec["gw"] == 3
    assert len(rec["squad"]) == 15
    assert "Recorded GW3" in plain(result.stdout)


def test_track_scores_nothing_on_its_first_run(live):
    """There is nothing on record yet, so nothing is scored.

    The first run records — that is something to do — but it cannot score any
    of the gameweeks already played, because nobody called them in advance.
    """
    result = runner.invoke(cli.app, ["track", "--strategies", "ep"])

    assert result.exit_code == 0
    assert "recommended XI scored" not in plain(result.stdout)
    assert "no gameweek called in advance yet" in plain(result.stdout)


def test_track_does_nothing_once_the_gameweek_is_on_record(live):
    """The idempotent case, which is what a four-hourly schedule mostly hits."""
    runner.invoke(cli.app, ["track", "--strategies", "ep"])
    result = runner.invoke(cli.app, ["track", "--strategies", "ep"])

    assert result.exit_code == 0
    assert "Nothing to do" in plain(result.stdout)
    assert "already on record" in plain(result.stdout)


def test_track_refuses_to_record_once_the_deadline_has_passed(live):
    """A record made after the deadline is worth nothing, so none is made."""
    results_dir = live.results_dir
    for event in live.bootstrap["events"]:
        if event["id"] == 3:
            event["deadline_time"] = "2024-09-03T11:30:00Z"

    result = runner.invoke(cli.app, ["track", "--strategies", "ep"])
    assert result.exit_code == 0
    assert "deadline has passed" in plain(result.stdout)
    assert not list((results_dir / "live").glob("*-gw03-*.json"))


def test_track_writes_a_status_file_for_automation(live, tmp_path):
    status = tmp_path / "status.json"
    result = runner.invoke(
        cli.app, ["track", "--strategies", "ep", "--status-file", str(status)]
    )

    assert result.exit_code == 0
    payload = json.loads(status.read_text())
    assert payload["recorded"] == ["ep"]
    assert payload["snapshot"] == 3
    assert payload["scored"] == []
    assert payload["outcomes"] == []
    assert payload["missed"] == []
    assert payload["next_gw"] == 3
    assert payload["api_players"] == 18


def test_track_will_not_overwrite_a_recommendation_already_on_record(live):
    first = runner.invoke(cli.app, ["track", "--strategies", "ep"])
    assert "Recorded GW3" in first.stdout

    second = runner.invoke(cli.app, ["track", "--strategies", "ep"])
    assert second.exit_code == 0
    assert "already on record" in second.stdout


def test_track_dry_run_decides_but_writes_nothing(live):
    results_dir = live.results_dir
    result = runner.invoke(cli.app, ["track", "--strategies", "ep", "--dry-run"])

    assert result.exit_code == 0
    assert "GW3 recommendation" in plain(result.stdout)
    assert not list((results_dir / "live").glob("*.json"))


def test_track_scores_a_gameweek_that_was_called_in_advance(live):
    """Record GW2 in advance, then let `track` find it finished and score it."""
    from fpl_optimizer.track import record, recommend
    from fpl_optimizer.rules import archive_season_rules

    data, results_dir = live.data, live.results_dir
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

    result = runner.invoke(cli.app, ["track", "--strategies", "ep"])
    assert result.exit_code == 0, result.output
    assert "GW2 ep: scored" in plain(result.stdout)
    assert "average manager 70" in plain(result.stdout)
    assert (results_dir / "live" / "2024-25.csv").exists()

    # The table gains its first row, with the gap to the average manager.
    scored = pd.read_csv(results_dir / "live" / "2024-25.csv")
    assert list(scored["gw"]) == [2]
    assert float(scored.iloc[0]["average_entry_score"]) == 70.0
    assert scored.iloc[0]["delta"] == scored.iloc[0]["points"] - 70.0

    # ...and every player's realized points are written beside the snapshot.
    outcomes = pd.read_csv(results_dir / "live" / "2024-25-gw02-outcomes.csv")
    assert len(outcomes) == 18
    assert set(outcomes.columns) == {"player_id", "points", "minutes", "starts"}


def test_track_rejects_an_unknown_strategy(live):
    result = runner.invoke(cli.app, ["track", "--strategies", "vibes"])
    assert result.exit_code != 0
    assert "unknown strategy" in plain(result.stderr)


# --- transfers --------------------------------------------------------------


@pytest.fixture
def real_team(monkeypatch, tmp_path):
    """A manager's team, the season, and the rules — all from fixtures.

    `fpl transfers` reads four entry endpoints and the season; the stub answers
    all of them, so the command is exercised without a request.
    """
    from fpl_optimizer.rules import archive_season_rules

    from .fixtures import StubEntryClient, entry_payloads
    from .fixtures.handbuilt import CLUBS, PLAYERS, hand_built_season

    data = hand_built_season()
    events = data.events.copy()
    events.loc[events["gw"] == 3, "finished"] = False
    upcoming = (datetime.now(timezone.utc) + timedelta(days=3)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    events.loc[events["gw"] == 3, "deadline_time"] = upcoming
    data.events = events
    data.source = "live"

    payloads = entry_payloads()
    bootstrap = {
        "events": [
            {
                "id": int(row["gw"]),
                "deadline_time": row["deadline_time"],
                "finished": bool(row["finished"]),
                "is_current": int(row["gw"]) == 2,
                "is_next": int(row["gw"]) == 3,
                "average_entry_score": row["average_entry_score"],
            }
            for row in data.events.to_dict("records")
        ],
        "elements": [{"id": pid} for pid in PLAYERS],
    }

    class Client(StubEntryClient):
        def bootstrap_static(self):
            return bootstrap

        def fixtures(self):
            return []

    monkeypatch.setattr(cli, "FPLClient", lambda *a, **k: Client(payloads))
    monkeypatch.setattr(cli, "load_live_season", lambda *a, **k: data)
    monkeypatch.setattr(
        cli, "rules_from_bootstrap", lambda *a, **k: archive_season_rules("2024-25")
    )
    monkeypatch.setattr(
        cli,
        "Config",
        lambda *a, **k: Config(
            excluded_clubs=(), horizon=2, results_dir=tmp_path
        ),
    )
    return SimpleNamespace(data=data, payloads=payloads, bootstrap=bootstrap)


def test_transfers_reads_the_team_and_recommends(real_team):
    result = runner.invoke(cli.app, ["transfers", "--team-id", "4242", "-s", "ep"])

    assert result.exit_code == 0, result.output
    assert "Hand Built FC" in plain(result.stdout)
    assert "is it worth it?" in plain(result.stdout)
    assert "hold (roll the transfer)" in plain(result.stdout)
    assert "starting XI" in plain(result.stdout)
    assert "bench, in order" in plain(result.stdout)


def test_transfers_reports_the_gain_against_rolling(real_team):
    result = runner.invoke(cli.app, ["transfers", "--team-id", "4242", "-s", "ep"])

    assert "Expected gain over the horizon versus rolling:" in plain(result.stdout)
    # Holding is the baseline, so its row is always zero against itself.
    assert "+0.00" in plain(result.stdout)


def test_transfers_says_when_no_hit_is_needed(real_team):
    result = runner.invoke(
        cli.app,
        ["transfers", "--team-id", "4242", "-s", "ep", "--free-transfers", "5"],
    )

    assert result.exit_code == 0, result.output
    assert "No hit needed" in plain(result.stdout) or "no transfer" in plain(result.stdout)


def test_transfers_can_be_told_not_to_consider_hits(real_team):
    result = runner.invoke(
        cli.app, ["transfers", "--team-id", "4242", "-s", "ep", "--max-hits", "0"]
    )

    assert result.exit_code == 0, result.output
    assert "Hits were not considered" in plain(result.stdout)


def test_transfers_warns_that_the_squad_may_be_stale(real_team):
    result = runner.invoke(cli.app, ["transfers", "--team-id", "4242", "-s", "ep"])
    assert "not necessarily the squad now" in plain(result.stdout)


def test_transfers_accepts_a_squad_given_by_hand(real_team):
    """For when the picks endpoint is behind — it only knows the last deadline."""
    squad = "1,7,3,5,8,10,13,2,6,9,11,14,4,12,15"
    result = runner.invoke(
        cli.app,
        [
            "transfers",
            "--team-id",
            "4242",
            "-s",
            "ep",
            "--squad",
            squad,
            "--bank",
            "1.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "1.5m in the bank" in plain(result.stdout)
    assert "rather than from the last deadline" in plain(result.stdout)


def test_transfers_accepts_names_in_a_hand_given_squad(real_team):
    names = (
        "Keeper,Reserve,Fullback,Anchor,Stopper,Overlap,Marker,"
        "Playmaker,Winger,Engine,Creator,Runner,Finisher,Poacher,Target"
    )
    result = runner.invoke(
        cli.app,
        ["transfers", "--team-id", "4242", "-s", "ep", "--squad", names, "--bank", "0"],
    )

    assert result.exit_code == 0, result.output
    assert "starting XI" in plain(result.stdout)


def test_transfers_refuses_a_squad_of_the_wrong_size(real_team):
    result = runner.invoke(
        cli.app,
        ["transfers", "--team-id", "4242", "-s", "ep", "--squad", "1,2,3", "--bank", "0"],
    )

    assert result.exit_code != 0
    assert "needs 15 players" in plain(result.stderr)


def test_transfers_refuses_an_ambiguous_name_rather_than_guessing(
    real_team, monkeypatch
):
    data = real_team.data
    frame = data.player_gws.copy()
    frame.loc[frame["player_id"] == 9, "name"] = "Keeper"
    data.player_gws = frame

    squad = (
        "Keeper,Reserve,Fullback,Anchor,Stopper,Overlap,Marker,"
        "Playmaker,Winger,Engine,Creator,Runner,Finisher,Poacher,Target"
    )
    result = runner.invoke(
        cli.app,
        ["transfers", "--team-id", "4242", "-s", "ep", "--squad", squad, "--bank", "0"],
    )

    assert result.exit_code != 0
    assert "matches more than one player" in plain(result.stderr)


def test_transfers_needs_a_team_or_a_squad(real_team):
    result = runner.invoke(cli.app, ["transfers", "-s", "ep"])
    assert result.exit_code != 0
    assert "--team-id" in plain(result.stderr)


def test_transfers_rejects_an_unknown_strategy(real_team):
    result = runner.invoke(
        cli.app, ["transfers", "--team-id", "4242", "-s", "vibes"]
    )
    assert result.exit_code != 0
    assert "unknown strategy" in plain(result.stderr)


def test_transfers_prices_each_move_separately(real_team):
    """A total gain says nothing about whether the fifth move is worth making."""
    result = runner.invoke(cli.app, ["transfers", "--team-id", "4242", "-s", "ep"])

    assert result.exit_code == 0, result.output
    out = plain(result.stdout)
    if "recommended transfers" in out:
        assert "margin behind each move" in out
        # One row per move, numbered.
        assert "#1" in out


def test_transfers_flags_a_margin_under_a_point_as_low_confidence(
    real_team, monkeypatch
):
    """The projections are not accurate to a fraction of a point.

    Forced by raising the threshold above every real margin, which is the same
    comparison the command makes — rather than hunting for a fixture whose
    numbers happen to land under it.
    """
    monkeypatch.setattr(cli, "THIN_MARGIN", 1000.0)
    result = runner.invoke(cli.app, ["transfers", "--team-id", "4242", "-s", "ep"])

    assert result.exit_code == 0, result.output
    out = plain(result.stdout)
    if "recommended transfers" in out:
        assert "low — under a point" in out
        assert "Low confidence on" in out


def test_transfers_does_not_attribute_a_margin_to_a_named_move_unless_it_can(
    real_team
):
    """The optimizer may change which players it wants when given another go.

    When the smaller plan is not a subset of the larger, the margins are per
    additional transfer allowed, and the output says so instead of naming a
    move that did not earn the figure.
    """
    result = runner.invoke(cli.app, ["transfers", "--team-id", "4242", "-s", "ep"])
    out = plain(result.stdout)

    if "not per named move" in out:
        assert "transfer(s) allowed" in out
    elif "margin behind each move" in out:
        # Nested: every row names the move it priced.
        assert "→" in out
