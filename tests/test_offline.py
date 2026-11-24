"""The guard in :mod:`tests.conftest` works, and the data paths respect it."""

from __future__ import annotations

import socket

import pytest

from fpl_optimizer.live import load_live_season

from .conftest import NetworkUsedInTest
from .fixtures import live_payloads, load_archive_fixture


def test_opening_a_connection_fails_the_test():
    with pytest.raises(NetworkUsedInTest):
        socket.socket().connect(("127.0.0.1", 9))
    with pytest.raises(NetworkUsedInTest):
        socket.create_connection(("127.0.0.1", 9))


def test_the_archive_path_reads_only_committed_files():
    """``load_season`` with explicit paths must not fall back to downloading."""
    season = load_archive_fixture()
    assert len(season.player_gws) > 0


def test_the_live_path_reads_only_injected_payloads():
    """``load_live_season`` with payloads must not construct a client."""
    bootstrap, fixtures, summaries = live_payloads()
    season = load_live_season(
        bootstrap=bootstrap, fixtures=fixtures, summaries=summaries
    )
    assert len(season.player_gws) > 0
