"""Shared test configuration.

The suite is offline by default, and this is what makes that true rather than
merely intended: every test runs with socket connections blocked. A test that
wants the network must say so with ``@pytest.mark.network``, which CI deselects.

The point is not tidiness. Both data sources — the FPL API and the archive —
fetch over HTTP, so a test that silently reached the network would pass against
whatever the server happened to be serving that day, and the committed fixtures
would stop being the thing under test.
"""

from __future__ import annotations

import socket

import pytest


class NetworkUsedInTest(RuntimeError):
    """Raised when an unmarked test tries to open a connection."""


_MESSAGE = (
    "this test tried to open a network connection: use the committed fixtures "
    "in tests/fixtures, or mark the test with @pytest.mark.network"
)


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return

    def blocked(*args, **kwargs):
        raise NetworkUsedInTest(_MESSAGE)

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
