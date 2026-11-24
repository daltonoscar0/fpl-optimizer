"""Read-only, cached, rate-limited access to the public FPL endpoints.

Three promises this module keeps:

* **Read-only.** Only ``GET`` is ever issued, and only to the public endpoints
  listed in :data:`ENDPOINTS`. There is no login, no session cookie and no
  write path. Attempting to fetch anything outside those endpoints raises.
* **Cached.** Every response is written to disk with a fetch timestamp. Within a
  single run, a URL is fetched at most once regardless of TTL.
* **Polite.** At most one request per second, process-wide.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from .config import API_BASE, Config

log = logging.getLogger(__name__)

#: Endpoint patterns this client is allowed to request. Anything else raises,
#: which is the mechanism that keeps the tool read-only by construction.
ENDPOINTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^bootstrap-static/$"),
    re.compile(r"^fixtures/$"),
    re.compile(r"^fixtures/\?event=\d+$"),
    re.compile(r"^element-summary/\d+/$"),
    re.compile(r"^event/\d+/live/$"),
    re.compile(r"^entry/\d+/$"),
    re.compile(r"^entry/\d+/history/$"),
    re.compile(r"^entry/\d+/event/\d+/picks/$"),
)


class ReadOnlyViolation(RuntimeError):
    """Raised if a caller tries to reach an endpoint outside the allowlist."""


@dataclass
class CacheEntry:
    """A cached response plus when it was fetched."""

    url: str
    fetched_at: datetime
    payload: Any

    @property
    def age(self) -> timedelta:
        return datetime.now(timezone.utc) - self.fetched_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "fetched_at": self.fetched_at.isoformat(),
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CacheEntry:
        return cls(
            url=d["url"],
            fetched_at=datetime.fromisoformat(d["fetched_at"]),
            payload=d["payload"],
        )


class _RateLimiter:
    """Process-wide minimum interval between outbound requests."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            elapsed = time.monotonic() - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


@dataclass
class FPLClient:
    """Cached, rate-limited, read-only FPL API client."""

    config: Config = field(default_factory=Config)
    offline: bool = False
    """When set, only the disk cache is consulted; a miss raises."""

    _memo: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _session: requests.Session | None = field(default=None, init=False, repr=False)
    _limiter: _RateLimiter | None = field(default=None, init=False, repr=False)
    fetch_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = _RateLimiter(self.config.min_request_interval)

    # --- public endpoints ---------------------------------------------------

    def bootstrap_static(self) -> dict[str, Any]:
        """Players, prices, clubs, positions, gameweeks, chips, game settings."""
        return self.get("bootstrap-static/")

    def fixtures(self, event: int | None = None) -> list[dict[str, Any]]:
        """All fixtures, or just one gameweek's, with difficulty ratings."""
        path = "fixtures/" if event is None else f"fixtures/?event={event}"
        return self.get(path)

    def element_summary(self, player_id: int) -> dict[str, Any]:
        """One player's per-gameweek history this season and future fixtures."""
        return self.get(f"element-summary/{int(player_id)}/")

    def event_live(self, event: int) -> dict[str, Any]:
        """Every player's realized stats for one gameweek, in one request.

        The same figures ``element-summary`` carries per player, which would be
        one request each — six hundred and seventy-four of them for a single
        gameweek's results. Points are already summed across a double
        gameweek's fixtures here, which is what the scorer wants.
        """
        return self.get(f"event/{int(event)}/live/")

    def entry(self, team_id: int) -> dict[str, Any]:
        """A manager's public entry summary (name, bank, squad value, chips)."""
        return self.get(f"entry/{int(team_id)}/")

    def entry_history(self, team_id: int) -> dict[str, Any]:
        """A manager's per-gameweek history, including bank and squad value."""
        return self.get(f"entry/{int(team_id)}/history/")

    def entry_picks(self, team_id: int, event: int) -> dict[str, Any]:
        """A manager's picks for one gameweek, with purchase/selling prices."""
        return self.get(f"entry/{int(team_id)}/event/{int(event)}/picks/")

    # --- plumbing -----------------------------------------------------------

    def get(self, path: str, *, force: bool = False) -> Any:
        """Fetch ``path`` relative to the API base, via memo then disk then HTTP."""
        self._check_allowed(path)
        if not force and path in self._memo:
            return self._memo[path]

        entry = self._read_cache(path)
        fresh_enough = (
            entry is not None
            and not force
            and entry.age < timedelta(hours=self.config.cache_ttl_hours)
        )
        if fresh_enough:
            assert entry is not None
            self._memo[path] = entry.payload
            return entry.payload

        if self.offline:
            if entry is not None:
                # Stale is better than nothing when explicitly offline.
                log.warning("offline: using cache for %s (age %s)", path, entry.age)
                self._memo[path] = entry.payload
                return entry.payload
            raise RuntimeError(f"offline and no cached response for {path!r}")

        payload = self._fetch(path)
        self._write_cache(path, payload)
        self._memo[path] = payload
        return payload

    def _check_allowed(self, path: str) -> None:
        if not any(p.match(path) for p in ENDPOINTS):
            raise ReadOnlyViolation(
                f"{path!r} is not one of the allowed read-only endpoints"
            )

    def _fetch(self, path: str) -> Any:
        url = urljoin(API_BASE, path)
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {"User-Agent": self.config.user_agent, "Accept": "application/json"}
            )
        assert self._limiter is not None
        last_error: Exception | None = None
        for attempt in range(3):
            self._limiter.wait()
            try:
                log.info("GET %s", url)
                resp = self._session.get(url, timeout=30)
                self.fetch_count += 1
                if resp.status_code == 404:
                    raise FileNotFoundError(f"{url} returned 404")
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 5 * (attempt + 1)))
                    log.warning("rate limited on %s; sleeping %.1fs", url, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except FileNotFoundError:
                raise
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                log.warning("GET %s failed (attempt %d): %s", url, attempt + 1, exc)
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"failed to fetch {url}: {last_error}")

    def _cache_path(self, path: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_") or "root"
        return self.config.cache_dir / f"{safe}.json"

    def _read_cache(self, path: str) -> CacheEntry | None:
        p = self._cache_path(path)
        if not p.exists():
            return None
        try:
            return CacheEntry.from_dict(json.loads(p.read_text()))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("discarding unreadable cache file %s: %s", p, exc)
            return None

    def _write_cache(self, path: str, payload: Any) -> None:
        entry = CacheEntry(
            url=urljoin(API_BASE, path),
            fetched_at=datetime.now(timezone.utc),
            payload=payload,
        )
        p = self._cache_path(path)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry.to_dict()))
        tmp.replace(p)

    def clear_cache(self) -> int:
        """Delete every cached API response. Returns the number of files removed."""
        n = 0
        for p in sorted(self.config.cache_dir.glob("*.json")):
            p.unlink()
            n += 1
        self._memo.clear()
        return n

    def cache_status(self) -> list[tuple[str, datetime, int]]:
        """``(filename, fetched_at, size_bytes)`` for each cached response."""
        rows = []
        for p in sorted(self.config.cache_dir.glob("*.json")):
            try:
                entry = CacheEntry.from_dict(json.loads(p.read_text()))
            except Exception:  # noqa: BLE001 - a corrupt file is still worth listing
                continue
            rows.append((p.name, entry.fetched_at, p.stat().st_size))
        return rows
