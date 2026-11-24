# fpl-optimizer

Integer-programming squad and transfer optimizer for Fantasy Premier League,
with leakage-free backtests.

The interesting part of FPL is not picking good players — it is picking the best
*legal* squad, which is a constrained optimization problem: 15 players, a budget,
quotas per position, at most three from any one club, and a starting XI that has
to make a formation. Greedy picking by expected points gets this wrong. An
integer program does not.

The other half is knowing whether a projection is actually any good, which is
where most FPL models quietly fall over: it is very easy to build a backtest that
reads a result before deciding on it and reports a season score no strategy could
have achieved. This project is built so that mistake is hard to make.

## Install

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11 is required (`jax` and `pulp` pins).

## Use

```sh
fpl seasons                                  # archive seasons, and what is on disk
fpl rules 2024-25                            # the rule set in force, and its provenance
fpl data 2025-26                             # load a season and print its shape and gaps
fpl project 2025-26 38 --strategy model      # expected points for one gameweek
```

Every command prints the data's *notes* alongside the numbers. A projection
running on a fallback, or a season whose expected-points column was never
collected upstream, is exactly the sort of thing that becomes a wrong conclusion
three steps later if it is not said out loud.

## How it works

**One schema, one gate.** Both sources — the live FPL API and the community
archive of completed seasons — normalize into a single `SeasonData`
(`schema.py`). Nothing downstream touches either source directly. Every
player-gameweek column is classified as either *pre-deadline* (published before
the deadline: price, FPL's own expected points, the fixture) or an *outcome*
(only exists once the match is played). A test compares the two sources'
schemas column by column and dtype by dtype, so a source that drifts fails
loudly rather than silently producing a second shape.

**No leakage, by construction and by test.** Deciding for gameweek *g* may read
outcome columns strictly before *g*, and pre-deadline columns of *g* itself.
`SeasonData.pre_deadline_at(g)` returns a frame with the outcome columns
*physically absent*, so a leak cannot be written by accident. The test suite
makes the stronger claim: it rewrites every outcome column from *g* onward to
nonsense and asserts that features, the training set, and all three strategies'
predictions are bit-identical. A control test asserts the features *do* move
when the past changes — without it, a feature builder that ignored history
entirely would pass perfectly.

Opponent strength comes from a rolling record of goals already scored and
conceded, not from FPL's difficulty ratings: for a completed season those
ratings are an end-of-season snapshot, so using them would leak.

**Three projections, one interface.** All three answer the same question — how
many points will this player score in this gameweek — so the backtester runs
them through identical machinery and any difference in results is attributable
to the projections rather than to the plumbing.

| Strategy | What it is |
|---|---|
| `model` | A three-part generalized linear model in JAX. Most of the variance in an FPL score is *whether the player played*, so rather than spend one regression's capacity blending the regimes it fits P(60 minutes), P(cameo) and Poisson points within each, then multiplies. Every head is linear, so the fitted model explains itself. |
| `ep` | FPL's own published expected points. Fits nothing, which is what makes it a fair baseline. |
| `naive` | Points per fixture so far, carried forward. What a manager does by eye; beating it is the minimum bar. |

All three are per *fixture*, so a double gameweek is worth two matches and a
blank is worth none.

## Results

Not yet filled in — the walk-forward backtest is the last layer and is still
being built. The table below is the shape it will take; the gameweek counts
matter as much as the totals, for the reasons in **Known gaps**.

| Season | Strategy | Gameweeks | Total points | vs `naive` |
|---|---|---|---|---|
| — | — | — | — | — |

## Known gaps

These are properties of the available data, not of the code, and they constrain
what the results table can honestly claim.

- **FPL's published expected points are patchy in the archive.** In 2025-26, 27
  of 38 gameweeks record `xP` as 0.0 for every player — a failed collection
  rather than a forecast that nobody will score. Those gameweeks are nulled, so
  the `ep` baseline reports itself unavailable instead of optimizing against a
  flat zero. Where `ep` is compared, it is compared only on the gameweeks that
  carry it, and the count is stated.
- **No average-manager benchmark for completed seasons.** The FPL API serves
  `events` (and therefore `average_entry_score`) for the current season only,
  and the archive carries none. Rather than estimate it, the comparison is
  omitted.
- **Past seasons' rules are asserted, not read.** Live rules come from
  `bootstrap-static`. The archive carries no rules snapshot, so
  `archive_season_rules` states them per season — the one place in the project
  where a number is hard-coded rather than read. `fpl rules <season>` prints
  which it is.

## Development

```sh
pytest                  # the suite is offline; sockets are blocked per test
pytest -m network       # the few tests that may reach the network
```

The suite runs with network access blocked, so a test cannot accidentally pass
against whatever the FPL API happened to be serving that day; the committed
fixtures in `tests/fixtures/` are the thing under test.

## Licence

MIT — see [LICENSE](LICENSE).
