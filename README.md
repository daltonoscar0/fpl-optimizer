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
fpl backtest all --strategy all              # walk every season forward, score it
fpl table                                    # the results table below, as Markdown
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
nonsense and asserts that the features, the training set, every strategy's
predictions, the candidate pool and — the one that actually matters — **the
squad, XI, armband and transfers chosen for gameweek *g*** are all identical.
Only the score moves, because only the score is entitled to read results.

Two control tests keep that honest: the features *do* move when the **past**
changes, and the score *does* move when the results do. Without them, a
backtest that ignored history, or ignored results, would pass the leakage test
perfectly.

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
| `ep` | FPL's own published expected points. Fits nothing — it is a read of someone else's number, which was meant to make it the fair middle baseline. On the archive it is not one: the column turns out to know who played. See **Why `ep` is not in the table**. |
| `naive` | Points per fixture so far, carried forward. What a manager does by eye; beating it is the minimum bar. |

`model` and `naive` are per *fixture*, so a double gameweek is worth two
matches and a blank is worth none. `ep` cannot be: FPL publishes one figure per
player per gameweek, which says nothing about how many matches are in it, so it
undervalues doubles. Doubling it would be inventing data.

**Picking is an integer program, not a ranked list.** Taking the fifteen
highest-projected players is wrong, and not by a little: the budget, the quota
per position, the three-per-club cap and the need to field a legal XI all
interact, so the best legal squad routinely contains players nowhere near the
top of the projection. One solve decides the fifteen, the eleven, the armband
and the transfers.

Transfer hits are *inside* the objective rather than deducted afterwards, which
makes "should I take a hit" a decision the program makes: a transfer happens
only if it buys back more than the four points it costs. The horizon beyond the
current gameweek is a decay-weighted **average**, not a sum — summed, a
transfer's gain gets counted once per gameweek in the horizon against a hit
charged once, and the squad churns. An early version did exactly that and took
88 points of hits in ten gameweeks.

**Scoring reads results, and nothing else does.** The scorer is handed realized
points and the squad chosen before them: both legs of a double count, the
armband doubles the summed gameweek (and passes to the vice if the captain does
not play), and starters who did not play are replaced from the bench in order —
a goalkeeper only by a goalkeeper, and only when the resulting XI is still a
legal formation. A player with no fixture did not play, so a blank is
substituted like any other absence.

**A strategy that cannot see a gameweek does not get to decide it.** Where a
projection is missing or flat — `ep` in the gameweeks the archive never
collected, `naive` before a ball has been kicked — the squad is held unchanged
and the gameweek is reported as held rather than decided. This is not
fastidiousness: letting a flat projection pick produced an arbitrary squad that
spent 74.5m of 100.0m and then cost 40 points in hits to unwind.

## Results

Reproduce with `fpl backtest all --strategy all && fpl table`. Per-gameweek
records are written to `results/`.

RESULTS_TABLE_PLACEHOLDER

Totals are measured over the **common window**: the gameweeks every strategy in
a season actually decided. They start in different places and that has to be
accounted for rather than averaged away — `naive` has no opinion until a
gameweek has been played, and the model has none until it has three gameweeks
to fit on, so the earliest gameweeks belong to neither. Comparing a 38-gameweek
total against a 34-gameweek one would flatter whoever played longest.

What the comparison is for: both strategies run through *identical* machinery —
same candidate pool, same integer program, same scorer, same rules — so the gap
between them is attributable to the projections and to nothing else.

### Why `ep` is not in the table

FPL's own published expected points were meant to be the honest middle
baseline. The backtest scores them at 3349 points in 2023-24 and 3185 in
2024-25, or 84–88 per gameweek. Nobody scores 88 points a gameweek over a
season; the record for ten million players is nearer 71.

That is the number doing the telling. The archive's `xP` rates 64–66% of
non-appearances at exactly zero, so a strategy reading it already knows who
played — see **Known gaps**. The figures above are reported here as the
evidence for excluding it, not as a result. `results/` keeps the full
per-gameweek records for anyone who wants to check the claim.

## Known gaps

These are properties of the available data, not of the code, and they constrain
what the results table can honestly claim.

- **The archive's `xP` column already knows who played, so it is not a
  baseline.** This was meant to be the fair comparison — FPL's own number,
  nothing fitted. It is not one. Across all three seasons, **64–66% of
  player-gameweeks with zero minutes are rated exactly 0.0**, against 1.3% of
  the player-gameweeks where someone played an hour or more. A pre-deadline
  forecast cannot do that: FPL quotes an `ep_next` for every registered player,
  including the nailed starter who turns out to be a surprise omission. The
  column has been recorded after the fact.

  Since playing time is most of the variance in an FPL score, a strategy
  reading it is being told the answer rather than forecasting it — which is why
  it "scores" 84–88 points a gameweek, well past any human. `fpl data` says so
  in the season notes, `ep_looks_like_hindsight` measures it, and the model
  refuses to use it as a fallback. It is excluded from the results table for
  the same reason the average-manager row is: a number that cannot mean what it
  appears to mean is worse than no number.

- **FPL's published expected points are also patchy.** In 2025-26, 27 of 38
  gameweeks record `xP` as 0.0 for *every* player — a failed collection rather
  than a forecast that nobody will score. Those gameweeks are nulled, so `ep`
  reports itself unavailable instead of optimizing against a flat zero, and the
  season falls below the 80% coverage threshold at which the strategy is run at
  all.

- **No average-manager benchmark for completed seasons.** The FPL API serves
  `events` (and therefore `average_entry_score`) for the current season only,
  and the archive carries none for any of the three. Rather than estimate it,
  the comparison is omitted.
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

## Data sources and credits

- Completed seasons come from
  [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League),
  a community archive of FPL data. Per season this project reads
  `gws/merged_gw.csv`, `teams.csv` and `fixtures.csv`. Downloaded files are
  cached under `data/raw/` and never refetched.
- The current season comes from the public FPL API. Access is read-only by
  construction: only `GET`, only an allowlist of public endpoints, at most one
  request per second, every response cached to disk. There is no login and no
  write path.

Neither source is affiliated with this project, and nothing here is endorsed by
the Premier League or FPL.

## Licence

MIT — see [LICENSE](LICENSE).
