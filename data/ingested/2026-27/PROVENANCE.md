# 2026-27 — where each field came from

Built by `fpl ingest` from the public FPL API, one gameweek at a time,
in the archive's own CSV format so the same loader reads both.

| field | source | exactness |
|---|---|---|
| `total_points` | event/{gw}/live | exact |
| `minutes` | event/{gw}/live | exact |
| `goals_scored` | event/{gw}/live | exact |
| `assists` | event/{gw}/live | exact |
| `clean_sheets` | event/{gw}/live | exact |
| `goals_conceded` | event/{gw}/live | exact |
| `own_goals` | event/{gw}/live | exact |
| `penalties_saved` | event/{gw}/live | exact |
| `penalties_missed` | event/{gw}/live | exact |
| `yellow_cards` | event/{gw}/live | exact |
| `red_cards` | event/{gw}/live | exact |
| `saves` | event/{gw}/live | exact |
| `starts` | event/{gw}/live | exact |
| `bonus` | event/{gw}/live | exact |
| `bps` | event/{gw}/live | exact |
| `defensive_contribution` | event/{gw}/live | exact |
| `fixture` | event/{gw}/live (explain) | exact |
| `kickoff_time` | fixtures | exact |
| `team_h_score` | fixtures | exact |
| `team_a_score` | fixtures | exact |
| `was_home` | fixtures + the player's club | exact unless the club moved |
| `opponent_team` | fixtures | exact |
| `name` | bootstrap-static | exact |
| `position` | bootstrap-static | today's position, not that gameweek's |
| `team` | fixtures + bootstrap-static | exact unless the club moved |
| `value` | element-summary history, or the weekly pre-deadline snapshot | exact when supplied, otherwise the nearest known price |
| `selected` | element-summary history | zero when not supplied |
| `transfers_in` | element-summary history | zero when not supplied |
| `transfers_out` | element-summary history | zero when not supplied |
| `xP` | unavailable | FPL does not retain past expected points |

A gameweek is ingested once FPL marks its data checked, and is never
rewritten afterwards: results are revised for a day or two after the
whistle, and rewriting them would silently change what a model was
trained on.

`xP` is empty throughout. FPL does not retain its own past
expected-points values, so no backfilled gameweek can carry one. The
pre-deadline snapshots in `results/live/` are that figure, recorded
weekly from the gameweek this project started watching.
