# LEAD-101

## Goal
News-delta read: a language model sees only items timestamped after the Tuesday freeze and forecasts the direction and size of each game's home-spread move to the deadline line (ROADMAP LEAD-101). Forward games only; nothing here is served; if it ever earns a place it enters the one probability as a fitted term.

## Declaration (frozen)
- Enrolment 2026-11-03: scripts/lead101_delta_read.py, prompt scripts/lead101_prompt.md sha256 4938b847...98e9, model claude-opus-5-5, no tools, artifacts/lead101/frozen.json. Forecast = signed home-spread move in points from the frozen pool line to the deadline line. Views: thursday (Thu 18:30 ET) and sunday (Sun 11:30 ET), scheduler jobs lead101_delta_read_*. Bar set 11-03: the read counts only if it beats the move already in the quote it was shown.
- Scoring declaration (2026-11-14 15:26 ET, before any score): views recorded on their weekday under the frozen hash and model; a row counts if recorded before its deadline (min(kickoff, Sunday 16:00 ET)) and the deadline has passed. Target T = deadline line minus frozen pool line. Deadline line = median across served books of each book's last spread quote after the record's run instant and before the deadline (nfl_ats.clv.load_decision_quotes, served move validity filter). No quote in the window = not scoreable, counted, never filled.
- Forecasts of T: the read; zero; no-change S (latest quote shown minus pool line); momentum S x (deadline - pool capture)/(quote time - pool capture). Metrics: MAE gain = baseline error minus read error, game bootstrap 95 pct interval, probability_positive; direction on T and on T - S among games moved at least the measured quote grid (0.5), statistic hits minus misses, p from permuting forecasts within week and view. Family: cells thursday, sunday, pooled x (3 error + 2 direction) = 15 looks per scoring.

## State (measured 2026-11-21 09:53 ET, `uv run --no-sync python scripts/lead101_delta_read.py score`, exit 0; artifacts/lead101/score/scores.json, rows.csv)
- The scorer already existed (added 11-14); this unit reran it unchanged on every stored run. Records 7; excluded 2 (dry, 11-03); scored 5: W9 thu, W9 sun, W10 thu, W10 sun, W11 thu. Skipped: W9 thu JAX at BAL (no quote between record and kickoff), W11 thu 12 rows (deadline not reached). 54 rows, 29 games, 3 weeks.
- Decisive record first: the read departed from the quote it was shown on 5 of 54 rows (all Thursday: W9 LA_WAS, CLE_NO, BUF_MIN, ARI_SEA; W10 NE_DET) and was closer to the deadline line on 1, farther on 4 (from rows.csv). On Sunday it equalled the shown quote on 25 of 25.
- Thursday (29 rows, 3 weeks): MAE read 0.640; gain vs zero +0.188 [-0.110, +0.507] P+ .889; vs no-change -0.105 [-0.267, +0.000] P+ .023; vs momentum +0.929 [+0.468, +1.475] P+ 1.000. Direction on T: 23 moved, 13-3, 7 no call (no-change 14-1), p .017. Remaining: 19 moved, 1-2, 16 no call (momentum 4-7), p .767.
- Sunday (25 rows, 2 weeks): MAE 0.380; vs zero +0.500 [+0.220, +0.840] P+ 1.000; vs no-change 0.000 [0, 0] .500 (dead heat, read = quote); vs momentum +0.028 [+0.013, +0.047] P+ 1.000. Direction on T: 20 moved, 13-0 (no-change 13-0), p .0001. Remaining: 11 moved, 0-0, all no call, p 1.0.
- Pooled (54 rows, 29 games): MAE 0.519; vs zero +0.332 [+0.057, +0.652] P+ .992; vs no-change -0.057 [-0.144, +0.000] P+ .024; vs momentum +0.512 [+0.266, +0.816] P+ 1.000. Direction on T: 43 moved, 26-3, 14 no call (no-change 27-1), p .0001. Remaining: 30 moved, 1-2, 27 no call (momentum 5-10), p .770.
- Reading (inferred): the whole-move record and the gain over zero are the quote in the packet; against that quote the read is a dead heat on Sunday and slightly worse on Thursday (its 5 departures went 1-4). No interval lies wholly below zero (vs no-change upper bound is 0.000), so nothing is refuted; all 15 cells `unresolved_below_power`. Nothing supports acting before the deadline on this read.
- Registry: 15 cells re-recorded with `--replace` (exit 0, "recorded": 15), family lead101_news_delta_read, units mae_improvement and rate_pp.

## Tried
- 11-03 unit 1: builder, enrolment, one dry read. 11-14 unit 2: `score` subcommand. 11-21 unit 3: rescore W9-W11 thursday, no code change, no model call.

## Next
1. Weekly call (not added; forward_weekly.py belongs elsewhere): `_run(lead101_delta_read, "score")` then `weak-signals record --batch artifacts/lead101/score/weak_signals_batch.json --replace`.
2. Tuesday 11-24: rescore; W11 Sunday games and the W11 sunday view (if recorded 11-22) enter.
3. Market-excluded arm: a second enrolled prompt whose packet drops the quote; without it the read cannot be separated from the quote.

## Open
- Capture gaps: no capture between Thursday 18:30 ET and kickoff in Week 9; Sunday deadlines read at 12:26 ET.
- Shown quote is one book (e.g. DraftKings); deadline line is the median of served books, so a cross-book gap can register as a remaining move.
- Week 10 freeze is Wed 11-11 20:22Z (the stored Tuesday packet ran Wednesday).
