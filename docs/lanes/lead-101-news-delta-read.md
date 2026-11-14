# LEAD-101

## Goal
News-delta read: a language model sees only items timestamped after the Tuesday freeze and forecasts the direction and size of each game's home-spread move to the deadline line (ROADMAP LEAD-101). Forward games only; nothing here is served; if it ever earns a place it enters the one probability as a fitted term.

## Declaration (frozen)
- Enrolment 2026-11-03: scripts/lead101_delta_read.py, prompt scripts/lead101_prompt.md sha256 4938b847...98e9, model claude-opus-5-5, no tools, artifacts/lead101/frozen.json. Forecast = signed home-spread move in points from the frozen pool line to the deadline line. Views: thursday (Thu 18:30 ET) and sunday (Sun 11:30 ET), scheduler jobs lead101_delta_read_*. Bar set 11-03: the read counts only if it beats the move already in the quote it was shown.
- Scoring declaration (2026-11-14 15:26 ET, written before any score existed). Records: views thursday and sunday recorded on that weekday under the frozen hash and model; a row counts if recorded before its deadline (packet deadline_et = min(kickoff, Sunday 16:00 ET)) and the deadline has passed. Target T = deadline line minus frozen pool line. Deadline line = median across books of each book's last spread quote observed after the record's run instant and before the deadline, from data/market/raw live captures via nfl_ats.clv.load_decision_quotes with the served move term's validity filter (sharp_book_movement_features.py:117-131). No quote in that window = not scoreable, counted, never filled.
- Forecasts of T: the read; zero; no-change S = packet's latest quote minus pool line (same as a no-change forecast of the remaining move T - S); momentum = S times (deadline - pool capture) / (quote time - pool capture). Metrics: mean absolute error, gain = baseline error minus the read's error, game bootstrap 95 pct interval and probability_positive; direction on T and on T - S among games that moved at least the quote grid (measured from the store: 0.5), zero forecast = no call, statistic hits minus misses, p from permuting forecasts within week and view.
- Family: 1 prompt, cells thursday, sunday, pooled, each 3 error baselines + 2 direction records = 15 looks.

## State (measured 2026-11-14 15:29 ET, `uv run --no-sync python scripts/lead101_delta_read.py score`, artifacts/lead101/score/scores.json, rows.csv)
- Records: 5 files. Excluded 2 (both 2026-11-03, view dry: the unit-1 trial and the scheduler `--run-job --dry` write). Scored 3: Week 9 thursday (recorded 11-05 23:31Z), Week 9 sunday (11-08 16:31Z), Week 10 thursday (11-12 23:31Z); every row recorded before its deadline. Rows: Wk9 thursday 14 of 15 (JAX at BAL had no capture between the record and kickoff), Wk9 sunday 13 of 13, Wk10 thursday 1 of 14 (13 deadlines not reached).
- Thursday (15 games, 2 weeks): error read 0.650; gain vs zero -0.017 [-0.333, +0.250] probability_positive 0.469; vs no-change -0.183 [-0.483, +0.017] 0.050; vs momentum +1.020 [+0.376, +1.851] 1.000. Direction on T: 12 moved, 6-2, 4 no call (no-change 7-0), permutation p 0.124. Remaining move: 11 moved, 1-2, 8 no call (momentum 2-4), p 0.797.
- Sunday (13 games, 1 week): error read 0.269; vs zero +0.385 [+0.192, +0.577] 1.000; vs no-change 0.000 [0, 0] 0.500, a dead heat because the read equalled the quote it was shown on all 13 games; vs momentum +0.033 [+0.010, +0.060] 0.999. Direction on T: 10 moved, 8-0, 2 no call (no-change 8-0), p 0.0003. Remaining: 5 moved, 0-0, 5 no call (momentum 1-2), p 1.0.
- Pooled (28 rows, 15 games, 2 weeks): error read 0.473; vs zero +0.170 [-0.025, +0.362] 0.956; vs no-change -0.098 [-0.259, +0.009] 0.050; vs momentum +0.562 [+0.208, +1.028] 1.000. Direction on T: 22 moved, 14-2, 6 no call (no-change 15-0), p 0.002. Remaining: 16 moved, 1-2, 13 no call (momentum 3-6), p 0.804.
- Reading (inferred): the direction record on the whole move comes from the quote in the packet; the read left that quote on 24 of 28 rows and its four departures went 1-2 with one on an unmoved game. The momentum gain says constant-rate extrapolation is a poor yardstick, not that the read is good. Two weeks (one for Sunday) is a small sample: neither a negative nor a confirmation; all 15 cells are `unresolved_below_power`.
- Card: the card already takes the realised move at the deadline as a fitted input, so a move forecast matters only if the card could act earlier at the frozen pool line. So far the read is no closer than the quote available at the same instant, so nothing here supports acting earlier; keep logging.
- Registry: 15 cells recorded, family lead101_news_delta_read (registry/weak_signals/lead101_news_delta_read.json), units mae_improvement and rate_pp, no reliability value (not measurable on two weeks). `weak-signals pool --league nfl --effect-units accuracy_points` loads (exit 0).

## Tried
- 11-03 unit 1: builder, enrolment, one dry read; novelty check vs LEAD-99, SKY-01, MKT-08, MKT-20. 11-14 unit 2: `score` subcommand added to the runner (ruff check and format clean); no model call; stored reads untouched.

## Next
1. Weekly call (not added; forward_weekly.py is being edited elsewhere): in scripts/forward_weekly.py add `import lead101_delta_read` with the other imports and `_run(lead101_delta_read, "score")` after `_run(eng50_online_stacker)`. Then the orchestrator re-records with `nfl-ats weak-signals record --batch artifacts/lead101/score/weak_signals_batch.json --replace`.
2. Tuesday 11-17: rerun `score`; Week 10 adds 13 thursday rows and the sunday view.
3. Market-excluded arm: a second enrolled prompt whose packet drops the quote, logged beside this one from the next Thursday; without it the read cannot be separated from the quote.

## Open
- Capture gaps limit the target: no capture between Thursday 18:30 ET and kickoff in Week 9; Sunday 16:00 ET deadlines are read at 12:26 ET; CIN at ATL (09:30 ET) at Saturday 10:01 ET.
- The quote shown on Sunday was DraftKings; the deadline line is the median of bovada, Caesars and mybookie (the only books passing the served filter), so a cross-book gap can register as a remaining move.
- Week 10 freeze is Wed 2026-11-11 20:22Z (the stored Tuesday packet ran Wednesday), so that week's packet starts later than Tuesday.
