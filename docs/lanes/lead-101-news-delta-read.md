# LEAD-101

## Goal
News-delta read: a language model sees only items timestamped after the Tuesday freeze and forecasts the direction and size of each game's home-spread move to the deadline line (ROADMAP LEAD-101). Forward games only; nothing here is served; if it ever earns a place it enters the one probability as a fitted term.

## Declaration (frozen)
- Enrolment 2026-11-03: scripts/lead101_delta_read.py, prompt scripts/lead101_prompt.md sha256 4938b847...98e9, model claude-opus-5-5, no tools, artifacts/lead101/frozen.json. Forecast = signed home-spread move in points from the frozen pool line to the deadline line. Views: thursday (Thu 18:30 ET) and sunday (Sun 11:30 ET), scheduler jobs lead101_delta_read_*. Bar set 11-03: the read counts only if it beats the move already in the quote it was shown.
- Scoring declaration (2026-11-14 15:26 ET, before any score): views recorded on their weekday under the frozen hash and model; a row counts if recorded before its deadline (min(kickoff, Sunday 16:00 ET)) and the deadline has passed. Target T = deadline line minus frozen pool line. Deadline line = median across served books of each book's last spread quote after the record's run instant and before the deadline (nfl_ats.clv.load_decision_quotes, served move validity filter). No quote in the window = not scoreable, counted, never filled.
- Forecasts of T: the read; zero; no-change S (latest quote shown minus pool line); momentum S x (deadline - pool capture)/(quote time - pool capture). Metrics: MAE gain = baseline error minus read error, game bootstrap 95 pct interval, probability_positive; direction on T and on T - S among games moved at least the measured quote grid (0.5), statistic hits minus misses, p from permuting forecasts within week and view. Family: cells thursday, sunday, pooled x (3 error + 2 direction) = 15 looks per scoring.

## Market-excluded arm (declared 2026-11-21 12:13 ET, before any read of the arm)
- Why: LEAD-103 W11 (dropping the market block moves the LEAD-99 read 5.2x read noise, 4 side flips) and this lane (read stays on the quote shown, 49 of 54 rows). This arm can only add what the market packet does not carry.
- Prompt: scripts/lead101_market_excluded_prompt.md sha256 b18ffdb4...d656 = LEAD-99 prompt (83800017...fe67, unchanged) with one clause removed ("the latest market spread and its move since the Tuesday opener, "); nothing else changed. Model claude-opus-5-5, tools WebSearch,WebFetch (same as LEAD-99). Enrolment artifacts/lead101/market_excluded/frozen.json; code scripts/lead101_market_excluded.py (enroll / run / summarize).
- Packet: the stored LEAD-99 deadline packet (hash verified) with the `market` key deleted from every game (as scripts/lead103_ablation.py does), plus beat posts and injury headlines matching the frozen line-number pattern deleted (W11: 1 post, NO_CHI "12-point favorites"). The pool line stays: it is the number the probability is about. K = 3 reads per packet; per game the arm probability = mean logit of the valid reads from the latest arm run recorded before the game's deadline.
- Population: W11 games not kicked off at the arm's last read (12; IND_HOU out), then every W12-18 game with a LEAD-99 deadline packet and an arm read before its deadline. Expected n about 118 (schedule measured: 109 W12-18 games; inferred loss of the W12 Wednesday game and two early Thanksgiving games).
- Graded once, after Week 18 (W11 is a first forward week, nothing tuned on it). Family lead101_market_excluded_read, 7 looks: (1) arm side at the pool line vs 50%, decisive record first (games where the arm and the market-included read pick different sides); (2) paired accuracy arm minus market-included read (the scheduled LEAD-99 read of the same packet), sign-flip permutation within week; (3) paired accuracy arm minus served card side at the game's deadline; (4) log-loss and (5) Brier gain vs the market-included read, game bootstrap, reliability table alongside; (6) direction of the arm's side vs the remaining move R = deadline line (LEAD-101 scoring definition) minus the packet quote the arm did not see, games with |R| >= 0.5, hits minus misses, within-week permutation; (7) correlation of arm logit with R. Contamination reported per read: tool turns, odds-site sources, line numbers other than the pool line.
- Detectable size at n 118, alpha .05 two-sided, power .8: 12.9 accuracy points vs 50%; paired vs market-included about 14.9 points at the LEAD-103 drop-market discordance (4 of 12). Anything smaller ends unresolved_below_power; zero crossing closes nothing.

- Proposed scheduler entry (not applied; primary owns capture_scheduler.py), after the lead101_delta_read block at scripts/capture_scheduler.py:1376. `run` with no --source takes the latest LEAD-99 deadline packet of the current week and skips a packet that already has 3 valid reads, so a catch-up firing costs nothing. Thu 19:15 follows the Thu 18:00 packet (read: :594) and ends before the 20:15 kickoff; Sun 09:30 follows the Sun 08:40 packet (:595) and ends well before 12:00. There is no LEAD-99 Saturday deadline packet (:593-598), so Saturday games (W15) use the Thursday packet; the W12 Wednesday game and early Thanksgiving games have no packet before kickoff.
```diff
+    *(
+        Job(
+            f"lead101_market_excluded_{day}_{at.replace(':', '')}",
+            day,
+            at,
+            30,
+            [str(UV), "run", "--no-sync", "python",
+             str(REPO / "scripts" / "lead101_market_excluded.py"), "run"],
+            True,
+            "LEAD-101 market-excluded arm: three reads of the latest LEAD-99 deadline "
+            "packet with the market block and line numbers removed, frozen prompt and "
+            "model, recorded before each game's deadline.",
+            added_on="2026-11-21",
+        )
+        for day, at in (("thu", "19:15"), ("sun", "09:30"))
+    ),
```

- Tie rule (added 12:17 ET after the W11 reads, before any outcome): an arm mean probability of exactly 0.5 is no call, counted and reported, never a side.
- W11 run (measured 2026-11-21 12:17 ET; `run --source <W11 deadline run> --k 3`, exit 0, reads 17:14-17:16Z; `summarize ... --expected-n 118`, exit 0; artifacts/lead101/market_excluded/summary-2026-week-11-deadline-20261119T230145Z.json). Descriptive only, no outcome. Valid 3 of 3, first attempt each; 12 games (IND_HOU kicked off). Contamination: 1 turn per read (no web use), 0 odds-site sources, 0 line numbers other than the pool line. Within-read SD arm 0.047 logit vs market-included 0.030; between-game SD 0.138 vs 0.123. Corr arm v market-included mean of 4 reads 0.091 [-0.620, +0.687] P+ .571 (v scheduled read 0.160 [-0.573, +0.743] P+ .637); v LEAD-103 drop-market ablation 0.975 [0.934, 0.993]. Mean abs shift 0.130 logit; signed toward home -0.001 [-0.090, +0.097] P+ .490. Sides differ from the market-included read on 4 of 12 (CIN_WAS, LV_DEN, NYJ_LAC, TEN_DAL); MIA_BUF arm exactly 0.5 (no call).
- Reading (inferred from the per-game table): the quote shown had moved off the pool line in 8 of 12 games; the market-included read sided with that move 8 of 8, the arm 3 of 8 (1 no call). All 4 side differences are moved games; 0 of the 4 unmoved games differ. The market-included read's side at the pool line is the quote's move; the arm is a separate read.
- ROADMAP LEAD-101 update: 2026-11-21 market-excluded arm declared and enrolled (prompt b18ffdb4, LEAD-99 prompt minus the market clause; market block and line-number posts dropped; K=3; 7 looks graded once after Week 18; detectable 12.9 pts vs 50%, 14.9 paired at n 118). W11 first forward run: 3/3 valid, no web or line contamination, corr with the market-included read 0.09 [-0.62, 0.69], 4 of 12 sides differ, all where the quote had moved. Nothing recorded. Next: scheduler entries Thu 19:15 and Sun 09:30 ET (diff above) for Weeks 12-18.

## State (measured 2026-11-21 09:53 ET, `uv run --no-sync python scripts/lead101_delta_read.py score`, exit 0; artifacts/lead101/score/scores.json, rows.csv)
- The scorer already existed (added 11-14); this unit reran it unchanged on every stored run. Records 7; excluded 2 (dry, 11-03); scored 5: W9 thu, W9 sun, W10 thu, W10 sun, W11 thu. Skipped: W9 thu JAX at BAL (no quote between record and kickoff), W11 thu 12 rows (deadline not reached). 54 rows, 29 games, 3 weeks.
- Decisive record first: the read departed from the quote it was shown on 5 of 54 rows (all Thursday: W9 LA_WAS, CLE_NO, BUF_MIN, ARI_SEA; W10 NE_DET) and was closer to the deadline line on 1, farther on 4 (from rows.csv). On Sunday it equalled the shown quote on 25 of 25.
- Thursday (29 rows, 3 weeks): MAE read 0.640; gain vs zero +0.188 [-0.110, +0.507] P+ .889; vs no-change -0.105 [-0.267, +0.000] P+ .023; vs momentum +0.929 [+0.468, +1.475] P+ 1.000. Direction on T: 23 moved, 13-3, 7 no call (no-change 14-1), p .017. Remaining: 19 moved, 1-2, 16 no call (momentum 4-7), p .767.
- Sunday (25 rows, 2 weeks): MAE 0.380; vs zero +0.500 [+0.220, +0.840] P+ 1.000; vs no-change 0.000 [0, 0] .500 (dead heat, read = quote); vs momentum +0.028 [+0.013, +0.047] P+ 1.000. Direction on T: 20 moved, 13-0 (no-change 13-0), p .0001. Remaining: 11 moved, 0-0, all no call, p 1.0.
- Pooled (54 rows, 29 games): MAE 0.519; vs zero +0.332 [+0.057, +0.652] P+ .992; vs no-change -0.057 [-0.144, +0.000] P+ .024; vs momentum +0.512 [+0.266, +0.816] P+ 1.000. Direction on T: 43 moved, 26-3, 14 no call (no-change 27-1), p .0001. Remaining: 30 moved, 1-2, 27 no call (momentum 5-10), p .770.
- Reading (inferred): the whole-move record and the gain over zero are the quote in the packet; against that quote the read is a dead heat on Sunday and slightly worse on Thursday (its 5 departures went 1-4). No interval lies wholly below zero (vs no-change upper bound is 0.000), so nothing is refuted; all 15 cells `unresolved_below_power`. Nothing supports acting before the deadline on this read.
- Registry: 15 cells re-recorded with `--replace` (exit 0, "recorded": 15), family lead101_news_delta_read, units mae_improvement and rate_pp.

## Tried
- 11-03 unit 1: builder, enrolment, one dry read. 11-14 unit 2: `score` subcommand. 11-21 unit 3: rescore W9-W11 thursday, no code change, no model call. 11-21 unit 4: market-excluded arm (scripts/lead101_market_excluded.py), W11 K=3.

## Next
1. Weekly call (not added; forward_weekly.py belongs elsewhere): `_run(lead101_delta_read, "score")` then `weak-signals record --batch artifacts/lead101/score/weak_signals_batch.json --replace`.
2. Tuesday 11-24: rescore; W11 Sunday games and the W11 sunday view (if recorded 11-22) enter.
3. Market-excluded arm: declared, enrolled and first run 11-21 (section above). Primary applies the scheduler diff; until then run `uv run --no-sync python scripts/lead101_market_excluded.py run` by hand after the Thu 18:00 and Sun 08:40 LEAD-99 deadline packets (W12 Thursday packet 11-26). Grade once after Week 18 under the 7 declared looks.

## Open
- Capture gaps: no capture between Thursday 18:30 ET and kickoff in Week 9; Sunday deadlines read at 12:26 ET.
- Shown quote is one book (e.g. DraftKings); deadline line is the median of served books, so a cross-book gap can register as a remaining move.
- Week 10 freeze is Wed 11-11 20:22Z (the stored Tuesday packet ran Wednesday).
