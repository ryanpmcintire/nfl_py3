# ENG-55

## Goal
Pool-line inference audit: which line the Splash pool copies each Tuesday, so the served hook anchor names that mechanism (ROADMAP ENG-55).

## State
Units 1-2 (11-03): artifacts/eng55_pool_line_audit.csv, eng55_anchor_compare.csv. Unit 3 (11-21 10:15): Wk7-8 row badges never rendered; their hook flags rest on the 10-31 `-reparse` folders; GB_NE Wk9 showed a false "¼ pt", Wk11 TEN_DAL shows "¾ pt" (books 8.0/8.0/8.5/8.5 -> 8.25); Wk5 had no Tuesday odds job until 10-13.
Unit 4 done 2026-11-21 10:32 EST (measured; tests/scratch/eng55/u4_books.py, u4_decide.py, u4_decisive.py and .out; rows u4_rows_{cap,noon,before}.csv). 163 pool games Wk1-11; 186 live folders (plain + `-odds-gap-private`, quotes.parquet plus each payload's all_books), 29 excluded (13 `-reparse`, 16 halves). Pool posts no whole numbers (0/163). Each book's line at the snapshot nearest the pool capture:
- FanDuel 163 games, 11 weeks: exact 127 (77.9%), consistent (half line equal, or whole line moved 0.5) 151 (92.6%); half lines equal 127/139, whole lines hooked 24/24. Excluding Wk5: 143/148, half lines 121/126 (96.0%). Per week consistent 16/16 16/16 15/16 15/16 8/15 14/14 14/14 12/14 15/15 13/14 13/13: no mid-season source change.
- Next best: betmgm 147/162, betrivers 145/163, MEDIAN_ALL 144/163 (55 games need a hook), bovada 142, mybookie 142, williamhill 140, DraftKings 137 (exact 93), MODE_ALL 140 (5 ties), served anchor (current code) 135/163 (exact 75). Exchanges (novig, polymarket, kalshi, fliff) post only half lines: exact 104-108 of 146.
- Decisive games (consistent-value sets disjoint), pool in FanDuel's set / other's / neither: v served anchor 10-0-1 (sign test p .002), DraftKings 14-0-1 (p .0001), median 7-0-1, mybookie 11-0-2, williamhill 11-0-1, bovada 8-0-1, novig 22-0-1, polymarket 23-0-1, pinnacle 13-1-1, betmgm 5-0-1.
- Nearest-noon rule: FanDuel 152/163, same picture. Last-snapshot-before-capture rule: FanDuel 141/163; it falls only in Wk4-5 (7/16, 6/15), where that snapshot is the prior Sunday (-51 h, -54 h).
- Whole FanDuel line -> pool side (24 games): favourite lays more 14, underdog gets more 10; home lays more 17, away 7. No side rule, so the anchor stays the whole number.
- FanDuel misses outside Wk5 (4): PHI_CHI Wk3, IND_JAX and PHI_WAS Wk8, SEA_LV Wk10 (fixed at noon): every book sits a point from the pool, so the pool copied an earlier number (inferred). ARI_NYG Wk4: pool NYG -1.5, every book ARI -1.0 to -2.5 (unexplained).
- Week 5 coverage gap: nearest snapshot is Wed 18:00 ET (+22.6 h); FanDuel 8/15. nflverse schedules spread_line fetched 10-06 18:23 ET (data/raw/20261006T222324Z): exact half 6, hook 3, other 6 of 15. Neither source reproduces the Wk5 pool; the copy time is earlier and unobserved. Unresolved.
Conclusion (inferred from the measured rows): the pool copies FanDuel's Tuesday line and moves a whole number a half point in a direction with no rule. The earlier mod57 claim (reported) holds. Looks: 72 book-rule cells (3 timing rules x 24 rows), 739 week cells, 24 decisive comparisons; nothing recorded (a measurement, no registry cell named).

## Proposed diffs (not applied; pinned)
1. pool_hook.py:85-118 anchor = FanDuel's Tuesday quote nearest the pool capture, no median, no fallback (no anchor -> term 0):
```
+POOL_SOURCE_BOOK = "fanduel"
-    observed = pd.to_datetime(quotes["observed_at_utc"], utc=True)
-    snapshots = (...)   (old lines 101-118 removed whole)
+    observed = pd.to_datetime(quotes["observed_at_utc"], utc=True)
+    captured = pd.Timestamp(capture.captured_at_et).tz_convert("UTC")
+    wednesday = (opened.tz_convert(POOL_TIMEZONE).normalize() + pd.Timedelta(days=1)).tz_convert("UTC")
+    copied = quotes.loc[observed.lt(wednesday) & quotes["bookmaker_key"].eq(POOL_SOURCE_BOOK)]
+    distance = (pd.to_datetime(copied["observed_at_utc"], utc=True) - captured).abs()
+    nearest = copied.assign(distance=distance).sort_values("distance").groupby("nflverse_game_id").first()
+    return {str(game): float(value) for game, value in nearest["home_spread_line"].items()}
```
2. Prerequisite (not pinned): scripts/capture_odds_gap_private.py:22 `LEADERS = ("bovada", "williamhill_us", "mybookieag", "fanduel")`. History has FanDuel only inside each live folder's response.json (Wk3-11) and the Odds API quotes (Wk1-2); a read-time loader of those stored payloads in market_data.load_quote_history is needed, never a rewrite of data/market/raw. `-reparse` folders hold no FanDuel rows, so diff 1 also removes their effect (unit 3 diff 1).
Effect (measured, unit push table): Wk11 TEN_DAL anchor 8.25 -> 7.5 = pool: term +2.6 -> 0, "¾ pt" badge gone. Wk9 GB_NE 3.0 (2.75 live then) -> 2.5 = pool: no term, no badge. Term-active games Wk1-11 88 -> 36 of 163; 58 change; quarter anchors 6 -> 0. Replay Wk1-11 and re-enroll validation before serving.

## Next
Primary: apply diffs 1-2 with the payload loader, re-enroll the validation cohort, replay Wk1-11 hook flags and the History strip. Weekly: rerun u4_books.py after each Tuesday capture (FanDuel consistent rate per week).

## Open
Payload loader versus forward-only FanDuel capture (history anchors vanish without it). Wk5 copy time unobserved. ARI_NYG Wk4 unexplained.

ROADMAP ENG-55 update: Unit 4 (2026-11-21): the pool copies FanDuel's Tuesday line (consistent 151/163 Wk1-11, 143/148 without the Wk5 gap; decisive games v served anchor 10-0-1, v DraftKings 14-0-1); a whole FanDuel number moves a half point with no side rule. Proposed pinned diff anchors pool_hook on FanDuel (TEN_DAL Wk11 8.25 -> 7.5, GB_NE Wk9 -> 2.5, both term 0); needs FanDuel in the quote loader first.

## Root decision (2026-11-21 10:38 EST): keep the median anchor; do not apply the FanDuel diff
- Measured by unit 4: the pool copies FanDuel's Tuesday line (151 of 163 fit; decisive games against the served anchor 10-0-1).
- The served hook term prices the pool line against the market's consensus number: it was fitted on 2020-25 rows where the pool line is a half point off the consensus market line (pool_hook.hook_rows), so the live analogue of that construction is the consensus (median of books), not the copy source. When FanDuel sits half a point off the other books, the pool line is genuinely better or worse than the market for one side, and that is the value the term measures. Anchoring on FanDuel would zero the term exactly when the copied book is off-market (TEN at DAL Week 11: pool 7.5 against books at 8, 8, 8.5, 8.5), which contradicts the fitted mechanism (inferred).
- What the copy-source finding does change: nothing served; it explains why the pool never posts a whole number and is the mechanism for predicting the pool line before it is posted (not needed: the board is read each Tuesday).
- Open for a declared season-boundary test: whether the term's value at quarter-point consensus anchors (even book counts) matches its fitted weight; measure on 2026 forward games by anchor type before any change.
