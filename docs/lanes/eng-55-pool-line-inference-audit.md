# ENG-55

## Goal
Pool-line inference audit: is the pool line always the Tuesday market plus or minus half a point (ROADMAP ENG-55).

## State
Units 1-2 (2026-11-03): 136 games Wk1-9, pool minus first post-noon snapshot (A) 0/0.5/1/other = 66/53/13/4; Wk1-8 hooks 44 of 121 = POOL-08. Artifacts: artifacts/eng55_pool_line_audit.csv, artifacts/eng55_anchor_compare.csv.
Unit 3 done 2026-11-21 10:15 EST. Scratch: tests/scratch/eng55/{badges_in_pages.py, anchor_now.py, book_split.py} (read git versions of docs/*.html; no render).
- Row badges never existed for Wk7-8 (read): UI-31 (6dfffe09, 11-03 16:35) shipped after Wk8 was final and board_terminal.py:1002 draws the badge only when `not game.final`. IND_MIN, MIA_NYJ, LV_NYJ, PHI_WAS rows show no badge in every docs/index.html since. The hook set reaches the page only through the History week strip (board_site_content.py:533-559): Wk7 "7 games" at 83593d52 and HEAD; Wk8 "5" at 83593d52 -> "4" at HEAD (PHI_WAS dropped by 940c8140); Wk6 6 -> 5 (BUF_LV); Wk9 "10" at HEAD.
- Wk7-8 anchors rest on the 10-31 back-fill (measured): the original odds-gap folders 20261020T182620Z and 20261027T182541Z have 0 rows for these games; every mybookie/williamhill quote comes from the `-reparse` folders. Live Bovada at lock: IND_MIN 6.0 (pool 6.5 = hook), MIA_NYJ 4.5 (= pool, no hook), LV_NYJ 1.0 (pool 1.5 = hook), PHI_WAS 5.5 (pool 4.5, 1 pt). With the re-parse (A and current code) each flips: IND_MIN none, MIA_NYJ hook, LV_NYJ none (re-parse mybookie -1.0 against Bovada +1.0, likely a sign error, inferred), PHI_WAS hook under A / 1 pt under current.
- GB_NE Wk9 badge wrong (read+measured): live page 83593d52..76acaf2d showed "¼ pt better than Tuesday", no cover effect. 12:10 snapshot bovada 3.0 / mybookie 2.5 -> median 2.75; both 3.0 by 14:27. Current pool_hook.py (940c8140, last quote per book) gives 3.0 = hook. Even splits still make quarter lines: live HEAD Wk11 TEN_DAL "¾ pt better than Tuesday, +2.6" from books 8.0/8.0/8.5/8.5 -> 8.25.
- Week 5 cause (read, data/scheduler_log.txt): Wk5 Tuesday was 10-06 (pool captured 18:25 ET), not 9-29 (Wk4). No Tuesday odds job existed until 5fc6fddb (10-13 18:38, "add Tuesday odds capture"). 10-06 log lines 2790-2852: no odds run; splash_board_tue FAIL x5, weekly_lock BLOCKED; pool read manually. Folders jump 20261004T162436Z (Sun) -> 20261007T220010Z (Wed odds_private_wed, Bovada failed "response must be an array", line 3031). Same gap explains B undefined Wk4 (9-29 manual dry-run 16:37, line 2290), Wk6 (manual 18:35 after 18:06 pool), Wk9 (12:10 job after 12:01 pool).

## Tried
Git-version page reads; per-folder quote reads. No code edit: the cause is pool_hook.py (pinned, cohort v8 per 940c8140) and load_quote_history reading `-reparse` folders; board_content.py display-only rounding would split the badge from the term's anchor (one-ledger rule).

## Proposed diffs (not applied)
1. pool_hook.py:92 (pinned): drop quotes from `*-reparse` snapshot folders before the anchor (needs a source-folder column from market_data.load_quote_history); re-enroll validation. Moves Wk7-8 hook flags back to the live-Bovada view; replay Wk1-10 before serving.
2. pool_hook.py:116 (pinned): median of an even book count yields x.25/x.75. Replace only after measuring which book the pool copies (Next); no tie rule without a mechanism.

## Next
Unit 4: per book (bovada, draftkings, mybookie, williamhill), Wk1-11, count pool == book and pool == book +-0.5 off a whole book line at the snapshot nearest noon, excluding `-reparse` folders; the book the pool copies becomes the anchor mechanism. Week 5: read-only compare against data/raw/20261006T222324Z/schedules.parquet spread_line (fetched 18:23 ET) in tests/scratch; never write into data/market/raw.

## Open
Whether the 10-31 re-parse folders should stay in data/market/raw (owner rule: no back-fill into the live store; they now move served hook flags and History counts). Is TEN_DAL Wk11 term (+2.6) built on 8.25 acceptable until unit 4?

ROADMAP ENG-55 update: Unit 3 (2026-11-21): Wk7-8 row badges never rendered (UI-31 shipped after those games were final); their hook flags depend on the 10-31 re-parse folders, which flip all four games versus live Bovada; GB_NE Wk9 showed a false "¼ pt" badge from a two-book median, and Wk11 TEN_DAL shows "¾ pt" now; Wk5 had no Tuesday odds job until 10-13. Next: unit 4 per-book pool-copy measurement.
