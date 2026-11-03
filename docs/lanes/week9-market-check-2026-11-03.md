# Week 9 market check vs pool numbers (2026-11-03)

## Goal
Compare the pool Week 9 spreads (read 11:38 ET = 16:38Z) with the newest market captures, flag hooks and favourite disagreements, explain JAX@BAL.

## State
- measured: data/market/raw/ newest folders (20261102T155148Z bovada, 20261102T155146Z odds-gap, 20261101T172241Z) hold Week 8 games only (2026_08_*). No Week 9 capture there.
- measured: newest Week 9 market is data/raw/odds_tape/20261103T125306Z/tape.jsonl (Bovada only, single book, polls 12:53-12:54Z = 07:53 ET; per-game provider_last_updated 10:33Z-12:47Z). No multi-book consensus exists for Week 9; it is about 3.75 h older than the pool read.
- measured, away line market vs pool (hook favours): JAX@BAL -9.0 v -9.5 (BAL); CIN@ATL -3 v -2.5 (CIN); DEN@CAR -3 v -2.5 (DEN); LA@WAS -6 v -5.5 (LA); CLE@NO +2.5 equal; DAL@IND -6 v -5.5 (DAL); NYJ@KC +9.5 equal; NYG@PHI -3 v -3.5 (PHI); DET@MIA -7 v -6.5 (DET); HOU@LAC -4.5 equal; ARI@SEA +11 v +10.5 (SEA); GB@NE +3 v +2.5 (NE); LV@SF +10 v +9.5 (SF); TB@CHI +6 v +6.5 (TB, away dog gets the extra half... pool is +6.5 so TB); BUF@MIN -3 v -3.5 (MIN).
- Every non-equal game is exactly 0.5 off a whole-number market line (12 of 15). Direction: the pool is half a point easier for the market favourite at CIN, DEN, LA, DAL, DET, SEA, NE, SF; harder for the favourite (dog gains the half) at JAX (BAL gains), NYG (PHI), BUF (MIN), CHI (TB).
- measured: no game has a different favourite between pool and Bovada.
- JAX@BAL: measured Bovada JAX -9.0 (updated 12:45Z), pool JAX -9.5. nflverse schedule snapshot (BAL -2.5, 20261027) is stale, pre-Week 8.
- read: beat post thebanner.com 2026-11-03 03:16Z (BAL): Ravens do not know who starts at QB for TNF; 2026-11-02 19:58Z: Lamar Jackson's expected return ended in one play Sunday (BAL lost 36-10 to BUF, per BUF beat post 15:51Z). PFT index (injury_news 20261103T010007Z) has no BAL QB headline after Oct 23 (Lamar questionable). JAX lost S Eric Murray and DT DaVon Hamilton for the season (2026-11-02). inferred: the 12-point swing is Lamar out/doubtful plus BUF blowout; the exact injury is not in a structured feed.

## Tried
Scratch scripts in the session scratchpad only; no network, no jobs.

## Next
Re-read market after Tuesday-noon freeze if a multi-book capture lands; confirm BAL QB status from a Wednesday injury report before TNF.

## Open
Single-book Bovada only; pool hooks are the pool-hook-term input in 12 games.
