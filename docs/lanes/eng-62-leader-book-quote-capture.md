# ENG-62 leader-book quote capture

Goal: bovada, williamhill_us, mybookieag spread quotes before Wed 00:00 ET and after, every week, for the move term.

State: Measured 2026-11-08. All three leaders come from capture_private_sunday_odds.py (scheduler.py:147 PRIVATE_SUNDAY_ODDS_CAPTURE): Bovada public endpoint plus The Odds Gap (capture_odds_gap_private.py:21 LEADERS, parser :56-58). Jobs at capture_scheduler.py:1452 (sun 12:10) and :1492-1512 (tue 12:10, tue 14:25, wed 18:00, fri 12:30, sat 10:00).
Gaps: Wk2 (Sep 14-20) paid odds_* jobs FAIL(1) (log 1186-1309) and odds_private_wed/fri/sat were disabled until 09-23. Wk5 (Oct 5-7): no Tue job (odds_private_tue added 10-13), first leader quote Wed 10-07 18:00. Wk9: williamhill_us absent from the Odds Gap feed on Mon 11-02 and Tue 11-03 (feed lists it Wed 18:01 and Fri-Sun; present Tue in Oct 20 and Oct 27 weeks), so upstream; daemon also MISSED 6 odds_private windows 11-04 08:36 (restart catch-up).
Change: added odds_private_mon (mon 12:00) to the tuple in capture_scheduler.py. SCHEDULE import shows 7 odds_private jobs. Daemon restart needed to load it (owner/orchestrator).
Tried: no live probe (script has no output-dir option; would write production).
Next: confirm williamhill_us Tue availability next week; if absent again, consider a Caesars-bearing free source.
Open: why Odds Gap omitted williamhill_us Mon-Tue wk9.
