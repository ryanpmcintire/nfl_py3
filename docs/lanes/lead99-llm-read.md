# lead99-llm-read

## Goal
Forward-test a sharp-style LLM read of each 2026 game (Weeks 7-18) as one added fitted term in the served pick probability; score after Week 18.

## State
Declared 2026-10-20 before any outcome (ROADMAP LEAD-99). Runner scripts/lead99_llm_read.py (packet|enroll|run; --week defaults to the captured week with a game still to kick off), prompt scripts/lead99_prompt.md, frozen artifacts/lead99_llm_read/frozen.json. Scheduler: tuesday view Tue 13:30; deadline views Thu 18:00, Sun 08:40, 11:40, 14:45 (daemon restarted 20:08 ET).
- Contamination: LLMs know pre-cutoff results; only 2026 games count; no historical grading.
- Blinding: the read never sees the served probability, pick or drivers.
- Inputs (point-in-time, saved with the output): matchup, kickoff, pool line, latest market spread and move since Tuesday, injury designations, inactives when out, beat-reporter posts since each team's last game, weather, rest/travel; web search for pregame news allowed, every source cited with its time.
- Output per game: home cover probability vs the pool line, one named mechanism, key facts. No clamps or hand constants.
- Runs: Tuesday after the lock (Tuesday view); Thursday before TNF and Sunday after early and late inactives (deadline view). Canonical row = latest before the game's deadline.
- Frozen: prompt file sha256 and model id at enrollment; later runs abort on mismatch.
- Score after Week 18: decisive record first, accuracy diff (week-block bootstrap, P+), LL/Brier vs served, market-only, model-only, reliability; coefficient leave-one-week-out; 6 looks, family lead99_llm_read; weak-signals record.

## Tried
Week 7 Tuesday run envelope (measured): 3 turns, 85 s, 0 web searches (packet only; sources are beat posts and injury headlines), reported cost 1.40 USD per run.
Week 7 Tuesday view recorded: artifacts/lead99_llm_read/runs/2026-week-07-tuesday-20261021T000817Z.json (14 rows, 0 dropped, no retry, range 0.44-0.56). Next: scheduler jobs for deadline views.
2026-10-20: built scripts/lead99_llm_read.py (packet|enroll|run) and scripts/lead99_prompt.md; enrolled (artifacts/lead99_llm_read/frozen.json). Packet 283k chars for 14 games, no beat-post cap needed. Gaps: SF_ATL has no market quote key match; nflverse injuries latest report is week 6; weather one model per game. Run: `uv run --no-sync python scripts/lead99_llm_read.py run --week 7 --view tuesday` (headless claude -p, tools WebSearch,WebFetch, setting-sources local, cwd artifacts/lead99_llm_read/cwd). Runs land in artifacts/lead99_llm_read/runs/.

## Next
Check Thu 18:00 deadline run lands (artifacts/lead99_llm_read/runs/, scheduler_log). No interim scoring; score after Week 18 per State.

## Open

## Week 9 Tuesday requirement (2026-11-03, read-only check)
- Job lead99_llm_read_tuesday_tue_1330 has no requires, so splash_board_tue status is irrelevant; it runs 13:30 regardless.
- Selection (lead99_llm_read.py:63-89): glob data/splash/2026_week09_*.json; per file needs captured_at_et (ISO with UTC offset, must be <= run instant, latest such file wins), games[] each with game_id (2026_09_AWAY_HOME), home, away, home_spread (positive = home favored), kickoff_et (ISO with offset). Week chosen = lowest captured week with any game kicking off after run instant. Other top-level keys (week, season, picks_lock_et, contest...) are copy-the-Week-8-schema, not read.
- Dry check (packet, no model call) now: "no captured pool week has a game still to kick off" (Week 8 fully kicked off 11-02). Expected to flip once the Week 9 file exists.
- Other inputs (market, injuries, headlines, inactives, weather) are best-effort and already present; none is Week 9 specific beyond the splash file.
- Manual capture satisfies it if the schema matches and captured_at_et is before 13:30 ET. Verify after saving: uv run --no-sync python scripts/lead99_llm_read.py packet (prints games count).

## Tuesday job repaired; one extra Week 10 file (2026-11-11 15:24 ET)
- Cause of the lost Tuesday views (measured, data/scheduler_log.txt): the job ran at 13:30 with a 30-minute window and no prerequisite; on 2026-10-27 and 2026-11-10 the board was read later, so it failed with no captured week. Week 8 has no Tuesday view. Weeks 7, 9 and 10 do.
- Fix (scripts/capture_scheduler.py): the Tuesday view now requires weekly_lock and its window stays open 600 minutes, to 23:30 ET; a same-day manual --run-job weekly_lock satisfies it. Scheduler tests 94 passed; daemon restarted 15:22 with no job running, one process tree, code and schedule current (measured).
- Exercise run and an error of mine: the edited job was exercised once with --run-job at 15:24 and wrote runs/2026-week-10-tuesday-20261111T202420Z.json (14 rows, range 0.38 to 0.57). The previous session had already recorded Week 10's Tuesday view by hand at 20:57 ET on 11-10 (runs/2026-week-10-tuesday-20261111T015754Z.json); that run does not appear in the scheduler log and the runs folder was not checked first. The Tuesday view of record for Week 10 is the 11-10 file. The 11-11 file is a Wednesday-afternoon read under the Tuesday label: keep it (the log is append-only and its inputs are timestamped), exclude it from any Tuesday-view comparison, and note that the canonical row (latest before each game's deadline) is unaffected once Thursday's and Sunday's deadline views run.
- Scoring note: when the views are scored after Week 18, take the earliest Tuesday-labelled file per week as the Tuesday view.

## Second term `lead99_first_view` declared (2026-11-14 15:22 ET, new look)
- Why: until today scripts/forward_ledger.py scored `lead99` from each game's earliest pregame record, not the canonical row declared in State (latest record of any view written before the game's pick deadline, min(kickoff, Sunday 16:00 ET)). The two rows take different sides on 13 of 43 Week 7-9 games (measured). Ruling: `lead99` is the declared row; the ledger now scores it (forward_ledger.py LATEST_BEFORE_DEADLINE, cutoff from nfl_ats.published_picks.game_deadlines).
- What it is: the same stored reads, prompt and frozen model; the row is the earliest record of any view written before the game's pick deadline (Tuesday view in Weeks 7 and 9, Thursday's in Week 8, which has no Tuesday view). Raw stated home cover probability at the pool line, side = home at 0.5 or above, no coefficient, no calibration. Research only, never served; monitor only, not a stacker input (forward_monitor.py MONITOR_ONLY), because adding it to the stacker after seeing its record would be a further look.
- Exploratory, not a test: the row was named after its results were seen. Weeks 7-9, 43 games (measured, forward_weekly.py): 28-15, decisive against the card 13-3, log-loss gain against the card +0.0632 [+0.0192, +0.1101] probability_positive .998, against 0.5 +0.0259 [+0.0047, +0.0471] .992, evidence value 7.74 (threshold 20). Declared `lead99` on the same games: 23-20, decisive 6-1, +0.0432 [+0.0102, +0.0782] .995 against the card, +0.0059 [-0.0231, +0.0349] .656 against 0.5, evidence value 11.60.
- Forward sample: Week 10 onward only (forward_monitor.py FORWARD_FROM_WEEK). Caveat (inferred from the calendar, result not looked at here): the Week 10 Thursday game was played before this declaration; its first-view read was stored 11-10, before kickoff, and the row rule is mechanical, but report the forward numbers with and without that game. Week 10's first view is the 11-10 file, the earliest record.
- Metrics on the forward sample, read once after Week 18 with LEAD-99: decisive record against the card first; paired log-loss gain against the card (game bootstrap, probability_positive); gain against a constant 0.5; sequential evidence value started at Week 10. 4 looks, family eng49_forward_ledger_vs_card, cell eng49_forward_lead99_first_view_vs_card_log_loss. Weekly Tuesday refreshes are interim reads; only the evidence value is valid at every look. The Weeks 7-9 reading is already among the 93 looks of lead99_forward_decision_read.
- Zero crossing closes nothing; nothing here serves or flips a pick; any served use is one fitted term chosen out of season.
