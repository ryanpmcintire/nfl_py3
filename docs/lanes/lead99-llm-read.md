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
