# LEAD-105

## Goal
Forward-log a frozen language read of the total points in each week's last game (the pool tiebreak game), 2026 Weeks 10-18, and grade it after the season against the market total and the served tiebreaker total (ROADMAP LEAD-105). Research only: nothing served, no pick or tiebreaker guess changes.

## State
Built and declared 2026-11-14 14:51 ET, before any outcome (Week 10 game kicks off Mon 2026-11-16 20:15 ET).
- Runner `scripts/lead105_tiebreak_total_read.py` (packet|enroll|run [--week] [--game-day-only]). Last game = the single latest kickoff in the newest schedule snapshot (`live_kickoff`); fails closed when the schedule names more than one. Packet = the LEAD-99 packet for that one game, same 13 fields, nothing added; call = `lead99_llm_read.call_claude` (same model, tools, headless CLI login). The read is not given the market total, the served guess or the pick; web search is allowed as in the parent.
- Prompt `scripts/lead105_prompt.md`, sha256 `761688ba6783e6454a124d1a12f617d71d1c6079113a5ff253cc041d36d8553e`, frozen in `artifacts/lead105_tiebreak_total_read/frozen.json` (enroll refuses to overwrite; run aborts on prompt, model or tools mismatch).
- Record: one exclusive-create file per run under `artifacts/lead105_tiebreak_total_read/runs/`; refuses at or after kickoff, before and after the call; aborts unless tiebreaker.json names the same game. Unparseable output goes to `failed/`, never `runs/`.
- Declaration. Family `lead105_tiebreak_total_read`. Per game, absolute error against the final total of the read (`total_points`), the market total (`market_total`: newest captured totals quote at or before the run instant, else tiebreaker.json) and the served tiebreaker total (`served_tiebreaker_total` = tiebreaker.json `projected_total`). Looks: read minus market and read minus served, on (a) the latest record before the pool deadline and (b) the latest record before kickoff: 4 looks, at most 9 games each. Report the closer/farther record first, then the mean paired difference with interval and probability_positive; `total_sd` is reported descriptively, not a look. Scored once, after the Week 18 last game is final (schedule: Sun 2027-01-10); nothing scored or served before; verdict through `nfl-ats weak-signals record`; zero crossing closes nothing; the read could only ever enter the tiebreaker as one term fitted out of season.
- Constants and sources: `MAX_ATTEMPTS = 2`, one retry, copied from the parent (`lead99_llm_read.py:463`); model, tools and call timeout are the parent's, not redefined; tiebreaker market precedence `current_median_total` then `market_total` as `board_content.py:3849`; served total field as `publishing.py:308`. No thresholds or clamps; validation is positivity and whole-number scores only.
- Week 10 record (measured): `runs/2026-week-10-20261114T195030Z.json`. LAC at BAL. Read total 38.0, sd 13.5, score LAC 17 BAL 21; market total 38.5 (one book, quote 2026-11-14T17:24:39Z); served tiebreaker total 37 (16-21; continuous 37.69). Recorded 2026-11-14T19:50:30Z, before the pool deadline. One attempt, 5 turns, 34.7 s, reported cost 0.25 USD.

## Tried
- `packet`: week 10, 2026_10_LAC_BAL, 7,526 chars. `run --week 9`: refused, kicked off (rc 1). `run --game-day-only` on Saturday: skipped, no call (rc 0). Second `enroll`: refused (rc 1). ruff check and format --check clean.

## Next
Orchestrator adds this to JOBS in `scripts/capture_scheduler.py` (not edited here):
```
    *(
        Job(
            f"lead105_tiebreak_total_read_{label}_{day}_{at.replace(':', '')}",
            day,
            at,
            30,
            [
                str(UV),
                "run",
                "--no-sync",
                "python",
                str(REPO / "scripts" / "lead105_tiebreak_total_read.py"),
                "run",
                *flags,
            ],
            True,
            "LEAD-105 forward log: a blind language read of the total points in the week's "
            "last game, the pool tiebreak game, frozen prompt and model, recorded with the "
            "market total and the served tiebreaker total. The game-day reads follow the "
            "19:00 odds capture and run only when the last game kicks off that day; the "
            "Sunday 15:05 read precedes the 16:00 pool deadline. Nothing is recorded after "
            "kickoff.",
            added_on="2026-11-14",
        )
        for label, day, at, flags in (
            ("gameday", "mon", "19:10", ("--game-day-only",)),
            ("gameday", "sun", "19:10", ("--game-day-only",)),
            ("deadline", "sun", "15:05", ()),
        )
    ),
```
Slots: Weeks 10-17 last games are all Mon 20:15 ET (measured, schedule 20261114T140059Z). 19:10 follows `odds_espn_mon` 19:00 so the recorded market total is that day's, and leaves 65 minutes; a run took 35 s. The Sunday game-day slot covers Week 18 and is a no-call skip otherwise. The Sunday 15:05 slot follows the parent's 14:45 view and is the only read made before the tiebreak guess locks; drop it if one run a week is wanted, and look (a) then rests on hand runs.
Score after 2027-01-10 per the declaration.

## Open
- Week 18 kickoffs are not set (16 games tied at Sun 13:00); the runner fails closed until the schedule names one last game.
- The Week 10 read cited sportsbook preview pages, so it likely saw a posted total; the read-minus-market look measures its adjustment, not an independent total.
- After the Sunday deadline tiebreaker.json is frozen, so a Monday read holds news the served guess could not use; look (b) against served is context only.
