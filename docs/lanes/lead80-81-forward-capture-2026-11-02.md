# LEAD-80 / LEAD-81 forward capture

## Goal
Make both source-gated rows forward-only with a running log. No outcome looks, no fit; research only.

## State
**Measured 2026-11-02:** `scripts/capture_travel_interruptions.py` ran once: 65 rows, 73,334 beat posts scanned, FAA ok, output `data/raw/travel_interruptions/20261102T215836Z/{rows.jsonl,manifest.json}`. `scripts/validate_pool_field_capture.py` ran on a scratch sample (valid; also rejected a total != away+home sample); `--schema` prints the schema. Week 9 lane step 1c names the field file to save.

### LEAD-81 rules (fixed before outcomes, no tuning)
Sources: public unauthenticated Bluesky beat captures (data/raw/beat_reporters, last 7 days, team from handle map, game = team's next game by gameday) and FAA NAS airport status (origin and destination airports per game this week). Post classes need a travel-context word, then first match in order: flight_cancelled, diverted_or_rerouted, mechanical_or_grounded, weather_disruption, delayed_or_late_arrival; else arrival_reported_no_interruption_term. FAA: ground stop/closure, ground delay, other delay; airport absent from the feed = faa_no_advisory_control (the only explicit control). Silence is missing, never on-time. Rows dedupe on (uri, team, game_id).

### LEAD-80 schema
`--schema` of the validator: entrants with tiebreaker_guess {away_score, home_score, total}, optional weekly_score, tiebreak_game_id, contest entries.

## Tried
First pass over-flags: "arrived in Chicago" posts class as arrival_reported; FAA "closed" matches runway closures (8 rows). These are classification noise; a human-read label is needed before any use. Not an outcome look.

## Next
Root adds after the Week 9 lock, in capture_scheduler.py (Job pattern like beat_reporters_*): `travel_interruptions_<day>` for day in (tue, wed, thu, fri, sat, sun), 08:00 (after beat_reporters 07:30), timeout 120, argv `[UV, run, --no-sync, python, scripts/capture_travel_interruptions.py]`; plus a second `travel_interruptions_sun_1000` at 10:00 sun for game-day arrivals. Week 9 Tuesday: save the field file (step 1c). Later: read-label a sample of rows; 2020-2025 stays unavailable, so evaluation is 2026+ prospective only.

## Open
No historical arrival/on-time table; both rows stay unresolved_below_power until a season of forward rows exists. Classification precision unmeasured. No registry write, no commit.

## LEAD-80 weekly tie frequency (measured 2026-11-03, Weeks 1-8, 241 of 250 entries; 9 missing could hide a higher unique top or extra ties)
Script: scratchpad tb.py (not in repo). Top score / entries tied at top / entries at or above 5th-best score / share of Best Picks that won:
W1 13/1/7/.572 | W2 14/1/5/.434 | W3 13/1/6/.383 | W4 13/1/7/.410 | W5 12/8/8/.548 | W6 11/2/9/.373 | W7 13/1/12/.517 | W8 13/1/6/.550.
Top tie in 2 of 8 weeks (25%); the near-top (top 5) is crowded every week (5 to 12 entries).
Inferred: Best Pick wins breaks a top tie unless 0 or 2+ tied entries won; P(falls to last-game total | tie) = 1 - k q (1-q)^(k-1): W5 .98 (k=8), W6 .53 (k=2). Expected 1.5 weeks in 8 (0.19/week) decided by the total, field-wide. An average entry sits in that tie with chance k/241 (W5 3.3%, W6 0.8%), so P(our total guess decides the weekly leaderboard) ~0.5%/week (inferred; higher for a strong entry, not estimated). Season order ignores the total. Implication: keep the simple unconditional total guess; a joint-score conditional model is not worth effort at this rate. No weak-signals cell (no effect measured); row stays unresolved_below_power.

## LEAD-81 retrospective Weeks 1-8 (measured 2026-11-03; unit script in scratchpad, not in repo)
Method: imported the collector's classify_post and handle map unchanged; scanned all 36 beat_reporters captures (159,443 posts, capture stamps 2026-10-01 to 2026-11-03). A flag counts only if post createdAt AND the first capture stamp that held it are both before the game's deadline (min(kickoff, Sun 16:00 ET)), and createdAt is after Tue 12:00 ET of that week. Game = team's next kickoff after createdAt.
Result: 567 classified posts; 0 confirmed post-Tuesday interruptions. Only non-arrival-only hit before a deadline is one Week 6 TEN "delayed_or_late_arrival" (a "struggle bus" joke, pre-freeze: false positive). Weeks 1-3 have no beat captures (archive starts Oct 1; old posts first-observed after their deadline are excluded). 12 arrival-only controls post-freeze/pre-deadline in Weeks 4-8 (W4 3, W5 2, W6 2, W7 3, W8 2). injury_news/pfr_transactions are injury/transaction feeds, not travel text; not scanned.
Looks: 0 outcome looks, 0 games scored; no pick/result table (no flagged game); nothing to record with weak-signals (no effect measured). Stays unresolved_below_power, source-limited.
