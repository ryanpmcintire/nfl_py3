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
