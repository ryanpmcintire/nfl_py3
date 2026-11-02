# backlog-triage-g5-2026-11-02

## Goal
Read-only triage of ROADMAP rows LEAD-28 29 43 53 55 59 61 64 65 (2026-11-02). No edits outside this file.

## State
Verdicts (labels: measured = log/lane read this session, inferred = reasoning):
- LEAD-28 NEXT-UNIT: post_bye_new_playcaller_back_overlay wired, 0 flips Wk1 because no 2026 OC rows in coordinator snapshot. Unit: capture fresh coordinator snapshot (data/raw/coordinators), confirm overlay fires on post-bye games. Light CPU.
- LEAD-29 FORWARD-ONLY: needs first real 2026 midseason OC/DC change plus hand adjudication; overlay interim_playcaller_first_game_back_overlay recorder is in publish-predictions.
- LEAD-43 CLOSE-NOW: screens run, 14 registry entries family altitude_fourth_quarter (unresolved_below_power); production retained.
- LEAD-53 FORWARD-ONLY: BEST-PICK-LEDGER tuesday 2026-10-27, sunday 2026-11-01 paired in data/scheduler_log.txt; scores as weeks settle (Wk9 settles 2026-11-03).
- LEAD-55 CLOSE-NOW (inferred): free compliant route exists, not Sportradar: Bluesky beat-reporter capture (scripts/mod24_u9_beat.py, jobs beat_reporters_sat/sun/mon, last OK 2026-11-02 run 20261102T123006Z, 139,597 posts, tagged limited/DNP/full rows weekly).
- LEAD-59 NEXT-UNIT: 2015+ done, battery closed 2026-09-26; 2009-2014 sweep is a 51-102 h background job.
- LEAD-61 CLOSE-NOW as source gap: Odds API cancelled, Bovada/Odds Gap carry no 2H markets; halves jobs gated. Not a refutation; unresolved_below_power stays.
- LEAD-64 FORWARD-ONLY: earliest Fri 2026-11-06 morning comparison of injury_headlines vs nflverse injuries.
- LEAD-65 FORWARD-ONLY: early-window challenger tracked; weeks 1-4 2026 already recorded.

## Tried
Read ROADMAP rows and lane State sections; grepped scheduler_log for ledger and beat jobs.

## Next
Orchestrator marks LEAD-43/55/61 per verdicts; spawn LEAD-28 snapshot unit.

## Open
Not verified on disk: 2026 coordinator snapshot presence; LEAD-65 2026 weeks 1-4 settled read.
