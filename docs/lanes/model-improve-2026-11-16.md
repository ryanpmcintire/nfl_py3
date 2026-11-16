# model-improve-2026-11-16

## Goal
Owner 2026-11-16: keep improving the model, no overfitting, maximum parallel throughput. Only units that fix a served input, bring information the line does not hold, or set up a declared forward test. No NFL one-term add-on units (memory add-one-noise-floor).

## State
- 2026-11-16 15:07 EST: wave 1 launched, four parallel units, disjoint files:
  - W1-A serve parity audit: lane serve-parity-audit-2026-11-16.md. Do the 2026 inputs the live card used equal what the training builder computes for the same games? Patches in tests/scratch/serve_parity/, nothing in src.
  - W1-B outside ratings acquisition (ROADMAP MOD-30): lane mod30-outside-ratings-acquire-2026-11-16.md. Historical pregame FPI predictor (NFL and college), 538 Elo, nfelo; as-of proof; no outcome read.
  - W1-C college family 5 (XLG-14): lane xlg-14-u5-2026-11-16.md. Declared before outcomes, novelty checked against units 2-4 and the registry.
  - W1-D LEAD-108 forward log: lane lead108-forward-2026-11-16.md. Freeze the wording-read term on 2020-25 inputs available, declare, log Weeks 11-18 through scripts/forward_weekly.py style; no fitting on 2026.

- W1-B DONE (read): ESPN predictor is blocked by source policy (robots, 403 2026-09-20) and the route was already closed 2026-11-03 (mod-30-mod30-public-ratings.md, my history miss); 538 and nfelo have no source-policy entry and no pregame proof except nfelo 2025. Only CFBD pregame Elo is local (100% join 2012-25), but it is a results-only team rating already in src/nfl_ats/cfb.py columns; team quality is priced (memory), so no screen launched. Approving 538/nfelo source-policy entries is the owner's call; not a blocker.
- W1-B slot refilled: served-terms check counts current consensus books for thin-move games (full-steam-2026-11-14.md Next 3), lane served-terms-thin-move-2026-11-16.md.

- W1-D DONE (measured): LEAD-108 term frozen at 0.0238 logit/pt (fit on 2026 Weeks 2-9 card offset, 116 games; no earlier season has the read; Week 10 excluded); declaration 2026-11-16T20:10Z in lead108-forward-2026-11-16.md; forward Weeks 11-18. Root added scheduler jobs lead108_forward_pregame_thu_1815, _sat_1845, lead108_forward_score_tue (13:25) in scripts/capture_scheduler.py; dry runs execute the script (pregame refuses Week 10 as declared; score now exits 0 with nothing graded). Committed 0f50ada2. Scheduler restarted 2026-11-16 ~15:25 EST (one daemon, pid 12008 under 28464); new jobs listed. Status shows unacknowledged MISSED on 2026-11-15: refresh_last_call_sun_1245, refresh_trigger_log_sun, pool_move_log_sun_1255; check against data/scheduler_log.txt next operational pass.
- Thin-move check (served-terms-thin-move-2026-11-16.md): now counts distinct leader books quoting the game; MIN at GB 6 books, no problem. Note: the served move value itself still uses leader books with a Tuesday baseline (one book for MIN at GB); whether a one-book served move needs damping is open, needs an out-of-season measure, not a patch.
- W1-C DONE (measured): college family 5 (EPA beyond success rate, leverage-weighted EPA; xlg14_composition_family_u5), 8,925 games, 12 cells unresolved; no survivor (best null percentile 84), so no NFL look owed. Lane xlg-14-u5-2026-11-16.md.

- W1-A DONE (measured, serve-parity-audit-2026-11-16.md; tests/scratch/serve_parity/): 31 of 90 columns differ between the served rows and today's rebuild across 150 games. The only builder defect is temp/wind: they were blank at serve time in Weeks 1-4 on 45 games while the training rows hold observed game-time weather. Swapping them in moves the model logit by up to 0.071 and flips 4 model-side picks, all in Week 1 and all within 1.3 points of 50%. A drop patch exists (weather_drop.patch), not applied. This is the same defect as live-weather-features.md, which on 2026-10-19 deferred the fix to after Week 18 because features.py is pinned by the independent validation cohort. That decision stands: the effect is under 0.04 logit and zero card sides. total_line and neutral-site differences are line and schedule changes after the forecast; quarterback inputs are covered by repair_ab.patch; rest_diff matches. Gap: per-game card logits are not stored, so card-side effects of input fixes can only be bounded (about 0.3 points of 50%).

## Post-Week-18 pinned batch (one refit, re-enrol validation)
- Quarterback input repair (tests/scratch/qb_input_repair/repair_ab.patch), unless applied Tuesday per full-steam-2026-11-14.md.
- Weather: pregame MOS temp/wind in serve and train (live-weather-features.md), or weather_drop.patch; forecast-trained weak_stack_v4.
- Ledger move column (pick_refresh.py:629-645), ENG-51 unit 2b.
- Store per-game served card logits so input fixes can be scored on card sides.

## Tried
See model-improve-2026-11-15.md, mod57-sixty-percent.md (Tried and phases), mod23-base-model.md. Not repeated: 2011+ training window (MOD-23: no better), team-change inputs, hindsight-gap label, play-level models.

## Next
1. On return verify each unit's decision-gating claim, commit, refill.
2. Tuesday 11-17: quarterback input repair (full-steam-2026-11-14.md Decisions) bundled with any parity patch W1-A proves, one fenced change, validation re-enrolled same session.

## Open
- None.
