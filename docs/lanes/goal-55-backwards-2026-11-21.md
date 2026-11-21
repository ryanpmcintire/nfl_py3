# goal-55-backwards-2026-11-21

## Goal
Owner 2026-11-21: get the MODEL's accuracy above 55% (model alone, walk-forward, Tuesday opener, 1,503 decisive 2020-25 games): 802-701 now, 827+ needed. Not the card replay, not the live record. Done = a servable model change that clears 827 out of season, declared before its grade, then confirmed forward.

## Route (read 2026-11-21)
- Market information cannot get there: following the eventual close gives 54.76% (below).
- The one measured route above 55% without the result or the market: exact who-plays knowledge in the base model, P2-L ALL oracle 835-668 (+2.20 pts [0.07, 4.31], P+ .977; O1 actives alone +0.73, O3 QB +0.67, O2 snap share in ALL); docs/lanes/mod57-p2l-availability-ceiling.md. O1 there is snaps > 0, which the official inactives list (posted 90 minutes before kickoff) nearly equals, and the pool deadline min(kickoff, Sun 4 PM ET) falls after inactives for every slot except SNF and MNF.
- Plan: make ALL servable. G55-U1 deadline actives and starting QB table with as-of proof (lane g55-u1-deadline-actives-2026-11-21.md); G55-U2 expected snap share from earlier games given who is active (lane g55-u2-expected-snaps-2026-11-21.md); then G55-U3, one declared grade of the servable ALL arm in the P2-L harness, model alone at the opener, LOSO, shuffled control and matched null. Hindsight team strength (843-660) is the other ceiling; its servable routes (P, P2-P label) are already refuted or worse.

## State (measured 2026-11-21 13:35 EST)
- Inputs: artifacts/opener_evaluation/20261121T171727Z/per_game.parquet (model a8ea1dadc06c2405, 1,503 non-push opener games 2020-25); `scripts/week_attribution.py --season 2026 --weeks 1-10`. Script: scratchpad backwards.py (not tracked).
- Model alone at the opener: 802-701 (53.36%). 55% = 827: +25 wins, about 4 a season. Seasons 49.6% (2023) to 55.6% (2024); 2021 and 2024 already at 55%.
- Confidence quintiles 52.5 / 55.7 / 49.2 / 56.3 / 53.2: the model's confident games are not its better games, so reweighting what it has cannot supply the 25 wins.
- Closing-line ceiling: following the opener-to-close move where the line moved (75% of games) and the model elsewhere gives 54.76% (823); moves of 1+ point followed go 376-277 (57.6%). Model agrees with the eventual move on 57.2% of moved games (371-277); where it fights the move it goes 232-253. Perfect hindsight of the close is 4 wins short of 55% at the opener.
- Live 2026 card: 77-73 (51.3%) through Week 10. Weeks 3-10 card 54-64 v current formula replay 62-56 v calibrated model 65-53; the 16 games where the card differed from its formula went 4-12 (zero such games in Week 10; fidelity check live). Flag set was chosen partly on 2026 results, so the replay flatters (inferred).
- Rest of season about 122 games: 55% for the season needs 73-49 (60%); 55% from here needs 68-54.
- Pool-line replay held out (goal-65 lane): 836-667 (55.6%); pooled stack 852-651 but its gain is the coach flag, which went 11-21 live.

## G55-U3 declaration (written 2026-11-21 ~14:00 EST, before any outcome read of a servable arm)
- Arm S_ALL: P2-L harness (scripts/mod57_p2l_availability_ceiling.py), base walk-forward model, with U1 deadline actives replacing O1, U1 starting QB replacing O3, U2 expected snap share replacing O2, on games known_at_deadline; other games keep base inputs. No position selection, no reweighting, nothing tuned on 2020-25 outcomes.
- Root pin 2026-11-21 ~14:55 EST after U1 (verified artifacts/g55_u1/summary.json: INA with snaps 0 of 19,847): O1 = gameday status from artifacts/g55_u1/deadline_actives.parquet per the U1 lane Next 1. O3 uses the actives list only: share 0 when the served projected QB is INA on a known_at_deadline game, else 1. Depth-chart starter is NOT used (pregame timing unverified, U1 Open).
- Grade: model alone at the Tuesday opener, 1,503 decisive 2020-25 games, LOSO as P2-L. Looks: all (primary), quiet, moved = 3, family g55_deadline_availability. Controls: S_ALL shuffled within season (50 draws) for a matched-null percentile; base reproduced 802-701 first.
- Serving candidate only if all of: S_ALL >= 827 wins, null percentile >= 95, log loss no worse than base. Then a forward read on 2026 Weeks 12-18 at the opener beside the served model, and landing in the pinned batch with validation re-enrolled. Anything else is recorded unresolved_below_power and the arm is not re-tuned on the same games.

## G55-U3 result (measured, root recomputed from artifacts/g55_u3/per_game.parquet; 492af846)
- S_ALL 809-694 v base 802-701 (+0.47 pts [-0.77, +1.47], P+ .785, null 87th pct, log loss .69904 v .69678 worse); quiet 450-387 v 437-400 (+1.55, P+ .99), moved 355-298 v 361-292 (-0.92); changed 95, 51-44. Serving test FAILS on all three legs; 3 cells unresolved_below_power (g55_deadline_availability). Not re-tuned.
- Inferred: the oracle's extra +26 wins over the servable arm sit in realized snap share, which carries game script (starters rest in blowouts), i.e. part of the 835 ceiling is the result leaking. G55-U4 measures that: oracle O2 rebuilt from one-score game states only (lane g55-u4-neutral-usage-ceiling-2026-11-21.md).
- Forward: S_ALL is a candidate for a frozen forward log Weeks 12-18 after Tuesday's capture patch (needs 2026 snap counts in the player snapshot).

## Tried
- Gaussian correlation mapping of residual to accuracy predicts 51.8% against 53.4% measured; not usable for sizing, dropped.

## Next
1. Sunday 11-22: confirm zero card-v-formula mismatches after the 12:45 refresh (backlog-2026-11-21.md Next 2).
2. U1 done and verified (4368410f); U2 done (span 2, lambda ~0.5, MAE .119 v naive .125, build check only). G55-U3 grade launched ~15:05 EST (lane g55-u3-servable-all-grade-2026-11-21.md), with no who-plays information on games not known_at_deadline and no realized-snap fallback. G55-U1b live-capture alignment running (lane g55-u1b-live-inactives-2026-11-21.md, patches only). U1b DONE (measured): coverage gaps were the NFL.com placeholder bug (fixed 09-24), missing passes before Week 7, the Week 10 12:20 pass delayed to 13:14 behind the odds tape (since fixed) and a single Thursday pass; patch tests/scratch/g55_u1b/inactives_capture.patch (T-90 cut, roster-id matching; roster INA found 89.2% -> 99.6%) applies cleanly; scheduler slot patch (12:20 grace 15 -> 40 min, Thursday 19:35 pass) specified in g55-u1b lane Next, not written. ROOT DECISION: no served-path change this weekend (captures feed refresh_triggers.py:278 through the inactives overlay); apply both at Tuesday 11-24 after the lock with one scheduler restart, before Week 12 Thursday. On U3 return: recompute the record from artifacts/g55_u3/per_game.parquet before reporting. The fights-the-move idea is dropped: market-side, capped at 54.76%.

## Open
- None.
