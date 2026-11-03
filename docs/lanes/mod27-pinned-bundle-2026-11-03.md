# mod27-pinned-bundle-2026-11-03

## Goal
Package the deferred pinned MOD-27 bundle as one unapplied patch (tests/scratch/mod27-pinned-bundle.patch) and measure it out of tree: (a) generalized pool-offset term as the stage-2 term of the two-stage hook, (b) thin-book guard on the leader-median move (LV at NYJ had two leader books).

## State
DECLARED BEFORE OUTCOMES (2026-11-03, before any bundle number was computed). Reuses the 11-02 replay inputs (artifacts/mod27_replay_inputs) and harness (scripts/mod27_fenced_repairs_replay.py); eval script scripts/mod27_pinned_bundle_eval.py, output artifacts/mod27_pinned_bundle/results.json.
- Patch touches only src/nfl_ats/pool_hook.py (offset_value generalizes hook_value to any pool-minus-market offset on the half-point grid; hook_rows stacks simulated offsets -1.0, -0.5, +0.5, +1.0 on every game, pushes at the pool line dropped; served_hook_frame accepts any offset) and src/nfl_ats/sharp_book_movement_features.py (MIN_LEADER_BOOKS_FOR_MEDIAN = 3, the median's breakdown point: a median of fewer than three values cannot outvote one outlier; thinner games get move 0 and availability 0). pick_probability.py, the fit and the stage-1 four-term fit are unchanged.
- Arms (4 cells counted as one family, mod27_pinned_bundle): S served four-term; G guard only; O offset term only on S stage 1; B bundle (offset on G stage 1).
- Grade: LOSO 2020-25 at the opener. Game level (real games carry offset 0 in history): S vs G. Offset effect on the stacked simulated rows (5,885-class stack): S vs O, S vs B. Report decisive (changed-side) record first, then accuracy, log loss, Brier with season-week block bootstrap probability_positive, per-fold coefficients, in-sample vs out-of-sample log loss gap.
- Forward: 2026 Weeks 1-8 harness replay (served = S plus the live whole-number hook 0.805; bundle = G plus fitted offset coefficient on every observed pool-versus-Tuesday-market offset); Week 9 read from the locked forecast after the lock.
- Push table for the stack is built from prior seasons only (hook_rows); serving uses the full-population table as the hook already does.

## Tried
RECONCILED 2026-11-03 (measured, artifacts/mod27_pinned_bundle/results.json). CAUSE: eval rebuilt the population with build_fit_population, which recomputes the counted-flag set from post-discovery 2026 records; as 2026 games accrued, flag_division joined flag_protection (264 rows differ), so composition_flag_sum differed from the stored fit (counted = flag_protection only, pick_probability/20261101T171318Z/metadata.json) and LOSO gave 829-674. Fix in the eval: pin the sum to the stored counted columns and pin the opener evaluation 20261101T151614Z (active model changed to 5d851f at 17:30, so the lookup failed). LOSO now 833-670.
- G guard, game level: 5 rows changed, 46 sides change after refit; decisive 25-21 served vs 21-25 guard; record 833-670 -> 829-674; LL gain +.00007 P+ .60, Brier P+ .59, accuracy -0.27 pts P+ .24; in-sample LL .68579 -> .68572, out-of-sample .68719 -> .68712. Matches the replay sign (negative on decisive).
- O offset (6,021 stacked rows, 4,144 nonzero): decisive 519-556 vs 556-519 (1,075 sides); 3305-2716 -> 3342-2679; LL gain .00291 P+ .998, Brier P+ .999, accuracy +0.61 pts P+ .77; coef per fold .876-1.039, full .937. Cells: abs0.5 LL P+ .988, abs1.0 .999, whole-market .991, half-market .985.
- B bundle vs served: decisive 545-579 vs 579-545 (1,124); LL gain .00297 P+ .997; accuracy +0.56 pts P+ .73. B vs G-only: LL .00291 P+ .998.
- 2026 W2-8 harness: bundle changes 5 sides (LAC-SEA, BAL-ATL, DET-ARI, HOU-TEN, LV-NYJ); record 60-44 -> 63-41; guard alone changes 3 (BAL-ATL, IND-JAX, LV-NYJ), IND-JAX a loss.
- MIN_LEADER_BOOKS_FOR_MEDIAN: LEADER_BOOKS has exactly 3 books; 792 of 799 historical games have all 3, 7 have 2 (4 nonzero moves, 2 of 4 correct sign). Not empirically estimable at that count. Derivation is structural: median breakdown point floor((n-1)/2) is 0 at n=2 and 1 at n=3, and 3 = len(LEADER_BOOKS) (full leader set). Patch should define it as len(LEADER_BOOKS), not a literal. Cited, not fitted.
- Week 9: artifacts/margin_predictions/2026-week-09-20261103T173056Z/pool_card.csv exists, but its pick_line equals the Tuesday market for all 15 games and home_cover_probability equals pick_probability, so no Splash pool line is observed and no hook is applied; offset values are all 0 and the bundle changes 0 Week 9 sides as of now. Guard: no Week 9 leader-book capture in the replay inputs; not computed.
- Recorded: weak-signals family mod27_pinned_bundle, 3 cells (G, O, B), log_loss_improvement, unresolved_below_power.

## Next
- Serving: apply the patch (MIN_LEADER_BOOKS_FOR_MEDIAN = len(LEADER_BOOKS)), regenerate the move table, run `nfl-ats fit-pick-probability`, enroll validation v8 before the Wed 2026-11-04 12:10 ET capture (v7 has zero captures, so nothing is lost), republish. When Splash pool lines are read, rerun the Week 9 offsets.
- RECOMMENDATION offset term: serve (LL P+ .998, per-fold coef .88-1.04 stable, rule: one fitted term, LOSO). Guard: do not serve alone (decisive 21-25, P+ .24 accuracy, .60 LL); it is a mechanism-named data-quality repair, so ship it only inside the bundle if v8 is enrolled anyway, else hold.

## Open
- Offset effect is on simulated rows; real-game history carries offset 0, so 2026 observed offsets (W2-8) are the only forward check.
