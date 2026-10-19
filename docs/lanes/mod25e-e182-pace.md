# MOD-25e E182 pace extension (PACE2)

## Goal
Close part of the E182 gap (sim Q4 possessions 3.576 v real 3.954; trailer s/play +1.27 Q4, +0.96 Q3) with a fitted, default-off elapsed-per-snap model (PACE2=1, scripts/mod25e_pace2.py).

## State
Declaration (before fitting), read from code:
- Target: real elapsed seconds per snap (bin 0..61), run/pass plays (code 0/1), penalties dropped, el<=hs; classes inc/term/run/pass.
- Correction to the brief: Q4 hs is gsr (0..900), so "Q4 hs>900" is empty. Coverage read from code: CDW = Q2/Q4 hs 45..420 all signs (hz 420); TPO = Q4 hs 420..900 classes run/pass only; URG/URG2 = hs 15..180 subset; CLQ=2 = Q1/Q3 pooled cells (inc/run/pass have sign none, no down/distance/lead band in Q3).
- Complement fitted: (a) Q3 all hs>0 excluding first snap of quarter (offset draw), classes inc/term/run/pass; (b) Q4 hs 420..900 classes inc/term (TPO skips them). Q4 leaders hs 300-900 outside CDW are covered by CDW (<=420) and TPO (>420, run/pass) already.
- Features: child = parent cell x one of {down, distance tercile, down x distance, lead band7, down x lead band7}; parent = the served CLQ (Q3) / CLK (Q4) best spec per class; timeouts and time left sit in the parent. Shrink child to parent with a in SMOOTH (7), parent to hs-bin marginal.
- Validation: nested LOSO by season (inner LOSO chooses feature and a), held-out per-play log likelihood and MAE of expected elapsed v parent; game-bootstrap 2000, folds positive, in-sample v out gap. Looks: 31 configs x 6 groups = 186 (family: feature x smoothing).
- Distance tercile cuts now fold-local (mod25e_pace2.fold_cuts: training seasons only, inner folds exclude both seasons); served cuts (8,10) stored in fit.json dist_cuts with provenance.

## Tried
Fit done (scripts/mod25e_pace2.py fit, artifacts/mod25e3/pace2/fit.json; nested LOSO, game boot 90%). Held-out LL gain per play, P+ , folds+, in-sample (gap):
q3 inc +.0022 [.0009,.0035] P+ .998, 8/9, in .0093; q3 term +.0171 [.0092,.0244] 1.0, 9/9, in .1035; q3 run +.0213 [.0192,.0235] 1.0, 9/9, in .0271; q3 pass +.0084 [.0070,.0098] 1.0, 9/9, in .0197; q4m(420-900) inc +.0002 [-.0007,.0010] P+ .644, 5/9; q4m term +.0259 [.0169,.0345] 1.0, 8/9, in .1195. n-weighted +.0123 [.0113,.0133] log loss (74.8k snaps). MAE of expected elapsed improves in all 6 groups (P+ 1.0).
Gain comes from down and down x distance (chosen per fold); lead band7 was not the winning feature.
Offline replay at real states (LOSO, s/snap real | parent | pace2): Q3 trail9+ 27.06|27.81|27.75; trail1-8 28.69|28.63|28.59; lead1-8 29.32|29.10|29.07; lead9+ 30.15|29.56|29.53; Q4 420-900 trail9+ 7.27|7.63|7.55. Lead-band gap barely moves (inferred: possessions per Q3/Q4 essentially unchanged, well under the 0.38 Q4 gap). So PACE2 is a per-play likelihood improvement, not the E182 carrier.
Recorded: weak-signals e182_pace2_q3_q4mid_elapsed_ll log_loss_improvement +.01229 [.01127,.01332] unresolved_below_power (186 looks).

Fold-local rerun 2026-10-19: every fold cut = (8,10), results identical to prior (n-weighted +.01229, groups unchanged, no verdict changed). Old fit.json sha256 ba6fb4a74ee988508bc03397066da0a7b1e0d9826080ce10ea5c0a16b2e031a8 (copy fit.old.json); new sha256 prefix 624cd2390a5415e5. Replay (tests/scratch/e182/replay.py) matches. Weak-signal record replaced. Queued run_p2_3seed.sh at end of e178/queue.txt.

## Next
crH install (not applied; do not edit until sole editor): in scripts/mod25e_crH.py after the TPO block (~line 189) add
    import mod25e_pace2 as p2

    if p2.enabled():
        p2.install_pace2()
and in the label chain after the PTO block (line ~510) add
    if os.environ.get("PACE2") == "1":
        label = label + "p2"
(suffix p2 unused, PACE2 unused by git grep). Run tests/scratch/e182/run_p2_3seed.sh (base = existing PTO label, cand = +PACE2, seeds 11-13); not launched.
Because lead-band pace gap is not closed, next fitted candidate is a lead-band x time x down elapsed model fit for trailers only with band as the sole child feature, or the leader complacency mechanism from the gap table.
## Open
PACE2 is applied after CLQ=2 draw in Q3 and after TPO in Q4; it skips the first snap of Q3 (offset draw). Q4 hs>900 does not exist (hs=gsr).
