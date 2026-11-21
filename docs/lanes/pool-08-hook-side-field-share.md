# POOL-08

## Goal
Half-point hook awareness: does the pool's field over- or under-pick the side the pool's half point helps (ROADMAP POOL-08). Context for the pool-standing panel and the standing-aware Best Pick question (revisit Week 15). Never a pick input: one fitted cover probability selects every side.

## State
Unit 3, 2026-11-21 09:59 local (Week 10 rerun), measured and recorded. Family `pool08_hook_field_share`. `scripts/pool08_hook_field_share.py --last-week 10 --subset 1-9 --subset 10-10 [--anchor first_snapshot]` -> `artifacts/pool_rank_replay/pool08_hook_field_share_w10.json` and `..._w10_first_snapshot.json` (Week 1-9 files untouched).
- Served anchor (multi-book Tuesday line), 150 games, 57 hook games (per week 7,4,7,4,4,5,7,4,10,5). First-snapshot anchor finds no Week 10 hook games (measured: `hook_by_week` has no week 10), so its pooled reading is Weeks 1-9 with a Week 1-10 proxy fit.
- Share on hook side minus leave-one-week-out line proxy, pp [95% game bootstrap] P+ / within-week permutation p / hook games:

| cell | served anchor | first-snapshot anchor |
|---|---|---|
| Weeks 1-10 pooled | +5.49 [1.36, 9.35] 0.995 / .078 / 57 | +6.20 [2.09, 10.34] 0.998 / .058 / 53 |
| Weeks 1-9 (in 1-10 fit) | +6.56 [2.30, 10.75] 0.998 / 52 | not recorded (identical games) |
| Week 10 alone | -5.66 [-12.79, 1.46] 0.074 / 5 | no hook games |

- Weekly means (served): 8.8, 24.8, 6.3, 7.0, 6.8, 14.6, -3.2, 2.3, 2.1, -5.7; 8 of 10 positive, exact sign p .055 one-sided, .11 two-sided (inferred from measured means).
- Hook coefficient in the home-share logit (150 games): 0.229 [0.057, 0.407] P+ 0.9945, folds 0.168-0.286, about +5.7 share points at an even split (first snapshot 0.256 [0.080, 0.437] P+ 0.998). Held-out log loss gain all games +0.0014 [-0.0025, 0.0051] P+ 0.754 (first snapshot +0.0018, P+ 0.805).
- Outcome (decisive games first): hook side covered 32 of 57 v mean model probability 0.509: +5.3 pp [-7.2, 18.0] P+ 0.792, null p .50.
- Implies: card unchanged. The pooled lean is smaller than at Week 9 (6.5 -> 5.5) because Week 10's five hook games went the other way; one week of five games cannot separate a fading lean from noise. All 10 cells `unresolved_below_power`.

## Tried
Unit 1 (2026-11-03, W1-8): +7.68 [3.22, 11.94]. Unit 2 (2026-11-14, W1-9): +6.53 [2.23, 10.57], 14 cells. Unit 3 looks: 2 anchors x (pooled, 2 subsets, log loss, coefficient, cover) = 12 (2 empty or duplicate first-snapshot subsets not recorded), plus one weekly sign count. Wiring into POOL-01 replay: `pool01_forward_sim.py` draws observed shares and has no fitted proxy.

## Next
1. Week 11+: `--last-week N --subset 1-(N-1) --subset N-N --out ..._wNN.json` for both anchors; replace pooled cells.
2. Check why the first-snapshot anchor has no Week 10 hook games (quote-store first snapshot for Week 10 may be missing).

## Open
- Predicted-share step for unplayed weeks still missing for the POOL-01 wiring (see POOL-05 Next 2).
- Intervals are game-level bootstraps; ten weeks is too few for a week-block interval. Zero crossing closes nothing; no side rule; no entrant names stored.
