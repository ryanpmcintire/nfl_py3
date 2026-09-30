# MOD-23 unit 2: MOD-22 constructions as inputs to the base model alone

Script `scripts/mod23_unit2.py`; outputs `artifacts/mod23_unit2/` (report.json, per-arm per_game parquet, rebuilt evidenced ratings table). All numbers **measured** this session.

## Protocol (declared before scoring)
Six arms, six looks, one family (`mod23_unit2_players_on_field`): (1) diff_full_strength_total (availability-free depth-chart rating), (2) diff_lineup_total, (3) diff_divergence, (4) lineup_total + divergence together, (5) qb_out_diff (unit 4), (6) qb_quality_loss_diff (unit 5). Each is added to the served 90-column set, ridge alpha 10, scored 2020-2025 at the opener by `clv.opener_pick_evaluation`, exactly as the baseline (reproduced 802-701 on 1,503 decisive games, row-identical count to `opener_evaluation/20260929T192743Z`). Missing values (pre-2017 for ratings, pre-2020 and non-decisive-population games for QB terms) are filled with 0, so QB arms learn only from 2020+ history. Units 2-3 (unpriced residual, unseen-move interaction) are add-on residualizations of the four-term probability and were not re-graded. Terms are home minus away, so positive favours home.

## Clock
Kickoff minus 24 hours, the clock the baseline's feature builder enforces (its own injury inputs use it). Injury-dependent arms use only evidenced (non-proxy `date_modified`) reports at or before that time: the ratings table was rebuilt with proxy-timed reports dropped (42,729 of 48,651 rows evidenced). Note the pool lock is Tuesday noon; under that stricter clock arms 2-6 would carry almost no information (unit 5 is defined by reports after Tuesday noon), and arm 1 is the only one clean on both clocks. Point-in-time ratings inherited from MOD-22, not re-audited.

## Results (decisive, opener, probability rule; baseline 802-701, LL 0.6969, Brier 0.2517; even 0.6931/0.25)
| arm | record | diff (pts) | 95% season-week block bootstrap (107 blocks) | probability_positive | LL | Brier |
|---|---|---|---|---|---|---|
| full_strength | 798-705 | -0.27 | [-0.93, 0.40] | 0.184 | 0.6970 | 0.2518 |
| lineup_total | 792-711 | -0.67 | [-1.39, 0.07] | 0.028 | 0.6973 | 0.2519 |
| divergence | 795-708 | -0.47 | [-1.47, 0.60] | 0.173 | 0.6972 | 0.2518 |
| lineup+divergence | 788-715 | -0.93 | [-2.08, 0.13] | 0.041 | 0.6975 | 0.2520 |
| qb_out | 797-706 | -0.33 | [-1.25, 0.60] | 0.223 | 0.6991 | 0.2526 |
| qb_quality_loss | 800-703 | -0.13 | [-0.92, 0.66] | 0.339 | 0.7066 | 0.2527 |

Per-season diff (2020..2025): full_strength 0,0,0,0,-0.8,-0.7; lineup_total 0,+0.4,-1.6,-0.8,-0.8,-1.1; divergence -0.9,-0.4,-0.8,-0.4,-0.8,+0.4; lineup+div -2.3,+0.4,-1.6,-0.8,-0.4,-1.1; qb_out +0.5,-0.4,-2.4,+1.1,-0.8,0; qb_quality +0.9,+0.4,-1.2,+1.1,-0.8,-1.1. No arm beats the baseline on record, log loss or Brier; none is better than market-even on log loss. QB terms are nonzero on only 53 of 1,503 games, so they are far below power.

## Verdict
Every arm is `unresolved_below_power` (recorded, six cells). No whole interval is on the wrong side; no closing ground applies. Nothing served changes.
