# flip-attribution-audit-2026-11-14

## Goal
Attribute every side change between the Tuesday card and the final card, 2026 Weeks 7-10, to a served fitted term, with records at the pool Tuesday line, for Tuesday's repair decision. Diagnosis only: no flip rule, no refit.

## State (2026-11-14, done; nothing served touched)
- Script scripts/flip_attribution_audit.py -> artifacts/flip_attribution_audit/ (games.csv, records.csv, transitions.csv, summary.json, record_batch.json). Ran once, exit 0, single process.
- Sources (read): ledger via load_pick_revisions (38 revision rows), Tuesday lock via original_card (57 games), waterfall_feed/<run>/feed.json (model and served probability; Tuesday feed side agrees 57/57), ledger late_week_net_move and late_week_eligible_books (serve-time move and leader depth), validation captures (flag sums), the coefficient set in force at each record (18 sets). Recomputed: the move at each revision cutoff with sharp_book_movement_features on today's quote history. Weeks 9-10 offset contribution is the remainder of the served logit.
- Fidelity (measured): Weeks 7-8 have no offset term; all 53 records reproduce the served logit to 4e-16 using the ledger move (22 revisions) or the replay (16).
- Net flips 18 (W7 5, W8 5, W9 4, W10 4). Driving term: line move 14 (3, 5, 4, 2), pool-line offset 2 (W10), flag set 1 (W7), model 1 (W7).
  - W7: DEN_ARI A>H move +0.5 on 1 book, L; GB_DET A>H flag no longer counted after the Sunday 10-25 18:16Z coefficient set (+.26; move +1.0 gave +.24), L; LA_LV A>H move +1.0, 3 books, L; NE_CHI A>H model +.005 (needed .002), W; SF_ATL A>H zero move, availability step alone, 1 book, W.
  - W8: ARI_DAL A>H +0.5, 1 book, three changes, L; IND_JAX A>H +1.5, 1 book, three changes, W; LV_NYJ A>H +1.75, 2 books, L; NE_MIA A>H zero move, availability step alone, 1 book, W; PHI_WAS A>H +0.5, 1 book, W.
  - W9: ARI_SEA H>A -2.5, L; DAL_IND A>H +0.5, W; DET_MIA A>H +0.5 plus model, L; LV_SF H>A -0.25 plus model, L; all 2 books.
  - W10: CAR_NO A>H +1.0, 3 books; MIN_GB H>A -1.5, 1 book; SF_DAL A>H offset +.021; WAS_NYG A>H offset +.086, L. Others unplayed.
- Flip and back 6: CIN_BAL, DAL_PHI, ATL_TB, LAC_LA left on the line move (served +2.5/3 books, +0.5/1, +0.5/1, -0.5/1) and returned on it; HOU_CLE, NE_DET (W10) left on the offset term at the anchor-fix refresh and returned on the line move. Side left 1-3, exact p .625.
- Records (measured, exact binomial two-sided v even): settled flips 15, final 6-9, Tuesday 9-6, p .607, P+ .23. Line move 5-7 p .774; flag set 0-1; model 1-0; offset 0-1 (p 1.0 each). Weeks 2-3, 3-2, 1-3, 0-1. All 44 settled games: Tuesday 21-23, final 18-26. Looks: 16, one family.
- Thin depth (served move on 1-2 of 3 leader books): 12 flips, settled 5-6 p 1.0; one book 4-2 p .69; two books 1-4; full depth 0-2; no move input 1-1.
- Serve time v today (measured): 17 Week 7-8 revision records carry a ledger move or leader count that today's quote history does not give at the same cutoff (served 1 book, today 2-3; CIN_BAL served +2.5 then 0.0, today -1.0 then -4.0). None in Weeks 9-10.
- Season context (read): pol11-ledger-check-2026-11-02.md:8, graded ledger through Week 8: 38 changed, refreshed 21-17 (W2-8 20-12, W1 1-5). fix-picks-u1-layer-table.md:9: W1-9 flips 30, flipped-to 15-15.
- Registry: 7 cells recorded (family flip_attribution_audit_2026_11_14, accuracy_points, unresolved_below_power); pool ran once, exit 0, 4574 signals, sign test p .105.

## Tried
- Replay of the move alone left remainders of exact half-point multiples in Weeks 7-8; the ledger's stored serve-time move closes them.

## Next
1. ruff check has one finding, RUF007 at scripts/flip_attribution_audit.py:496 (use itertools.pairwise(chain)); not fixed, call cap. ruff format --check is clean.
2. Find why serve-time leader depth was 1 book where today's history has 2-3 (late-arriving or re-dated baseline quotes is the candidate), before the Tuesday decision.
3. Orchestrator verifies the gating numbers, commits script, artifact pointers, registry shard, lane.

## Open
- Weeks 9-10 offset contribution is a remainder. It equals the replayed term on MIN_GB and NE_DET; on SF_DAL, WAS_NYG, HOU_CLE it is smaller (.021 v .035, .086 v .159, -.044 v -.075), consistent with the conversion still in force at the 01:46Z refresh (inferred).
- 12 of 18 flips started within .05 logit of even; 8 of 14 line-move flips rested on a move of half a point or less (settled 4-4), two on the availability step alone.
- summary.json key season_audit_table_changed_sides and games.csv result_source were written but not re-read.
