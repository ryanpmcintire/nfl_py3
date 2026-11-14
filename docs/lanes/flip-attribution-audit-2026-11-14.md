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
- Serve time v today (measured): 20 Week 7-8 revision records (7, 13) carry a ledger move or leader count that the audit replay does not give at the same cutoff; on 17 the ledger value is the served one (served 1 book, today 2-3; CIN_BAL served +2.5 then 0.0, today -1.0 then -4.0). None in Weeks 9-10. Cause established in the block below.
- Season context (read): pol11-ledger-check-2026-11-02.md:8, graded ledger through Week 8: 38 changed, refreshed 21-17 (W2-8 20-12, W1 1-5). fix-picks-u1-layer-table.md:9: W1-9 flips 30, flipped-to 15-15.
- Registry: 7 cells recorded (family flip_attribution_audit_2026_11_14, accuracy_points, unresolved_below_power); pool ran once, exit 0, 4574 signals, sign test p .105.

## Tried
- Replay of the move alone left remainders of exact half-point multiples in Weeks 7-8; the ledger's stored serve-time move closes them.

## Next
1. ruff check has one finding, RUF007 at scripts/flip_attribution_audit.py:496 (use itertools.pairwise(chain)); not fixed, call cap. ruff format --check is clean.
2. Done, see Serve-time move reproduction. Orchestrator decides the four repairs there; none is applied.
3. Orchestrator verifies the gating numbers, commits script, artifact pointers, registry shard, lane.

## Open
- Weeks 9-10 offset contribution is a remainder. It equals the replayed term on MIN_GB and NE_DET; on SF_DAL, WAS_NYG, HOU_CLE it is smaller (.021 v .035, .086 v .159, -.044 v -.075), consistent with the conversion still in force at the 01:46Z refresh (inferred).
- 12 of 18 flips started within .05 logit of even; 8 of 14 line-move flips rested on a move of half a point or less (settled 4-4), two on the availability step alone.
- summary.json key season_audit_table_changed_sides and games.csv result_source were written but not re-read.
- Where the exit code of check_served_terms.py is consumed was not read; the depth alarm below fires on 3 Week 10 games today and would have fired on every Week 9 game (Caesars absent).
- The alarm for late-written files covers Tuesday baselines only; a back-dated Wednesday-or-later file (the 10-21 and 10-28 siblings) is caught only by repair 1.

## Serve-time move reproduction (measured, artifact stamp 2026-11-14 21:02Z)
Script tests/scratch/serve_time_move/serve_time_move.py -> artifacts/flip_attribution_audit/serve_time_move.json (.out beside the script); two runs, exit 0, single process; nothing served touched.
- Serve-time values are the ledger's late_week_net_move and late_week_eligible_books (the leader-book count; read pick_refresh.py:629-645). Recomputed with the served function on today's store or the store as it stood at the pass (quote files written by then), on current code or the code before 52555216 (2026-10-31 12:07Z). A per-book rebuild agrees with the function on all 152 checks.
- Cause 1, quotes that arrived after the serve: 17 of 17. Four re-parse siblings (10-20, 10-21, 10-27, 10-28 `*-odds-gap-private-reparse`) were written 10-31 20:44Z carrying their original observation times (read 1bec13f4). They are the only late-written files since Week 7 Monday (61 others, largest write lag 0.5 min). Without them all 17 reproduce: 10 on current code, 7 only with the old code as well.
- Cause 2, change in the served code: 7 of 17, never alone (0 reproduce on today's store with the old code). Before 52555216 a book first seen after Wednesday counted from its first late reading: SF_ATL 10-21; CIN_BAL 10-23 and 10-24; DAL_PHI 10-24; ARI_DAL, ATL_TB, IND_JAX 10-30. Already repaired.
- Cause 3, parser week selection: the root of all 17. The Tuesday parse kept 1 game per book where the response held 13-14 (Week 7 14:26 ET: bovada 13, Caesars 14, MyBookie 14; Week 8 14:25: 13 each). That was the four-day window, fixed 10-31 20:49Z (1bec13f4). Bovada's baseline came from its public endpoint, so the move ran on bovada alone.
- Cause 4, a book that posted after the serve: 0 of 17. On 2 (DAL_PHI 10-21, PHI_WAS 10-28) Caesars and MyBookie had no stored quote at all, but the Wednesday response held them; the same parser drop.
- Cause 5, a different last pre-Wednesday quote: 0 of 17. On 15 the baseline was absent then and present now; no baseline changed value or time.
- The other 3 (GB_DET 10-25, ATL_TB and LV_NYJ 11-01, Sunday passes) are not a mismatch: the ledger column is the weekday read (sharp_book_movement_features.py:227), the served term reads through Sunday (pick_probability.py:767). Today's history reproduces the ledger column on all 3.
- The re-parse moved live picks: the 10-31 20:49Z pass served ATL_TB +0.5 on 3 books (without it 0.0 on 1) and LAC_LA 0.0 on 3 (without it -0.5 on 1), both revised to HOME; all 5 Week 8 records after it rest on re-parsed baselines.
- Wider check, the served move per game in the validation captures before kickoff: Week 7 29 of 119 rows and Week 8 13 of 78 are not reproduced today; all 42 are explained by causes 1 and 2 (2 by the old code alone), none unexplained.
- Weeks 9-10: not present. Ledger 0 of 13, captures 0 of 113 and 0 of 96, the store at each pass equals today's, no late-written file, Tuesday parses equal the response on every book. Still open in the parser: the Monday run writes none of the coming week (Week 9 response bovada 8, MyBookie 13; Week 10 bovada 11, Caesars 14, MyBookie 12; :39-42 takes the week of the nearest kickoff).
- Week 10 unplayed, leader books behind the move now: 3 books BUF_NYJ 0.0, CAR_NO +1.0, HOU_CLE +0.5, JAX_TEN 0.0, KC_ATL 0.0, MIA_IND +2.0, NE_DET -0.5, PIT_CIN 0.0, SEA_LV -2.0, SF_DAL 0.0; 2 books LAC_BAL -0.5 and LA_ARI -0.25 (no MyBookie baseline); 1 book MIN_GB -1.5 (Caesars only). MyBookie quoted LA_ARI and MIN_GB on Monday 12:34 ET and the week pick dropped both. Fit population: 3 books on 792 of 799 games (read 52555216 message).
- Sunday (inferred): the served moves are what the history shows and depth is fixed since Wednesday. No back-fill or re-parse into data/market/raw before Monday night; that event moved Week 8 picks. Thin games stay as served: the alarm reports, no rule flips.
- Tuesday bundle (inferred): its 2023-25 evidence is untouched. A 2026 replay of the move term for Week 8 or earlier reads baselines the card never had, so those rows are as-if, not served history. The late-Tuesday capture lands through the daemon and cannot recreate the mismatch. Adding eng-62 Next 4 would have given LA_ARI 3 books and MIN_GB 2 this week, on a Monday baseline 35 h before Wednesday.
- Repairs, none applied: (1) write any re-parse or back-fill with a capture_kind other than live (clv.py:247 serves only live), or after the week settles; (2) capture_odds_gap_private.py:39-42 per eng-62 Next 4; (3) pick_refresh.py:629-645 store the served through-Sunday move and leader count in the ledger row; (4) the alarms below.

### Alarm lines for scripts/check_served_terms.py (not applied)
Dry run tests/scratch/serve_time_move/guard_dry_run.py, exit 0: Week 10 names LAC at BAL (2 of 3, -0.50), LA at ARI (2 of 3, -0.25), MIN at GB (1 of 3, -1.50) and no late file; replayed at the Week 8 Wednesday it names 20261027T182541Z-odds-gap-private-reparse. Placement: add MARKET_MOVE_FEATURE_SUNDAY to the import at :21-25 and the two imports after :26; the constant after :30; both functions before `check` (:190); `thin_moves` and `late_files` start as empty lists beside :212; first call after :232, second after :253; the two problems after :264; both lists into the result. The three books are the fit's own depth; the two days are the Monday-to-Wednesday span of the served window (sharp_book_movement_features.py:65-70).
```python
from nfl_ats.clv import LIVE_CAPTURE_KIND, load_decision_quotes, load_snapshot_manifest_index  # noqa: E402
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS, sharp_book_movement_features  # noqa: E402

MOVE_FIT_LEADER_BOOKS = len(LEADER_BOOKS)


def _thin_move_games(
    data_root: Path, pending: pd.DataFrame, now: pd.Timestamp, version: str
) -> list[str]:
    quotes = load_decision_quotes(data_root / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    kickoff = pd.to_datetime(pending["kickoff"], utc=True, errors="coerce")
    games = pd.DataFrame(
        {
            "game_id": pending["game_id"].astype(str),
            "commence_time_utc": kickoff,
            "week_first_commence_utc": kickoff.min(),
            "cutoff_utc": now,
        }
    ).dropna(subset=["commence_time_utc"])
    depth = sharp_book_movement_features(
        quotes, games, include_sunday=version == MARKET_MOVE_FEATURE_SUNDAY
    )
    thin = depth.loc[depth["leader_books"].gt(0) & depth["leader_books"].lt(MOVE_FIT_LEADER_BOOKS)]
    return [
        f"{row.game_id.split('_', 2)[2].replace('_', ' at ')} ({int(row.leader_books)} of "
        f"{MOVE_FIT_LEADER_BOOKS}, move {row.leader_median_net_move:+.2f})"
        for row in thin.itertuples()
    ]


def _baselines_written_late(data_root: Path, wednesday: pd.Timestamp) -> list[str]:
    index = load_snapshot_manifest_index(data_root / "market" / "raw")
    index = index.loc[
        index["capture_kind"].eq(LIVE_CAPTURE_KIND)
        & index["snapshot_timestamp_utc"].ge(wednesday - timedelta(days=2))
        & index["snapshot_timestamp_utc"].lt(wednesday)
    ]
    late = []
    for row in index.itertuples(index=False):
        path = Path(str(row.dir)) / "quotes.parquet"
        if path.is_file() and pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC") >= wednesday:
            late.append(Path(str(row.dir)).name)
    return late


        thin_moves = _thin_move_games(data_root, pending, now_ts, model.market_move_feature_version)

        late_files = _baselines_written_late(
            data_root, (tuesday + timedelta(days=1)).tz_convert("UTC")
        )

    if thin_moves:
        problems.append(
            "The line-move input rests on fewer leader books than it was fitted on: "
            + ", ".join(thin_moves)
            + "."
        )
    if late_files:
        problems.append(
            "Tuesday quote files for this week were written after Wednesday began, so the "
            "line-move baseline changed after picks were served: " + ", ".join(late_files) + "."
        )
```
