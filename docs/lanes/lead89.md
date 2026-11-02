# LEAD-89 - option to wait

## Goal
Compare early Best Pick locking with the declared legal waiting policy.
Own `scripts/lead89_unit2.py`, `docs/lead89_unit2.md` and this lane only.

## State
**Measured:** upstream season-exclusion and opener checks pass for 1,537 games.
A separate source blocker remains: 814/816 outer-season Tuesday rows have late
injury inputs; five have missing timestamps; their union is all 816 rows.
No eligible Tuesday nominee exists in any of the 54 scoring weeks.
These are censuses, so intervals do not apply. No outcomes read or scored.
Full evidence and counts: `docs/lead89_unit2.md`.

### Unit-2 amendment saved before scoring
Pin `artifacts/loso_upstream/20260929T235904133782Z/` instead of the rejected
companion margins. Its upstream/calibration LOSO is retrospective, not a claim
of chronological upstream training. Reconstruct stage movement from unit-1
quotes and require all upstream inputs to precede the nomination timestamp.
The original declaration and pre-run amendment are saved in
`tests/scratch/codex/lead89_unit2/protocol_before_run.md`.

Keep 2020-2025 frozen historical openers, pushes and source-complete subsets;
outer 2023/2024/2025; downstream fit through Y-3, tune Y-2, calibrate Y-1.
Keep Tuesday noon/Thursday capture/Sunday 12:30 legal nominations, one joint
weekly transition law and one calibrated probability selecting the side.
Keep the five declared arms and six metrics, IS/OOS/gaps, calibration,
season/week uncertainty, probability_positive, and **717 looks** (F=3/B=7/K=6).
Zero outcome looks consumed. No policy or protocol selection occurred.

## Tried
**Measured:** real unit-2 command completed its input audit, exit 2; zero scored
replays. Initial attempt hit a reporting assertion (exit 1), removed before the
completed audit. Final scoped Ruff check and format check both exit 0.
Script, source hashes, clock rows and censuses are saved under
`tests/scratch/codex/lead89_unit2/`; commands/logs are listed in the report.

## Record commands
None: no scored candidate-versus-served result exists. No registry command ran.
The orchestrator records eventual results serially with `--plain-summary` and
`--source docs/lead89_unit2.md`; inventing effect numbers now would be invalid.

## Next
Supply season-held-out upstream forecasts constructed from inputs available
at each decision stage, including injury provenance and discrete opener masses;
then implement and run the unchanged transition/policy replay in a fresh thread.

## Open
Weekly records, differing-week results, exact paired null, Wilson intervals,
IS/OOS/gaps and probability_positive remain uncomputed. Zero crossing closes
nothing; no signal is adjudicated or closed. No serving, publishing or Git writes.
## Unit 3 (stage-timed rebuild) - blocked on pin drift
**Measured 2026-11-02:** `scripts/lead89_stage_timed.py` (Tuesday/Thursday = 11 injury+QB columns zeroed, stage-matched same-book median move, nested-LOSO Replay per stage; Sunday = production features, injury-clock-passing games only; nominee contrasts, 48 looks, declaration in `tests/scratch/codex/lead89_stage_timed/protocol_before_run.md`) fails at `Replay.__init__` with "Source model mismatch": `artifacts/active_ats_model.json` was re-activated today (model 9b7b41..., 12:10 ET) and `data/processed/game_features_weak_stack.parquet` rebuilt (sha 42d669..., 12:06) vs the pinned a3eedb.../b578fbea. The pinned feature table is not on disk (backup is Sep 8, sha 34acbe...). Zero outcomes read, nothing scored, nothing recorded.
**Next:** owner/orchestrator decision: restore the Sep-29 feature table + active model (e.g. from the E: mirror) or re-baseline LOSO upstream on the new model (new pin, new look budget); then run the script unchanged.

## Unit 3 progress (2026-11-02, second agent)
**Measured:** step 1 done. New LOSO upstream `artifacts/loso_upstream/20261102T213642884071Z` (source `artifacts/pick_probability/20261102T171412Z`, model 9b7b416622f7538b), 865/1,503 = 57.552%, log loss 0.681819, Brier 0.244338: same as the 2026-09-29 baseline to the digits shown. Needed one edit in `scripts/build_loso_upstream.py`: composition guard is now "source flags subset of the 7 protocol flags" (served artifact now counts only flag_protection; LOSO keeps all 7 as baseline). Log `tests/scratch/codex/loso_upstream_rerun_20261102.log`.
**Step 2 written, not yet validated:** `scripts/lead89_stage_features.py <control|tuesday|thursday>` rebuilds the 11 injury/QB diff columns with `enrich_with_player_features` at each game's stage cutoff (input_clock_audit decision_at), carrying last known prior-week report rows for non-playing players, sentinel row if all played, zero only if no earlier report; writes `stage_features_<mode>.parquet` and `stage_cases_<mode>.parquet` (case counts). `scripts/lead89_stage_timed.py` edited to load those, pin the new LOSO/pick_probability artifacts. Revised protocol appended to `tests/scratch/codex/lead89_stage_timed/protocol_before_run.md`. Control run (production cutoff, must match active table columns) hit and fixed ImportErrors and a depth-snapshot pin (now uses manifest id); its last run was killed by me before finishing (no result). Zero outcomes read.
**Next:** run `control` (background, single process, OMP/MKL=1), confirm max_abs_diff_vs_active ~0, then `tuesday`, `thursday`, then `scripts/lead89_stage_timed.py` once, then `nfl-ats weak-signals record --batch tests/scratch/codex/lead89_stage_timed/record_batch.json`. Write `docs/lead89_stage_timed.md` (SOURCE_DOC) before record. Watch: games outside input_clock_audit keep real kickoff; script may need debugging (ruff, no comments).
