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
