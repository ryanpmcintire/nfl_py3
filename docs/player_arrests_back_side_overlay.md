# Player-arrest 14-day back-side prospective overlay

## Decision

- **[Measured]** The frozen opener evaluation recorded production at 53.3599% and the candidate at 53.7591% on 1,503 graded games, for +0.3992 accuracy points, a week-primary interval of [-0.2688, +1.0774], and `probability_positive=0.8562`; the authoritative record is `artifacts/player_arrests_policy_eval/20260820T162321Z/metadata.json`.
- **[Read]** The binding taxonomy keeps this result `unresolved_below_power`, as recorded in `registry/weak_signals.json` under `player_arrests_recent_14d_back_side_policy_opener`; this prospective registration is therefore a play/track decision, not a resolved claim.
- **[Read]** Challenger `player_arrests_recent_14d_back_side_overlay` is `ACTIVE_PROSPECTIVE` in `artifacts/prospective/challengers.json`, and nothing in `nfl_ats.player_arrests_back_side_overlay` changes the active model, production picks, or the published card.

## Frozen rule

- **[Read]** The implementation reproduces `docs/player_arrests_policy_eval.md`: compute each game's Tuesday decision date, consider broad incidents dated exactly 1 through 14 days before Tuesday, and never admit a same-Tuesday or later incident.
- **[Read]** When exactly one team is flagged, flip only if the production probability-rule pick opposes that team; if production already backs it, both teams are flagged, or neither is flagged, preserve the production pick.
- **[Read]** A flip complements `home_cover_probability`, while all other card fields and every unflipped probability remain unchanged; `tests/test_player_arrests_back_side_overlay.py` pins these additivity boundaries.
- **[Read]** Only `record_id`, `incident_date`, and `team` are read from the safe index; retrospective outcomes, descriptions, and links are excluded by construction and a mutation regression test.

## Freshness and refusal contract

- **[Read]** `load_latest_complete_arrest_snapshot` inspects the lexically newest directory under `data/raw/player_arrests/` and refuses rather than falling back if that newest attempt is incomplete or malformed.
- **[Read]** Recording requires `complete=true`, a matching `snapshot_id`, a present safe index named by the manifest, and an exact SHA-256 match to the manifest.
- **[Read]** `fetched_at_utc` must be no more than 36 hours before the recording instant and must not be future-dated; missing, stale, or future timestamps refuse recording.
- **[Read]** The publisher catches that refusal and returns `player_arrests_back_side_challenger_ledger: {recorded: 0, error: ...}`; it never substitutes an all-unflagged baseline week.
- **[Read]** The recorder also enforces the standard active-card configuration fingerprint, synchronized-artifact check, seven-day recording lock, strictly pre-kickoff filter, and append-only `(challenger_id, game_id)` ledger identity.

## Weekly operation

**[Read]** The registry's weekly command obtains a new default snapshot first and publishes only after a successful ingest:

```powershell
.\.tools\uv.exe run --no-sync python scripts/ingest_player_arrests.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.\.tools\uv.exe run --no-sync nfl-ats publish-predictions --record-decisions
```

**[Read]** A publish without `--record-decisions` returns a structured skipped payload for this challenger and writes no prospective row.

## Week 1 dry run

**[Measured]** A read-only application against synchronized Week 1 artifact `artifacts/margin_predictions/2026-week-01-20260820T005017Z` and fresh snapshot `data/raw/player_arrests/20260820T153000Z` found 0 sole-flagged games and 0 flips on the 16-game card; the dry-run command and ledger before/after SHA-256 are recorded in the verification section below.

**[Inferred]** Zero Week 1 exposure is expected because the snapshot's latest incident is dated 2026-06-23, well outside the frozen 14-day window for Week 1's Tuesday decision date; this is an exposure observation, not evidence against the mechanism.

## Verification

- **[Measured]** The read-only dry run used `load_latest_complete_arrest_snapshot(Path('data'), now=datetime(2026,8,20,16,30,tzinfo=UTC))` and `apply_player_arrests_back_side_overlay` directly; both before and after checks reported the live challenger ledger as `ABSENT`, so it wrote no live decision row.
- **[Measured]** Focused verification passed 12 overlay tests plus 11 CLI tests (`23 passed`) with `pytest -q tests/test_player_arrests_back_side_overlay.py tests/test_cli.py`; the final repository-wide checks are reported in the session handoff.
- **[Measured]** Repository-wide `pytest -q` passed 1,627 tests with 51 warnings in 124.84 seconds; repository-wide Ruff check and `mypy src` also passed, while the exact all-tree Ruff format traversal separately encountered access-denied shared temporary directories after the source checks.
