# Player-arrest 14-day back-side production overlay

## Decision

- **[Measured]** The frozen opener evaluation recorded production at 53.3599% and the candidate at 53.7591% on 1,503 graded games, for +0.3992 accuracy points, a week-primary interval of [-0.2688, +1.0774], and `probability_positive=0.8562`; the authoritative record is `artifacts/player_arrests_policy_eval/20260820T162321Z/metadata.json`.
- **[Read]** The binding taxonomy keeps this result `unresolved_below_power`, as recorded in `registry/weak_signals.json` under `player_arrests_recent_14d_back_side_policy_opener`; this prospective registration is therefore a play/track decision, not a resolved claim.
- **[Read]** The forced-pick EV decision therefore plays the candidate: the final card is raw model → year-1-coach fade → player-arrest back-side policy. This is a production decision under AGENTS.md's “promotion bar is not a decision bar” rule, not a claim that the historical effect is resolved.
- **[Read]** The former candidate identity `player_arrests_recent_14d_back_side_overlay` is retained as `SUPERSEDED_BY_PROMOTION`; `player_arrests_recent_14d_no_overlay_incumbent` is the active paired control and records the former coach-only card.

## Frozen rule

- **[Read]** The implementation reproduces `docs/player_arrests_policy_eval.md`: compute each game's Tuesday decision date, consider broad incidents dated exactly 1 through 14 days before Tuesday, and never admit a same-Tuesday or later incident.
- **[Read]** When exactly one team is flagged, flip only if the production probability-rule pick opposes that team; if production already backs it, both teams are flagged, or neither is flagged, preserve the production pick.
- **[Read]** A flip complements `home_cover_probability`, while all other card fields and every unflipped probability remain unchanged; `tests/test_player_arrests_back_side_overlay.py` pins these additivity boundaries.
- **[Read]** Only `record_id`, `incident_date`, and `team` are read from the safe index; retrospective outcomes, descriptions, and links are excluded by construction and a mutation regression test.

## Freshness and refusal contract

- **[Read]** `load_latest_complete_arrest_snapshot` inspects the lexically newest directory under `data/raw/player_arrests/` and refuses rather than falling back if that newest attempt is incomplete or malformed.
- **[Read]** Recording requires `complete=true`, a matching `snapshot_id`, a present safe index named by the manifest, and an exact SHA-256 match to the manifest.
- **[Read]** `fetched_at_utc` must be no more than 36 hours before the recording instant and must not be future-dated; missing, stale, or future timestamps refuse recording.
- **[Read]** `publish_active_predictions`, the public-site builder, the CLI, and the primary paper-ledger recorder require this fresh source by default. A refusal occurs before the tracked Markdown write, so stale or missing data cannot silently substitute an all-unflagged production week.
- **[Read]** The primary paper ledger stores the raw model side, the post-coach/pre-arrest side, the final played side, both frozen arrest flags, flip markers, snapshot ID/fetch time, and safe-index hash. `pick_side` is the final played side.
- **[Read]** Late-week refresh re-applies the coach policy, then the arrest policy from those recorded Tuesday flags, then the observed-movement override. It never loads a later arrest snapshot, which prevents a later source revision from changing Tuesday exposure retrospectively.

## Weekly operation

**[Read]** `weekly-run` now has a fatal `ingest-player-arrests` step immediately before publication. The equivalent manual sequence is:

```powershell
.\.tools\uv.exe run --no-sync nfl-ats ingest-player-arrests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.\.tools\uv.exe run --no-sync nfl-ats publish-predictions --record-decisions
```

**[Read]** A publish without `--record-decisions` still applies the production policy but writes neither paper nor challenger rows. At the deliberate lock-day recording, the paper ledger stores the played arm and the no-arrest incumbent recorder stores the coach-only control at the same line.

**[Read]** Every tracked public surface discloses that this policy is active even when the current week has zero qualifying incidents or flips. The weekly card, README, findings page, and model record distinguish the 53.3599% raw-model baseline from the 53.7591% frozen arrest-policy evaluation; the latter is a component evaluation beneath the live coach-then-arrest composition, not a claim that the full composition has already been graded at 53.7591%.

## Composition and comparison arms

- **[Read]** Side transforms are ordered: coach fade first, arrest policy second, and any late-week ≥1-point observed-movement override last.
- **[Read]** `player_arrests_recent_14d_no_overlay_incumbent` is identical to the `hc_year_one_fade_overlay` candidate arm under the current policy stack. They are correlated views, not independent evidence, and must not be pooled together.
- **[Read]** The paper ledger's `model_pick_side` preserves the raw-model arm, `pre_arrest_pick_side` preserves the coach-only arm, and `pick_side` preserves the submitted production arm, so later audits do not need to reconstruct a historical card from mutable source data.

## Week 1 dry run

**[Measured]** A read-only application against synchronized Week 1 artifact `artifacts/margin_predictions/2026-week-01-20260820T005017Z` and fresh snapshot `data/raw/player_arrests/20260820T153000Z` found 0 sole-flagged games and 0 flips on the 16-game card; the dry-run command and ledger before/after SHA-256 are recorded in the verification section below.

**[Inferred]** Zero Week 1 exposure is expected because the snapshot's latest incident is dated 2026-06-23, well outside the frozen 14-day window for Week 1's Tuesday decision date; this is an exposure observation, not evidence against the mechanism.

## Verification

- **[Measured]** The read-only dry run used `load_latest_complete_arrest_snapshot(Path('data'), now=datetime(2026,8,20,16,30,tzinfo=UTC))` and `apply_player_arrests_back_side_overlay` directly; both before and after checks reported the live challenger ledger as `ABSENT`, so it wrote no live decision row.
- **[Read]** Regression coverage includes coach/arrest composition conflict, stale-source no-write, final-card paper-ledger provenance, incumbent append-only recording, frozen-flag refresh, and fatal weekly ingest ordering. Current session-wide verification is recorded in `HANDOFF.md`.
