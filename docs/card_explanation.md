# Card-level explanation contract (ENG-12)

Written 2026-09-04. Every claim is labeled per `AGENTS.md`: **measured** (a
command run this session, command given), **read** (a file opened this
session, path given), or **inferred** (reasoning, labeled as such).

Implementation: `src/nfl_ats/card_explanation.py`.
Tests: `tests/test_card_explanation.py` (**measured**,
`pytest tests/test_card_explanation.py`, 32 passed).

## What this answers

A published pick collapses a lot of machinery into three table cells: a
team, a number, a decision score. This module builds the DESCRIPTIVE record
behind one pick and renders it as one plain paragraph:

* the market line used, which snapshot it came from, and when that snapshot
  was captured;
* this game's own model probability for the picked side — labeled
  explicitly as a single-game estimate, never as accuracy (AGENTS.md: "Never
  describe the current historical forced-pick accuracy as proof of a
  profitable or stable edge. Keep historical accuracy distinct from each
  game's model probability.");
* which overlays fired, their direction, the input value that triggered
  them, and whether they changed the pick;
* freshness per input source (as-of instant plus
  complete/degraded/blocked/`no_data`);
* whether anything has changed since the Tuesday card
  (`none`/`pick_flipped`/`line_moved`/`overlay_added_removed`/`no_refresh_yet`).

Every field the caller does not supply degrades to an explicit `no_data`
state — never a guess, never silence — the same discipline
`nfl_ats.lineage` and `nfl_ats.source_freshness_policy` already established
(a lineage entry may carry `lineage: null` only with a stated `reason`; an
unobserved freshness source is reported, never folded into "healthy").

## The entry points

```python
explain_pick(row, *, lineage=None, source_report=None, overlays=None, refresh_changes=None) -> PickExplanation
explain_card(rows, *, lineage=None, source_report=None, overlays_by_game=None, refresh_changes_by_game=None) -> list[PickExplanation]
```

`row` is any `Mapping` with at least `game_id`/`home_team`/`away_team`/
`spread_line`/`home_cover_probability` — a plain `dict`, a `pandas.Series`,
or one record of a forecast/recommendations frame all satisfy it.
`lineage` is an optional `nfl_ats.lineage.CardLineage` (ENG-16) supplying
the market line's source snapshot and capture instant. `source_report` is
an optional `nfl_ats.source_freshness_policy.SourcePolicyReport` (ENG-14)
supplying per-source freshness. `overlays` is a pre-normalized sequence of
`OverlayFiring` records — pass `()` (not `None`) once overlay evaluation
has actually run with nothing firing, so the pick reads as "evaluated, none
fired" (`measured_from_artifact`) rather than "not supplied" (`no_data`).
`refresh_changes` is an optional `RefreshChangeInput` (or an equivalent
mapping) describing the latest Tuesday-to-refresh delta for that game.

## Provenance, per component

Every component (`market_line`, `model_probability`, `overlays`,
`freshness`, `refresh`) carries its own `provenance` field in
`{measured_from_artifact, computed_now, no_data}`:

* `measured_from_artifact` — read directly off a stored artifact (the
  card's own `spread_line` column, a `lineage.json` snapshot record, a
  `SourcePolicyReport`, the pick-revision ledger).
* `computed_now` — derived from a stored value by a small, deterministic
  computation this call performs (the pick-side-oriented model probability
  is `home_cover_probability` or its complement, not a column stored
  verbatim).
* `no_data` — the caller did not supply this input; the component still has
  its full fixed shape, just with empty/`None` fields and an honest label.

## Overlay adapters

Only FIRED overlays are represented (mirrors `nfl_ats.lineage`'s own rule:
"an overlay that did not fire changed nothing, so it has nothing to
justify"). Two tiers of adapter build one normalized `OverlayFiring`:

* `overlay_firing_from_coach_fade_flip` / `overlay_firing_from_arrest_flip`
  / `overlay_firing_from_division_revenge_flip` /
  `overlay_firing_from_spread_gap_flip` — rich, team-level detail from each
  overlay module's own flip record (`OverlayFlip`, `ArrestFlip`, the two
  `TiltFlip` classes).
* `overlay_firings_from_composition(composition, game_id)` — a generic
  fallback built from the four-member production policy's own provenance
  (`nfl_ats.four_overlay_composition.FourOverlayCompositionResult`): member
  id plus the raw/final `home_cover_probability`. This is what
  `publish_active_predictions` actually calls, once per game in the card —
  **every** game gets an entry (an empty tuple for one where nothing
  fired), not only the flipped subset `composition.games` itself lists, so
  an unflipped pick reads as "no overlay fired" and not "not supplied".

## Refresh: the ENG-18 join point

`RefreshChangeInput` is a generic, source-agnostic Tuesday-to-refresh delta
for one game (`previous_pick_side`, `new_pick_side`, `movement_delta`,
`overlays_added`/`overlays_removed`, `note`). Classification precedence is
a pick flip first, then a line move, then an overlay change, else `none`.

**Read** 2026-09-04: `src/nfl_ats/snapshot_diff.py` (ENG-18's own
Tuesday-vs-refresh diff module) did not exist yet in this tree when this
module was built. `refresh_change_from_pick_revision` is the fallback this
task's own instructions named: it reads one row of
`nfl_ats.pick_refresh`'s append-only pick-revision ledger
(`load_pick_revisions`) directly and adapts it into a `RefreshChangeInput`.
When `snapshot_diff.py` lands, prefer its own per-game summary directly —
adapt it into a `RefreshChangeInput` (or pass an equivalent mapping) and
hand it to `refresh_changes`/`refresh_changes_by_game`; this fallback stays
correct either way as the direct-ledger read.

## The language contract

`LANGUAGE_CONTRACT` is a literal, case-insensitive substring blocklist —
`"will win"`, `"lock"`, `"guaranteed"`, `"guarantee"`, `"profitable"`,
`"edge proven"`, `"because of"`, `"caused"`, `"beats the market"`, `"sure
thing"`, `"can't lose"`, `"cannot lose"`, `"no risk"`, `"risk-free"` — not a
semantic classifier. `check_language(text)` raises `LanguageContractError`
if any phrase appears anywhere in `text`, including inside an
otherwise-safe negation (a substring check cannot distinguish "not
profitable" from "profitable"), so the template is written to avoid every
phrase outright rather than negate it. `explain_pick` runs this check on
its own generated text before returning — a caller can never receive a
`PickExplanation` whose `text` violates the contract.

## Wiring into `publish_active_predictions` (additive)

**Read** 2026-09-04, `src/nfl_ats/publishing.py`:

* `explanations.json` is **always** written beside the linked forecast
  artifact (`forecast_dir / "explanations.json"`, via
  `nfl_ats.io.atomic_json`) — a new file, so no existing card-writer test
  needed to change.
* `_publication_context`'s returned tuple gained one element, the raw
  (overlay-applied, display-unformatted) predictions frame, because
  `_published_card`'s own `Date`/`Matchup`/`ATS prediction`/`Decision
  score` projection drops `game_id`/`home_team`/`away_team`/`spread_line`/
  `home_cover_probability` — exactly what `explain_card` needs. It is a
  private, single-call-site helper (**read**, `git grep
  _publication_context` — the only call is inside `publish_active_
  predictions` itself; the two other repo references are docstring
  mentions, not calls), so extending its return shape carries no external
  blast radius.
* `lineage.json` (read via `nfl_ats.lineage.read_card_lineage`) and the
  pick-revision ledger (`nfl_ats.pick_refresh.load_pick_revisions`) are
  both OPTIONAL inputs, read read-only and degraded to `None`/empty on
  `FileNotFoundError`/`OSError`/`ValueError`/`KeyError` — matching every
  other optional artifact already on this publish path (the coach-fade
  overlay's own schedule-snapshot fallback, the arrest overlay's disabled
  path, etc.).
* An inline "Pick explanations" section can additionally be appended to the
  tracked Markdown card (`render_markdown`), gated behind
  `include_pick_explanation_lines: bool = False` on
  `publish_active_predictions`. **It defaults OFF.** The existing
  card-writer tests (`tests/test_publishing.py`) assert on the exact
  Markdown table/header text `publish_active_predictions` produces, and
  verifying the appended section would not break any of those specific
  assertions was not exercised end-to-end for the ON path this session —
  defaulting off keeps that verification unnecessary for this change to
  ship. Turning it on is one keyword argument at the call site.
* The publish summary dict gained one additive key,
  `"pick_explanations_path"` (the path `explanations.json` was written to,
  or `None` on the unreachable path where no forecast artifact resolves).

## Example (read-only, live Week 1 artifact)

**Measured** 2026-09-04, read-only against
`artifacts/margin_predictions/2026-week-01-20260903T143253Z/` (nothing
under `artifacts/` was modified; output written only under
`%TEMP%\eng12\week1_explanations.json`):

> ARI at LAC: the market line used for this pick is +10.5. The model's own
> probability for ARI to cover this game is 65.3%; this is a single-game
> estimate, not the project's historical accuracy. No overlay fired on this
> pick. Source freshness at publish time: 6 complete, 4 degraded. No
> late-week refresh has run yet for this pick. This is a descriptive
> research summary, not a wagering recommendation.

> BAL at IND: the market line used for this pick is +3.5. The model's own
> probability for IND to cover this game is 53.7%; this is a single-game
> estimate, not the project's historical accuracy. 1 overlay fired on this
> pick: coach fade (complemented toward home, triggered by raw home-cover
> probability 0.463 complemented to 0.537). Source freshness at publish
> time: 6 complete, 4 degraded. No late-week refresh has run yet for this
> pick. This is a descriptive research summary, not a wagering
> recommendation.

`lineage.json` did not exist for this artifact at read time (no snapshot
metadata to surface, hence no snapshot clause in the market-line sentence);
the pick-revision ledger held no rows for this season/week (hence
`no_refresh_yet` on every game). Both degrade exactly as designed rather
than raising or inventing a value.
