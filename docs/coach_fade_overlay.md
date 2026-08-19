# Year-1 head-coach fade overlay (PER-07 offshoot)

Owner decision **2026-08-18**: play the clean-case year-1-head-coach fade as
a pick-level overlay on the real weekly card, weeks 1-8 of the 2026 season,
and record both arms of the comparison (the un-overlaid model pick and the
overlaid pick) prospectively regardless of the play decision. This document
is the implementation record for that decision; the decision brief itself is
`week1_fade_brief.md` (owner-facing, not tracked in this repo).

## The rule, exactly as decided

Weeks 1-8 of the regular season only. Flip a game's forced pick from the
active model's own choice to the opposite side **only** when:

1. the model's pick lands on a team whose head coach is in year 1 of their
   tenure (a modal-coach change from the immediately prior, fully completed
   season -- see "How the flag is derived" below), **and**
2. the opponent's head coach is **not** also in year 1 (the "clean case").

Games where **both** coaches are year-1 are flagged but never flipped: the
registered measurement is a year-1-vs-kept-coach contrast and has no
direction to give a year-1-vs-year-1 game (Section 2 of the decision brief).
Weeks 9 and later are never touched -- the registered effect is a weeks-1-8
measurement and makes no claim outside that window.

Implemented in `src/nfl_ats/coach_fade_overlay.py`:
`year_one_by_game` derives the flag; `apply_coach_fade_overlay` applies the
rule at pick level; `overlay_disclosure_note` produces the plain-English
provenance sentence the published card shows.

## How the flag is derived (pregame-safe, not hand-typed)

Coach identity comes straight from the local nflverse raw schedule feed
(`data/raw/<snapshot>/schedules.parquet`, `home_coach`/`away_coach`
columns) -- the same source `scripts/hc_year_one_fade.py` and the Week 1
2026 decision brief used, never a hardcoded team list. For every (team,
season):

- `team_season_primary_coach` computes the **modal** credited REG-season
  coach for that team-season, from a season taken as a whole. This is only
  ever looked up for a **prior, fully completed** season -- it is never used
  to describe the CURRENT season a game belongs to, which would reintroduce
  a within-season lookahead.
- A side is "year 1" for a specific game when (a) its franchise has an
  OBSERVED, immediately-prior REG season in the schedule data (`season - 1`,
  looked up directly, so an expansion team, a franchise's first season in the
  data, or any gap year is excluded rather than guessed at), and (b) **that
  game's own credited coach** (a pregame-known fact once the schedule
  snapshot carries it) differs from the prior season's modal coach.

Using the game's own coach field -- never a same-season aggregate -- is what
keeps the flag leak-safe *within* a season; this is the production-safe
design `docs/hc_year_one_fade.md` ("What needs to be built") specified,
distinct from the season-mode shortcut the research script and the Week 1
brief used for their own retrospective measurement.

**Verified against local data (2026-08-18):** running `year_one_by_game`
against `data/raw/20260817T235649Z/schedules.parquet` reproduces the exact
7-of-32 teams the decision brief reported, derived independently rather than
copied: BAL, CLE, LV, MIA, NYG, PIT, TEN.

Two AGENTS.md-mandated leakage regression tests
(`tests/test_coach_fade_overlay.py`) prove this empirically: mutating a
*later* week's coach value never changes an earlier week's already-computed
flag, and mutating a *later season's* schedule never changes an earlier
season's flags.

## Design choice: pick level, not a training-time feature

The overlay is implemented as a **post-prediction pick transform**
(`apply_coach_fade_overlay` takes an already-scored `recommendations.csv`
and flips `home_cover_probability` to its complement on the clean-case
rows), not as a new `bias_hc_year_one_*` feature column feeding the model.

This was a deliberate choice, not the only option -- `docs/hc_year_one_fade.md`
already specs the feature-column version (`bias_hc_year_one_{home,away,diff}`
inside `constants.BIAS_METRICS`/`features.add_bias_features`), and that
design remains available if this project later wants the effect to
participate in training rather than only in card assembly. The pick-level
design was chosen here because:

1. **It sidesteps the frozen-inputs invariant entirely**, the same way
   `docs/postseason_support.md` had to build a two-pass feature pipeline to
   *preserve* that invariant for playoff rows. A training-time feature column
   changes the feature table's content and hash, which desynchronizes the
   active manifest and requires re-running `margin-backtest`/`margin-predict`
   before anything can publish -- exactly the workflow `docs/postseason_support.md`
   had to engineer around. A pick-level transform touches zero feature
   columns, zero training frames, and zero stored model artifacts: the
   active model's historical evaluation is provably unaffected by this file
   existing, with no two-pass build required to prove it. (The overlay's own
   "frozen inputs" analogue -- every column of the un-flipped rows, and every
   column but `home_cover_probability` even on flipped rows, stays
   byte-identical -- is asserted directly in
   `test_overlay_changes_only_home_cover_probability_on_the_flipped_row`.)
2. **It matches the decision brief's own framing.** Section 4 of the brief
   describes the overlay as "the active model's picks with the year-1 fade
   applied wherever it changes the side" -- a transform of an existing card,
   not a retrained model.
3. **The effect is not a training-time claim.** ROADMAP PER-07 is explicit
   that "the active model does not capture it" (it sides with the year-1 team
   51.6% of the time and those picks cover only 47.6%) -- the finding is
   about which SIDE to prefer on already-identified games, which is exactly
   what a pick-level rule expresses, not a magnitude to learn inside a
   regression.
4. **It keeps the rotation-registry window unspent.** ROADMAP PER-07's
   standing instruction is "do NOT ship on this measurement" for a
   rotation-registry `opener` window, because the effect lives entirely
   inside the mined 2018-2025 era it was discovered in (see below). A
   feature-column version would need a confirmation window before it could
   train inside the frozen model; a pick-level overlay is a separate, owner
   -made card decision that spends no window and needs none to be applied.

No hard blocker ruled out the feature-column design -- this is a choice, made
explicit as the task requires, not a workaround for something that could not
be built the other way.

## The EV arithmetic (from the decision brief)

**Inputs** (read from `registry/weak_signals.json`, `hc_year_one_fade`):
year-1 cover rate 46.73% vs kept-coach 50.99% on 856 games,
`probability_positive` 0.932, standard error 0.5036.

**Per-flip delta.** Flipping one pick moves its expected win probability from
the year-1 rate `p` to its complement `1 - p` (a single ATS game: if the
year-1 side doesn't cover, the other side did, modulo the push probability
this arithmetic ignores, matching the registered headline figures). At the
measured magnitude, per-flip delta = `(1 - 0.4673) - 0.4673` = **+6.54
points**.

**Season-level (weeks 1-8, 2026).** 121 games in the slate; 47 involve at
least one year-1 team (38.8%, roughly 2.2x the historical 17.7% rate --
2026 has unusually high coaching turnover); of those, 7 are both-year-1 and
excluded, leaving 40 clean subject games. The active model sides with the
year-1 team 51.6% of the time on clean subject games (ROADMAP PER-07's own
historical figure), giving an expected ~20.6 flip-candidate games across the
8 weeks:

| Scenario | Year-1 cover rate | Per-flip delta | Expected extra wins, season |
|---|---|---|---|
| Measured magnitude | 46.73% | +6.54 pts | **+1.35** |
| Half magnitude | 48.37% | +3.27 pts | **+0.68** |
| Zero (mined-era null) | 50.00% | 0 pts | **0.00** |

**Week 1, 2026, the one game actually in front of the owner:** BAL at IND,
flipping BAL -3.5 to IND +3.5 is worth **+0.065 expected wins** at measured
magnitude, **+0.033** at half magnitude, and exactly **0** at the null.

**The downside is bounded at zero, not negative.** This is a forced-pick
pool -- a pick is made on every game regardless -- so the overlay never adds
a new bet, it only reassigns the side of a bet already required. If the true
effect is null, both sides of a flip candidate are a 50/50 coin flip in
expectation, so swapping sides changes nothing. The one way this is not
literally costless is if the registered *direction* itself is backwards:
`probability_positive` 0.932 leaves a residual ~6.8% chance of that, already
priced into the split above, not an extra unaccounted-for downside.

## The mined-era caveat

ROADMAP PER-07's standing instruction is **do NOT ship on this measurement**
for a rotation-registry `opener` confirmation window: the effect is null in
2009-2017 (49.00% cover, P=0.321) and lives entirely inside 2018-2025
(44.79%, P=0.011) -- the same era the effect was mined from. Every opener-
graded window this project has left to spend already sits inside 2018-2025
(`GRADE_POOLS["opener"] = (2020, 2025)`), so no opener window can test this
without re-confirming the effect on its own discovery data. That circularity
is why no rotation-registry window has been declared or spent for this
family, and why this overlay is a **separate owner decision** to play the
clean case on the real card -- distinct from, and not contingent on, a
window-based confirmation that structurally cannot happen inside the
project's remaining opener blocks.

## Weeks 1-8 of 2026 score this prospectively

Registered as challenger `hc_year_one_fade_overlay` in
`artifacts/prospective/challengers.json`, status `ACTIVE_PROSPECTIVE`,
pinned to the active model's own configuration fingerprint (it is not a
retrained model -- its "model" IS the active model, transformed
post-prediction). `nfl-ats publish-predictions --record-decisions` writes
BOTH arms at publish time, kickoff-gated by the same
`RECORDING_LOCK_WINDOW` guard every other ledger write respects:

- the **un-overlaid** arm: the active model's own pick, already recorded to
  `artifacts/clv_ledger/decisions.parquet` by `record_paper_decisions`
  (unchanged by this work);
- the **overlaid** arm: `record_overlay_challenger_decisions` appends the
  overlay's pick for every game on the card (flipped or not) to
  `artifacts/prospective/challenger_decisions.parquet` under
  `challenger_id="hc_year_one_fade_overlay"`.

`nfl-ats prospective-score` settles both ledgers against the same games at
both grades (recorded line, primary; close, secondary), producing a paired
accuracy comparison with **no rotation-registry window spent** -- the same
non-circular pattern MOD-07's weak-signal stack already uses. `bet_side` is
always recorded as `PASS` and `edge` as `NaN` for this challenger: it tracks
forced-pick (`decision_line`) accuracy only, never a fabricated paper-bet
edge for the post-flip side.

## Verified Week 1 2026 dry run (2026-08-18, this implementation)

Against the active model's real Week 1 2026 card
(`artifacts/margin_predictions/2026-week-01-20260818T013139Z/recommendations.csv`,
model `118f31d9a98c815b`) and the local schedule snapshot
(`data/raw/20260817T235649Z/schedules.parquet`):

- **Exactly one flip**: `2026_01_BAL_IND`, BAL (year-1, Jesse Minter) at IND
  (kept coach, Shane Steichen) -- BAL -3.5 becomes **IND +3.5**.
- **One flagged, not flipped**: `2026_01_MIA_LV` -- both MIA (Jeff Hafley)
  and LV (Klint Kubiak) are year-1, so the clean-case rule does not fire.
- Every other year-1 game in Week 1 (ATL at PIT, CLE at JAX, DAL at NYG,
  NYJ at TEN) already has the model siding AGAINST the year-1 team, so the
  overlay does not touch them either.

This reproduces the decision brief's Section 2 table exactly, derived from
data rather than copied from the brief. The card was **not** published or
republished by this work -- `CURRENT_PREDICTIONS.md` stays exactly as it was
until the real 2026-09-08 Tuesday lock.

## Configuration

`src/nfl_ats/coach_fade_overlay.py`:

- `OVERLAY_ENABLED = True` -- the owner-decision switch. Off restores the
  un-overlaid card everywhere the overlay is wired in; the challenger ledger
  keeps recording both arms regardless, since it reads the rule itself, not
  this constant's effect on the published card.
- `OVERLAY_WEEK_MAX = 8` -- the registered window; not owner-configurable
  without a new measurement.

`publish_active_predictions(..., data_root=...)`
(`src/nfl_ats/publishing.py`) is the overlay's explicit opt-in on the
tracked card path: omit `data_root` (or point it somewhere with no local
schedule snapshot) and the overlay degrades to a no-op, mirroring how a
missing `line_sweep.parquet` silently degrades the Best Pick nomination
rather than failing the publish. `nfl-ats publish-predictions` (the CLI,
and therefore `weekly-run`'s step 7) always passes the real `data_root`.

## Scope note: the public HTML dashboard is not wired to the overlay

This implementation wires the overlay into the tracked Markdown card
(`CURRENT_PREDICTIONS.md` / `README.md`, via `publishing.py`) -- the
deliverable AGENTS.md calls out as "the tracked Markdown prediction card...
the deliberate exception for public forecast visibility," and the exact
surface the decision brief itself reads picks from. `src/nfl_ats/public_board.py`
(the public HTML site) independently rebuilds its own per-game cards,
including a per-game natural-language "explanation" derived from the
model's own SHAP-style feature attributions for the ORIGINAL pick. Flipping
`home_cover_probability` there without also re-deriving each flipped game's
explanation text would leave the page describing reasons for a pick it no
longer shows -- a real (not just cosmetic) inconsistency, and a separate
scope of work from this change. Before the 2026-09-08 real publish, wire
`apply_coach_fade_overlay` into `public_board.py`'s card-building path too,
or accept that the public site and the tracked card disagree on flipped
games until that follow-up lands.
