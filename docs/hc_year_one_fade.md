# `hc_year_one_fade` — predeclaration for an `opener` confirmation window

Predeclared 2026-08-18, before any NFL rotation-registry window is assigned
or spent for this family. This follows the recorded directive in
`ROADMAP.md` (Phase 4, PER-07): "Do NOT ship on this measurement -- declare
`hc_year_one_fade` with `acknowledges_mined_2018_2025` and buy the answer
with one opener window." Nothing in this document, or in producing it, has
assigned or spent that window.

## The recorded finding (ROADMAP PER-07, 2026-08-17)

A first-year head coach's team underperforms the spread in weeks 1-8:
**46.72% cover on 762 games / 104 team-season clusters** (cluster 95%
`[43.46, 50.06]`), **-1.30 raw ATS points** (-1.45 with a cubic quality
control), both blocked intervals excluding zero. Checked against three
alternative explanations and ruled out: not a quality confound (equal-quality
teams that *kept* their coach cover 50.70%), not a QB-change proxy (the
new-QB coefficient is null in the 2x2), and not already captured by the
active model (which sides with the year-1 team 51.6% of the time, and those
picks cover only 47.6%). Independently replicated on CFB at 3.5x the sample
(year-1 all-weeks 48.73% on 483 clusters; weeks 1-4 47.14%), same sign,
magnitude, and era boundary. **The catch**: the effect is null in 2009-2017
(49.00%, P=0.321) and lives entirely in 2018-2025 (44.79%, P=0.011) — inside
the mined-era ledger. Estimated worth: ~+0.57pp of full-slate pool accuracy.

## Independent reproduction (this session, `scripts/hc_year_one_fade.py`)

Rebuilt from scratch on a different pipeline than whatever produced the
original figures: coach identity read directly from nflverse's raw
`home_coach`/`away_coach` schedule columns (`data/raw/*/schedules.parquet`,
already local — no new ingestion), franchise relocations normalized with
this project's own `TEAM_ABBREVIATION_ALIASES` (OAK->LV, SD->LAC, STL->LA),
"year 1" requiring an *observed* prior season with a *different* primary
(mode) coach, and a team-season cluster bootstrap (5,000 resamples) for
every interval. Full output: `artifacts/hc_year_one_fade/20260818T003000Z/`.

| Quantity | Recorded | Reproduced here |
|---|---|---|
| Year-1 cover, weeks 1-8 | 46.72% on 762 games / 104 clusters | **46.70%** `[43.67, 49.71]` on 863 games / 118 clusters |
| Kept-coach cover (quality control) | 50.70% | **50.98%** `[49.36, 52.64]` on 2,897 games / 394 clusters |
| Raw ATS points, year-1 | -1.30 | **-1.093** |
| 2009-2017 cover (year-1) | 49.00%, P=0.321 (null) | **50.00%** `[45.81, 54.27]` on 398 games / 56 clusters (null) |
| 2018-2025 cover (year-1) | 44.79%, P=0.011 | **43.87%** `[39.66, 47.98]` on 465 games / 62 clusters |

Every headline number reproduces within a few points using an independently
built pipeline; the era boundary reproduces almost exactly (49.00%/44.79%
recorded vs 50.00%/43.87% here). The small game-count gap (863 vs 762) is
methodology detail (this reproduction's mode-based primary-coach detection
is slightly more permissive around in-season interim-coach games than
whatever exclusion the original used) and does not change the conclusion.

**Quality control, extended**: this reproduction additionally splits both
groups by prior-season scoring-margin tercile (computed within-season, so
"quality" is always relative to that year's league). The gap survives
*inside* the dominant (`low`-quality) bucket specifically — the bucket most
year-1 coaches actually inherit, since teams that fire a coach tend to have
been playing badly: year-1 teams cover 45.75% there (623 games / 84
clusters) vs kept-coach teams in the SAME bucket at 51.46% (684 games / 93
clusters). Kept-coach cover is flat across all three quality buckets
(50.6-51.5%), which is exactly what a clean comparison group should look
like. This is independent evidence the effect is not "bad teams stay bad."

The CFB replication was not independently rebuilt here (it needs a separate
CFB coach-identity pull this session did not build); it is reported as
recorded, and nothing found here contradicts it.

## Does it escape the standing closes? Yes — checked, not assumed

Two closes in `docs/pool_edge_plan.md`/`ROADMAP.md` PER-07 could plausibly
swallow this candidate. Neither does:

**1. The coach-identity/reputation close** (PER-07: split-half reliability
of a coach's mean ATS residual is +0.063, Spearman-Brown +0.119 — coaches
who covered well keep covering at 49.5-51.0%, i.e. a coach's *track record*
does not predict future performance). That close is a claim about a
**trait** (is this specific coach good?) and is refuted because the trait
itself barely correlates with itself across time. `hc_year_one_fade` makes
no claim about which coaches are good — it is an **event/transition**
claim (did this team's coach change?), which needs no stable coach-quality
trait to exist at all. An unreliable trait and a real short-run transition
cost are not in tension; "new-manager effects" are documented in other
domains for exactly this reason. This reproduction's quality-tercile split
provides an independent check that supports the same conclusion: the effect
holds up against teams of the SAME measured quality that simply didn't
change coaches.

**2. "Measuring team quality more precisely is bounded near zero"**
(`docs/pool_edge_plan.md`, ceiling ≈ +0.0129 points from the deliberate-leak
opponent-adjustment positive control). `hc_year_one_fade` is not a
measurement of team quality at all — it is invariant to it by construction,
and both the original write-up (equal-quality kept-coach comparison, 50.70%)
and this reproduction (flat 50.6-51.5% kept-coach rate across quality
terciles, with the gap surviving inside the dominant bucket) confirm it
empirically rather than by argument. It also is not covered by the "market
already prices team quality" logic, since the mechanism here is a
transition/installation cost the spread-setter would need a DIFFERENT kind
of information (roster-independent, about coaching-staff churn specifically)
to price — exactly `docs/pool_edge_plan.md`'s category 2, "prices something
the market prices badly," the same category availability occupies.

**3. Not the same family as the already-failed opener biases.** The three
bias features already built and ablated in MOD-07 (playoff holdover,
prior-week ATS, week-2 anchoring) are a different, already-tested set;
`hc_year_one_fade` is untested at NFL scale and distinct from all three (it
is about coaching continuity, not schedule-position bias). It is not "one
more bias feature that already failed."

**Verdict: escapes both closes.** This is a genuinely new, not-yet-spent
candidate, not the same closed family under a different name.

## What needs to be built (spec — no source file this item owns was edited)

This candidate is implemented as a new column in the existing bias family
(`constants.BIAS_METRICS`, `features.add_bias_features`), the same pattern
`bias_playoff_holdover`/`bias_prior_week_ats`/`bias_week2_anchor` already
use, so it plugs into `margin_feature_columns(..., "weak_stack")` and the
existing paired-comparison harness with no new plumbing. Building it
requires editing `constants.py` and `features.py`, which this item does not
own; **the exact patch is provided in this session's final report, not
applied here.** Summary of the design (leak-safety argued, not yet
regression-tested):

- `bias_hc_year_one_{home,away,diff}`: 1.0 when `week <= 8` AND the side's
  team has an observed, contiguous prior REG season AND that specific game's
  *own* credited coach (`schedules.home_coach`/`away_coach`, a pregame-known
  fact) differs from the team's prior season's *modal* REG-season coach
  (a fully completed season, so no lookahead). Using the CURRENT game's own
  coach value (rather than aggregating the current season) avoids any
  within-season lookahead entirely — the comparison never touches a future
  game to label an earlier one.
- Requires a new leakage regression test (AGENTS.md: "Add a leakage
  regression test for every new feature family"), following this project's
  existing pattern for the other bias features.

## Grade and window

**Grade: `opener`** (the recorded directive's explicit intent, and the
effect is a forced-pick-moving flag, not a calibration-only change — the
opener grade is this project's primary goal and where this candidate's
value would actually be realized in the pool).

Traced against `src/nfl_ats/rotation.py` (not executed): a new family with
no `inherits` and no prior windows draws the earliest eligible block in the
`opener` pool `(2020, 2025)` at the default 2-season width, i.e.
**`[2020, 2021]`**. That block already overlaps two OTHER families'
spent windows (`best_pick_ranker_opener`, `mod07_weak_signal_stack`), which
rule 4 explicitly permits (windows retire per-family). It intersects the
mined 2018-2025 ledger, so `--acknowledge-mined` is REQUIRED at declaration
— consistent with the recorded finding itself living entirely inside that
era.

**Inherits**: none (a genuinely new family; no existing spent window
constrains it).

**Training policy**: forward-chaining, `min_train_games=500`, player
feature profile, `market_residual` target — the frozen active model's own
recipe, with `weak_stack` (or a narrower profile containing only the bias
family) as the candidate arm once `bias_hc_year_one_*` exists.

## Frozen decision rule

Primary metric: paired forced-pick accuracy improvement of the candidate
(player profile + `bias_hc_year_one_*`) over the frozen baseline (player
profile alone) on the assigned `[2020, 2021]` opener window, week-blocked,
via `paired_feature_comparisons`. Clears if `probability_positive >= 0.75`
(the same SPEC-5/`best_pick_ranker` screening bar), matching this registry's
established practice for a small, mined-era-adjacent window. Brier/log-loss
are reported as secondary coherence checks only. One run; no threshold
(week <= 8), coach-source, or quality-control retuning after seeing the
window's results.

## Commands to run (NOT executed by this document)

```powershell
.\.tools\uv.exe run --no-sync nfl-ats rotation declare `
  --name hc_year_one_fade `
  --description "Does a bias_hc_year_one flag (team is in weeks 1-8 of a new head coach's tenure) improve forced-pick accuracy over the frozen player profile at the Tuesday opener? See docs/hc_year_one_fade.md; the underlying effect lives inside 2018-2025 (ROADMAP PER-07)." `
  --grade opener `
  --acknowledge-mined

.\.tools\uv.exe run --no-sync nfl-ats rotation assign --name hc_year_one_fade
```

Expected assignment: `[2020, 2021]` (verify against `nfl-ats rotation
status` at run time).

```powershell
.\.tools\uv.exe run --no-sync nfl-ats rotation record `
  --name hc_year_one_fade `
  --artifact <artifacts/.../<confirmation-run-id>> `
  --verdict <confirmed|closed_negative|unresolved> `
  --probability-positive <accuracy P(positive) from that run> `
  --notes "<one-line summary>"
```

## Declared limitations

1. `bias_hc_year_one_*` does not exist in the feature table yet; it must be
   built (spec above, patch in the session report) and pass a leakage
   regression test before any confirmation run.
2. The CFB replication is reported as recorded, not independently rebuilt
   this session (needs a separate CFB coach-identity source).
3. This reproduction's coach-of-record detection (modal coach per REG
   season) is slightly more permissive than whatever the original used
   (863 vs 762 games); the production feature must use the stricter,
   leak-safe per-game design in the spec above, not the season-mode
   shortcut used for this research reproduction.
4. A 2-season opener window is small; the 0.75 `probability_positive` bar is
   a screening gate, not a claim the window fully resolves the effect.
