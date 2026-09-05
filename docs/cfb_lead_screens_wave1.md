# CFB free-screen wave 1: LEAD-48, LEAD-50, LEAD-46, LEAD-44

Written **before any ATS outcome, cover rate, accuracy delta or sign is
computed for any of the four columns below**. Sections under each lead's
"Predeclaration" heading were frozen before `scripts/cfb_lead_screens_wave1.py`
was run in `--mode screen`; each lead's "Results" subsection was appended
after the look and changes nothing above it.

This is a **cross-league screen batch**, not a new NFL look. It spends **no
NFL evaluation window and no rotation window** -- CFB is this project's
sanctioned free replication ground, the same one
`scripts/cfb_option_prep_screen.py` (LEAD-45) and
`docs/cfb_rest_bye_replication.md` used. **Each lead is its own weak-signal
family and its own signed/flag column** on top of the frozen XLG-03
benchmark arm: none of the four is pooled with another, and none is pooled
with `cfb_rest_bye_replication` or `cfb_option_prep_screen`'s families either
(different constructs, disclosed per-lead below).

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
**bounded by a positive control** -- the instrument was PROVEN able to detect
an effect that size and it was absent. Everything else is
`unresolved_below_power`: record it with
`nfl-ats weak-signals record --league cfb`, report `probability_positive`,
never the binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. A promotion threshold governs only what the docs may CLAIM; it
never governs which card is PLAYED, which is expected value.

## Shared comparator, metric, and controls (all three scored leads)

| element | value |
|---|---|
| Population | `data/processed/cfb_game_features.parquet`, restricted to `nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS` (2012-2019, 2021-2025; **read**, `src/nfl_ats/cfb_benchmark.py:46`) |
| Baseline arm | `nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS` (35 columns, the frozen XLG-03 contract), `nfl_ats.cfb_benchmark.fit_cfb_residual_model`, ridge alpha 10.0 |
| Candidate arm | the same 35 columns **plus exactly one** of the three columns below |
| Walk-forward | every scored week's models train on completed table games kicking off strictly before that week's earliest kickoff, 500-game floor (`CFB_BENCHMARK_MIN_TRAIN_GAMES`) |
| Metric | paired candidate-minus-baseline forced-pick accuracy delta, `accuracy_points`, picks at `home_cover_probability >= 0.5`, graded with `nfl_ats.clv.pick_correct` (pushes NaN, excluded) |
| Uncertainty | week-blocked bootstrap primary (1,000 samples, seed 20260905), season-blocked secondary, never averaged (within-week correlation is ZERO by owner mandate) |
| Positive (leak) control | the candidate column REPLACED by the realized `ats_margin` -- must read hugely positive (P+ 1.000); if it does not, the harness is blind |
| Null | 200 within-week permutations of the settling margin; the observed delta is reported at its null percentile |
| Era split | declared before scoring: **2012-2019** and **2021-2025**, the benchmark's own regime-gap boundary; per-era magnitudes reported separately, never averaged across a sign flip |
| Grade | CFB is close-graded (`spread_line`, the median-across-books close proxy; no verified CFB opener exists) -- settles no NFL play/no-play or promotion decision by itself |

Every candidate column here is either a deterministic schedule/calendar
identity or a deterministic team/pair identity -- built from team ids,
kickoff dates, and (for the rivalry lead) the full local schedule history,
never from a score, a line, or any outcome column. That is what the
leakage tests in `tests/test_cfb_lead_screens_wave1.py` pin: every column is
bit-identical after `result`, `ats_margin`, `home_cover`, `home_points`, and
`away_points` are permuted.

---

## 1. LEAD-48: CFB post-bye prep asymmetry

### Predeclaration

**Mechanism and predeclared direction.** Extra install time matters more in
college football (bigger playbooks, younger players who rely more on
practice-week installation than professional muscle memory), so the team
coming off a bye should show an ATS edge over an opponent that is not --
**the opposite prediction from the NFL bye fade** (where the market is read
as OVERpricing a professional team's extra rest,
`bye_overval_home_edge_post2011`, `docs/bye_overvaluation_screen.md`).
**Predeclared direction: BACK the team off a 13+ day bye when its opponent is
not.** The cross-league contrast is the point: if CFB backs the bye team
while the NFL fades it, that adjudicates a rest-vs-prep mechanism
difference, not just a direction flip.

**Is this a re-score of `docs/cfb_rest_bye_replication.md`?** Read in full
before building this lead (**read**, that document's cells 1-3). It is
**not** the same construct, on three independent grounds:

1. **Threshold and combination.** Cells 1/2 (`home_off_bye` /
   `away_off_bye`) are two SEPARATE unsigned columns at `rest >= 13` with no
   "and the other side is not" condition. Cell 3 (`bye_edge_home`) IS an
   edge condition but at a **12-day** gap, **home-only** (no away-side
   mirror in the same column), and unsigned.
2. **Signedness.** This lead is ONE signed column combining both directions
   (+1 home edge, -1 away edge) at the 13-day threshold -- no existing
   column in that document does this.
3. **Predicted direction.** Cell 3's predicted direction is **negative**
   (market OVERprices the CFB bye, mirroring the NFL mechanism). This lead's
   predicted direction is **positive** (the opposite mechanism: prep
   asymmetry, not rest overvaluation). Scoring the same numbers under two
   opposite predictions is not a re-score; it is a different hypothesis
   about the same calendar fact.

Per this document's own instruction, this is therefore scored, not skipped.

**Population.** The XLG-03 clean core (2012-2019, 2021-2025). Both sides'
rest must be known to resolve to a sign or 0; a season opener on either side
leaves the row NaN (never 0 -- the project's standing rest-missingness
convention, identical to `rest_diff`'s own treatment).

**Encoding (frozen).** `cfb_lead48_post_bye_signed`: **+1** when
`home_rest >= 13` and NOT `away_rest >= 13`; **-1** when `away_rest >= 13`
and NOT `home_rest >= 13`; **0** when both sides are known and neither/both
qualify (Army-Navy-style both-off-bye games included at 0, not excluded);
**NaN** when either side's rest is undefined. Per-side rest is derived by
`nfl_ats.cfb_rest_bye_feature.derive_side_rest` -- the exact helper that
built the frozen `rest_diff` column and the `cfb_rest_bye_replication`
cells -- reused, not re-derived, from the FULL CFB schedules snapshot
(`data/cfb/schedules`), never the filtered benchmark table.

**Comparator, metric, controls, era split.** All as in the shared table
above.

**Reliability.** The underlying per-team trait ("this team's propensity to
arrive off a 13+ day bye") is the SAME trait `docs/cfb_rest_bye_replication.md`
section 5 already measured as `own_off_bye_13` on the identical clean-core
team-season panel: across-season odd/even-year Pearson r **+0.7052** [+0.5180,
+0.8300], Spearman-Brown **0.8271**, `probability_positive` **1.0000**
(**read**, that document, section 5 table). Reused here rather than
re-derived, because this lead reads the identical trait off both sides of the
game and combines it into one signed edge column -- a different COMBINATION
of the same measured propensity, not a new trait needing its own reliability
run. That reliability (0.8271) is far above the `no_split_half_reliability`
ceiling of 0.10 (**read**, `src/nfl_ats/weak_signals.py`,
`NO_SPLIT_HALF_RELIABILITY_MAX`), so that ground is inadmissible for this
lead regardless of the interval; independently, this is a deterministic
calendar fact with zero measurement error, the same argument
`docs/cfb_option_prep_screen.md` section 7 and
`docs/cfb_rest_bye_replication.md` section 5 both freeze.

**Decision rule and recording.** Expected value, never a threshold. The
pooled entry `cfb_post_bye_home_on_benchmark` records `unresolved_below_power`
unless its own interval resolves; era slices are recorded separately under
the same family **because the two eras disagree in sign** (the same
judgement `docs/cfb_rest_bye_replication.md` section 8 exercises), and any
era slice whose WHOLE week-blocked interval sits on one side of zero records
that as `refuted_mechanism` / `wrong_sign_resolved` for that era-restricted
population -- a narrower, mechanically justified conclusion about that
specific window, not a claim about the pooled full-window construct.

### Results (2026-09-05)

Coverage (`--mode coverage`, predictor-only): clean core 9,093 games, **1,500
flagged nonzero** (781 `+1` home-edge, 719 `-1` away-edge), 548 missing
(season openers on one or both sides).

Positive control (`--mode positive-control`, artifact
`artifacts/cfb_lead_screens_wave1/post_bye/20260905T023959Z/results.json`):
**pooled +48.405 accuracy points, week-blocked 95% [+47.311, +49.408], P+
1.000** on 8,933 games / 199 weeks -- identical in magnitude to LEAD-45's and
`cfb_rest_bye_replication`'s own leak controls, as expected (same leak
mechanism, same 36-column ridge fit). Era controls: 2012-2019 **+48.345**
(P+ 1.000, [+46.905, +49.758]); 2021-2025 **+48.493** (P+ 1.000, [+46.888,
+50.240]). The harness is not blind in either era.

Null (`--mode null`, 200 within-week permutations of the settling margin):
mean **+0.0074 pts**, 95% [-0.202, +0.224], observed delta at the **39.5th
percentile** -- clean, no home-tilt artifact to discount.

Screen (`--mode screen`, artifact
`artifacts/cfb_lead_screens_wave1/post_bye/20260905T024109Z/results.json`):

| cut | delta (pts) | week 95% CI | week P+ | season 95% CI | season P+ | n games / weeks / flagged |
|---|---|---|---|---|---|---|
| pooled | **-0.022** | [-0.289, +0.258] | 0.428 | [-0.323, +0.290] | 0.407 | 8,933 / 199 / 1,500 |
| era 2012-2019 | **+0.262** | [-0.098, +0.627] | **0.915** | [-0.056, +0.605] | 0.937 | 5,349 / 122 / 946 |
| era 2021-2025 | **-0.446** | **[-0.763, -0.166]** | **0.000** | [-0.695, -0.196] | 0.000 | 3,584 / 77 / 554 |

**Era 2021-2025's whole week-blocked AND season-blocked interval sits below
zero.** That is the admissible `wrong_sign_resolved` ground, met exactly and
mechanically -- not inferred from "the number looks bad": the predeclared
direction (back the bye-edge team) is resolved WRONG for this
era-restricted population. Era 2012-2019 leans the predicted way with P+
0.915-0.937, but its own interval still contains zero
([-0.098, +0.627] pts), so it stays `unresolved_below_power`. The pooled
full-window entry averages the two eras' opposite signs down to -0.022 pts
(P+ 0.428) and its interval contains zero, so it too stays
`unresolved_below_power` -- **the pooled number is not a fair summary of
this lead**, and the per-era table is the one that carries the finding, per
"era magnitude, not presence."

**What this implies for the decision, before the caveat.** On the
project's EV rule, era 2021-2025 says: taking the baseline over this
specific signed column in the most recent window is favored at P+ 1.000
(the whole interval says so) -- this specific combination-and-threshold does
NOT belong on a CFB card built for recent seasons, full stop, regardless of
what era 2012-2019 says. Era 2012-2019 leans the ORIGINAL predicted
direction at P+ 0.915 but is not resolved. **Caveat:** this is one signed
column built from a threshold (13 days) and a combination rule (edge-only)
declared before scoring; it does not resolve the broader "does CFB prep
asymmetry exist" question, only this specific operationalization of it in
this specific era.

---

## 2. LEAD-50: CFB rivalry home dog

### Predeclaration

**Mechanism and predeclared direction.** Rivalry games compress talent gaps:
effort and emotion run higher for both sides, and extra scouting/familiarity
narrows tactical surprise, both of which should help the side the market
prices as an underdog close the gap more than in a non-rivalry game.
**Predeclared direction: BACK the HOME underdog in a rivalry game.** The NFL
cousin is the deployed division-revenge overlay; CFB rivalry flags are a new
family, not a replication of it (no local rivalry field lets a like-for-like
transcription happen, and revenge and rivalry are different mechanisms:
revenge is about a specific PAST grievance, rivalry is a persistent pairing).

**Population.** The XLG-03 clean core. The flag can only be 1 for games
where the home team is the market underdog; the mechanism makes no claim
about rivalry games where the home team is favored, so those score 0 (same
convention as `bye_edge_home`'s edge restriction).

**Rivalry identity: no local field exists, so the deterministic proxy
applies, exactly as this lane's task predeclares it.** **Measured** this
session: the CFBD schedule rows carry a free-text `notes` field (1,586 of
36,915 non-empty) with zero systematic rivalry marker -- a case-insensitive
search for "rival" turns up exactly 3 rows, all a single neutral-site bowl
name ("Allstate Red River Rivalry"), not a usable flag. **Encoding (frozen):
a team pair is a rivalry if they met (regular season, completed) in at least
8 consecutive seasons anywhere in the full local schedule history**
(`data/cfb/schedules`, all 25 seasons it carries, not just the clean core).
**Measured**: 521 pairs meet this bar. **Disclosed limitation, not a
defect**: this proxy has high recall for genuine historic rivalries (Army-
Navy, Auburn-Alabama, Ohio State-Michigan all qualify) but also captures
long-running conference-mandated annual pairings that are not "rivalries" in
the colloquial sense (e.g. Georgia-South Carolina, Alabama-Arkansas) --
no local field exists to separate the two, so the mechanical definition
given in this lane's task is used exactly as specified. This construction
reads only team identity, completion status, and season from the full
schedule -- never a score or a line -- so it cannot encode any game's own
outcome. It DOES use schedule knowledge from seasons after a given game
(a pair's later meetings can establish the 8-consecutive-season run that
retroactively flags an earlier game), the same identity-style argument
`docs/cfb_option_prep_screen.md` uses for the Georgia Tech 2008-2018 option
window: an institutional/scheduling fact known in hindsight, not a
game outcome.

**Encoding (frozen).** `cfb_lead50_rivalry_home_dog`: **1** iff the game's
team pair is a rivalry pair (above) AND `spread_line < 0` (home team is the
market underdog; a pick'em at exactly 0 does not qualify as underdog); **0**
otherwise. Always defined (`spread_line` has zero missingness on the
benchmark table, **measured**).

**Comparator, metric, controls, era split.** As in the shared table above.

**Reliability.** Rivalry-pair membership is a deterministic identity fact
(schedule history, zero measurement error) and market-underdog status is a
deterministic market fact (the benchmark's own `spread_line`, also zero
measurement error): `no_split_half_reliability` is inadmissible for the same
reason it is inadmissible for LEAD-45's option-team identity and the
rest/bye calendar facts -- there is no noisy construct here for a sample
size to fail to rescue.

**Decision rule and recording.** Expected value, never a threshold. One
pooled entry, `cfb_rivalry_home_dog_on_benchmark`. Era slices are reported in
this document and in `--notes` but **not** recorded as separate registry
rows unless they diverge materially (judged after the look, section 8 style
of `docs/cfb_rest_bye_replication.md`).

### Results (2026-09-05)

Coverage (`--mode coverage`): clean core 9,093 games, **4,459** rivalry-pair
games (any spread), **3,439** home-underdog games (any pair), **1,852**
flagged (both conditions), 0 missing.

Positive control (artifact
`artifacts/cfb_lead_screens_wave1/rivalry_home_dog/20260905T024240Z/results.json`):
pooled **+48.405 pts, P+ 1.000**, [+47.311, +49.408], n=8,933/199 -- identical
control magnitude, as expected (same leak mechanism). Era controls: 2012-2019
+48.345 (P+ 1.000); 2021-2025 +48.493 (P+ 1.000).

Null (200 within-week permutations): mean **+0.0236 pts**, observed delta at
the **44.5th percentile** -- clean.

Screen (artifact
`artifacts/cfb_lead_screens_wave1/rivalry_home_dog/20260905T024348Z/results.json`):

| cut | delta (pts) | week 95% CI | week P+ | season 95% CI | season P+ | n games / weeks / flagged |
|---|---|---|---|---|---|---|
| pooled | **+0.011** | [-0.237, +0.290] | 0.555 | [-0.137, +0.171] | 0.490 | 8,933 / 199 / 1,852 |
| era 2012-2019 | **+0.037** | [-0.315, +0.394] | 0.547 | [-0.171, +0.283] | 0.583 | 5,349 / 122 / 1,208 |
| era 2021-2025 | **-0.028** | [-0.388, +0.363] | 0.429 | [-0.165, +0.114] | 0.299 | 3,584 / 77 / 644 |

Every interval crosses zero. Both era magnitudes are noise-scale
(|delta| < 0.04 points, P+ within 0.06 of a coin flip): not recorded as
separate registry rows -- unlike LEAD-48's clear -0.45-vs-+0.26-point
divergence with one era fully resolved, this pair does not diverge enough
to earn its own rows, and recording two more near-null entries would
proliferate the registry without adding information the pooled row and this
table do not already carry.

**What this implies for the decision, before the caveat.** On EV grounds
this lean is a near-exact coin flip (pooled P+ 0.555): taking the candidate
over the baseline is barely favored, nowhere near a promotion bar, and not
worth acting on alone. Recorded `unresolved_below_power`. **Caveat**: the
rivalry proxy's disclosed imprecision (conference-mandated pairs mixed with
true rivalries) means a genuine rivalry-specific effect could be diluted by
the broader pool this construction admits; a tighter identity (e.g. a
curated named-trophy list) was not built here because no such list exists
locally without a fetch.

---

## 3. LEAD-46: CFB altitude-plus-cold home

### Predeclaration

**Mechanism and predeclared direction.** Altitude imposes a real
conditioning cost on visiting teams (reduced oxygen availability degrades
repeated-sprint performance), and that cost compounds with late-season cold
at exposed Mountain-West stadiums. **Predeclared direction: BACK the
altitude home team, October on.**

**Elevation data: not on disk, so the frozen named list stands in, exactly
as this lane's task instructs.** **Measured** this session: `data/cfb`
carries no elevation, latitude/longitude, or venue-detail table --
`data/cfb/team_info` (a directory outside `nfl_ats.cfb.CFB_SOURCES`, read
directly as a raw parquet partition) carries only `venue_name`, `city`,
`state`, no elevation figure. **The frozen named list (Colorado State,
Wyoming, Air Force, Utah -- the ROADMAP-declared Mountain-West altitude
outs) is used unchanged.** All four team names are confirmed present in the
benchmark table's `home_team`/`away_team` columns (**measured**).

**"Cold" is calendar-proxied, disclosed rather than assumed.** No local
weather/temperature column exists on the benchmark table (**measured**: the
60-column schema carries no temperature field). "October onward" (calendar
month >= 10, read from `gameday`) stands in for the late-season cold
component; this is a coarser proxy than an actual temperature reading, and
the result below should be read as an altitude-plus-calendar-lateness
effect, not a verified altitude-plus-cold effect.

**Population.** The XLG-03 clean core.

**Encoding (frozen).** `cfb_lead46_altitude_cold_home`: **1** iff
`home_team` is in `{Colorado State, Wyoming, Air Force, Utah}` AND
`gameday.month >= 10`; **0** otherwise. Always defined (no missingness: team
identity and calendar month are never undefined for a completed game).

**Comparator, metric, controls, era split.** As in the shared table above.

**Reliability.** Deterministic identity (frozen team list) plus a
deterministic calendar fact (month), zero measurement error:
`no_split_half_reliability` is inadmissible for the same reason as LEAD-45's
option-team identity.

**Decision rule and recording.** Expected value, never a threshold. Pooled
entry `cfb_altitude_cold_home_on_benchmark`; era slices recorded separately
under the same family because the two eras diverge materially (judged after
the look).

### Results (2026-09-05)

Coverage (`--mode coverage`): clean core 9,093 games, **191 flagged**
(~14-17/season, every season -- thin by construction, four programs'
October-on home slates), 0 missing.

Positive control (artifact
`artifacts/cfb_lead_screens_wave1/altitude_cold/20260905T024427Z/results.json`):
pooled **+48.405 pts, P+ 1.000**, [+47.311, +49.408] -- identical control
magnitude. Era controls: 2012-2019 +48.345 (P+ 1.000); 2021-2025 +48.493
(P+ 1.000).

Null (200 within-week permutations): mean **+0.0010 pts**, observed delta at
the **24.0th percentile** -- clean.

Screen (artifact
`artifacts/cfb_lead_screens_wave1/altitude_cold/20260905T024538Z/results.json`):

| cut | delta (pts) | week 95% CI | week P+ | season 95% CI | season P+ | n games / weeks / flagged |
|---|---|---|---|---|---|---|
| pooled | **-0.090** | [-0.366, +0.184] | 0.230 | [-0.361, +0.149] | 0.218 | 8,933 / 199 / 191 |
| era 2012-2019 | **+0.075** | [-0.219, +0.393] | 0.657 | [-0.076, +0.192] | 0.820 | 5,349 / 122 / 115 |
| era 2021-2025 | **-0.335** | [-0.806, +0.144] | 0.086 | [-0.867, +0.282] | 0.122 | 3,584 / 77 / 76 |

No cut's interval clears zero (era 2021-2025 comes closest at P+ 0.086 but
[-0.806, +0.144] still contains it). Both era rows are recorded despite
neither resolving, because the divergence is large relative to the pooled
number (a 0.41-point swing between eras) and the sample per era is thin (76
and 115 flagged games) -- exactly the case "era magnitude, not presence"
exists to preserve rather than average away.

**What this implies for the decision, before the caveat.** On EV grounds,
the pooled column is a mild lean AGAINST the predeclared direction (P+
0.230), while era 2012-2019 alone leans FOR it at P+ 0.657. Neither is
strong enough to act on, and the era split does not agree on even the sign.
Recorded `unresolved_below_power` at every cut. **Caveat**: this is thin by
construction (four programs, one 3.5-month window) -- the widest interval in
this whole wave (era 2021-2025's week-blocked span is 0.95 points) reflects
that, not a defect in the harness (the positive control on the identical
population detects a 48-point effect cleanly).

---

## 4. LEAD-44: CFB sandwich/lookahead fade -- data gap, no scoring

**Mechanism (not tested).** AP-ranked favorites hosting an unranked team the
week before a rivalry or top-10 opponent should play flat -- the classic
lookahead trap. Predeclared direction (per the ROADMAP row, never
scored): FADE the sandwich favorite.

**The gap.** Identifying an "AP-ranked favorite" and its "next-week
rivalry/top-10 opponent" requires a weekly AP Top-25 poll table. **Measured**
this session: `data/cfb` carries no rankings/poll directory (its thirteen
directories are `draft_picks`, `espn_betting`, `lines`, `participants`,
`pbp`, `portal`, `recruiting_players`, `recruiting_teams`,
`returning_production`, `rosters`, `schedules`, `team_info`, `usage` -- no
polls), `data/raw` carries nothing CFB-poll-shaped either, and
`docs/cfb_data.md`'s provider table lists only odds/line sources across every
season regime. A case-insensitive grep for `rival|poll|ranking` across
`src/`, `scripts/`, and `docs/` returns nothing that names an AP or coaches'
poll ingestion path.

**Disposition.** Per the fleet brief's no-fetch rule, this lead is **skipped,
not screened**. `scripts/cfb_lead_screens_wave1.py --lead sandwich` (any
`--mode`) prints this gap and exits without reading the feature table. No
registry entry is recorded (there is no measurement to record). The ROADMAP
LEAD-44 row carries a dated note pointing here; it is **not** marked done,
because the row's own definition of done is a screen, which this data gap
blocks.

---

## 5. Registry summary

Seven entries recorded under `--league cfb`, `--effect-units accuracy_points`,
week-blocked interval and `probability_positive`, season range 2012-2025
(era rows use their own restricted range). Every `--notes` field discloses:
(i) close-graded CFB, no verified opener; (ii) each lead's own family is
never pooled with another lead, with `cfb_option_prep_screen` or
`cfb_rest_bye_replication`; (iii) flagged-game counts; (iv) per-era
magnitudes; (v) the comparison to any overlapping existing construct
(LEAD-48 vs `bye_edge_home`, spelled out above).

| entry | family | category | classification | closing ground |
|---|---|---|---|---|
| `cfb_post_bye_home_on_benchmark` | `cfb_post_bye_home_on_benchmark` | schedule | unresolved_below_power | -- |
| `cfb_post_bye_home_on_benchmark_era_2012_2019` | `cfb_post_bye_home_on_benchmark` | schedule | unresolved_below_power | -- |
| `cfb_post_bye_home_on_benchmark_era_2021_2025` | `cfb_post_bye_home_on_benchmark` | schedule | **refuted_mechanism** | **wrong_sign_resolved** |
| `cfb_rivalry_home_dog_on_benchmark` | `cfb_rivalry_home_dog_on_benchmark` | onfield | unresolved_below_power | -- |
| `cfb_altitude_cold_home_on_benchmark` | `cfb_altitude_cold_home_on_benchmark` | environment | unresolved_below_power | -- |
| `cfb_altitude_cold_home_on_benchmark_era_2012_2019` | `cfb_altitude_cold_home_on_benchmark` | environment | unresolved_below_power | -- |
| `cfb_altitude_cold_home_on_benchmark_era_2021_2025` | `cfb_altitude_cold_home_on_benchmark` | environment | unresolved_below_power | -- |

**For the NFL card: nothing changes, and nothing was ever going to.** Every
result here is a CFB replication/screen on a close-graded, sanctioned free
ground; none by itself changes an NFL card. The one resolved finding
(LEAD-48's era 2021-2025 slice) is a narrow, mechanically justified
conclusion about that specific signed-column-and-threshold in that specific
recent window -- it says this exact operationalization of "back the CFB
post-bye team" should not be carried forward for recent seasons; it does not
settle the broader prep-asymmetry mechanism question, which era 2012-2019's
own P+ 0.915 lean keeps open.

## Files added

- `docs/cfb_lead_screens_wave1.md` (this document).
- `scripts/cfb_lead_screens_wave1.py` -- `--lead {post_bye,rivalry_home_dog,
  altitude_cold,sandwich}` x `--mode {coverage,null,positive-control,screen}`.
- `tests/test_cfb_lead_screens_wave1.py` -- construction, sign-convention,
  underdog-restriction and leakage tests for all three scored columns.
- `artifacts/cfb_lead_screens_wave1/<lead>/<UTC stamp>/results.json` -- one
  positive-control and one screen artifact per scored lead (six total).
