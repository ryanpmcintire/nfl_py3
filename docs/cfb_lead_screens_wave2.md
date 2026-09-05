# CFB free-screen wave 2: LEAD-47, LEAD-49

Written **before any ATS outcome, cover rate, accuracy delta or sign is
computed for either column below**. Both leads mirror the structure of
`docs/cfb_lead_screens_wave1.md` (LEAD-48/50/46/44): each is its own
weak-signal family, its own signed/flag column on top of the frozen XLG-03
benchmark arm, and neither spends an NFL evaluation window or a rotation
window (CFB is this project's sanctioned free replication ground). The
"Results" subsection under each lead was appended **after** `--mode screen`
ran; nothing above that line was touched afterward.

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
validator. Verdicts flow only through `nfl-ats weak-signals record`, never
through prose. A promotion threshold governs only what the docs may CLAIM; it
never governs which card is PLAYED, which is expected value (`probability_positive`
above 0.5 favours the candidate).

## Shared comparator, metric, and controls (both leads)

| element | value |
|---|---|
| Baseline arm | `nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS` (the frozen XLG-03 contract), `nfl_ats.cfb_benchmark.fit_cfb_residual_model`, ridge alpha 10.0 |
| Candidate arm | the same columns **plus exactly one** of the two columns below |
| Walk-forward | every scored week's models train on completed table games kicking off strictly before that week's earliest kickoff, 500-game floor (`CFB_BENCHMARK_MIN_TRAIN_GAMES`) |
| Metric | paired candidate-minus-baseline forced-pick accuracy delta, `accuracy_points`, picks at `home_cover_probability >= 0.5`, graded with `nfl_ats.clv.pick_correct` (pushes NaN, excluded) |
| Uncertainty | week-blocked bootstrap primary (1,000 samples, seed 20260905), season-blocked secondary, never averaged (within-week correlation is ZERO by owner mandate) |
| Positive (leak) control | the candidate column REPLACED by the realized `ats_margin` -- must read hugely positive (P+ 1.000); if it does not, the harness is blind |
| Null | 200 within-week permutations of the settling margin; the observed delta is reported at its null percentile |
| Grade | CFB is close-graded (`spread_line`, the median-across-books close proxy; no verified CFB opener exists) -- settles no NFL play/no-play or promotion decision by itself |

Every candidate column below is a deterministic function of pregame-knowable
identity facts (play-by-play passer identity, roster class fields, portal
transfer records, and schedule/calendar facts) -- **never** a score, a line,
or any outcome column. That is what the leakage tests in
`tests/test_cfb_lead_screens_wave2.py` pin: both columns are bit-identical
after `result`, `ats_margin`, `home_cover`, `home_points`, and `away_points`
are permuted.

## Shared starter-identification method (reused, not rebuilt)

Both leads need "who started at QB". No local depth-chart/injury-report
source exists for CFB (**read**, `src/nfl_ats/cfb_qb_dependence.py`
module docstring: "CFB has no pregame injury/availability signal of any
kind"). This module's own QB-identity work is reused verbatim rather than
re-derived: `nfl_ats.cfb_qb_dependence.build_cfb_qb_game_metrics` credits
dropbacks to `passer_player_id` on competitive (5%-95% win-probability) pass
plays, the exact same construction already used and tested for the
QB-dependence interaction feature. This script's
`leading_passer_per_game_team` groups that output to one row per
`(game_id, team_id)` -- the passer with the most dropbacks that game, the
**post-hoc** in-game starter identification the task calls for ("the QB with
the most dropbacks/pass attempts in that game per the pbp/usage tables"),
ties broken deterministically (highest dropbacks, then lowest
`passer_player_id`).

A second, **strictly pregame-safe** variant chains that same table
chronologically per team: `attach_previous_game_starter` walks every
identified `(game_id, team_id)` row in kickoff order and, for each row,
records whichever passer led that TEAM's immediately preceding
identified game -- built and looked up **before** that row's own game is
processed, so it never reads anything about the game it is predicting.
This is the CFB analogue of `cfb_qb_dependence.py`'s own `latest_passer`
lookback loop, applied to raw starter identity instead of an EWM state.

## 1. LEAD-49: CFB portal-QB early starts

### Predeclaration

**Mechanism and predeclared direction.** A transfer-portal quarterback
brings talent but not timing: install time with a new playbook, new
receivers, and a new coordinator's verbal cadence takes real games to
resolve, so the market should overprice a portal QB's arrival before that
integration cost is paid down. **Predeclared direction: FADE the team
starting an early-tenure (first three games with the new team) transfer-portal
QB.**

**Population.** CFBD's `/player/portal` endpoint is populated locally from
season **2021** on (`nfl_ats.cfb.CFB_SOURCES["portal"].first_season = 2021`,
**read**, `src/nfl_ats/cfb.py`); local partitions confirmed present for
seasons 2021-2026 under `data/cfb/portal/raw/20260816T164011Z/` (**measured**,
directory listing). **The scored population is therefore the 2021-2025 XLG-03
clean-core seasons only** -- there is no way to evaluate this construct
before the portal era, and no attempt is made to backfill or proxy it. No
era split is reported (the population already spans a single regime); the
`--mode screen` table below is the whole story for this lead.

**Portal-to-roster identity: no athlete ids, so a disclosed name+school+season
match is built, exactly as CFBD's own identity contract requires.**
**Measured**, `nfl_ats.cfb.CFBD_PORTAL_IDENTITY_CONTRACT`: "CFBD
`/player/portal` publishes no athlete ids: rows are name-keyed only. Linking
portal entries into any id space requires a reviewed name+school+date match
and must never silently join on names." This script performs exactly that
reviewed match, disclosed in full:

1. Filter `portal.parquet` to `position == "QB"` and a non-null
   `destination`.
2. Resolve `destination` (a bare school name, e.g. `"UCF"`) to the same
   ESPN `team_id` space the benchmark table uses, via a `team_name -> team_id`
   lookup built from the full local schedules snapshot's own
   `home_team`/`away_team` columns -- the identical naming convention
   `cfb_game_features.parquet`'s `home_team`/`away_team` already carry
   (**measured**: both use bare school names like `"Ohio State"`, `"UCF"`).
   Rows whose destination does not resolve to a known FBS team id are
   excluded and counted, never guessed.
3. Within the resolved destination team's roster for the transfer's
   `season`, match on case/whitespace-normalized `(first_name, last_name)`
   against the portal row's `(firstName, lastName)`. A `(team_id, season,
   name)` key that resolves to more than one distinct `athlete_id` is an
   ambiguous match, counted and excluded rather than guessed.
4. Any portal QB row surviving both joins contributes its resolved
   `athlete_id` to that `(team_id, season)`'s set of "portal-transfer QBs".

**"Starting" and "first three games": the pregame-safe variant is the one
that is SCORED.** Per this lane's task instruction: the **post-hoc**
same-game leading passer (most dropbacks that game) is the natural
identification of "who started", but using it directly as a same-game
predictor risks (rarely) misreading a QB competition or an in-game injury
substitution as the wrong identity for THIS game. The column that is
actually scored therefore uses the **strictly pregame-safe** variant: a
team's presumed starter for game *g* is whoever led that team's most
recently completed EARLIER game (`attach_previous_game_starter`, above).
`nfl-ats` records the **agreement rate** between the two variants (pregame
proxy vs. the same game's own post-hoc identity) as a disclosed diagnostic,
never as part of the flag itself.

**A structural consequence of this choice, disclosed before any number is
seen:** because a portal transfer, by construction, cannot have started an
EARLIER game for their new team (they were not on that roster yet), the
pregame-safe variant can **never** flag a team's first game of the season
for its incoming portal QB -- only games 2 and 3 of the "first three", and
only if the QB in fact started game 1 (post-hoc) and the pregame proxy
correctly carries that forward. This is a real, disclosed asymmetry in
statistical power, not a defect to paper over: the construct as scored is
closer to "started an early-tenure portal QB who also started the
previous game" than to the full "any of the first three games" population.
The coverage table below reports both counts.

**"First three games with the new team".** A team's own season game index
(1, 2, 3, ...) is built from the full local schedules snapshot, restricted
to regular-season, completed games, ranked chronologically by kickoff within
`(team_id, season)` -- the same `season_type == "regular" & completed`
restriction `docs/cfb_lead_screens_wave1.md`'s rivalry lead uses. "Early
tenure" is `season_game_index <= 3`.

**Encoding (frozen).** `cfb_lead49_portal_qb_early_signed`: **+1** when the
AWAY team's pregame-safe presumed starter is one of that team's portal-QB
arrivals for the current season AND the game is within that team's first
three games of the season (favours home, the FADE direction); **-1** for the
mirror (home team qualifies); **0** otherwise, including when both sides
qualify simultaneously (rare; the same "0 for both" convention
`docs/cfb_lead_screens_wave1.md`'s LEAD-48 post-bye column uses) or when
either side's presumed starter cannot be identified at all. Always defined
(0, never NaN) -- a missing identification is treated as "not known to be an
early-tenure portal QB", the same convention `cfb_lead_screens_wave1.md`'s
identity flags (rivalry, altitude) use for their own unknown cases.

**Comparator, metric, controls.** As in the shared table above, scored on
seasons 2021-2025 only (`LEAD49_SEASONS`).

**Reliability.** Every input is a deterministic identity/schedule fact
(passer identity from play-by-play, portal transfer records, schedule
kickoff order) with zero measurement error in the ordinary sense; the one
genuine source of noise is the name-matching join itself, which is measured
and disclosed via its match rate rather than a split-half correlation.
`no_split_half_reliability` is therefore inadmissible here for the same
reason it is inadmissible for `cfb_lead_screens_wave1.md`'s option-team and
rivalry-pair identities: there is no noisy trait for a sample size to fail
to rescue, only a measured (and disclosed) join-coverage rate.

**Decision rule and recording.** Expected value, never a threshold. One
pooled entry, `cfb_portal_qb_early_on_benchmark`. Never pooled with the
high-school recruiting-rating entries (`XLG-06`) -- this construct never
reads a recruiting rating or star count, only a portal transfer record and
play-by-play passer identity.

### Results (2026-09-05)

Coverage (measured, part of every mode's printed `diagnostics`): the
loaded feature table carries 12,500 games total, 3,653 of which fall in
`LEAD49_SEASONS` (2021-2025). Portal-QB identity match: **807** `position ==
"QB"` portal rows, **39** unresolved destinations (school name not found in
the schedule's `home_team`/`away_team` space), **35** ambiguous roster-name
keys excluded (never guessed), **658 matched (81.5%)**, covering **490**
distinct `(team, season)` pairs. Starter-agreement diagnostic (general QB
continuity -- the previous known starter vs. the SAME game's own post-hoc
starter, not portal-specific): **81.7%** on 6,772 comparable games. Flagged:
**106** home-side, **95** away-side, **19** games where both sides
independently qualify and cancel to 0 per the signed convention.

Positive control (`--mode positive-control`, artifact
`artifacts/cfb_lead_screens_wave2/portal_qb_early/20260905T174714Z/results.json`):
**pooled +48.493 accuracy points, week-blocked 95% [+46.888, +50.240], P+
1.000** on 3,584 games / 77 weeks -- identical in magnitude to wave 1's own
leak controls, as expected. The harness is not blind.

Null (`--mode null`, 200 within-week permutations of the settling margin):
mean **+0.107 pts**, observed delta at the **4th percentile**.

Screen (`--mode screen`, artifact
`artifacts/cfb_lead_screens_wave2/portal_qb_early/20260905T174737Z/results.json`):

| cut | delta (pts) | week 95% CI | week P+ | season 95% CI | season P+ | n games / weeks / flagged |
|---|---|---|---|---|---|---|
| pooled (2021-2025) | **-0.251** | [-0.687, +0.188] | 0.098 | [-0.643, +0.164] | 0.110 | 3,584 / 77 / 163 |

**What this implies for the decision, before the caveat.** On the project's
EV rule, this leans STRONGLY AGAINST the predicted FADE direction (P+
0.098): if forced to choose today, holding the market side (NOT fading the
early-tenure portal QB team) is favoured at roughly 9-to-1 odds in this
window. That is a real, disclosed lean, not a null result -- but per the
binding taxonomy, **the week-blocked interval's upper bound (+0.188 pts)
sits above zero**, so this is not `wrong_sign_resolved` (the WHOLE interval
must sit below zero) and is not treated as a rejection of the mechanism.
Recorded `unresolved_below_power`, reporting `probability_positive` rather
than "contains zero", exactly as the taxonomy requires. **Caveat:** this is
the strongest single-cut lean measured in either wave-2 lead, and the
closest to resolving of anything in this document; a future re-look with
more portal seasons (2026 on) is the natural next test of whether it
crosses.

**Structural disclosure, worth restating with the numbers in hand:** because
the pregame-safe flag can never fire on a team's first game of a portal QB's
tenure (see predeclaration above), the 163 flagged games are concentrated in
games 2-3 of tenure, not the full "first three games" population -- the
measured effect is closer to "fading a team riding a portal QB into his
2nd/3rd start" than the literal full-window construct named in the ROADMAP
row. This is disclosed as a scope narrowing, not hidden in the number.

Registry: `cfb_portal_qb_early_on_benchmark` (`unresolved_below_power`,
registry total 770 after this session's four records).

---

## 2. LEAD-47: CFB true-freshman road QB fade

### Predeclaration

**Mechanism and predeclared direction.** A true freshman thrown into a
hostile road environment for the first time faces a double inexperience
tax: unfamiliarity with the speed/complexity of the sport itself, compounded
by unfamiliarity with playing in front of a hostile crowd (snap-count
communication, silent-count discipline, tempo under noise). Both costs are
steepest for a player in his very first year of exposure. **Predeclared
direction: FADE true-freshman road starters.** Distinct from XLG-06's
recruiting-RATING question (**read**, `src/nfl_ats/xlg06_prior.py`
docstring): XLG-06 asks whether a numeric recruiting rating predicts NFL
rookie production; this lead never reads a recruiting rating or star count
at all, only a roster-derived class/experience identity crossed with home/
away, matching the ROADMAP row's own framing ("class x venue, not rating").

**Population.** The XLG-03 clean core, **restricted to seasons 2014-2019 and
2021-2025** (11 of the 13 clean-core seasons) -- 2012-2013 are excluded, and
the reason is a **measured data-quality finding, disclosed in full below**,
not a convenience cut.

**Class identification: the obvious local field is broken; disclosed and
worked around.** `nfl_ats.cfb.CFB_ROSTER_SNAPSHOT_COLUMNS` carries
`experience_years` (1.0-4.0, mapping to Freshman/Sophomore/Junior/Senior on
the raw ESPN source's `experience_display_value`/`experience_abbreviation`,
which are NOT in the canonicalized snapshot contract and were read from the
raw `source/game_rosters_<season>.parquet` file to confirm the mapping).
**Measured this session**: `experience_years` for a given `athlete_id` is
**frozen** across seasons in the local archive -- e.g. athlete_id 487698
("Darius Smith") reads `experience_years == 3.0` in game rows from 2018,
2019, AND 2021 alike; athlete_id 3914348 ("Kyle Horn") reads `4.0` across
every 2018 and 2019 game row. This is a scrape-time-snapshot artifact of the
same character as `nfl_ats.cfb.CFB_ROSTER_QUARANTINED_COLUMNS`'s already-
documented static-attribute columns (`active`, `starter`, `valid`, etc.),
though `experience_years` itself is not currently in that quarantine list.
**`experience_years` cannot be trusted to tell a true-freshman season from
any other season for the same player and is not used for classification.**

**The proxy used instead, and its own measured regime boundary.** A player's
**first season appearing on ANY local roster snapshot** (the full 2004-2025
archive, any team, any week) is used as the true-freshman proxy: `is_true_
freshman(athlete_id, season) := first_local_roster_season(athlete_id) ==
season`. This is not merely a fallback -- it is a *better* instrument for
"true freshman, not redshirt" than the broken `experience_years` field would
have been even if it worked, because `nfl_ats.cfb.CFB_ROSTER_AVAILABILITY_
CONTRACT` establishes that a roster listing is a **season roster listing,
not a play-credit** ("listed does not mean played"): a redshirt freshman is
listed on his team's roster during his redshirt year even though he never
plays, so his "first local roster season" correctly lands on his redshirt
year, one season BEFORE his true-freshman-of-record season -- which the
proxy then correctly excludes (his "first roster season" is not the current
one). **With one measured exception**: `nfl_ats.cfb.CFB_ROSTER_SOURCE_
REGIMES` documents that 2004-2013 files contain "only stat-credited players
(~32 per team-game)", confirmed this session (**measured**, mean players per
team-game: 2012 = 31.76, 2013 = 32.37, 2014 = 105.51, 2015 = 112.14). In the
stat-credited-only regime, a redshirting player who does not play generates
**no roster row at all** that year, so his "first local roster season" lands
on his first PLAYING season regardless of whether that is a true or a
redshirt freshman year -- the proxy cannot distinguish the two before 2014.
**This is exactly why the scored population excludes 2012-2013**: the
"not redshirt if distinguishable" instruction is honored by restricting the
claim to the seasons where it genuinely is distinguishable (2014 on), rather
than silently degrading the construct for two seasons the task never asked
to be included at any cost.

**Starter identification: post-hoc, disclosed, reused from LEAD-49's shared
method.** The AWAY team's starter for a game is `leading_passer_per_game_
team`'s post-hoc same-game identification (most dropbacks that game, via
`cfb_qb_dependence.build_cfb_qb_game_metrics`). Unlike LEAD-49, this lead
does **not** additionally require the strictly-pregame "previous game's
starter" proxy -- the task's pregame-variant instruction is scoped
specifically to LEAD-49's early-tenure/first-three-games construct, where
precise game-to-game alignment with a specific tenure count matters; here,
a true freshman's CLASS is constant for his entire season regardless of
which specific game is checked, so the risk the pregame variant exists to
manage (misreading a specific game's starter identity) only ever
misclassifies a single game, not the whole-season construct, and the
post-hoc identity is what the task's general starter-identification
guidance ("the starter is the QB with the most dropbacks/pass attempts in
that game ... a post-hoc record, disclose") describes directly. This is a
disclosed design choice, not an oversight; the leakage test still confirms
the column is invariant to `result`/`ats_margin`/`home_cover`/points
permutation, matching this repo's own operational definition of pregame-
safety for identity-style features (**read**, `docs/cfb_lead_screens_wave1.md`,
"an institutional/scheduling fact known in hindsight, not a game outcome").

**Encoding (frozen).** `cfb_lead47_true_freshman_road_qb_flag`: **1** iff
the AWAY team's post-hoc starter is a true freshman by the proxy above
(favours home, the FADE direction); **0** otherwise, including when the
AWAY starter cannot be identified at all (no qualifying pass play that game)
or when it is the HOME starter who is a true freshman (a different
construct, per this lane's task -- its own count is reported as a
diagnostic, `_lead47_home_true_freshman_starter`, and is never folded into
this column or pooled with it). Always defined (0, never NaN).

**Comparator, metric, controls, era split.** As in the shared table above,
scored on `LEAD47_SEASONS = (2014..2019, 2021..2025)`; era split
**2014-2019** vs **2021-2025** (shifted two seasons later than
`cfb_lead_screens_wave1.md`'s 2012-2019 boundary, for the reason above).

**Reliability.** Class identity (first local roster appearance) and starter
identity (leading passer that game) are both deterministic facts read off
administrative/play-by-play records, not noisy latent traits;
`no_split_half_reliability` is inadmissible for the same reason it is
inadmissible throughout `cfb_lead_screens_wave1.md`. The one measured,
disclosed source of imprecision is the proxy's own regime boundary above,
handled by population restriction rather than a reliability estimate.

**Decision rule and recording.** Expected value, never a threshold. One
pooled entry, `cfb_true_freshman_road_qb_on_benchmark`; era slices recorded
separately under the same family if they diverge materially (judged after
the look, same convention as wave 1). Never pooled with XLG-06's
recruiting-rating entries (no recruiting rating is read anywhere in this
lead) or with any `cfb_lead_screens_wave1.md` family.

### Results (2026-09-05)

Coverage (measured): of 7,799 games in the restricted population
(2014-2019 + 2021-2025), the away-team post-hoc starter is identified in
**6,818 (87.4%)** and the home-team starter in **6,864 (88.0%)** -- the
remaining ~12-13% have no passer with >=5 competitive-play dropbacks that
game (option-heavy teams, blowouts, weather-shortened games), consistent
with the same floor `cfb_qb_dependence.py`'s QB-dependence feature already
uses. **604** away starters and **657** home starters resolve to a true
freshman by the first-local-roster-appearance proxy; home true-freshman
starters are a diagnostic only (`_lead47_home_true_freshman_starter`), never
folded into the scored column.

**A cross-source identity bug found and fixed before any number below was
trusted.** `cfb_qb_dependence.build_cfb_qb_game_metrics` casts pbp's
`passer_player_id` straight from float64 to `str` (e.g. `"4775196.0"`) --
internally consistent for that module's own same-source joins, but
**measured** this session to produce **zero** overlap against roster
`athlete_id` values (clean int64, e.g. `"4775196"`) on a 2024 sample check
(0 of 294 distinct passer ids matched 27,471 roster ids). Fixed by
re-casting through `int64` inside `leading_passer_per_game_team` before any
cross-source join -- disclosed here because it is exactly the kind of
silent-zero failure this project's "measured before quoting" rule exists to
catch; the first (unfixed) run of this lead's coverage diagnostic read
**zero** true freshmen identified at all, which is what surfaced the bug.

Positive control (`--mode positive-control`, artifact
`artifacts/cfb_lead_screens_wave2/true_freshman_road_qb/20260905T174527Z/results.json`):
**pooled +48.460 accuracy points, week-blocked 95% [+47.327, +49.539], P+
1.000** on 7,660 games / 168 weeks. Era controls: 2014-2019 **+48.430** (P+
1.000, [+46.970, +49.976]); 2021-2025 **+48.493** (P+ 1.000, [+46.888,
+50.240]). The harness is not blind in either era.

Null (`--mode null`, 200 within-week permutations): mean **-0.003 pts**,
observed delta at the **69th percentile** -- clean.

Screen (`--mode screen`, artifact
`artifacts/cfb_lead_screens_wave2/true_freshman_road_qb/20260905T174603Z/results.json`):

| cut | delta (pts) | week 95% CI | week P+ | season 95% CI | season P+ | n games / weeks / flagged |
|---|---|---|---|---|---|---|
| pooled | **+0.091** | [-0.251, +0.427] | 0.682 | [-0.403, +0.482] | 0.645 | 7,660 / 168 / 604 |
| era 2014-2019 | **-0.049** | [-0.634, +0.488] | 0.428 | [-0.912, +0.640] | 0.476 | 4,076 / 91 / 405 |
| era 2021-2025 | **+0.251** | [-0.115, +0.593] | 0.889 | [-0.057, +0.569] | 0.952 | 3,584 / 77 / 199 |

Every interval crosses zero; none resolves. The pooled lean (P+ 0.682) and
era 2021-2025's stronger lean (P+ 0.889 week / 0.952 season, the closest cut
in this whole document to resolving without doing so) both point the
predicted FADE direction; era 2014-2019 leans mildly the other way (P+
0.428) on a materially different point estimate (-0.049 vs +0.251 pts, a
0.30-point swing) -- recorded as separate era rows per "era magnitude, not
presence," the same judgement `docs/cfb_lead_screens_wave1.md`'s LEAD-48
divergence used.

**What this implies for the decision, before the caveat.** On EV grounds,
2021-2025 alone favours fading the road true-freshman starter at roughly
8-to-1 odds (P+ 0.889-0.952) -- a real, actionable lean for that specific
recent window, though not resolved. Pooled across the whole restricted
population the lean is milder (P+ 0.682, about 2-to-1). 2014-2019 alone is
close to a coin flip leaning the other way. **Caveat:** this is one specific
operationalization (first-local-roster-appearance proxy, post-hoc same-game
starter identity) of "true freshman"; it does not resolve the broader
experience-times-hostile-environment mechanism question, and the excluded
2012-2013 seasons (measured to be undistinguishable from redshirt freshmen)
are not part of any claim here.

Registry: `cfb_true_freshman_road_qb_on_benchmark` (pooled,
`unresolved_below_power`) + `..._era_2014_2019` + `..._era_2021_2025` (both
`unresolved_below_power`), registry total 769 after these three records
(770 after LEAD-49's fourth record below).

---

## 3. Registry summary

Four entries recorded under `--league cfb`, `--effect-units accuracy_points`,
category `onfield`, week-blocked interval and `probability_positive` primary
(season-blocked reported in notes, never averaged with week-blocked).

| entry | family | classification | closing ground |
|---|---|---|---|
| `cfb_portal_qb_early_on_benchmark` | `cfb_portal_qb_early_on_benchmark` | unresolved_below_power | -- |
| `cfb_true_freshman_road_qb_on_benchmark` | `cfb_true_freshman_road_qb_on_benchmark` | unresolved_below_power | -- |
| `cfb_true_freshman_road_qb_on_benchmark_era_2014_2019` | `cfb_true_freshman_road_qb_on_benchmark` | unresolved_below_power | -- |
| `cfb_true_freshman_road_qb_on_benchmark_era_2021_2025` | `cfb_true_freshman_road_qb_on_benchmark` | unresolved_below_power | -- |

Every `--notes` field discloses: (i) close-graded CFB, no verified opener;
(ii) each lead's own family is never pooled with the other or with any
`cfb_lead_screens_wave1.md`/XLG-06 family; (iii) flagged-game counts and
identification/match coverage rates; (iv) per-era magnitudes where
applicable; (v) the two measured data-quality findings behind each
construct (LEAD-47's frozen `experience_years` field and the pre-2014
stat-credited-only roster regime; LEAD-49's cross-source
`passer_player_id` float-string bug and the structural game-1 blind spot of
its pregame-safe proxy).

**For the NFL card: nothing changes, and nothing was ever going to.** Both
leads are CFB replications/screens on a close-graded, sanctioned free
ground; neither by itself changes an NFL card. LEAD-49's -0.251-point,
P+ 0.098 lean is the strongest single-cut result in this document and the
one most worth a future re-look as more portal-era seasons accumulate; it
does not resolve today.

## Files added

- `docs/cfb_lead_screens_wave2.md` (this document).
- `scripts/cfb_lead_screens_wave2.py` -- `--lead {portal_qb_early,
  true_freshman_road_qb}` x `--mode {null,positive-control,screen}`.
- `tests/test_cfb_lead_screens_wave2.py` -- construction, tie-break,
  point-in-time lookback, portal-identity-match, season-game-index,
  sign-convention, restriction, and leakage tests (13 tests, all passing).
- `artifacts/cfb_lead_screens_wave2/<lead>/<UTC stamp>/results.json` -- one
  positive-control and one screen artifact per lead (four total; an early
  LEAD-47 null-mode run before the `passer_player_id` fix is not among
  these four and was not used for any recorded number).
