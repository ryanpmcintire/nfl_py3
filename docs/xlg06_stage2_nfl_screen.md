# XLG-06 Stage 2: recruiting rating vs NFL rookie production (predeclaration)

**Status:** predeclared 2026-09-03, BEFORE any outcome number for this
comparison was computed. Sections 0–10 are frozen; section 11 (Results) is
appended after the look and nothing above it is edited afterwards.

**Owning work package:** XLG-06 Stage 2. Files: this document,
`scripts/xlg06_stage2_nfl_screen.py`,
`tests/test_xlg06_stage2_nfl_screen.py`,
`artifacts/xlg06_stage2_nfl/`.

**Parents:** `docs/xlg06_rookie_prior_screen.md` (Stage 1, pure CFB) and
`docs/xlg06_stage2_crosswalk.md` (identity audit, 2026-09-03). This screen
reuses the crosswalk's linked population and Stage 1's correlation machinery
(shape, not code) and modifies neither.

---

## 0. Binding closing-grounds taxonomy (AGENTS.md, pasted verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

Decisions are expected value. `probability_positive` above 0.5 favours the
candidate; predeclared thresholds govern only what a document may CLAIM, never
which card is played. This experiment measures a correlation gate, not an ATS
pick: it changes no card, no model, and no rotation window either way
(section 10).

---

## 1. What Stage 1 left open, and what this look asks

Stage 1 (pure CFB) found recruiting rating does not predict a true-freshman
QB's realised usage (13-cohort r=-0.0018, P+ 0.484) while the RB secondary read
leaned positive (r=+0.0644, CI entirely positive, P+ 0.9971; dedicated WP46
confirmation in the same document). Neither result touches the NFL. The open
Stage-2 question is whether the recruiting rating predicts **NFL production**
for drafted skill players — the minimum gate before any recruiting prior can
be proposed as an NFL feature. This screen is that gate, and only that gate:
a positive result proposes a prior family, it does not wire one.

## 2. Population (frozen)

Crosswalk-linked recruits (`gsis_id` non-null via
`build_recruit_to_nfl_crosswalk`, no name join) satisfying ALL of:

- recruiting `position` exactly in `{WR, RB, TE}` (excludes ATH/PRO/DUAL/APB
  hybrids whose NFL side of ball is not predeclared);
- usable numeric recruiting `rating` (null ratings dropped and counted);
- at least one `REG` row in `player_stats.parquet` (snapshot
  `data/players/values/raw/20260817T184911Z`);
- rookie season — the minimum `season` with any stats row — is `<= 2024`, so
  every included rookie year is complete (2025 debuts are in-progress and
  excluded by eligibility, not by outcome);
- at least one `REG` row inside that rookie season (postseason-only debuts
  dropped and counted).

Design-time counts (eligibility only, no outcome viewed): 405 linked skill
rows, 8 null ratings, 331 with any stats, 274 with a completed rookie season.
The realised n is reported from the run, not from this paragraph.

## 3. Predictor and outcome (frozen)

- **Predictor:** recruiting `rating` as stored (0–1 composite scale).
- **Outcome:** rookie-season total `rushing_epa + receiving_epa` summed over
  that player's `REG` rows only. Postseason rows are excluded because extra
  games accrue to good-team rookies and would confound a total. No snap
  normalization: per-snap rates would divide by a second measured quantity
  with its own key risk; the total is the declared construct, disclosed as
  conflating opportunity with efficiency.

## 4. Statistic and uncertainty (frozen)

- Primary: Pearson r, with a recruiting-cohort-blocked percentile bootstrap
  (whole `year` cohorts resampled), 10,000 samples, seed 20260903.
- Secondary: Spearman rho with the same bootstrap (same seed).
- `probability_positive` is the blocked-resample fraction above zero.

## 5. Reliability (frozen)

Split-half reliability of the outcome construct: within each included
player's rookie REG weeks, odd- vs even-indexed weeks' EPA sums, Pearson r
across players, Spearman-Brown corrected. Eligibility for this calc only:
players with `>= 4` rookie REG weeks with any row. Reported n alongside.
A reliability indistinguishable from zero rules out a refuted-mechanism
reading in that direction (no sample size rescues pure noise); a healthy
reliability with a near-zero correlation is the classic
`unresolved_below_power` shape, not a closure.

## 6. Positive control (frozen, diagnostic only)

200 within-cohort shuffles of the predictor; report the null center and the
observed value's percentile. This checks the machinery runs (a shuffled null
centres near zero), not a candidate-sized sensitivity — it is NOT
`bounded_by_control`-eligible.

## 7. Leakage and chronology (frozen, fail-closed)

The predictor (recruiting rating, public before the draft) strictly predates
the outcome (rookie NFL season) by construction. The script asserts
`rookie_first_season > recruiting_year` for every included row and fails
closed otherwise. Design-time check: zero violations in the eligible frame.
A dedicated leakage test pins the assertion with a synthetic violation.

## 8. Test contract (release-blocking)

`tests/test_xlg06_stage2_nfl_screen.py` covers, without network access:

- eligibility filtering (null rating, unlinked, 2025 debut, postseason-only
  debut, chronology violation → excluded/counted, never silently kept);
- REG-only outcome construction on synthetic weekly rows;
- bootstrap determinism (same seed → identical r and interval);
- the chronology fail-closed on a synthetic violation;
- reliability helper on a synthetic two-half frame.

## 9. Decision rule (frozen)

Record one `unresolved_below_power` entry (`xlg06_rookie_prior_stage2_nfl`,
units `correlation`) with the Pearson r, cohort-blocked interval, P+, n,
reliability, and null percentile — unless the interval sits wholly on one
side of zero (positive: evidence toward a prior family, still not a wiring;
negative: `wrong_sign_resolved` admissible ONLY if the whole interval is
below zero) or reliability is zero (mechanism refuted as noise). No card,
model, profile, rotation window, or ATS comparison follows from this screen
under any outcome.

## 10. What this screen may therefore claim

At most: whether pre-draft recruiting ratings carry linear signal for
same-player NFL rookie production among drafted skill players, on a
disclosed n with a disclosed reliability. It may not claim a feature, a
prior weight, a decay schedule, or any ATS consequence — those need their
own predeclarations on windows this screen does not spend.

## 11. Results (added after the look, 2026-09-03)

Measured by `scripts/xlg06_stage2_nfl_screen.py` in one run (artifact
`artifacts/xlg06_stage2_nfl/20260903T170528Z/results.json`,
per-player audit `rookie_epa.parquet` in the same directory):

- Realised n = **272** drafted skill players (recruiting WR/RB/TE, usable
  rating, GSIS-linked, completed rookie season 2017–2024). Excluded and
  counted: 53 incomplete-2025 debuts, 72 linked with no production rows,
  8 null ratings, 0 postseason-only debuts, 0 chronology violations.
- **Pearson r = +0.1004**, recruiting-cohort-blocked 95% [-0.0322, +0.2292],
  `probability_positive` **0.9275** (10,000 samples, seed 20260903, 8 cohort
  blocks). Spearman rho = +0.0655, [-0.0448, +0.2224], P+ 0.8111.
- Outcome reliability is healthy, not noise: odd/even-week split-half
  r = 0.4775, Spearman-Brown **0.6464** (n = 238 players with 4+ rookie
  REG weeks).
- Shuffle null centres at +0.0020 with the observed value at the 96th
  percentile (200 within-cohort shuffles, seed 20260904) — machinery sane,
  diagnostic only.

What this implies for the decision, before what is wrong with it: the point
estimate leans positive at P+ 0.93 with a reliable outcome construct, so the
recruiting-prior family earns its next predeclared step (a prior-weight
proposal needs its own design); it does not earn a feature, a weight, or any
ATS consequence today. The interval crosses zero, reliability is well above
zero, and the null control is not candidate-sized — so under the frozen §9
rule this records `unresolved_below_power`, not a closure and not a finding.
The magnitude (≈0.10 correlation) is small, and the total (not per-snap)
construct conflates opportunity with efficiency by declared design.

## Registry record

Recorded via `nfl-ats weak-signals record` (command and output pasted in the
session report): `xlg06_rookie_prior_stage2_nfl`, units `correlation`,
`unresolved_below_power`, `--league nfl`, seasons 2017–2024.
