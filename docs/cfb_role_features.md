# CFB role-continuity feature family — predeclaration

Predeclared: 2026-08-17 (US), before any run in which the feature columns
below touched ATS outcomes, spreads-with-results, or the XLG-03 benchmark
evaluator. The frozen constants are mirrored in
`src/nfl_ats/cfb_role_features.py`; the benchmark runner records
`hypothesis_frozen_before_scoring: true` and this document's path in its
artifact metadata. Results are appended in a separate section below the
predeclaration and never edit it.

This is the XLG-04 follow-up called for by the roadmap: ONE frozen CFB
role-loss/role-continuity feature family (dropback and carry only), scored
against the frozen XLG-03 benchmark. It is a CFB-only experiment; no NFL
rows or outcomes are involved. Whatever the verdict, an NFL transfer claim
(XLG-05) remains a separate, separately predeclared step.

## Mechanism being operationalized

XLG-04 (`docs/cfb_role_replication.md`) replicated cross-league that a
player with a material trailing role who participates at all delivers
approximately that full role (CFB median delivered/prior 1.043 for
dropbacks, 0.995 for carries). The pregame-knowable trace of that mechanism
is *participation continuity*: whether the players holding a team's role
mass actually appeared in the team's most recent game. When they did, the
team's realized roles are highly predictable; when they did not, the team
enters the game with a disrupted, less predictable role structure that the
market may or may not fully price.

Receptions are excluded: XLG-04 recorded reception delivery as **not
replicated** (severe under-delivery gate), and that verdict stands.

## The departure prerequisite (satisfied first, participation data only)

The roadmap required separating permanent departures from temporary
absences before any absence-derived feature. The descriptive study
(`nfl-ats cfb-absence-separation`, artifact
`artifacts/cfb_role_experiments/20260817T105651Z`, participation data only
— no spreads or outcomes read) measured, over 2013–2025 FBS-vs-FBS games:

- **Season boundaries are dominated by departures.** Of qualified role
  holders at the end of a season, only **15.6%** (dropback) / **18.7%**
  (carry) ever appear for the same team the following season. A role mass
  that carried prior-season holders forward would therefore consist mostly
  of graduated/transferred/drafted players every September.
- **Within-season, reappearance probability decays fast with the missed-game
  streak.** Same-season return rates conditional on an absence episode
  reaching k consecutive missed valid games (dropback / carry):
  k=1: 40.3% / 45.2%; k=2: 25.4% / 24.7%; k=3: 16.4% / 12.4%;
  k=4: **10.2% / 6.8%**; k=5: 6.0% / 3.8%. By four straight missed games,
  roughly 80% of episodes never see the player again at all.

Frozen consequences: the active role mass is **scoped to the current
season** (a player enters it only after their first current-season
appearance), and a holder who misses **4** consecutive valid team-games
leaves it (treated as departed/out-for-season). Both rules are
pregame-knowable from participation alone.

## Frozen feature definition

Six columns (`CFB_ROLE_FEATURE_COLUMNS`), computed per canonical game:
`{home,away,diff}_{dropback,carry}_continuity`, with `diff = home − away`.

For one team, action type, and game t, evaluated strictly before game t's
own credits update any state:

- **Role state**: the XLG-04 appearance-only span-8 EWM share per
  (team, player, action type) — identical update rule, thresholds
  (dropback ≥ 0.50, carry ≥ 0.20), and ≥ 3 prior appearances, computed on
  the same credited-action definitions over the same valid team-games
  (≥ 10 team dropbacks / ≥ 10 carries; XLG-04's per-(season, action)
  coverage gate applies unchanged).
- **Active mass at game t**: qualified holders (state ≥ threshold,
  appearances ≥ 3) who have appeared for this team in game t's season and
  whose current missed-valid-game streak is < 4.
- **Continuity** = (state-weighted mass of active holders with streak 0,
  i.e. who appeared in the team's most recent valid game) ÷ (total active
  mass). Empty active mass → **1.0** (neutral: "no known disruption"). A
  canonical game side without a computable row (week one, invalid
  team-game) is likewise imputed 1.0.

Continuity lives in [0, 1]; 1 means every accustomed role holder
participated last game, 0 means none did.

## Frozen evaluation protocol

- **Evaluator**: the XLG-03 walk-forward recipe, unchanged — Ridge alpha 10,
  no calibration, `market_residual` target, ≥ 500 strictly-earlier training
  games, out-of-time empirical residual distribution, forced picks.
- **Three matched arms on identical weeks**: `market` (no-vig control),
  `market_residual` (the frozen benchmark contract), and
  `market_residual_roles` (the identical recipe with exactly the six
  continuity columns appended). Any week skipped for one arm is skipped for
  all.
- **Decision metric**: paired per-game **accuracy improvement** of
  `market_residual_roles` over `market_residual` on the **clean core**
  (2012–2019, 2021–2025), with the week-blocked bootstrap interval from
  `paired_feature_comparisons` (2,000 samples, seed 20260817).
- **Decision rule (frozen)**: the family **clears** only if the week-blocked
  95% interval on paired accuracy improvement excludes zero from below
  (lower bound > 0). Season-blocked intervals, Brier, and log-loss
  improvements are reported as coherence checks but do not override the
  rule in either direction.
- **One run.** No cap retuning, threshold tuning, imputation changes, or
  season-window changes after seeing results. Any variant is a new
  predeclaration. If the family does not clear, it is recorded as-is and no
  NFL transfer claim is predeclared from it.

Sensitivity context, fixed in advance: the XLG-03 positive-control audit
found the CFB evaluator's week-blocked machinery detects synthetic
0.5/1/2-point-per-SD effects in 1/8, 5/8, and 8/8 replicas. A non-cleared
result is therefore evidence of "not resolvably large," not proof of zero.

## Declared limitations

1. **Absence of credit is not absence.** Continuity measures observed
   participation only; a quiet game with zero credited dropbacks/carries by
   a role holder in a valid team-game reads as a miss. The XLG-04
   participation contract is inherited verbatim.
2. **The market prices injuries too.** CFB spreads move on QB news; the
   family may be entirely redundant with the close. That is exactly what
   the market-residual design tests, and a null is an informative outcome.
3. **Week-one blindness.** Season scoping makes every opener neutral; the
   family deliberately claims nothing about offseason roster change (that
   is XLG-06's territory, with roster/recruiting data).
4. **Mid-season departures inside the cap window** (weeks 1–3 of an
   absence) still depress continuity as if they were temporary; the study
   shows these are a minority of episode-games but they are not zero.
5. The neutral imputation (1.0) biases toward "no disruption"; teams with
   chronically thin volume read as continuously undisrupted.

---

## Results

### Voided first run (instrument failure, no information revealed)

The first execution (artifact `artifacts/cfb_role_experiments/20260817T110002Z`)
was **void**: the feature join keyed teams by name, but play-by-play uses
display names ("Minnesota Golden Gophers") while the canonical table uses
schedule names ("Minnesota"), so every game received the neutral value and
the candidate arm reproduced the baseline bit-for-bit (paired improvements
exactly 0.0 across all 8,933 games). Because the candidate arm was
numerically identical to the already-known baseline, this run revealed
nothing about the family's ATS value; repairing the join and re-running is
an instrument fix, not tuning. Two changes were made before the re-run,
neither touching the frozen definitions, gates, seeds, or decision rule:
the join now goes through ESPN team ids (pbp `pos_team_id` against the
canonical `home_id`/`away_id`), and the runner now **fails closed** if all
six role columns are constant, so a vacuous comparison can never complete
silently again.

### Frozen run (2026-08-17, artifact `artifacts/cfb_role_experiments/20260817T110541Z`)

Join verified: 31.7% of the 12,500 canonical games carry at least one
non-neutral continuity value (features exist for 2013–2025 only; pre-2013
games are all-neutral by construction and leave the pre-2013 arms
identical, as expected).

Clean core, 8,933 paired non-push games, candidate minus baseline
(positive = candidate better), 2,000 samples, seed 20260817:

| Metric | Estimate | Week-blocked 95% | Season-blocked 95% |
|---|---|---|---|
| Accuracy improvement | **−0.0067** (−0.67 pts; 50.92% vs 51.60%) | [−0.0133, +0.0001] | [−0.0168, +0.0040] |
| Brier improvement | −0.00058 | [−0.00102, −0.00015] | [−0.00115, −0.00010] |
| Log-loss improvement | −0.0018 | [−0.0038, −0.0004] | [−0.0042, −0.0002] |

Margin error also resolved worse (clean-core MAE 12.285 vs 12.256).

### Verdict: **not cleared** (frozen rule), recorded as a real negative

The week-blocked accuracy interval does not exclude zero from below — it
nearly excludes zero on the *wrong* side — and both probability metrics
resolved significantly worse under week and season blocking. The honest
mechanistic reading: participation-continuity disruption is information the
CFB market already prices (QB and lead-back news moves college spreads),
so conditioning the residual model on it added variance without signal.
XLG-04's participation-level replication stands — players who play deliver
their roles — but the pregame trace of that mechanism carries no residual
ATS value in this form.

Consequences, per the predeclaration:

- **No NFL transfer claim is predeclared from this family.** The XLG-05
  role-continuity transfer path is closed in this form.
- The family is retained as a negative result; no cap/threshold/imputation
  retuning on these outcomes. Any successor (e.g. roster-aware departure
  handling, replacement-quality weighting, or availability semantics from
  XLG-07 data) is a new predeclaration against this same benchmark.
- The benchmark instrument worked as designed: it resolved a ~0.6-point
  probability-metric degradation that the NFL evaluator could not have,
  which is exactly the detection power XLG-03 was built to provide.
