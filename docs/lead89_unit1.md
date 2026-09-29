# LEAD-89 unit 1 - legal-state and clock replay

**Measured:** `.tools/uv.exe run --no-sync python scripts/lead89_unit1.py`.
Metadata/quote replay only; no outcome columns read, fits or scores.

## Scope and declaration

**Read:** ROADMAP.md:872 separates this legal-state/clock unit from transition fitting
and policy scoring. The unchanged declaration was saved in `docs/lanes/lead89.md`
before execution. Historical openers are the frozen pool-line proxy.

**Read:** outer 2023/2024/2025; fit through Y-3, tune Y-2, calibrate Y-1.
One four-term calibrated discrete-margin probability selects every side.
Weekly Best Pick reward is primary: F=3, B=7, K=6; 717 planned looks.
This unit consumes zero outcome looks.

## Source and clock results

**Measured:** 1537 opener games / 107 weeks;
1503 four-term rows;
34 additional opener rows retained without outcome filtering.

**Measured:** 393 three-stage quote files /
444762 source rows; 173450 spread rows.
Excluded 37240 off-target linked rows and
396 inadmissible clock/line rows.
Missing files: 0.

**Measured:** 107 weeks have legal candidates at all three stages;
107 also have same-book opener/stage pairs at each stage.
61075 book/game/stage rows saved.
These are census counts; confidence intervals do not apply.

| Season | Weeks | Three-stage complete | Paired complete | Tue games | Thu games | Sun games |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 | 17 | 17 | 17 | 227 | 226 | 208 |
| 2021 | 18 | 18 | 18 | 239 | 237 | 216 |
| 2022 | 18 | 18 | 18 | 255 | 253 | 216 |
| 2023 | 18 | 18 | 18 | 272 | 270 | 239 |
| 2024 | 18 | 18 | 18 | 272 | 268 | 238 |
| 2025 | 18 | 18 | 18 | 272 | 268 | 237 |

**Measured:** checks enforce quote digests, target week, teams, handicap signs,
observation/book update and available market update clocks, and quote/schedule kickoffs.
Tuesday uses captures available by noon, Thursday its actual pre-kickoff capture,
and Sunday 12:30. Same-book movement is the stage quote minus the Tuesday anchor;
absent pairs remain unavailable.

**Read:** Tuesday captures are earlier anchors, not evidence of noon captures.
The final schedule is retrospective; quote kickoffs are also enforced.
This does not establish when every schedule revision became known.

## Legal nomination witnesses

**Measured:** 642 structural stage states;
102 earliest-nominee paths lock before Sunday;
107 latest-kickoff paths remain unlocked at Sunday.
One nominee is retained after initial nomination; locked nominees never change.
No side or reward is assigned and no future quote chooses a witness.

**Inferred:** these paths establish legality only; they do not price waiting or add
fitted policy variants. Early locked games need no Sunday quote. Replacement occurs
only before the nominee's known kickoff.

## Upstream cutoff and next-unit gates

**Measured:** 1537 upstream matches;
0 missing;
1443 training-cutoff violations against season start.

| Prediction season | Latest upstream training game |
| --- | --- |
| 2020 | 2020-12-28 |
| 2021 | 2022-01-03 |
| 2022 | 2023-01-01 |
| 2023 | 2023-12-31 |
| 2024 | 2024-12-30 |
| 2025 | 2025-12-29 |

**Read:** `src/nfl_ats/clv.py:2184` fits opener models on completed games before
each target week's first date, including earlier games in the prediction season.
At lines 2207-2218 it recomputes forecasts with the opener as the model input line.
The companion ledger is not verified direct lineage for those recomputed points.

**Inferred, decision gate:** season-held-out training is not established by these
sources. Unit 2 scoring is blocked until the orchestrator supplies compatible
forecasts, verifies exact opener lineage and checks stage-specific feature timing.
This is not evidence of future-game leakage or a refuted waiting mechanism.
No full-history rebuild was attempted in this shared-resource unit.

**Read:** base terms: `model_logit, composition_flag_sum, market_move_toward_home, market_move_available`.
Movement version: `leader_median_through_sunday_prekick_v1`.
Stored Sunday probabilities cannot support Tuesday/Thursday decisions.
Unit 2 must reconstruct stage-matched terms, fit/calibrate in the declared blocks,
retain push mass and verify all other term timestamps.

**Read:** frozen metadata limitation: Features were selected using these seasons; this is not an untouched outer test. Weekly ranking uncertainty is assessed separately from average calibration.

**Inferred:** one earlier-season joint weekly transition/resampling law must retain
cross-game dependence and compare early expected reward with the expected best
still-playable reward. Training-cutoff checks alone do not certify stage availability
or untouched model selection.

## Requested scoring fields

Decisive-game record, IS/OOS and gap, fold coefficients/stability,
accuracy/Brier/log loss/RPS, reliability bands, weekly reward, nominee Brier,
season/week-block 95% intervals and `probability_positive`:
**not computed in unit 1**. No score or verdict exists to record.
The mechanism is unadjudicated; AGENTS.md:65-85 forbids closing on an interval
spanning zero.

## Saved output and continuation

`tests/scratch/codex/lead89_unit1/` holds quote panels, source hashes, upstream cutoffs,
weekly coverage, structural state traces and summary JSON. Next is the predeclared
transition-fit/policy-replay unit, with 717 planned looks. The orchestrator records
any eventual research results serially.
