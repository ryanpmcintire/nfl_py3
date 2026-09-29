# LEAD-89 — option to wait

## Goal
Execute the predeclared legal-state/clock replay unit; save `scripts/lead89_unit1.py` and `docs/lead89_unit1.md`. Only the orchestrator records results or changes the served card.

## State
**Measured:** unit 1 complete: one replay, exit 0; 1,537 opener games/107 weeks, 61,075 book/game/stage rows, all 107 weeks complete, 642 legal states. 102 early paths lock before Sunday; 107 waiting paths stay unlocked. Companion margin ledger: 1,443/1,537 cutoffs reach the prediction season; direct opener lineage is unverified (`upstream_linkage.json`). Current opener code also fits before each target week, including that season (clv.py:2184). **Inferred:** unit 2 requires compatible upstream forecasts; this does not establish future-game leakage or refute the mechanism. Full report: `docs/lead89_unit1.md`.
Unit-1 structural witnesses use schedule-order ties only: retain the earliest available nominee, or keep an unlocked latest-kickoff nominee through the three stages. They assign no side or reward, inspect no outcomes, and are not additional policy specifications.

### Declaration (verbatim protocol, ROADMAP.md:872 and Protocol C)
**Inferred mechanism:** news missing from Tuesday's frozen lines can create a better Sunday candidate after a Thursday nominee becomes irrevocable; the largest current edge need not justify giving up that opportunity. Predeclare a three-stage nomination policy at Tuesday noon, Thursday's pre-kickoff capture and Sunday 12:30. Preserve one legal nominee throughout: nominate a later game when retaining the option, compare the early game's expected reward with the earlier-season learned expected best still-playable reward, and stop before the nominee locks. Estimate one joint weekly transition/resampling law from pre-deadline quote panels, retaining cross-game dependence; no future realized outcomes or quotes enter a decision. The same fitted calibrated probability chooses every side; only nomination timing changes.

Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting. Enforce target-week, observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. LEAD-82 uses the hourly noon boundary. No closing inputs.

Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing missing-archive zeros. LEAD-82 alone fits 2023, calibrates 2024, scores 2025 with existing ridge; one outer season cannot establish stability. Verify upstream training cutoffs. Reused archives are retrospective; freeze survivors prospectively.

One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first; optimistic IS/OOS and gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and `probability_positive`. Zero crossing closes nothing; default `unresolved_below_power`.

Looks: L=(27K+B+4)(F+1)+25. F=outer seasons; B=candidate/nuisance specifications plus four baselines; K=endpoint/population series. Each fold/pooled panel: five arms and four contrasts × IS/OOS/gap × K, B fit/coefficient summaries, four decisive records; add 25 reliability cells. Identical refits add no specification. K=4 ordinarily; 85/89 add weekly reward/nominee Brier; 88 adds all-game/last-game total MAE. No unlisted variants. LEAD-89: weekly Best Pick reward primary, F=3, B=7, K=6, **717 looks**.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead89_unit1.py` ran once, one thread. Scoped ruff format/check and `mypy src` pass; `pytest -q`: 1,645 passed (67.76s), using a fresh temporary root after a permissions failure. Global format/lint fail outside this packet (400 lint errors). Temporary uv cache avoids default-cache permissions. No test files/functions added. Protocol copy, hashes and row outputs: `tests/scratch/codex/lead89_unit1/`.

## Record commands
None: unit 1 fits/scores/adjudicates nothing; zero outcome looks of 717 planned. No registry commands were run. The orchestrator records scored results serially after unit 2.

## Next
Orchestrator supplies season-compatible upstream forecasts and verifies stage-timed features; then assign the declared transition-fit/policy-replay unit in a fresh thread. No full-history rebuild here.

## Open
Decisive records, IS/OOS/gap, fold coefficients, effect intervals and `probability_positive` are uncomputed pending unit 2. Counts are censuses, so no interval applies. Quote sources are present; the upstream training/provenance gate blocks scoring. No serving, publication or Git mutation.
