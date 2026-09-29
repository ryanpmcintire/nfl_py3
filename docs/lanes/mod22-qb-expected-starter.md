# MOD-22 unit 4: expected-starter report repair

## Goal
Require evidenced pregame injury times, repair QB suffix matching, remeasure once.

## Protocol (declared before corrected outcomes, 2026-09-29; unchanged)
2020?2025 REG `build_fit_population` games with Tuesday opener; target `home_covered`;
player snapshot `20260910T205112Z`. Starter: highest-snap QB in previous team game
before kickoff minus 24h. Same `qb_out_diff` = away minus home Out/Doubtful flag,
requiring evidenced times by that cutoff; exclude/count guessed/missing/late reports.
Normalize punctuation and terminal Jr./Sr./II/III/IV/V; report matching without selection.
Same four base terms plus this fitted term, LOSO settings/folds and IS diagnostic.
One calibrated probability selects the side; family `qb_expected_starter_v1`, look **1**,
a remeasurement. Same accuracy/Brier/log-loss, decisive/null, calibration, IS/OOS gaps,
season/fold reports; 2,000 season/week bootstrap draws, seed 20260923, 95% intervals,
reliability edges [0,.4,.45,.5,.55,.6,1]. Report probability_positive; zero crossing
never closes a signal. No protocol revision.

## State
**Measured:** exit 0; 1,503 games/107 blocks, 55 evidenced flags (29 home/26 away),
53 nonzero games; 18 guessed flags excluded from 73 cutoff-only flags, one late
report excluded. Matches 3,230/3,230 source team-games (100%), 3,006/3,006 scored;
all 39 Gardner Minshew II matches restored; zero missing or ambiguous identities.
Decisive record **3?10**, exact-null p=0.0922852, before headline comparison:
accuracy -0.465735 points, 95% [-1.144499, 0.064857], probability_positive=0.04825;
Brier -0.000615381, 95% [-0.001312914,-0.000083369], probability_positive=0.008;
log-loss -0.001280602, 95% [-0.002717481,-0.000163914], probability_positive=0.0075.
IS/OOS Brier improvement -0.000000040/-0.000615381, gap +0.000615341;
candidate accuracy 57.4185%/56.6866%, gap +0.7319 points (base 57.3520%/57.1524%).
OOS Brier candidate/base/model/market: .246002912/.245387531/.251715273/.250000000.
QB coefficients 2020?25: +.23615,+.00186,-.11373,+.17714,-.26013,+.01475;
Brier worsened in all six seasons; full coefficients and reliability tables saved.
**Inferred:** this binary fitted addition meets AGENTS.md's `wrong_sign_resolved`
ground on corrected Brier; proposed closure is scoped to this addition, pending
root recording. Broader QB injury mechanisms remain open; no serving change.

## Tried
Ran `.tools/uv.exe run --no-sync python scripts/mod22_unit4.py` **once**, exit 0,
with repository-local UV cache and OMP/OpenBLAS/MKL/NumExpr threads capped at 2.
New artifact: `tests/scratch/codex/mod22_unit4/20260929T213636Z/`
(summary, per-game fitted probabilities, team-game evidence); run log:
`tests/scratch/codex/mod22-unit4-remeasurement.log`. **Measured:** saved population,
targets, opener and base IS/OOS probabilities equal the old artifact; all 55 flags
pass the cutoff. Replacement CLI parsing, coherence and closure validation passed
without invoking its handler; diff check passed. Old run remains diagnostic history.

## Next
Orchestrator reviews/runs replacement below serially; worker did not record it.

## Open
Binary absence omits QB quality; fold signs vary; split-half reliability unmeasured.

## Record commands
```bash
nfl-ats weak-signals record --replace \
  --name qb_expected_starter_out_pick_probability_term --family qb_expected_starter_v1 \
  --category health --league nfl --season-start 2020 --season-end 2025 \
  --description "MOD-22 unit 4: evidenced expected-starter absence; same single-look LOSO protocol remeasured" \
  --source "tests/scratch/codex/mod22_unit4/20260929T213636Z/summary.json" \
  --effect=-0.0006153808242152587 --effect-units brier_improvement \
  --interval-low=-0.001312913598693687 --interval-high=-8.336869502941536e-05 \
  --probability-positive 0.008 --sample-games 1503 --sample-blocks 107 \
  --classification refuted_mechanism --closing-ground wrong_sign_resolved \
  --classification-evidence "Corrected paired Brier improvement is wholly negative at 95%; 55 evidenced flags; closure limited to this binary fitted addition" \
  --notes "Look count remains 1 (remeasurement); 18 guessed flags excluded; all 39 Minshew II matches restored; no split-half reliability estimated; broader QB injury effects remain open" \
  --plain-summary "When the previous starting quarterback was ruled out or doubtful at least a day before kickoff, this simple adjustment made our predictions worse. Keep this adjustment out of the picks; quarterback injuries may still matter."
```
