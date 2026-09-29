# LEAD-69

## Goal
Execute LEAD-69's declared horizon interaction and descriptive slot sensitivity on local predeadline captures.

## State
**Measured:** complete; 799 starting games minus 114 structural SNF/MNF exclusions = 685 scored; 2 research looks.
Primary decisive OOS 8-6; accuracy gain +0.292 pp [95% -0.249, +0.779], probability_positive 0.7963.
Horizon IS/OOS accuracy 57.080%/56.350%, gap 0.730 pp; OOS log-loss gain -0.000980
[-0.002285, +0.000110], probability_positive 0.1481. c1 folds: -0.434823, -0.029133, -0.653790.
Descriptive slots decisive 23-19, OOS gain +0.584 pp [-0.343, +1.716], probability_positive 0.7222.
**Inferred:** unresolved_below_power; no serving proposal. Full IS/OOS, coefficients and calibration in results.

## Declaration
| LEAD-69 | ⬜ | Horizon-scaled move weight by kickoff slot (rank 4 of 8) | **Added 2026-09-29 (unmeasured).**
Mechanism: the pool line freezes Tuesday noon, so a Thursday game reaches its deadline about 2 days later and a
Sunday 1 PM game about 5 days later; the market has had different amounts of time to absorb news the frozen line
cannot contain, so the move's evidential weight should grow with hours from freeze to the pick deadline (min of
kickoff and Sunday 4 PM ET, `pick_refresh.pick_deadline`), and a single pooled `c` misweights both ends.
Predeclared: `c = c0 + c1*log(hours_to_deadline)` (one added parameter, not slot dummies), 799 games 2023-2025,
LOSO by season, base = served four-term, deadline and move read only from captures at or before the deadline
(leakage check in the builder), 2 looks (added term; slot-dummy sensitivity descriptive only), metrics as
LEAD-67. SNF and MNF are excluded structurally (deadline precedes any late capture). Checked, not duplicated:
`sunday-market-probability` lane (cutoff choice for one instant), MKT-08 (timing of refresh jobs, not the move
coefficient), MOD-20 unit 5 (interactions with flag sum and availability, not horizon). About 12 tool calls. |

## Tried
Declared before outcomes in this lane and `docs/lead69_protocol.md`; implemented `scripts/lead69_unit1.py`.
Ran `.tools/uv.exe run --no-sync python scripts/lead69_unit1.py`: one inventory halt, two scoring passes.
Departed from the requested single execution to repair source handling and 20 stale slot labels; no tuning.
Primary results exactly unchanged after correction. Final pass has 12 fits; both scoring passes total 24 fits.
Scoped Ruff format/check and eight registry payload validations passed; 685 rows, zero timestamp/slot violations.
Evidence: `docs/lead69_results.md`, `docs/lead69_predictions.md`, and source/run diagnostics under `docs/lead69_*`.

## Next
Orchestrator runs the batch below serially, then reconciles kickoff-clock revisions before any serving decision.

## Open
Registry pending; no terminal closure. Three seasons only; selected base features, no untouched outer test.
456 games had revised clocks; conservative deadline contractions reach 180 minutes. No served or Git mutation.
Prediction-level Markdown is local processed evidence; keep it untracked under the repository's data rule.

## Record commands
JSON payload contains eight correlated metric cells (2 comparisons x 4 metrics); not eight independent findings.
```powershell
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --batch docs/lead69_registry_payload.md
```
