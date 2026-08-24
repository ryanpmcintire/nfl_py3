# Drift monitoring (RWB-12)

Implemented 2026-08-25 on `swarm/blog-rwb12-drift`. Every Tuesday the pipeline
rebuilds the tables, re-fits the walk-forward models and publishes a card;
nothing before this work asked whether the *inputs* still look like the data
the model was designed for. The 2026-08-19 Tuesday-visibility audit
(`docs/prospective_evidence.md`) showed how a live arm can quietly stop seeing
its own inputs while everything keeps "working": `injury_value_lost_tilt_overlay`
read 0.0 for essentially every current-week game because its source rows do not
exist yet at build time, and nothing in the weekly run would have said so.
Drift monitoring is the standing instrument for that class of failure.

## What it monitors — four signals, one read-only report

`nfl_ats.drift` compares one published week against a reference window of the
six most recent completed weeks strictly before the target week's earliest
gameday (the same leak-safe cutoff rule `score_outcome_week` uses). All four
signals are computed from information available before kickoff plus results of
games already settled; nothing feeds back into any model or feature.

1. **Feature drift** — per-column standardized mean shift
   ((current mean − reference mean) / reference sd) and PSI (population
   stability index) binned on reference quantiles, organized by the
   feature-family registry (`nfl_ats.constants.FEATURE_FAMILIES`, RWB-02).
2. **Missingness drift** — per-column missing-rate change in percentage
   points. A column that turned half-null between snapshots — a broken join,
   a renamed source field — trips this even when every surviving value looks
   normal. Non-numeric garbage counts as missing: for a numeric feature it is
   operationally identical.
3. **Probability drift** — this week's published `home_cover_probability`
   distribution versus the same configuration's earlier cards: mean shift and
   share of games outside the reference central 90% band.
4. **Calibration drift** — Brier score and ECE over the most recent four
   settled weeks versus all prior settled history, for already-published
   probabilities only.

### Alert thresholds

| signal | warn | alert |
|---|---|---|
| feature mean shift (\|sd units\|) | ≥ 0.50 | ≥ 1.00 |
| feature PSI | ≥ 0.10 | ≥ 0.25 |
| missingness delta (pp) | ≥ +10 | ≥ +25 |
| probability drift | >20% outside reference band OR \|Δmean\| > 0.05 | — |
| calibration drift (Δ Brier, recent − prior) | ≥ 0.02 | ≥ 0.04 |

Two floors keep the monitor from crying wolf:

- **PSI needs ≥ 50 current-window games before its status is scored**
  (`FEATURE_PSI_MIN_GAMES`). A 16-game week against decile bins averages 1.6
  games per bin and reads ~0.2 PSI under a true null (measured this session on
  gaussian draws); below the floor the value is reported but marked
  `insufficient_history`, so September weeks are judged on shift and
  missingness instead of a statistically meaningless PSI.
- **Calibration comparison needs ≥ 32 recent and ≥ 200 prior settled games**
  (`CALIBRATION_MIN_RECENT_GAMES` / `CALIBRATION_MIN_PRIOR_GAMES`). Until
  those exist — roughly mid-season of the prospective era — the section reads
  `insufficient_history`.

A column absent from either frame is reported as fully missing rather than
dropped: a column that vanished from this week's table is precisely the
regression worth shouting about. A constant reference column is compared as
two bins (equal vs not), so a constant going non-constant is caught.

## What this deliberately is NOT

**Drift reports are telemetry, not evidence.** They adjudicate no candidate
against any baseline, so they spend no rotation-registry window and may never
be cited as a result about any signal — including "the feature drifted, so the
signal died", which would be a mechanism claim requiring the weak-signal /
rotation registry machinery, not a distribution comparison. That note is
written into every artifact the module produces. Per AGENTS.md binding rule 1,
nothing here closes anything: an alert means "look at this", never "discard".

## Storage format

Each run writes `artifacts/drift/<season>-week-NN-<run_id>/`:

- `drift_report.json` — the full report: overall status, reference window
  identity, per-section summaries (status, alerts, warnings, headline
  numbers), and the telemetry-not-evidence note;
- `feature_drift.csv` — the wide per-column table (family, means, shift,
  PSI, both missing rates, deltas, per-signal statuses).

Runs are append-only by construction (fresh timestamped directory), matching
the margin-predict artifact convention.

## Weekly-pipeline hook

Step 13 (`drift-report`) trails the publish, after POL-10 steps 9–12. Like
those steps it is **optional**: a failure is reported loudly and recorded in
the run summary's `optional_failures` but never aborts the run — the card is
already published by then, and read-only monitoring must not be able to take
the deliverable down. `--skip-drift` opts out. The step is read-only over
`data/` and `artifacts/`; it writes only its own artifact directory.

The card used for probability/calibration sections is matched by
**configuration fingerprint, not recency**: season, week, feature profile and
probability method must all match the query. This is the same lesson
`prospective-record` learned (see `docs/prospective_evidence.md`, "step 10
finds its card by configuration fingerprint") — the active model's card and
every challenger's card share one `margin_predictions` namespace, and picking
the newest would silently monitor the wrong model. History cards are
deduplicated per game keeping the FIRST occurrence, mirroring the ledger's
first-write-wins rule, so rehearsal reruns cannot rewrite history.

## CLI

```powershell
# Standalone (weekly-run step 13 runs exactly this):
.\.tools\uv.exe run python -m nfl_ats drift-report `
    --season 2026 --week 3 `
    --features data/processed/game_features_player.parquet `
    --feature-profile player --probability-method gaussian
```

Exit is a JSON payload ending in `artifact_directory`. Missing matching cards
fail loudly rather than reporting empty drift.

## Required tests

All in `tests/test_drift.py` (21) plus updated plan assertions in
`tests/test_weekly.py`:

- PSI ≈ 0 on identical distributions, large on a level shift; constant-column
  detection both ways.
- Stable week reports no alerts; wholesale level shift alerts; vanishing
  column = full-missingness alert; half-null column alerts; non-numeric
  garbage counts as missing.
- Probability drift flags a mean shift, passes a size-and-center matched
  sample, reads `insufficient_history` without history.
- Calibration drift detects recent miscalibration (alert), passes a genuinely
  calibrated recent window, reads `insufficient_history` below the game
  floors. Constructions are deterministic near thresholds (stratified-uniform
  outcomes), because seeded RNG draws at n≈16 sit close enough to the warn
  boundary to flake under `-k` selection.
- End-to-end report assembly (sections, statuses, evidence-disclaimer note)
  and artifact writing.
- Weekly plan: step 13 exists, optional, numbered 13, strictly after
  `publish-predictions`; `--skip-drift` removes it; CLI parses the standalone
  command; monitored columns come from the registry.

## Provenance

- **Measured this session:** the small-sample PSI null level (~0.2 at n=16,
  bins=5 and ~0.1 at n=100, bins=10) on gaussian draws, which set
  `FEATURE_PSI_MIN_GAMES = 50`.
- **Read:** `docs/prospective_evidence.md` (Tuesday-visibility audit and the
  fingerprint lesson), `src/nfl_ats/weekly.py`, `src/nfl_ats/outcomes.py`,
  `src/nfl_ats/cli.py`, `src/nfl_ats/constants.py` (feature-family registry),
  `src/nfl_ats/prospective_scoring.py`.
- **Inferred:** the specific threshold values other than PSI's standard
  tiers are engineering judgment (documented above), chosen conservative so
  warnings stay rare; they are defaults to tune against observed seasons, not
  measurements.
