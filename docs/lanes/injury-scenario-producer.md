# Injury scenario producer (PER-10)

## Goal
Build a research-only joint lineup-scenario producer wired to the injury
scenario mixture kernel, run once for 2026 Week 3, and compare its mixed
discrete cover probability to the served one per game.

## State
Not built. Both documented blockers in `docs/injury_scenario_mixture.md`
still hold today, and a third blocker was discovered this session: **the
kernel itself no longer exists in `src/`.**

## Tried (this session, read-only)
- `docs/injury_scenario_mixture.md` (2026-09-02): names two blockers —
  (1) no joint multi-player lineup probability, only marginals; (2) no
  accepted mapping from player EPA/value state to a scenario margin-point
  center.
- `git log --oneline -- src/nfl_ats/injury_scenarios.py` and
  `tests/test_injury_scenarios.py`: both were **deleted** in commit
  `b7ed31d` ("Repository cut", 2026-09-10) as one of "60 src modules
  imported by nothing but their own tests" — the kernel was orphaned
  (no producer ever called it) and got swept in the hygiene cut. The
  271-line module is recoverable verbatim at `git show b7ed31d~1:src/nfl_ats/injury_scenarios.py`.
  ROADMAP.md:549 (`PER-10` row) still asserts "Distribution kernel
  complete" — that line is now stale/inaccurate; nothing in `src/`
  imports or exposes `injury_scenarios.py` today (confirmed via
  `find src -iname "*injury*scenario*"` -> only a stale `.pyc`).
- Checked whether blocker 1 (joint probability) is closeable with
  existing measured results: `docs/absence_pairwise_dependence.md` /
  `registry` (ROADMAP.md:549) already measured pairwise absence coupling
  (independence rejected; observed/null coupling ~1.13-1.25x, all pairs
  sit together 20-27% less than pooled independence implies) — this is
  reusable, not a new decision, and would plausibly close blocker 1 for
  small "questionable" sets (full-unit sits never occur per the same
  doc).
- Checked whether blocker 2 (margin-center mapping) is closeable with
  data already in the repo:
  - `artifacts/latent_ratings_on_production/expected_lineup_ratings.parquet`
    (built by `scripts/latent_ratings_on_production.py`) has per-game
    `lineup_total` / `full_strength_total` / `divergence`, built from
    per-player offense/defense ridge ratings
    (`data/processed/player_participation_ratings.parquet`) times EWMA
    role shares times **marginal** play probability
    (`scripts/latent_ratings_on_production.py:300-345`). This is in raw
    EPA-weighted rating units, not margin points, and is used nowhere in
    `src/` (`grep -rln "lineup_total|diff_divergence" src` -> no hits):
    it is a standalone research artifact with no fitted conversion to
    margin points.
  - `src/nfl_ats/players.py:1441-1442` (`_injury_value_features`) builds
    `injury_skill_epa_value_lost` / `injury_defense_disruption_value_lost`
    as a **linear sum over players** of
    `severity(marginal) * role_share * player_value_rate`, which would in
    principle let a scenario plug in a deterministic 0/1 in place of
    `severity` — **if** a frozen margin-point coefficient on these
    features existed.
  - No such coefficient exists. The only consumers of these two columns
    are `src/nfl_ats/surgical_gating.py:11-13` (a magnitude-threshold
    gate that picks between two already-computed model outputs, not a
    margin adjustment) and
    `src/nfl_ats/injury_value_tilt_overlay.py:82-108`
    (`apply_injury_value_tilt_overlay`), which only **flips the pick's
    side** when `value_lost_diff` has a favorable sign for the healthier
    team — no magnitude-to-points conversion anywhere, and its own
    disclosure text says "Prospective evidence only -- not applied to
    the published card" (`injury_value_tilt_overlay.py:125-131`). A
    sign-only flip is exactly the inadmissible flip pattern AGENTS.md's
    "One calibrated probability decides every pick" section bans from
    the served path, so it cannot be reused as the kernel's per-scenario
    margin center either.

## Next
Blocker 2 is unresolved and is a genuine new modeling decision (fit and
leave-one-season-out validate a mapping from lineup EPA divergence, or
from `injury_*_value_lost`-style per-player deltas, to margin points),
not a wiring task. Before a producer can be built:
1. Restore `src/nfl_ats/injury_scenarios.py` and its test from
   `git show b7ed31d~1:...` (or re-derive) if/when a producer is ready
   to consume it.
2. Predeclare and fit the margin-point mapping out of season (candidate
   input: the already-computed `diff_divergence` /
   `injury_*_value_lost` features), report in-sample/out-of-sample gap
   per AGENTS.md's calibration section, before any scenario center is
   trusted.
3. Combine per-player marginal play probabilities
   (`nfl_ats.availability.resolve_unavailability`) with the measured
   pairwise coupling multiplier from `docs/absence_pairwise_dependence.md`
   to build the joint scenario set (blocker 1 path, already closeable).
4. Only then wire both into the restored kernel and run 2026 Week 3.
5. Separately: fix ROADMAP.md:549's stale "kernel complete" claim, since
   the module is gone from `src/`.

## Open
No Week 3 comparison was produced this session — blocked as above, no
code was written. `ruff` was not run (nothing changed under `src/` or
`scripts/`).
