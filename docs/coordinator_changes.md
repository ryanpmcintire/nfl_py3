# Point-in-time coordinator changes (PER-07)

## Backlog audit

**Read (`ROADMAP.md`, PER-07; `src/nfl_ats/coach_fade_overlay.py`;
`src/nfl_ats/interim_hc_first_game_tilt_overlay.py`):** the head-coach portion
is implemented as year-one and interim-transition states, with prospective
challenger recording already covered by focused tests. **Read
(`registry/weak_signals.json`; `artifacts/prospective/challengers.json`):**
the head-coach measurements and their unresolved classifications are recorded,
and the two head-coach overlays have explicit prospective identities.

**Measured (2026-09-02, `rg --files src tests docs scripts | rg -i
'coach|coordinator|interim'` and recursive `data/raw` inventory):** before this
change, the repository had no coordinator feature module or OC/DC historical
snapshot. The only coaching-specific raw capture was
`data/raw/interim_coaches`, which contains head-coach events. **Read
(`docs/archive/data_source_scout_v4.md` section 12;
`docs/era_mechanism_screens_20260901.md`):** the existing source audits reach
the same limit: schedules contain head-coach identity only, while historical
OC/DC revision reconstruction is unverified and no structured feed has been
accepted.

Therefore PER-07 is not complete. A source-independent point-in-time adapter
is now implemented, but no fabricated coordinator history fills the source
gap.

## Builder contract

`src/nfl_ats/coordinator_changes.py` accepts two caller-owned tables:

- games: `game_id`, season/week, both teams, `decision_at`, and `kickoff`;
- assignments: team, exact role (`OC` or `DC`), coordinator name,
  `effective_at`, `observed_at`, and `source_url`.

For each team-game and role, the builder selects only an assignment whose
effective and observation timestamps are both no later than that game's
decision timestamp. It compares that assignment with the assignment actually
known at the same team's preceding game's own decision timestamp. This makes
offseason and in-season changes the same deterministic transition operation.

The numeric family is:

- `home_oc_changed`, `away_oc_changed`;
- `home_dc_changed`, `away_dc_changed`;
- home/away coordinator change counts and their home-minus-away difference.

Names, effective timestamps, observation timestamps, and source URLs remain
in the result as audit columns. A first observed game, incomplete OC/DC
coverage, or a post-decision assignment produces nullable change state rather
than assumed continuity. Missing timestamps, non-pregame decisions, ambiguous
roles, or conflicting observations raise `DataContractError`.

## Leakage and operational status

**Measured (`tests/test_coordinator_changes.py`):** focused tests cover an
offensive-coordinator transition, stable defensive coordinator, missing and
post-decision assignments, ambiguous roles, conflicting revisions, and a
decision-at-kickoff rejection. The leakage regression mutates completed-game
scores/results, appends a later game, and appends a historical correction
observed after the tested decisions; all prior feature rows remain identical.

**Read (implementation):** this family is not registered in feature profiles,
does not fetch a source, and is not wired to an experiment, registry decision,
challenger, or active model. **Inferred:** it closes the reusable transformation
contract but cannot close PER-07 until a lawful historical OC/DC source with
revision or announcement timestamps is captured and passes this adapter's
coverage rules. No ATS experiment or adjudication was run.
