# XLG-06 Stage 2 identity crosswalk

**Status:** implementation and local coverage audit measured 2026-09-03; no
NFL outcome or model measurement has run.

This slice makes the identity contract executable before any Stage-2 feature
work. The permitted path is:

```text
recruiting.athleteId
  -> draft_picks.collegeAthleteId
  -> nflverse players.espn_id
  -> nflverse players.gsis_id
```

The implementation is `src/nfl_ats/xlg06_crosswalk.py`. It preserves every
recruiting row, reports coverage at each hop, and fails closed on conflicting
one-to-many mappings or zero/empty identifiers. Ambiguous historical draft
keys are excluded and counted rather than guessed. It intentionally does not use
names or CFBD's `nflAthleteId`: the latter is an independent CFBD identifier
space, not the nflverse ESPN key.

This is an audit primitive, not evidence that a recruiting prior predicts NFL
performance. A later Stage-2 predeclaration must still specify the player
population, NFL feature timing, outcome, chronological evaluation, and any
source snapshot before those measurements are run.

## First local audit

Measured by `scripts/xlg06_crosswalk_audit.py` on 2026-09-03, using CFB
snapshots `20260818T214704Z` (recruiting) and `20260816T164451Z` (draft), plus
`nflreadpy.load_players()`; the immutable local output is under
`artifacts/xlg06_crosswalk/20260903T104848Z/`.

- 56,924 recruiting rows; 31,160 have a usable `athleteId`.
- 13,080 draft rows reduce to 5,904 usable college IDs; 8 ambiguous reused
  historical IDs are excluded.
- 25,065 NFL player rows reduce to 16,771 with a usable ESPN ID.
- Recruiting-to-draft coverage is 2.967%; recruiting-to-GSIS coverage is
  2.649%, measured over recruiting rows with usable IDs.

The low aggregate rate is a coverage result, not an outcome result: this
snapshot spans many recruiting cohorts while the draft table is a sparse
historical bridge. It must be stratified by cohort/player population before
any Stage-2 feature is proposed.

The cohort audit makes the timing issue explicit: measured recruiting-to-GSIS
reach is 7.1–11.4% for 2015–2020, 5.0% in 2022, 1.2% in 2023, and effectively
zero for 2025–2026 because those cohorts are not yet fully draft-observable.
These rows define a population/eligibility question for the next slice; they
do not justify dropping a cohort or claiming predictive value.
