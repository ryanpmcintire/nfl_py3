# Player-arrest violent-person severity screen

## Predeclared design

**Measured** (`data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet`,
category-only count command, 2026-08-20): the safe 1,116-row incident index contains
125 literal category strings. No ATS outcome was read before this design was frozen.

**Inferred**: allegations involving violence against a person may create a more
coherent short-lived disruption mechanism than pooling every arrest category,
because availability, preparation, and teammate attention can plausibly change
around a serious incident. This is a research mechanism, not evidence.

**Read** (`registry/experiment_specs/player_arrests_violent_person_recent_14d_fade_close.json`
and `player_arrests_violent_person_recent_14d_fade_opener.json`): the family was
fixed as a case-insensitive literal contains-any match on `assault`, `battery`,
`domestic violence`, `murder`, `manslaughter`, `rape`, `kidnapping`,
`sexual assault`, `sexual abuse`, `child abuse`, `injury to elderly`, `robbery`,
or `strangulation`. Composite labels remain eligible. The exposure window is
fixed at 14 calendar days strictly before the Tuesday decision date, and the
direction is fixed as a fade of the exposed team.

**Read** (same specs): the jointly declared cells are close grade for 2009-2025
and opener grade for 2020-2025, with week-primary/season-secondary blocking,
20,000 resamples, and seed 20260820. Split-half reliability is not applicable
to this transient per-game exposure.

**Read** (`src/nfl_ats/experiment_runner.py`): the builder reads only record ID,
incident date, team, and the safe category field when this filter is enabled.
Retrospective outcomes and case resolutions are not feature inputs. The join
uses only incidents strictly before the Tuesday decision date; a same-Tuesday
incident is excluded because the source supplies no intra-day publication time.

## Results

**Measured** (`registry/experiments/experiment-run/20260820T160814Z.json` and
`20260820T160823Z.json`): the safe category filter selected 314 of 1,116 source
rows before team mapping and 308 after mapping to schedule team codes.

| Grade/window | Flag/complement team-games | Slate share | Full-slate effect (accuracy points) | Week-blocked estimate and interval | `probability_positive` | Classification |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Close, 2009-2025 | 76 / 8,558 | 0.88% | -0.0000 | -0.0001 [-0.0980, +0.0993] | 0.4732 | `unresolved_below_power` |
| Opener, 2020-2025 | 22 / 2,984 | 0.73% | -0.0335 | -0.0332 [-0.1839, +0.1225] | 0.2893 | `unresolved_below_power` |

**Measured** (same registry records): the close cell used 294 week blocks and
the opener cell used 107 week blocks, with 20,000 resamples and seed 20260820
in each cell.

**Measured** (`artifacts/experiment_runner/20260820T160814Z/metadata.json`):
the close season-blocked diagnostic used 17 seasons, estimated +0.0007 accuracy
points, and reported `probability_positive=0.4702`.

**Measured** (`artifacts/experiment_runner/20260820T160823Z/metadata.json`):
the opener season-blocked diagnostic used six seasons, estimated -0.0270
accuracy points, and reported `probability_positive=0.2960`; the runner marked
that secondary diagnostic degenerate because it is below the measured
block-count floor.

## Decision and recording

**Measured** (opener registry record): the decision-relevant research cell's
point estimate is against the predeclared fade, at -0.0335 full-slate accuracy
points and `probability_positive=0.2893`.

**Inferred**: if this isolated research choice were the only input, current
expected value favors retaining the comparison side rather than applying the
violent-person fade. This does not change any production or prospective pick;
the experiment was scoped as research only.

**Measured** (`registry/weak_signals.json`): both cells were recorded by
`nfl-ats experiment run` as `unresolved_below_power`, with no closing ground.
The lane remains category 3; neither cell establishes a refuted mechanism or a
positive-control bound.

**Read** (`AGENTS.md`, research taxonomy): an interval is not a rejection
criterion, and a terminal verdict requires an admissible closing ground.
Accordingly, these results are retained with their `probability_positive`
values rather than described as settled.

## Exact commands and artifacts

**Measured** (2026-08-20): the recorded commands were:

```powershell
.\.tools\uv.exe run nfl-ats experiment run registry/experiment_specs/player_arrests_violent_person_recent_14d_fade_close.json
.\.tools\uv.exe run nfl-ats experiment run registry/experiment_specs/player_arrests_violent_person_recent_14d_fade_opener.json
```

**Measured** (command output): the corresponding artifact directories are
`artifacts/experiment_runner/20260820T160814Z` and
`artifacts/experiment_runner/20260820T160823Z`; the versioned registry records
are `registry/experiments/experiment-run/20260820T160814Z.json` and
`registry/experiments/experiment-run/20260820T160823Z.json`.
