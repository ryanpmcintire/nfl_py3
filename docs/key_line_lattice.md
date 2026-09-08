# MOD-18 Lane T: the key-line restriction of the mass-preserving read

## Predeclaration (frozen before any computation, 2026-09-08)

Read: `AGENTS.md`, "Football margins are multimodal, not Gaussian" -- the served
cover probability is computed against the DISCRETE margin distribution
conditional on the line, with mass at the key numbers 3/7/10/14.

Read: `docs/mass_preserving_lattice.md` (lane K's frozen predeclaration and its
measured results) and `artifacts/research/laneK/cells.json`. Lane K's MP1 keeps
the empirical integer atoms at their ABSOLUTE positions and lets the model enter
as an exponential tilt. Measured by lane K and read from that doc's bucket table
(`docs/mass_preserving_lattice.md:198-204`): MP1 LOSES for side selection
through the played three-member card overall (-0.399 accuracy points, 95%
[-2.088, +1.318], `probability_positive` 0.311), while its entire standalone
gain sits in the "7" bucket -- games quoted exactly at 7 or -7 -- where it goes
51.35% -> 59.46% right (+8.108 points, 95% [-2.500, +18.919],
`probability_positive` 0.920) and pays it back on lines between the atoms
(bucket 3.5-6.5, -3.080 points).

### The mechanism this lane tests

Stated as a mechanism before any number is computed. A discrete read differs
from a smooth read in exactly one place: at the integer where the empirical mass
piles up. When the quoted line SITS ON such an atom, the atom's weight is the
push, and the cover / loss split either side of it is decided by mass the smooth
read spreads across a continuum -- so the discrete read is answering the
question the football actually asked. When the line sits BETWEEN atoms (3.5,
4.5, 6.5 ...), the push atom is empty by construction, the discrete read's only
contribution is sampling noise in the band's integer histogram, and the smooth
read was already calibrated there. So the prediction is: serve the discrete
read where the line is quoted on a key atom, keep the served smooth read
everywhere else.

This is a MECHANISM, not a threshold flip (AGENTS.md, "No unexplained threshold
flips on the played card"): the boundary is the set of key integers the margin
distribution actually piles on, named in MOD-05 and in the AGENTS.md rule, not a
spread value chosen because accuracy dipped there.

### KL1, the primary form (declared before computing)

For every game in the matching opener evaluation, with served opener line `L`
(`tue_open_home_spread`) and served S3 home-cover probability `p_S3`:

- If `abs(L)` is exactly 3, 7, 10 or 14, the served home-cover probability is
  lane K's MP1 read for that game -- its declared band (`h = 2.5`, the frozen
  `nfl_ats.conditional_margin.BANDWIDTH`, widened in 0.5-point steps only until
  the declared 200-game floor is met), its declared prior pool (five trailing
  seasons, strictly prior completed games, whole target week excluded), its
  declared exponential tilt onto the served S3 point, and its declared read
  `cover + 0.5 * push`. Nothing about MP1 is re-tuned; lane K's construction is
  imported from `scripts/mass_preserving_lattice_opener_eval.py`, not
  reimplemented.
- Otherwise the served probability is `p_S3`, bit-for-bit.

The push probability follows the same rule: MP1's push on key lines, the served
S3 push elsewhere.

### KL1b, the one predeclared sibling

Identical, restricted to `abs(L)` exactly 3 or 7. It is declared here, before
the numbers are seen, as the narrower reading of the same mechanism -- the two
atoms that carry the most mass -- and not as a search: no third key set is
computed or scored.

### How it is graded (declared before computing)

1. **Replay gate.** Re-run lane K's frozen S3 replay (weekly walk-forward ridge
   refit with the served home-side offsets) against the matching opener
   evaluation. If any served probability differs by more than 1e-9, stop before
   any candidate is constructed.
2. **Cross-lane reproduction.** The MP1 read recomputed here must reproduce lane
   K's `artifacts/research/laneK/scored.parquet` `p_MP1` on the games it touches;
   the maximum gap is reported.
3. **Games touched.** Report how many archive games and how many Week 1 2026
   games each arm touches, and how many of those the arm's pick differs from S3.
4. **Standalone accuracy at the OPENER**, paired against S3 on the same games:
   accuracy points, week-blocked bootstrap, 20,000 draws, seed 20260817,
   within-week correlation ZERO (never estimated, never padded).
5. **Per season**, and **per key line separately (3, 7, 10, 14)** with `n`,
   plus the per-`nfl_ats.spread_regime.spread_bucket` table for continuity with
   lanes H and K.
6. **Brier and log loss** on the same served-comparable probability.
7. **Push rate predicted versus realised** on each key line, and the push
   calibration Brier against S3.
8. **THROUGH the played three-member card**, via `overlay-composition
   --per-game-artifact` on a research `per_game` artifact whose metadata carries
   `research_arm` and `active_model_id` `research_laneT_KL1` /
   `research_laneT_KL1b`, never the active identity.
9. **Week 1 2026 sides that would change**, read from the linked forecast's
   `predictions.csv` and its `home_side_offset.json` sidecar (replayed to 1e-9),
   standalone and through the card. No forecast is regenerated.

Every comparison cell is recorded through `nfl-ats weak-signals record`, family
`mod18_conditional_margin_v1`, names `mod18_conditional_margin_v1_kl1_*`,
classification `unresolved_below_power`, each with a plain-English summary.

### Decision rule, declared before the numbers are seen

Through-card `probability_positive` above 0.5 favours PLAYING KL1 in place of S3
on the played card; that reading is reported before any limitation. Standalone
accuracy and the loss reads are reported alongside and do not veto it
(AGENTS.md, "A promotion bar is not a decision bar"). The pool is forced picks,
so declining a candidate that is more likely than not better is taking the other
side of the bet.

### Disclosure

This is a POST-HOC RESTRICTION, and the restriction was chosen after seeing the
result it is built on. The 1,537-game opener archive is the mined archive lanes
K and S were selected on; S3 is itself a post-hoc restriction fitted on it; MP1
is a further look at the same window; and the "7" bucket result that motivated
KL1 was measured on exactly the same 74 games KL1 will be graded on. Whatever
this lane measures on the "7" and "3" atoms is therefore descriptive reuse of an
already-mined slice, not independent confirmation. Only the 10 and 14 atoms
enter for mechanism alone -- lane K published no separate reading on them -- and
even those sit inside buckets (7.5-10, 10.5+) whose overall behaviour was seen.
The 2026 rows are the evidence that settles it at no window cost.

### Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator. Never state that anything "needs more games"; decide
on expected value (P+ above 0.5 favours playing), and state what the numbers
imply for the DECISION before what is wrong with them.

## Measured results (appended after computation)

### The decision first

Measured (`artifacts/research/laneT/cells.json`): **serve the discrete read on
key lines.** Through the played three-member card KL1 reads **+0.133 accuracy
points, 95% [-0.396, +0.666], `probability_positive` 0.6456**, and the
predeclared sibling KL1b **+0.200 [-0.271, +0.726], `probability_positive`
0.7426**. Both clear the declared 0.5 bar, so on expected value both are the
better card, and the narrower KL1b -- the atoms at 3 and 7 only -- is the
stronger of the two. Card accuracy over the 1,503 graded archive games:
55.888% (S3) -> 56.021% (KL1) -> 56.088% (KL1b). Standalone at the opener the
same ordering holds: 54.558% -> 54.691% (P+ 0.6162) -> 54.757% (P+ 0.6880).

The mechanism this lane predeclared is what shows up. Lane K's MP1 served
everywhere loses the card (-0.399, `probability_positive` 0.311); the same read
served only where the line sits ON an atom wins it. Nothing is closed: all 88
recorded cells are `unresolved_below_power` and no cell carries a closing
ground.

### Games touched (measured, `mapping.json`)

| Arm | Atoms | Games touched | Of 1,537 | At 3 | At 7 | At 10 | At 14 | Picks changed | Card flips (graded) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KL1 | 3, 7, 10, 14 | 307 | 20.0% | 199 | 73 | 23 | 12 | 32 (30 graded) | 18 |
| KL1b | 3, 7 | 272 | 17.7% | 199 | 73 | -- | -- | 30 (29 graded) | 17 |

Measured invariant (`invariants.json`): on the 1,230 games KL1 does not touch
(1,265 for KL1b) the maximum probability gap against S3 is exactly **0.0** and
the card disagrees on **zero** games. The restriction cannot move a game the
mechanism does not name.

### Reproduction gates (measured, `reproduction.json`, `mapping.json`)

Matching archive `artifacts/opener_evaluation/20260908T115957Z`, active model
`a4c757efd2525da6`, feature digest `457aafb7...`. Maximum S3 served-probability
replay gap **3.774758e-15**, maximum offset gap 6.661338e-16 -- the gate passes
far inside 1e-9. The MP1 read recomputed here reproduces lane K's
`scored.parquet` `p_MP1` on all 1,537 rows to **exactly 0.0**. 1,503 graded
games over 107 weeks; week-blocked bootstrap, 20,000 draws, seed 20260817,
within-week correlation zero, never estimated or padded.

### Per key line (measured, `cells.json`), delta points [95%], P+

| Line | n graded | S3 standalone / card % | KL1 standalone / card % | Standalone delta (P+) | Card delta (P+) |
|---|---:|---|---|---|---|
| 3 | 179 | 53.07 / 59.22 | 51.40 / 59.78 | -1.676 [-5.979, +2.488] 0.186 | **+0.559** [-3.141, +4.167] 0.555 |
| 7 | 70 | 52.86 / 51.43 | **61.43** / 54.29 | **+8.571** [-1.449, +19.403] 0.934 | **+2.857** [-4.001, +10.000] 0.729 |
| 10 | 23 | 60.87 / 78.26 | 56.52 / 73.91 | -4.348 [-14.286, +0.000] 0.000 | -4.348 [-14.286, +0.000] 0.000 |
| 14 | 11 | 36.36 / 36.36 | 36.36 / 36.36 | no graded pick moved | no graded pick moved |
| all touched (KL1) | 283 | 53.00 / 57.95 | 53.71 / 58.66 | +0.707 [-2.996, +4.437] 0.610 | +0.707 [-2.084, +3.497] 0.645 |
| all touched (KL1b) | 249 | 53.01 / 57.03 | 54.22 / 58.23 | +1.205 [-2.834, +5.344] 0.688 | +1.205 [-1.754, +4.264] 0.742 |

Measured reading: the 7 atom carries the effect, the 3 atom costs standalone
accuracy and returns it through the card, and the 10 atom is one flipped pick
in 23 games that went the wrong way. On the 12 games quoted at 14 the read
changes the stated chances but never the side, so that accuracy comparison is
identically zero and is reported here rather than recorded as a signal.

### Per season (measured, `cells.json`), delta points [95%], P+

| Season | n | KL1 standalone | KL1 card | KL1b standalone | KL1b card |
|---|---:|---|---|---|---|
| 2020 | 220 | +0.909 [+0.000, +2.326] 0.880 | +0.455 [+0.000, +1.415] 0.644 | +0.909 [+0.000, +2.326] 0.880 | +0.455 [+0.000, +1.415] 0.644 |
| 2021 | 236 | no graded pick moved | no graded pick moved | no graded pick moved | no graded pick moved |
| 2022 | 248 | +0.806 [-0.800, +2.390] 0.779 | -0.403 [-1.250, +0.000] 0.000 | +0.806 [-0.800, +2.390] 0.779 | -0.403 [-1.250, +0.000] 0.000 |
| 2023 | 266 | -1.504 [-3.422, +0.380] 0.045 | -0.752 [-2.264, +0.741] 0.097 | -1.128 [-3.030, +0.746] 0.083 | -0.376 [-1.569, +0.761] 0.186 |
| 2024 | 266 | +1.504 [-0.763, +3.650] 0.875 | +2.256 [+0.377, +4.044] 0.987 | +1.504 [-0.763, +3.650] 0.875 | +2.256 [+0.377, +4.044] 0.987 |
| 2025 | 267 | -0.749 [-2.239, +0.741] 0.092 | -0.749 [-1.880, +0.000] 0.000 | -0.749 [-2.239, +0.741] 0.092 | -0.749 [-1.880, +0.000] 0.000 |

Measured: 2020, 2022 and 2024 favour the restriction standalone; 2023 and 2025
go against it; 2021 never moved a graded pick at all. Two cells have an interval
wholly on one side of zero -- the 2024 card, for the candidate, on both arms --
and three have an upper bound of exactly zero (2022 card, 2025 card, key line
10), which is one or two flipped picks in a season, not a resolved sign.

### Losses (measured, `cells.json`)

Overall, both arms are very slightly WORSE calibrated on the served-comparable
probability: KL1 Brier -0.0000372 [-0.000493, +0.000423] `probability_positive`
0.4361 and log loss -0.0000740 [-0.001008, +0.000871] 0.4389; KL1b Brier
-0.0000141 [-0.000459, +0.000425] 0.4773 and log loss -0.0000230 [-0.000941,
+0.000880] 0.4827. On the 7 atom alone both losses improve (Brier +0.001357
`probability_positive` 0.6308, log loss +0.002867 at 0.6334); on the 3 atom
both worsen (Brier -0.000649 at 0.2847). So on this archive the restriction
buys side selection at the 7 atom and pays a little calibration at the 3 atom.

### Push rate, predicted versus realised (measured, `push.json`)

| Line | n | Realised push % | S3 predicted % | KL1 predicted % | KL1b predicted % | Push Brier improvement (P+) |
|---|---:|---:|---:|---:|---:|---|
| 3 | 199 | 10.050 | 3.499 | **9.095** | **9.095** | +0.003652 [-0.000521, +0.008260] 0.955 |
| 7 | 73 | 4.110 | 3.429 | 5.571 | 5.571 | +0.000166 [-0.001917, +0.002752] 0.533 |
| 10 | 23 | 0.000 | 3.860 | 3.786 | 3.860 (untouched) | +0.000008 [-0.000359, +0.000394] 0.518 |
| 14 | 12 | 8.333 | 3.992 | 7.324 | 3.992 (untouched) | +0.003990 [-0.004422, +0.021808] 0.649 |

The push read is the part of MOD-05's premise this construction exists for, and
it is right where the mass is: at a line of 3 the smooth read says 3.5% and the
football says 10.1%, and the discrete read says 9.1%.

### Per spread bucket (measured, `buckets.parquet`), for continuity with lanes H and K

| Bucket | n graded | S3 standalone / card % | KL1 standalone / card % | KL1b standalone / card % |
|---|---:|---|---|---|
| 0-3 | 552 | 56.16 / 57.43 | 55.62 / 57.61 | 55.62 / 57.61 |
| 3.5-6.5 | 552 | 56.16 / 57.79 | 56.16 / 57.79 | 56.16 / 57.79 |
| 7 | 74 | 51.35 / 50.00 | 59.46 / 52.70 | 59.46 / 52.70 |
| 7.5-10 | 194 | 51.55 / 51.55 | 51.03 / 51.03 | 51.55 / 51.55 |
| 10.5+ | 131 | 47.33 / 51.15 | 47.33 / 51.15 | 47.33 / 51.15 |

The bucket and the atom are not the same slice: the "7" BUCKET holds 77 archive
rows (74 graded) because four of them are quoted at 6.75 or 7.25, which are not
on the atom and which KL1 therefore leaves alone; the key line 7 itself is 73
rows (70 graded).

### Construction diagnostics (measured, `scored.parquet`)

Across the 307 key-line games the band stays at the declared 2.5 points for
83.4% and widens for the rest (maximum 8.5), with a median 395 prior games
inside and a mean 62.8 distinct integer atoms. The solved tilt is small
everywhere -- mean absolute theta 0.0119, maximum 0.0404 -- so the model's
information is a gentle reweighting of the empirical shape, never a distortion
of it.

### Week 1 2026 (measured, `week1.json`); no forecast was regenerated

Linked forecast `artifacts/margin_predictions/2026-week-01-20260908T124514Z`;
its `home_side_offset.json` sidecar replays to **9.992007e-16** on all 16 games.

Two of the sixteen games are quoted on an atom: DEN at KC (line 3) and NO at
DET (line 7). Both arms touch both games and both change **one side**: NO at
DET, where the served read has Detroit at 0.5158 to cover and the discrete read
has 0.4720 -- New Orleans. It survives the three-member card. DEN at KC moves
from 0.5178 to 0.5152 and keeps its side.

### Limitation, disclosed

Read (the predeclaration above): this is a post-hoc restriction, and the
restriction was chosen after seeing the result it is built on. The 74-game "7"
bucket that motivated it supplies 70 of the 283 graded games KL1 is scored on
here, so that part of the measurement is descriptive reuse of an already-mined
slice, not confirmation. The 1,537-game archive is the mined archive lanes K
and S were selected on, S3 is itself a post-hoc restriction fitted on it, and
the two arms are nested and correlated with lane K's MP1 and lane H's M1. Only
the 10 and 14 atoms entered for mechanism alone, and between them they carry 34
graded games and one changed pick. The 2026 rows are the evidence that settles
it at no window cost.

## Registry and verification

Coordinator instruction, 2026-09-08 (read: the coordinator's brief): the shared
`registry/weak_signals.json` was corrupted this morning by two lanes recording
at once, so research lanes EMIT their recorder argv instead of running it. All
88 comparison cells are emitted as ready-to-run PowerShell in
`artifacts/research/laneT/record_commands.ps1` (family
`mod18_conditional_margin_v1`, names `mod18_conditional_margin_v1_kl1_*`, every
one `unresolved_below_power` with no `--closing-ground`, a plain-English
`--plain-summary` on every row, `--replace` so a re-run is idempotent); the name
list is in `registry_names.json`. Eight degenerate slices are deliberately not
recorded and are named in `push.json` under `skipped_degenerate`: they are
comparisons in which no graded pick moved at all, so the paired difference is
identically zero and a bootstrap of zeros would report a
`probability_positive` of 0.0 for what is really "no change".

Measured: six representative commands (accuracy points, Brier, log loss, the
push Brier, a negative cell and a KL1b cell) were executed against an ISOLATED
registry directory under the session scratchpad and all returned 0, so the
emitted argv is admissible to the real validator. The live registry was not
written: it holds 3,627 signals and zero lane-T rows.

Measured checks: 61 tests passed in 22.97 seconds (18 of them new)
(`tests/test_key_line_lattice.py`, `tests/test_mass_preserving_lattice.py`,
`tests/test_conditional_margin.py`, `tests/test_experiment_registry.py`, which
includes the artifact-provenance scanner); scoped `ruff format --check` and
`ruff check` pass.

```powershell
.\.tools\uv.exe run --no-sync ruff format --check scripts/key_line_lattice_opener_eval.py tests/test_key_line_lattice.py
.\.tools\uv.exe run --no-sync ruff check scripts/key_line_lattice_opener_eval.py tests/test_key_line_lattice.py
.\.tools\uv.exe run --no-sync pytest tests/test_key_line_lattice.py tests/test_mass_preserving_lattice.py tests/test_conditional_margin.py tests/test_experiment_registry.py -n 2 --basetemp <scratch>/laneT -q
.\.tools\uv.exe run --no-sync python scripts/key_line_lattice_opener_eval.py --stage {replay,map,score,week1,record}
```

The predeclaration is frozen byte-for-byte at
`artifacts/research/laneT/predeclaration.md`, sha256
`e5043b013bbb93d667f111ba771a80f0ecdcb0074097f4139b4aba0a7729541b`, which is
the digest stamped into `reproduction.json` before any number was produced. No
production path, forecast, card or published page was touched, and nothing was
committed or pushed.
