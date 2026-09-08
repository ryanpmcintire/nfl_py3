# MOD-18 Lane K: the mass-preserving conditional margin read

## Predeclaration (frozen before any computation, 2026-09-08)

Read: `AGENTS.md`, "Football margins are multimodal, not Gaussian" -- the
served cover probability is computed against the DISCRETE margin distribution
conditional on the line, with mass at the key numbers 3/7/10/14. Read:
ROADMAP.md row MOD-18 -- lane K's K2 translated the empirical lattice so its
median met the prediction and predicted a 4.1% push rate at |line| = 3 against
10.1% realised; read: `docs/big_spread_lattice.md:55` -- lane H's K1-on-big-
spreads still reads 3.499% predicted against 10.050% realised on 199 games.
Both failures have the same cause: the mapping MOVED the atoms. MOD-05's
constraint is that recentring must not displace the ABSOLUTE key numbers.

This lane's one idea: keep the atoms exactly where the football put them and
let the model's information enter as a REWEIGHTING of those atoms.

### MP1, the primary form (one form, declared before computing)

For a target game with served opener line `L` (`tue_open_home_spread`) and
served S3 point `p`:

1. **Prior pool.** Completed regular-season games with a recorded final home
   margin and a recorded home line, from seasons at or after (target season
   minus five), whose gameday plus a one-day completion allowance is strictly
   before the target week's earliest gameday, with the whole target week and
   every later week excluded. These are exactly the exclusions of
   `nfl_ats.home_side_location.prior_games_for_week` (read:
   `src/nfl_ats/home_side_location.py:96`), which is the gameday form of
   `prior_rows_before` (read: same file, line 239). Each prior game's line is
   its archived Tuesday opener where the matched opener archive carries that
   game, otherwise the nflverse `spread_line` -- lane K's frozen construction,
   reused verbatim by lane H (read: `scripts/big_spread_lattice_opener_eval.py:58`).
2. **Line band.** The base atoms are the prior-pool games whose line satisfies
   `|line_prior - L| <= h`, with `h = 2.5` for MP1. That is the already frozen
   lane-K bandwidth `nfl_ats.conditional_margin.BANDWIDTH` (read:
   `src/nfl_ats/conditional_margin.py:16`), reused rather than a new constant.
   If fewer than `MIN_BAND_GAMES = 200` prior games fall inside, `h` widens in
   0.5-point steps until 200 games are inside or `h` reaches 20.0. Derivation
   of the 200 floor, stated before it is used: the smallest quantity this read
   exists to get right is the push mass at an integer key line, about 10% of
   games, so 200 prior games put roughly 20 games on that atom -- a relative
   standard error near 22%. It is a floor, applied identically to both arms,
   not a tuned value.
3. **Base atoms, at their absolute positions.** `q(m)` = share of band games
   whose final home margin was exactly the integer `m`. Nothing is translated,
   smoothed, or kernel-widened; the atom at 3 stays at 3.
4. **The model enters as an exponential tilt.**
   `p_theta(m)` proportional to `q(m) * exp(theta * (m - L))`, with `theta`
   solved so that `sum m * p_theta(m) = p`. The tilted mean is strictly
   increasing in `theta`, so the root is unique; it is found by Brent's method
   on `theta` in [-1, +1]. That bracket is far wider than anything attainable
   is expected to need (the band's margin variance is on the order of 190
   points squared, so one point of mean shift costs `theta` about 0.005); if
   the target mean lies outside the range the bracket can reach, `theta` is
   clamped to the nearer bracket end. Exponential tilting is the
   minimum-Kullback-Leibler way to impose a mean on a discrete distribution:
   atom positions are invariant and only their weights move, which is exactly
   the mass-preserving property K2's translation violated.
5. **The read.** strict cover = mass at `m > L`; push = mass at `m == L`
   (identically zero at a half-point line); loss = mass at `m < L`. The
   served-comparable decision probability is `cover + 0.5 * push`, the frozen
   convention of `ConditionalMarginLattice.decision_probability` (read:
   `src/nfl_ats/conditional_margin.py:45`) and the quantity lane H graded. The
   pick is that probability at or above 0.5. Brier and log loss are scored on
   the same quantity so the arms stay paired with lanes G and H; the
   push-renormalised read `cover / (cover + loss)` is recorded once at overall
   level as a declared diagnostic, never as the primary.
6. **Scope.** MP1 replaces the served mapping on EVERY game, in every spread
   bucket. The owner's directive is about the cover probability as such, and
   the restricted variants (lane H's M1 on big spreads only) have already been
   measured; the honest test of the directive is the whole card.

### MP1b, the one predeclared sibling

Identical in every respect with `h = 5.0` -- double the primary band, the same
200-game floor, the same tilt, the same read. It is a band-width sensitivity,
not a search: no third width is computed or scored.

### How it is graded (declared before computing)

Replay the served S3 read first from the active model's matching opener
evaluation via `MarginModel.predict(center_offset=...)` with the walk-forward
offsets of `fit_home_side_offsets(prior_rows_before(...))`. Stop before any
candidate construction if a served probability differs by more than 1e-9. The
S3 point `p` is the location of the served predictive distribution,
`predicted_margin` (offset included) plus the fitted week's residual median --
the mean of the served `gaussian_median` read, and the same centre lane H
tilted its lattice onto.

Then, at the OPENER, versus S3: paired non-push accuracy points; week-blocked
bootstrap, 20,000 draws, seed 20260817, within-week correlation ZERO, never
estimated or padded; overall, per season, and per `nfl_ats.spread_regime.spread_bucket`
bucket; Brier and log loss; push rate at |line| = 3 and |line| = 7, predicted
versus realised; and THROUGH the played three-member card via
`overlay-composition` on a research `per_game` artifact carrying
`research_arm` and `active_model_id` `research_laneK_MP1` / `research_laneK_MP1b`.
Week 1 sides are read from the linked forecast's `predictions.csv` and its
`home_side_offset.json` sidecar, replayed to 1e-9; no forecast is regenerated.

Every cell is recorded through `nfl-ats weak-signals record`, family
`mod18_conditional_margin_v1`, names `mod18_conditional_margin_v1_mp1_*`,
classification `unresolved_below_power`.

### Decision rule, declared before the numbers are seen

Through-card `probability_positive` above 0.5 favours PLAYING MP1 in place of
S3; that reading is reported before any limitation. Standalone and loss reads
are reported alongside and do not veto the through-card read (AGENTS.md, "A
promotion bar is not a decision bar").

### Disclosure

The 1,537-game opener archive is the MINED archive lanes K and S were selected
on, and S3 is itself a post-hoc restriction fitted on it. This is a further
correlated look at the same window, not independent confirmation. The two arms
are correlated with each other and with lane H's M1.

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

Inferred decision from measured `artifacts/research/laneK/cells.json`: **keep S3
on the played card.** Through the played three-member card MP1 reads -0.399
accuracy points, 95% [-2.088, +1.318], `probability_positive` 0.311, and the
band sibling MP1b -1.397 [-3.000, +0.196], `probability_positive` 0.037. Both
sit below the declared 0.5 bar, so on expected value the card stays as it is.
Neither arm is closed and no cell carries a closing ground; every comparison
stays `unresolved_below_power`.

**What the lane did establish, measured:** the mass-preserving construction is
the first arm in this family whose discrete read reproduces the key-number
mass MOD-05 demanded. At |line| = 3 (199 games, realised push 10.050%) MP1
predicts **9.095%** and MP1b 7.791%, against S3's 3.499% -- the same 3.5%
lane H's M1 could not move (read: `docs/big_spread_lattice.md:55`) and better
than lane K's earlier K2 translation (4.1%). Across all 1,537 archive rows the
realised push share is 2.212% and MP1 predicts 2.156% (S3: 1.522%); the mean
modelled mass on a margin of exactly +/-3 is 14.15% against a realised 15.09%.
The push-calibration Brier improvement at |line| = 3 is +0.00365
[-0.00052, +0.00826] `probability_positive` 0.955 for MP1 and +0.00336
[+0.00016, +0.00691] `probability_positive` 0.980 for MP1b -- the sibling's
whole interval sits above zero.

So the premise is now demonstrated on the archive and the mapping still does
not win the card. The reading (inferred): getting the push and key-number mass
right is a calibration gain, not a side-selection gain, because the pick turns
on which side of the line the mass balance falls, and reallocating ~6 points
of probability onto the push atom moves both sides almost equally.

### Decision table (measured, `cells.json`, `push_calibration.json`, `week1.json`)

| Arm | Standalone % | Card % | Card delta pts [95%] | P+ | Brier | Log loss | Push at 3 % | Push at 7 % | Week 1 model / card changes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| S3 | 54.5576 | 55.8882 | reference | reference | 0.25158399 | 0.69662362 | 3.499 | 3.429 | reference |
| MP1 | 53.4265 | 55.4890 | -0.399 [-2.088, +1.318] | 0.31085 | 0.25150380 | 0.69646666 | **9.095** | 5.571 | 7 / 4 |
| MP1b | 53.0273 | 54.4910 | -1.397 [-3.000, +0.196] | 0.03665 | 0.25144629 | 0.69638051 | 7.791 | 5.294 | 5 / 3 |

Measured realised: push at |line| = 3 is 10.050% on 199 games; at |line| = 7 it
is 4.110% on 73 games. Standalone delta MP1 -1.131 [-3.098, +0.867]
`probability_positive` 0.126 (217 picks change); MP1b -1.530 [-3.405, +0.336]
`probability_positive` 0.051 (213 change). Card flips 146 / 143.

Measured loss improvements versus S3 (positive favours the candidate): MP1
Brier +0.0000802 [-0.0014398, +0.0015971] `probability_positive` 0.543, log
loss +0.0001570 [-0.0029329, +0.0032498] 0.542; MP1b Brier +0.0001377
[-0.0013654, +0.0016217] 0.574, log loss +0.0002431 [-0.0028353, +0.0032776]
0.565. The declared push-renormalised diagnostic (`cover / (cover + loss)`,
overall only) is slightly worse for both: MP1 Brier -0.0000921
`probability_positive` 0.458, MP1b -0.0000088 0.499.

### Reproduction gate (measured, `reproduction.json`)

Matching archive `artifacts/opener_evaluation/20260908T115957Z` on active model
`a4c757efd2525da6`, feature digest `457aafb7...`. Maximum S3 served-probability
replay gap **3.774758e-15**; maximum offset gap 6.661338e-16. 1,537 archive
rows, 1,503 non-push games over 107 weeks. Week-blocked bootstrap, 20,000
draws, seed 20260817, within-week correlation zero, never estimated or padded.
The weekly `gaussian_median` residual medians span [-2.2354, +0.7384] points,
which is the offset between the card's fair spread and the point MP1 tilts to.

### Per spread bucket (measured, `cells.json`)

| Bucket | n | S3 standalone / card % | MP1 standalone / card % | MP1 delta standalone; card (P+) | MP1b standalone / card % | MP1b delta standalone; card (P+) |
|---|---:|---|---|---|---|---|
| 0-3 | 552 | 56.16 / 57.43 | 54.89 / 57.61 | -1.268 [-4.283, +1.795] 0.191; **+0.181** [-2.348, +2.722] 0.526 | 53.80 / 55.98 | -2.355 [-4.954, +0.177] 0.027; -1.449 [-3.503, +0.539] 0.061 |
| 3.5-6.5 | 552 | 56.16 / 57.79 | 53.08 / 57.25 | -3.080 [-6.142, +0.000] 0.019; -0.543 [-3.002, +1.808] 0.298 | 54.17 / 57.43 | -1.993 [-5.043, +0.945] 0.085; -0.362 [-2.730, +1.880] 0.356 |
| 7 | 74 | 51.35 / 50.00 | **59.46** / 52.70 | **+8.108** [-2.500, +18.919] 0.920; +2.703 [-4.819, +10.256] 0.705 | **59.46** / 51.35 | +8.108 [-1.389, +18.182] 0.937; +1.351 [-5.714, +8.451] 0.573 |
| 7.5-10 | 194 | 51.55 / 51.55 | 51.55 / 48.97 | +0.000 [-5.914, +5.770] 0.474; -2.577 [-8.252, +2.874] 0.162 | 50.52 / 47.94 | -1.031 [-7.576, +5.781] 0.357; -3.608 [-9.605, +2.500] 0.109 |
| 10.5+ | 131 | 47.33 / 51.15 | 48.09 / 50.38 | +0.763 [-4.724, +6.061] 0.565; -0.763 [-5.882, +3.968] 0.328 | 45.04 / 47.33 | -2.290 [-8.661, +4.032] 0.207; -3.817 [-9.774, +2.206] 0.085 |

Measured reading: the gain is concentrated in the ONE bucket where the line
sits exactly on a key number. Bucket "7" is every game quoted at 7 or -7, the
atom this construction most changes, and both arms take it from 51.35% to
59.46% right standalone. The cost is in 3.5-6.5 (lines of 3.5-6.5 sit between
key numbers, where the smooth read was already calibrated) -- the same trade
lane K's K1 and lane T's T2 made, arrived at from a different direction.

Measured stated confidence versus realised accuracy (`scored.parquet`): MP1's
stated confidence is 55.7-56.7% in every bucket, essentially unchanged from
S3's 55.4-56.5%. Reweighting the atoms fixed the push mass without narrowing
the flat-confidence weak spot the Model page publishes.

### Per season (measured, `cells.json`), delta pts [95%], P+

| Season | n | MP1 standalone | MP1 card | MP1b standalone | MP1b card |
|---|---:|---|---|---|---|
| 2020 | 220 | +1.818 [-1.878, +5.357] 0.800 | +1.818 [-1.357, +4.505] 0.855 | +0.455 [-3.636, +4.505] 0.547 | +0.000 [-3.587, +3.365] 0.464 |
| 2021 | 236 | +2.119 [-1.235, +5.150] 0.884 | +1.271 [-1.688, +4.237] 0.765 | +1.695 [-3.333, +5.762] 0.749 | +0.424 [-4.202, +4.150] 0.569 |
| 2022 | 248 | -1.210 [-5.668, +3.292] 0.267 | -1.613 [-5.490, +2.041] 0.175 | -1.613 [-5.882, +2.682] 0.201 | -2.016 [-5.738, +1.210] 0.101 |
| 2023 | 266 | -6.015 [-11.111, -0.379] 0.017 | -2.632 [-7.576, +2.996] 0.146 | -6.391 [-10.448, -2.264] 0.001 | -4.135 [-8.088, +0.000] 0.017 |
| 2024 | 266 | +1.880 [-3.571, +7.380] 0.718 | +3.008 [-1.538, +7.605] 0.883 | +0.000 [-5.118, +4.887] 0.476 | +1.504 [-2.642, +5.455] 0.736 |
| 2025 | 267 | -4.494 [-8.425, -0.382] 0.011 | -3.745 [-7.037, -0.749] 0.005 | -2.622 [-6.593, +1.504] 0.089 | -3.745 [-6.767, -0.758] 0.003 |

Measured: 2020, 2021 and 2024 favour MP1 through the card and 2022, 2023, 2025
go against it, so the -0.399 overall is a mean of an unstable per-season split,
not a steady drift. Five cells have an interval wholly on one side of zero:
MP1 season_2023 standalone, MP1 season_2025 standalone and card, MP1b
season_2023 standalone and MP1b season_2025 card (all against the candidate),
plus MP1b's push-at-3 Brier (for the candidate). These are single-season slices
of a correlated family; the closing reclassification is left to the next look
rather than applied from a lane report, exactly as lane G's was.

### Construction diagnostics (measured, `scored.parquet`)

MP1's band stays at the declared 2.5 points for 83.9% of games and widens for
16.1% (maximum 12.0 points), with a median 384 prior games inside; MP1b stays
at 5.0 for 95.3% with a median 703 games. The solved tilt is small everywhere
-- mean |theta| 0.0121, maximum 0.0636 -- so the model's information is a gentle
reweighting, never a distortion of the empirical shape, and the bracket
[-1, +1] was never approached.

### Week 1 2026 (measured, `week1.json`); no forecast was regenerated

Linked forecast `artifacts/margin_predictions/2026-week-01-20260908T124514Z`;
its `home_side_offset.json` sidecar replays to 9.992007e-16 on all 16 games.

MP1 changes 7 model sides -- BUF at HOU, CHI at CAR, CLE at JAX, DAL at NYG,
NO at DET, NYJ at TEN, TB at CIN -- and 4 of them survive the three-member
overlay (BUF at HOU, CHI at CAR, NO at DET, TB at CIN). MP1b changes 5 model
sides (CHI at CAR, CLE at JAX, DAL at NYG, NO at DET, TB at CIN) and 3 through
the card. Every changed game is a near-coin-flip on both reads: S3's
probabilities on the seven MP1 changes run 0.484-0.542 and MP1's 0.422-0.507.

### Limitation, disclosed

Read (the predeclaration above): the 1,537-game archive is the mined one lanes
K and S were selected on and S3 is itself a post-hoc restriction fitted on it,
so this is a further correlated look at the same window. The two band widths are
correlated with each other and with lane H's M1. Nothing here is independent
confirmation, and the 2026 rows are the evidence that settles it at no window
cost.

## Registry and verification

Coordinator instruction, 2026-09-08 (read: the coordinator's mid-task message):
`registry/weak_signals.json` was corrupted by two lanes recording at the same
time, so lanes EMIT their recorder argv instead of running it. Measured root
cause, read `src/nfl_ats/io.py:46`: `atomic_json` writes every payload through
a FIXED sibling temp path (`<destination>.tmp`), so two processes writing the
same registry open the same file and interleave; the corrupted file on disk was
one complete 2,700-signal document followed by 530 orphan bytes of a longer one.

All 104 comparison cells are emitted as ready-to-run PowerShell in
`artifacts/research/laneK/record_commands.ps1` (family
`mod18_conditional_margin_v1`, names `mod18_conditional_margin_v1_mp1_*`, every
one `unresolved_below_power`, no `--closing-ground`, `--replace` set so a re-run
is idempotent); the full name list is in
`artifacts/research/laneK/registry_names.json`. Fifteen of the 104 are already
in the registry, written before the serialisation instruction arrived.

Measured evidence of the second half of the race, from this lane's own log:
`artifacts/research/laneK/commands.json` holds **36** recorder invocations that
returned 0, yet only **15** of those rows survive in `registry/weak_signals.json`
-- a concurrent lane rewrote the file from a snapshot loaded before they landed.
Recording is therefore not merely corruption-prone under concurrency; it
silently loses verdicts. Running the emitted commands serially restores all
104 regardless.

Measured checks: 43 tests passed in 23.18 seconds
(`tests/test_mass_preserving_lattice.py`, `tests/test_conditional_margin.py`,
`tests/test_experiment_registry.py`, which includes the artifact-provenance
scanner); scoped `ruff format --check` and `ruff check` pass.

```powershell
.\.tools\uv.exe run --no-sync ruff format --check scripts/mass_preserving_lattice_opener_eval.py tests/test_mass_preserving_lattice.py
.\.tools\uv.exe run --no-sync ruff check scripts/mass_preserving_lattice_opener_eval.py tests/test_mass_preserving_lattice.py
.\.tools\uv.exe run --no-sync pytest tests/test_mass_preserving_lattice.py tests/test_conditional_margin.py tests/test_experiment_registry.py -n 2 --basetemp <scratch>/laneK -q
```

Measured execution: `python scripts/mass_preserving_lattice_opener_eval.py
--stage {replay,map,score,week1,record}` under the locked uv environment. The
successful `overlay-composition` argv for both research arms is retained in
`artifacts/research/laneK/commands.json`; research `per_game` metadata carries
`research_arm` and `research_laneK_MP1` / `research_laneK_MP1b`, never the
active identity. No production path, forecast, card or published page was
touched.
