# SIM-04 / SIM-05 build plan: full play-by-play simulator and counterfactuals

Written 2026-09-25. Planning only — no model was trained and no `src/` code
changed while writing this document. All figures below are **measured** this
session against the current repository state.

## Plain English

This plan is for a program that plays out a simulated football game, drive by
drive and play by play, thousands of times per matchup, so we get a full
distribution of final margins instead of one number. The case for building it
is **not** "it will pick more winners" — a 2026-08-17 audit
(`docs/play_level_audit.md`, ROADMAP MOD-09) already measured that play-level
rows are nearly independent (ICC 0.013) and a sequence model over them does not
beat the game-level summary features the served model already uses. This plan
does not re-litigate that; it targets three different jobs a simulator is
actually good for: (1) generating the lumpy, multi-peaked shape real NFL
margins have around 3, 7, 10, 14, and 17 points from the scoring mechanics
themselves rather than assuming it, so a game or a slate gets a real
probability of landing exactly on the number the pool cares about; (2) answering
"what if the starting QB is out" or "what if this team always goes for it on
4th down" without re-fitting anything; and (3) a dashboard page people can look
at. If it ever produces a probability worth using in a pick, it goes in as one
more fitted input to the single calibrated probability, the same way the other
signals do — it never gets its own flip rule.

## What exists today (measured)

- **Served discrete margin read.** `src/nfl_ats/mass_preserving_lattice.py`
  defines `DiscretePushReader` (build from prior lines/margins via
  `DiscretePushReader.for_week`) and `serve_discrete_three_way`, which reads a
  `(line, point)` pair against a band of historical margins and returns
  cover/push/loss mass preserving the discrete lattice
  (`DISCRETE_PUSH_READ_POLICY = "mass_preserving_lattice_mp1_v2"`, key numbers
  `(3, 7, 10, 14)`, `MIN_BAND_GAMES = 200`). This is the baseline any simulator
  output must beat, not a Gaussian.
- **NFL play-by-play capture.** `data/pbp/raw/20260817T184927Z/`, 17 season
  partitions (2009–2025), one season sampled at 48,771 rows (2025). Columns are
  a fixed allowlist, `PBP_SNAPSHOT_COLUMNS` in `src/nfl_ats/pbp.py` (45 columns
  total, extending `PBP_REQUIRED_COLUMNS`). It has `down`, `ydstogo`,
  `yardline_100`, `qtr`, `game_seconds_remaining`, `play_type`, `yards_gained`,
  `interception`, `fumble_lost`, `touchdown`, `first_down`, `penalty`,
  `penalty_yards`, `score_differential`, `posteam_score`/`posteam_score_post`,
  `epa`/`wp`/`cpoe`/`xpass`, and `fixed_drive`/`fixed_drive_result`. It is
  missing **timeouts remaining (either side), penalty team/type, personnel or
  formation groupings, and non-passer player IDs** — all present upstream in
  the `nflreadpy` source this project already fetches from
  (`nflverse_current_season.py`, `pbp.py`), just not carried into the allowlist.
  A capture-column widen (Unit 2 below) is a config change, not a new data
  source.
- **College play data.** `data/cfb/pbp/raw/`, 44 parquet files across capture
  runs, ~1.3 GB. Useful as a large auxiliary/positive-control corpus (more
  games, different scoring-rate mix), not as a source of NFL rotation-registry
  looks.
- **Aggregated features.** `data/processed/game_features_pbp.parquet`, 4,902
  rows (games, seasons 2009–2026) x 201 columns. This is the already-rejected
  summary/drive-layer bundle from the MOD-09 audit (screened nested 2018–2025:
  50.36% ATS no edge; drive layer worsened Brier 0.250803 → 0.250891). The
  simulator must not re-propose this bundle as a side-pick input; it consumes
  the underlying play-level mechanics for a different output (a distribution),
  not this table.
- **Player/participation data.**
  `data/processed/game_features_player_participation.parquet` and
  `data/processed/player_participation_ratings.parquet` exist for
  pregame-available player-strength conditioning.

## Architecture

**Game state.** A tuple `(offense, defense, quarter, game_seconds_remaining,
down, distance, yardline_100, score_off, score_def, timeouts_off,
timeouts_def, half)`. Advances one play at a time; a drive ends on score,
turnover, turnover-on-downs, or punt; a game ends at `game_seconds_remaining ==
0` after four quarters (plus a simple overtime rule, coin-flip start, sudden
death on TD / first-possession FG per current rule since 2012, split by season
since the rule changed).

**Per-play outcome submodel.** Conditioned only on the state above plus
**pregame** team-strength features already computed elsewhere in the pipeline
(`off_epa_per_play`, `def_epa_per_play`, `off_pass_rate`, `off_sack_rate`, and
the equivalent defense-side columns, all as of the Tuesday-opener cutoff before
that week's games — the same leakage-safe features the served model reads) and,
where available, pregame player-participation ratings (starting QB out is a
pregame-known state, not a leak). Two output heads: (a) categorical play type
(run / short pass / deep pass / sack / scramble / punt / FG attempt / kneel /
spike), (b) yards gained conditional on play type, both fit as multinomial or
gradient-boosted classifiers over the widened play-level table.

**Clock submodel.** Play duration and clock runoff by play type and
late-game/two-minute state; conditioned on pregame pace priors
(`off_plays_per_game` or equivalent). Governs how many plays a drive and a game
get, which is part of what produces realistic final-margin variance.

**Penalty submodel.** Penalty occurrence and yardage by situation and team
pregame penalty-rate priors, applied as a state perturbation after a play
outcome is drawn (accept/decline logic follows the standard rule: whichever
side the penalty favors chooses).

**Turnover submodel.** Interception and fumble-lost rate conditioned on
situation, pressure proxy (sack rate), and a return-yardage tail (pick-six /
scoop-and-score) drawn from the historical return-yardage distribution
conditional on turnover type and field position.

**Fourth-down policy submodel.** Go/kick/punt as a function of
score-differential, field position, quarter, and time remaining, fit to
observed coach behavior (empirical logistic on situation only — this is a
policy fit to what coaches do, not a team rating, so it carries no leakage
risk). SIM-05 counterfactuals swap this function out (e.g., an "always go for
it inside opponent territory on 4th-and-3-or-less" policy) without touching any
other submodel.

**Sampling.** For a matchup, draw pregame team/player ratings once (fixed for
the whole slate of N simulated games), then sample N independent games
(N = 10,000 default) by chaining the submodels play by play from opening
kickoff. Output is the empirical distribution of final home-minus-away margin,
which is read as a discrete histogram, not smoothed — the multimodal shape is
the point.

## Leakage rules

- Every conditioning feature (team strength, pace, penalty rate, player
  participation, 4th-down policy priors) is computed strictly before that
  week's Tuesday-opener cutoff, using the same warm-up/leakage guards already
  enforced in the feature builder (`nfl_ats.pbp`, `nfl_ats.modeling`). No
  submodel may condition on same-game or same-week outcomes.
- The per-play, clock, penalty, and turnover submodels are fit on **prior
  seasons only** relative to any season they are used to simulate — chronological
  fit, never in-sample on the games being evaluated.
- Player-participation conditioning uses only designations known before kickoff
  (the existing inactives/participation capture), never final-box-score
  participation.

## Chronological evaluation protocol (predeclared before any fit runs)

- **Comparison baseline:** the served `DiscretePushReader` /
  `serve_discrete_three_way` read at the Tuesday-opener line, exactly as the
  card is graded today.
- **Metric:** log loss (and Brier) of the simulator's discrete
  cover/push/loss read at the opener line, against the same metric for the
  served discrete read, on held-out games only.
- **Method:** leave-one-season-out. Fit every submodel on all seasons except
  the held-out one, simulate that season's games, score at the opener.
  Repeat per season, report the per-season deltas plus the pooled delta with a
  week-blocked confidence interval and `probability_positive`, per AGENTS.md
  ("An interval crossing zero is not grounds for rejection" — a crossing
  interval here is `unresolved_below_power`, not a rejection, unless the whole
  interval sits on the wrong side).
- **Acceptance bar:** the simulator read is a candidate discrete-margin
  challenger, run through the normal registry (`nfl-ats weak-signals record` /
  rotation `record-look`), not auto-served. Passing 0.5 log-loss improvement is
  not sufficient to serve it and failing 0.90/0.95 is not grounds to close it,
  per the promotion-bar rule in AGENTS.md — closure requires a refuted
  mechanism or a proven positive control, same as any other signal.
- **Positive control:** before grading the fitted version, run the
  scoring-mechanics-only baseline (Unit 1, no team conditioning) through the
  same LOSO protocol against a synthetic ground truth built by resampling
  actual historical margins — this bounds what the harness can detect and
  keeps a null result honest instead of silently closing the family.
- **How it would enter a pick:** if and only if a simulator-derived probability
  clears registry review, it is added as one more standardized term in the
  fitted design (`nfl_ats.pick_probability_fit`) and refit LOSO alongside the
  existing four served terms — never as a standalone rule and never able to
  flip a side by itself, per "One calibrated probability decides every pick."

## Build units

Each unit is sized for one subagent (a few dozen tool calls). Units write
prototype code under `scripts/` or `tests/scratch/`, not `src/`, until a unit
clears its own verification — matching how other challengers in this repo stay
out of `src/` until registered.

1. **Drive-outcome key-number baseline (ships first, measurable).** No team
   conditioning at all: from the existing 45-column NFL pbp snapshot
   (2009–2025), build the empirical drive-ending distribution (TD / FG / punt
   / turnover / turnover-on-downs / end-of-half / safety) conditioned only on
   starting field position and down-and-distance bucket. Chain drives (using
   the empirical possession-count and clock-runoff distribution, no per-play
   loop needed yet) into full games between two league-average teams, sample
   N=10,000 games, and compare the resulting margin histogram's mass at 3, 7,
   10, 14, 17 against the actual historical margin histogram (same source
   `game_features_pbp.parquet` margins). Input:
   `data/pbp/raw/20260817T184927Z/`. Output: a histogram comparison artifact
   under `tests/scratch/` or a scratch registry experiment JSON; a printed
   table of simulated vs. historical key-number mass. Verification: run the
   script once, report the mass table — no registry look yet since there is no
   comparison to a served probability at this stage, just a mechanism check.
2. **Capture-column widen.** Extend `PBP_SNAPSHOT_COLUMNS` in `src/nfl_ats/pbp.py`
   to add timeouts-remaining (both sides), penalty team/type, `play_type_nfl`,
   personnel/formation if present upstream, and rusher/receiver/fumble player
   IDs. Re-run the capture for 2009–2025, diff row counts and manifest hash
   against the existing `20260817T184927Z` capture to confirm no game loss.
   Output: new capture directory + manifest. Verification:
   `uv run --no-sync python -m nfl_ats.pbp` capture entrypoint (or the CLI
   command that wraps it) plus a row-count diff against the prior capture.
3. **Per-play outcome submodel.** Fit the play-type/yards-gained model on the
   widened 2009–2024 play table, pregame team-strength features only. Output:
   a fitted model artifact under `artifacts/` (research artifact, not `src/`)
   and a chronological held-out log-loss number for play-type classification.
4. **Clock submodel.** Fit play-duration/clock-runoff by play type and
   late-game state; verify by comparing simulated plays-per-game and
   drives-per-game distributions against the historical ones.
5. **Penalty and turnover submodels.** Fit penalty rate/type/yardage and
   turnover rate/return-yardage; verify by comparing simulated
   penalties-per-game and turnovers-per-game distributions against historical.
6. **Fourth-down policy submodel.** Fit the observed go/kick/punt function;
   verify by comparing simulated go-for-it rate by situation bucket against
   the historical rate.
7. **Full game assembly.** Chain units 3–6 into a play-by-play loop, sample
   N=10,000 games per matchup for a full slate, produce the discrete margin
   distribution per game. Verification: rerun the Unit 1 key-number mass check
   with the fitted submodels in place of the unconditional baseline, confirm
   it does not regress the mass reproduction.
8. **Chronological evaluation.** Run the predeclared LOSO protocol above
   against `DiscretePushReader`, report per-season deltas, pooled delta,
   `probability_positive`, and the positive-control bound. Record every
   unresolved cell with `nfl-ats weak-signals record` before any write-up
   calls a season settled.
9. **SIM-05 counterfactuals.** Reuse the Unit 7 assembly with one submodel
   swapped (player-participation input for an injury counterfactual, a wind/
   precipitation-conditioned play-mix for a weather counterfactual, or an
   alternate Unit 6 policy for a 4th-down counterfactual), rerun N games, and
   report the shift in simulated margin distribution and cover probability at
   the actual line. Every counterfactual output is labeled inferred/simulated
   and is a research and dashboard artifact — it never re-picks a served card
   game on its own, per "One calibrated probability decides every pick."
10. **Dashboard deliverable.** A game-page simulated-margin histogram (line
    marked, key numbers labeled) fed by Unit 7's output for that week's slate.
    Build this only after Unit 8's registry look completes, so the page's
    framing (illustrative distribution vs. a number that changed a pick)
    is accurate either way.

## Compute estimates

- Per-play submodel fit (Unit 3): ~700k historical play rows (17 seasons x
  ~41k plays/season, consistent with the 781,712-row figure in
  `docs/play_level_audit.md`), a few dozen features, gradient-boosted or
  multinomial — minutes on CPU, not GPU-bound.
- Simulation throughput: a vectorized per-play sampler should reach roughly
  50k–200k simulated plays/second in numpy on a laptop CPU. A full week's
  slate (16 games x 10,000 sims x ~155 plays/game ≈ 24.8M plays) is on the
  order of two to five minutes at that rate. A naive per-play Python loop
  would be far slower and should be avoided; batch the play-outcome draw
  across all in-flight simulated games at a given state instead of looping
  game by game.
- No GPU or distributed compute is required for any unit in this plan.

## Risks and what would change the plan

- **The widened capture (Unit 2) is not actually available upstream** for some
  older seasons (timeouts/personnel coverage often starts later than yardage
  in nflverse's own history) — if so, submodels 4–6 get seasons.min() moved
  forward and the LOSO evaluation window shrinks; state that explicitly rather
  than back-filling with an assumption.
- **Unit 1 fails to reproduce key-number mass** even without team
  conditioning — that would mean the discrete lattice's shape comes from
  something the drive-chain architecture does not capture (e.g., score-driven
  end-game behavior, kicker-distance-specific FG rates) and the architecture
  needs a rethink before Units 3–7 are worth building; this is the single
  biggest go/no-go gate and is why it ships first.
- **The per-play submodel does not clear a positive control** — bounded by
  the ICC 0.013 finding, it is plausible the team-conditioning layer adds
  nothing beyond what the summary features already give the served model; if
  the positive control (Unit 8) cannot detect an injected effect of realistic
  size, later units are `unresolved_below_power`, not evidence to keep
  iterating past the predeclared bar.
- **Compute or wall-clock cost exceeds the estimate above** by an order of
  magnitude — if a naive implementation is too slow to run weekly, vectorize
  before adding submodel complexity, not after.
- **A submodel needs a data source this repo does not have** (e.g.,
  play-by-play personnel groupings never existed for a given era) — narrow
  that submodel's scope (drop personnel conditioning, keep play type) rather
  than importing an unaudited third-party dataset.
