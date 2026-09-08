# The discrete push read (MOD-18 lane S, 2026-09-08)

## The contract

Read: `AGENTS.md`, "Football margins are multimodal, not Gaussian" -- "Any
served cover probability, push probability, flip line or alternative-line
answer is computed against the DISCRETE margin distribution conditional on
the line ... A smooth pooled-residual read ... may run only as a CHALLENGER
and must beat the discrete read on the opener grade through the played card
to be served."

Lane K (read: `docs/mass_preserving_lattice.md`, `artifacts/research/laneK/`)
measured both halves of that sentence on the same 1,537-game opener archive,
and they came out differently:

| Served quantity | Discrete (MP1) vs smooth (S3) | Lane K cell | Who serves it now |
|---|---|---|---|
| Push probability at \|line\| = 3 | MP1 predicts **9.095%** vs realised 10.050%; S3 says 3.499% | `push_calibration.json` (n = 199) | **discrete** |
| Push Brier at \|line\| = 3 | +0.00365 [-0.00052, +0.00826], `probability_positive` **0.955** (MP1); +0.00336 [+0.00016, +0.00691], 0.980 (MP1b) | `mp1_mp1_push_at_3_brier`, `mp1_mp1b_push_at_3_brier` | **discrete** |
| Push share, all 1,537 rows | realised 2.212%, MP1 2.156%, S3 1.522% | `docs/mass_preserving_lattice.md` | **discrete** |
| Side, through the played three-member card | MP1 -0.399 pts [-2.088, +1.318], `probability_positive` **0.311**; MP1b -1.397, 0.037 | `mp1_mp1_overall_card_2020_2025` | **smooth** (`gaussian_median`) |
| Side, standalone | MP1 -1.131 [-3.098, +0.867], 0.126 | `mp1_mp1_overall_standalone_2020_2025` | **smooth** |

So the served card is split by QUANTITY, on a named mechanism, not by a
threshold:

1. **The pick and its two-way `home_cover_probability` stay on the smooth
   `gaussian_median` read** (with the served home-side offset). That is the
   rule's own escape clause applied honestly: on the opener grade through the
   played card the smooth read beats the discrete one for the SIDE
   (`probability_positive` 0.689 in its favour), so it keeps serving the
   side. The pick does not change when the discrete read is switched on or
   off; a test pins that bit-for-bit.
2. **The push probability and every three-way (cover / push / loss) answer
   -- at the quoted line and at every alternative line -- come from the
   discrete read.** That is the card's `push_probability`,
   `home_cover_probability_excluding_push` and `home_loss_probability`
   columns, the `line_sweep.parquet` three-way split at every offset, the
   spread explorer's `spread_explorer_three_way`, and `scripts/cover_odds.py`.
3. **The smooth read's push / alternative-line numbers are the paired
   challenger**, recorded beside the served ones in the forecast sidecar
   `discrete_push_read.json` (`games[*].smooth` next to `games[*].served`,
   per game, plus `point`, `theta`, `band`, `band_games`, `atoms`,
   `key_mass_3`). It is not registered as a prospective-challenger ledger
   entry: the existing recorder pattern
   (`home_side_offset_incumbent_overlay.py`) scores PICKS, and the two reads
   here produce the same pick by construction, so a pick ledger would record
   nothing. The quantity to grade is push calibration, and the sidecar
   carries exactly the two numbers that grading needs.

### What the flip line does, and why

The board's "Flips at" cell is the first half-point line at which the
PLAYED pick switches sides. That is a side decision at a nearby line, and it
stays on the same smooth read that decides the side at the quoted line, so
the column can never disagree with the pick beside it. Inferred, not
measured: serving the discrete side read there would put the flip cell in
contradiction with the served pick on every game where the two reads
disagree at the quoted line -- lane K measured seven of sixteen Week 1
games (`week1.json`) -- and `board_content._flip_line` already refuses to
render a flip line whose source disagrees with the card. What the discrete
read adds at alternative lines is the push mass, which the sweep and the
explorer now carry at every line.

## The construction (`src/nfl_ats/mass_preserving_lattice.py`)

Lane K's MP1, moved verbatim so its artifacts replay bit-for-bit (pinned by
`tests/test_discrete_push_read.py::test_lane_k_research_numbers_replay_bit_for_bit_through_the_module`
against `artifacts/research/laneK/replay.parquet`; the research script
`scripts/mass_preserving_lattice_opener_eval.py` now imports from the module).

1. **Prior pool** (`prior_pool`): completed regular-season games with a line
   -- the archived Tuesday opener where the opener evaluation carries the
   game, else the feature table's `spread_line` -- and the integer final home
   margin.
2. **Walk-forward window** (`prior_pool_for_week`, `DiscretePushReader.for_week`):
   seasons at or after the target season minus five; gameday plus a one-day
   completion allowance strictly before the target week's earliest gameday;
   the whole target week, every later week and every target game id
   excluded. The same exclusions as `home_side_location.prior_games_for_week`.
3. **Band** (`band_read`): prior games with `|line_prior - L| <= 2.5`
   (`BAND_HALF_WIDTH`, the frozen lane-K bandwidth), widened in 0.5-point
   steps until `MIN_BAND_GAMES = 200` are inside or the band reaches 20.
4. **Atoms at absolute positions**: the share of band games whose final
   margin was exactly each integer. Nothing translated or smoothed.
5. **Tilt** (`tilted_atoms`): `p(m) ~ q(m) exp(theta (m - L))`, `theta` solved
   (Brent on [-1, +1]) so the tilted mean equals the served point --
   `predicted_margin` (home-side offset included) plus the residual median
   of the fitted week (`residual_location`, the `gaussian_median` centre lane
   K tilted to). Documented fallback: a target mean the bracket cannot reach
   clamps `theta` to the nearer end.
6. **Read**: cover = mass strictly above `L`, push = mass at `L` (exactly
   zero at a half-point line), loss = mass below.

Served flag: `DISCRETE_PUSH_READ_SERVED = True` in the module; policy id
`mass_preserving_lattice_mp1_v1`. With the flag `False`, `margin-predict`
serves today's smooth split again and writes no sidecar.

## The one per-game function (for the key-line pick override)

Coordinator scope note, 2026-09-08: lane T measured that serving the
discrete read for the PICK only on games quoted exactly on 3 or 7 gains
+0.20 pts through the played card (`probability_positive` 0.74,
`artifacts/research/laneT/cells.json`, `docs/key_line_lattice.md`). That
override is NOT in this patch (the pick is untouched here); the module
exposes the entry point it will call, and the served push path and the
line sweep already go through the same core (`band_read`), so the override
and the card's push chance can never disagree on the lattice:

```python
from nfl_ats.mass_preserving_lattice import DiscreteRead, discrete_read, discrete_reads_for_frame, prior_pool

def discrete_read(
    prior_stream: pd.DataFrame,      # the full pool from prior_pool(features, opener_lines)
    line: float,                     # home-oriented line the answer is asked at
    served_point: float,             # predicted_margin (offset included) + residual_location(...)
    *,
    season: int, week: int, cutoff: pd.Timestamp,   # the walk-forward window, applied INSIDE
    game_id: str | None = None,      # excluded from its own window
    band: float = BAND_HALF_WIDTH,   # 2.5
    prior: int = MIN_BAND_GAMES,     # 200-game floor the band widens to
) -> DiscreteRead                    # .cover .push .loss .atoms .theta .band .band_games .key_mass_3
                                     # .home_cover_probability = cover + push / 2 (the research decision number)
                                     # .three_way() -> (cover, push, loss)

def discrete_reads_for_frame(
    prior_stream: pd.DataFrame,
    frame: pd.DataFrame,             # game_id, season, week, gameday, <line_column>, <point_column>
    *,
    line_column: str = "spread_line",
    point_column: str = "point",
    band: float = BAND_HALF_WIDTH,
    prior: int = MIN_BAND_GAMES,
) -> pd.DataFrame                    # aligned to frame.index; cover, push, loss, home_cover_probability,
                                     # theta, band, band_games, atoms, key_mass_3, prior_rows
```

`tests/test_discrete_push_read.py::test_one_pure_per_game_function_backs_every_served_answer`
pins `discrete_read == DiscretePushReader.read == band_read` on the same
prior rows, and the walk-forward exclusions inside `discrete_read`.

## Where it is wired

- `nfl_ats.outcomes.score_outcome_week(discrete_read=..., discrete_read_log=...)`
  and `score_outcome_week_line_sweep(discrete_read=...)`: the served
  `market_residual` method's three-way split only; every other column and
  every companion method untouched.
- `nfl_ats.cli_commands.prediction`: `_served_discrete_push_read` fits the
  week's reader (never raises -- a failure degrades to the smooth split with
  the error in the sidecar and `metadata.discrete_push_read.error`), passes
  it to both scorers, writes `discrete_push_read.json` and the metadata
  summary, and labels the line sweep's `push_read`.
- `nfl_ats.spread_explorer.spread_explorer_three_way(..., discrete_read=...)`
  and `scripts/cover_odds.py`.
- `nfl_ats.card_explanation`: one plain sentence at a whole-number line,
  read from the card's own served `push_probability`.
- Not changed: `nfl_ats.tiebreaker` reads its push probability off MOD-05's
  joint score lattice (`score_lattice.ScoreLattice.push_probability`), a
  different discrete object; unifying the two is MOD-17's open question.
  `nfl_ats.lines.rescore_at_lines` (the ad hoc rescoring at supplied lines)
  still uses `MarginModel.predict`'s smooth split.

## What a reader sees change

- On the deep dive's "Why this pick" paragraph, whole-number lines gain one
  sentence, e.g. "The line sits right on 3, a number games land on a lot:
  about 9 in 100 games like this finish exactly there, a push, and that
  chance is counted here." Half-point lines say nothing about pushes.
- The pool card's push column (`push_probability_at_pick` on the
  at-lines card) and the sweep behind the cover chart carry the discrete
  push mass: roughly 9-10 in 100 on a line of 3 instead of 3-4.
- Nothing else moves: not the pick, not the cover probability, not the
  flip line.

## Verification

```powershell
.\.tools\uv.exe run --no-sync pytest tests/test_discrete_push_read.py tests/test_mass_preserving_lattice.py tests/test_outcomes.py tests/test_spread_explorer.py tests/test_margin.py tests/test_card_explanation.py tests/test_board_content.py tests/test_board_flip_line.py tests/test_public_board.py tests/test_experiment_registry.py tests/test_cli.py -n 2
.\.tools\uv.exe run --no-sync ruff format --check .
.\.tools\uv.exe run --no-sync ruff check .
.\.tools\uv.exe run --no-sync mypy src
```

## Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close
a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is `unresolved_below_power`. Nothing in this document closes anything: the
side comparison stays `unresolved_below_power` in lane K's registry rows,
and the served split is an expected-value decision per quantity.
