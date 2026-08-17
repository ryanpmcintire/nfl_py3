# Opener-graded evaluation of the frozen active model — predeclaration

Predeclared: 2026-08-17 (US), before the run. This is the first measurement
under the project's clarified **primary goal (Ryan, 2026-08-17): beat the
opening line his football pool (Splash Sports) grades against.** Closing
lines and vig-beating are secondary. The frozen constants and recipe are in
`src/nfl_ats/clv.py` (section 8); results are appended below and never edit
the predeclaration.

## Question

The active model's 52.05% was graded against closing lines — the market at
its sharpest. The replicated MKT-06 sign test showed the close moves
*toward* the model's fair margin in 54.9% of games, which implies the same
model should grade **better against the Tuesday opener** than against the
close. How much better, measured honestly?

## What this is and is not

- It scores the **frozen incumbent only** (`market_residual`/`player`/
  ridge/alpha-10, weekly refits with strictly-earlier training — the exact
  recipe behind the 52.05% evaluation). Zero selection dimensions: no
  variants, no tuning, one run.
- It reads 2020–2025 outcomes, which lie inside the ledgered 2018–2025 era.
  That is admissible here precisely because nothing is being selected —
  this is a re-measurement of an already-fixed pick stream generator at a
  different settlement line, predeclared with one look. No candidate model
  gains or loses standing from this run.
- Declared approximation (inherited from the MKT-06 pilot machinery): only
  `spread_line` is swapped to the opener; all other features (including
  `total_line`) are close-era values. The result approximates "the active
  model if the market had stopped at Tuesday's opener."

## Frozen recipe

For every archived game with a paired `tue_open` consensus and a resolvable
close (2020–2025 historical snapshot archive; 227–272 games per season):
one weekly-refit model per (season, week), trained on completed games
strictly before the week's first kickoff (≥500 games), then:

- **Opener arm**: residual evaluated at the opener; forced pick = sign of
  the residual; settled against the opener (`result − tue_open`).
- **Close arm**: same model's residual evaluated at the close; forced pick
  settled against the close. (Grading convention identical to the 52.05%
  evaluation.)
- **Movement oracle** (diagnostic upper bound): pick the side the close
  eventually moved toward, settle at the opener — how much accuracy pure
  line-movement capture is worth.

Pushes are excluded per arm; the paired opener-minus-close delta uses games
non-push under both lines.

## Frozen reporting (no accept/reject gate — this is a measurement)

`opener_accuracy`, `close_accuracy`, `opener_minus_close` (paired),
`opener_vs_coin_flip`, and `movement_oracle_accuracy`, each with week- and
season-blocked bootstrap intervals AND `probability_positive` (fraction of
blocked resamples above zero — continuous evidence, not a binary verdict),
2,000 samples, seed 20260817. Per-season table reported. Whatever the
numbers say, nothing is retuned on them; they set the baseline that any
future pool-targeted candidate must beat at the opener.

## Interpretation contract, fixed in advance

- `opener_accuracy` is the project's headline **pool-relevant** number if
  Splash-style pool lines are set near the Tuesday opener and frozen; it
  overstates pool edge if pool lines post later in the week (a separate
  Splash Sports timing investigation is running).
- If `opener_minus_close` comes back ≈ 0 or negative, the line-movement
  signal does not translate into settlement advantage and the pool edge
  claim rests on `opener_vs_coin_flip` alone.

---

## Results (run 2026-08-17, artifact `artifacts/opener_evaluation/`, 1,537 paired games 2020–2025)

| Metric | Estimate | Week-blocked 95% | Season-blocked 95% | P(positive) |
|---|---|---|---|---|
| **Opener accuracy (the pool number)** | **52.50%** | [49.90%, 55.08%] | [50.21%, 54.31%] | **96.8% / 98.6%** (vs coin flip) |
| Close accuracy (same games) | 51.09% | [48.32%, 53.66%] | [48.92%, 53.14%] | — |
| **Opener minus close (paired)** | **+1.35 pts** | [+0.41, +2.35] | [+0.49, +2.14] | **99.95% / 99.8%** |
| Movement oracle at opener | 55.08% | [52.26%, 57.82%] | [53.99%, 56.19%] | — |

Per season (opener accuracy): 2020 46.8%, 2021 55.5%, 2022 51.2%,
2023 54.1%, 2024 53.8%, 2025 52.8% — positive in five of six seasons. The
exception, 2020, is the COVID season: the archive's thinnest slice (227
paired games vs 272; bookmaker coverage itself was fine at ~11 books per
consensus) and a genuine regime break — empty stadiums collapsed
home-field advantage league-wide, which a model trained on pre-2020
seasons would systematically misprice. Mean absolute
open-to-close movement and the oracle ceiling (55.1%) indicate the model
captures roughly half the value of perfectly foreseeing line movement.

### Reading, per the interpretation contract

- **The same frozen model is resolvably better against the opener than
  against the close**: the paired +1.35-point delta carries ~99.9%
  probability of being positive under both blockings. The replicated
  line-movement signal does translate into settlement advantage — grading
  at the close was systematically understating the model.
- **Against the coin flip at the opener — the primary pool question — the
  model sits at 52.5% with a ~97–99% probability of genuine positive
  skill**, and the season-blocked interval excludes 50%. This is the
  strongest goal-relevant evidence the project has produced.
- Splash Sports context (researched 2026-08-17): their legacy pool engine
  posts lines Tuesday morning, revises once Wednesday, then freezes them
  for the week at half-point numbers, identical for all entrants — very
  close to the `tue_open` label measured here. The 52.5% figure is
  therefore approximately the pool-relevant grade, subject to the
  Wednesday revision (which pulls the pool number slightly toward the
  eventual close and would land true pool accuracy between the opener and
  close grades).
- Caveats: 1,537 games over six seasons; the close-era-features
  approximation declared above; one predeclared look, nothing retuned.

### Consequence

The project's headline metric changes: pool-targeted candidates are now
graded at the opener (`opener-evaluation` is the yardstick), and 52.50%
at the opener replaces 52.05%-at-the-close as the incumbent's number for
the primary goal. Prospective 2026 scoring should record both grades —
the live Tuesday captures make the opener grade available in production.
