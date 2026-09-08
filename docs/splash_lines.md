# The Splash Sports board: the lines the pool actually grades

Built 2026-09-08 (lane: Splash ingestion), wired into the feature build the
same day (lane: decision line). **The board's spread is now the decision line**
— the number the model sees, the number a pick is expressed against, and the
number `ats_margin` is computed from — for every week the board was captured.
nflverse `schedules.spread_line` and the Odds API consensus stay exactly where
they were: archive and research inputs, and the decision line for every week
with no capture (which is all of 2009–2025). See
[Wiring: how a capture reaches the card](#wiring-how-a-capture-reaches-the-card).

## What the pool grades on

The owner plays a Splash Sports NFL Pick'Em contest — "FTPL", channel
a private channel, 250 entries, format "Winner (ATS)". Every entry picks one side
of every game against a spread **printed on the Splash contest board**, and
that printed number is what settles the pick. It is fixed when the board's
spreads lock, **Tuesday at 12:00 PM Eastern** (the same lock
`nfl_ats.market_data.POOL_SPREAD_LOCK_ET` already names), and it does not move
afterwards no matter what the market does for the rest of the week.

Until 2026-09-08 this project had never captured that number.

## Why the two sources already in the repo are not it

| Source | What it is | Why it is not the graded line |
| --- | --- | --- |
| `schedules.spread_line` (nflverse) | The game's **closing** spread, published after the fact | It is a different number at a different moment — the market's last word, not the pool's frozen Tuesday word |
| Odds API consensus (`nfl_ats.market_data`) | A consensus across sportsbooks, captured point-in-time | Useful as research signal, but no sportsbook grades this pool. Splash sets its own number and it is frequently off the consensus |

Both are proxies, and measured on the first real capture the proxy error is
not small. On the 2026 Week 1 board (coordinator, 2026-09-08, hand-read):

- the **published card** sat on a different number than Splash for **4 of 16**
  games;
- the card the noon lock **regenerated** was on a different number for
  **8 of 16**.

A pick graded on a spread the model never saw is a pick made in the wrong
game. That is what this ingestion layer removes.

**Neither source is retired.** They are still the right answer to different
questions, and both remain wired exactly as they were:

- nflverse `schedules.spread_line` is the decision line for every week with no
  capture — the whole 2009–2025 archive, on which the opener evaluation and
  every weak-signal registry cell were scored. Those numbers must never move,
  and the override is built so they cannot (see the fail-closed rules below).
- The Odds API consensus stays the point-in-time market series: opener/close
  quotes, line movement, CLV, and every market-microstructure signal. It is a
  measurement of the market, which is a different thing from the number this
  pool settles on.

## Wiring: how a capture reaches the card

The override happens **once**, at the earliest point the table is assembled,
and everything downstream inherits it:

```
build-features
  load_snapshot(...)                       -> nflverse schedules
  splash_decision_line_overrides(data/)    -> one override per captured week
  apply_decision_lines(schedules, ...)     -> spread_line replaced, week by week
  build_game_features(...)                 -> game_features.parquet
     -> build-pbp-features                 -> game_features_pbp.parquet
        -> build-learned-availability-...  -> game_features_weak_stack.parquet
           -> margin-predict / the card
```

Because the swap lands on the **schedules frame**, before a single feature is
derived from it, no enrichment step and no research table needs a change of its
own: `game_features_pbp`, `game_features_player`, `game_features_weak_stack`
and everything cut from them all read one line per game.

| Piece | Where |
| --- | --- |
| The override + its refusals | `nfl_ats.features.apply_decision_lines` |
| Finding and validating captures on disk | `nfl_ats.pool_decision_lines` |
| Applying it in the build | `nfl_ats.cli_commands.features._cmd_build_features` |
| Provenance forward to the card | `nfl_ats.feature_manifest` → `nfl_ats.lineage` |

`build-features --splash-decision-lines off` turns the whole thing off and
rebuilds on nflverse's line everywhere, which is bit-for-bit the behaviour
before the board was ever captured.

### What it refuses to do

`apply_decision_lines` raises `DataContractError` rather than produce a table
nobody can trust:

1. **A partial week.** A capture that covers only some of a week's games would
   leave that week half on the pool's line and half on the nflverse close —
   the worst of both, and invisible once the card is built. Missing games, or
   a line for a game that is not on that week's schedule (a mis-read board, or
   a team-abbreviation mismatch), both raise.
2. **A completed game.** `ats_margin` is derived (`result - spread_line`), so
   moving the line under a played game rewrites the graded outcome the archive
   and every registry cell were scored on. Any covered row with a recorded
   `result` raises.
3. **A non-finite line.** A capture that cannot state a number is a defect.

Weeks with no capture are not touched at all. Measured 2026-09-08 on the full
4,902-row build: rebuilding with the Week 1 capture applied changed
**0 of 4,886 historical rows** (every column, exact comparison), and the only
column that differed anywhere was `spread_line`, on the 8 Week 1 games where
Splash and nflverse disagree.

### An open decision, for the week after Week 1 is played

Rule 2 above will fire the first time `build-features` runs with results
recorded for a week that still has a capture on disk — that is the point of
it, and it is a question a human should answer rather than a build should
guess:

- keep the archive on nflverse's number for played weeks (retire the capture
  from `data/splash/`, or rebuild with `--splash-decision-lines off`); or
- archive played weeks on the pool's own graded number, which is the honest
  record of how the pool actually settled and which would move `ats_margin`
  and `home_cover` for those rows.

Nothing here picks one. The build stops and says so.

### Provenance

The build manifest records which weeks were overridden, by which capture:

```json
"decision_lines": {
  "policy": "pool_capture",
  "builder_module": "nfl_ats.pool_decision_lines",
  "builder_version": "v1",
  "weeks": [{"season": 2026, "week": 1, "source": "splashsports.com",
             "capture_id": "2026_week01_20260908_noon",
             "captured_at_utc": "2026-09-08T12:45:00-04:00",
             "games": 16, "changed_games": 8, "changed_game_ids": ["..."]}]
}
```

That block rides the same ENG-22 inheritance chain the nflverse
`source_snapshot` does, so it survives every enrichment step and reaches the
card. On a card whose week is covered, `lineage.json`'s `market_line` record
then names the board capture instead of the nflverse snapshot:

```json
{"card_field": "market_line", "feature_family": "market",
 "source_snapshot": "2026_week01_20260908_noon",
 "source_captured_at": "2026-09-08T12:45:00-04:00",
 "builder_module": "nfl_ats.pool_decision_lines"}
```

A week with no capture keeps resolving to the nflverse snapshot exactly as
before — nothing may claim the pool's board as its source unless the board was
really read. The `model_input:*` families are unaffected either way: only the
graded line moved, not the rest of the table.

## Every line is a half point

Observed directly on all 16 Week 1 games: **every Splash number ends in .5.**
No exceptions, and the board has never been seen to post a whole number.

What follows from that, and why it matters more than it sounds:

1. **No push is possible in this pool.** Every game resolves win or loss;
   there is no third outcome to model, price, or display.
2. **Push probability is not a quantity this pool has.** Any push mass a
   model computes against the graded line is by construction zero, so a card
   that shows one is showing an artifact of a different line, not of this
   pool's line.
3. **A whole-number line is therefore a contract violation, not data.**
   `nfl_ats.splash_lines.validate_splash_capture` raises
   `DataContractError` on one, and says so in the message: a whole number
   would silently re-enable push machinery that cannot apply here. If Splash
   ever really does post a whole number, that is a change in the pool worth a
   human decision, and it should arrive as a loud failure rather than as a
   quietly different card.

This does **not** license a smooth cover-probability read. AGENTS.md's binding
rule stands unchanged: margins are discrete and multimodal, and a cover
probability is asked at one spread where the mass points decide it. The
half-point fact removes the push *outcome*; it says nothing about the shape of
the margin distribution, and the key-number lattice is still the read.

## Sign convention

`home_spread` positive means the **home** team is favored — the repository-wide
nflverse convention (`docs/bye_overvaluation_screen.md`). Splash prints the same
fact as two sides:

```
Patriots     NE +3.5
Seahawks     SEA -3.5
```

so `home_spread` is the home side's printed number negated: `+3.5` here,
meaning Seattle is favored by 3.5. `away_line` in a capture is the away side's
printed number, which under this convention is numerically the same value; the
validator enforces that the two agree, because they are one fact recorded
twice.

## Where captures live

```
data/splash/{season}_week{week:02d}_{YYYYMMDD}_{label}.json
```

for example `data/splash/2026_week01_20260908_noon.json`. `label` is `noon`
for a capture taken in the pool's noon lock hour and `HHMM` otherwise, so a
re-read later in the week is a new file rather than an overwrite. A verbatim
copy of the first capture is kept at
`tests/fixtures/splash/2026_week01_20260908_noon.json` so the contract tests
run in a clone with an empty `data/`.

Multiple captures per week are fine and are the point of the timestamp:
`load_splash_capture` returns the one with the latest `captured_at_et`, and
every sibling for that week is validated on the way past. A malformed older
capture is a defect to fix, not a file to skip.

## Taking next week's capture

Do this every Tuesday, as soon after 12:00 PM ET as possible — before the
board's numbers are of any interest to anything downstream, and never before
the lock (a pre-lock read is a different, moving number).

1. **Open the contest board** on splashsports.com — the FTPL contest, channel
   the owner's channel, and its NFL Pick'Em slate for the week. Confirm the header
   says the spreads are locked.
2. **Copy the board text.** Select the full list of matchups and paste it into
   a plain text file. The page renders one block per game and the parser is
   written against that shape:

   ```
   CHI   Sun, Sep 13 1:00 PM   CAR
   Winner (ATS)
   Bears      CHI -2.5
   Panthers   CAR +2.5
   ```

   Only the team abbreviations and the signed numbers are load-bearing. Team
   nicknames vary and are ignored, extra page chrome between blocks is ignored,
   and Splash's own abbreviations (`JAC`, `LAR`, `WSH`) are folded onto the
   nflverse identity (`JAX`, `LA`, `WAS`).
3. **Dry-run the parse first** and read the games back:

   ```powershell
   .\.tools\uv.exe run --no-sync python scripts/capture_splash_lines.py `
       --season 2026 --week 2 --text-file board.txt --dry
   ```

   Nothing is written. Check the game count against the slate and spot-check
   two or three lines against the page.
4. **Write it**, adding the identifiers the page shows:

   ```powershell
   .\.tools\uv.exe run --no-sync python scripts/capture_splash_lines.py `
       --season 2026 --week 2 --text-file board.txt `
       --contest-id contest_... --slate-id slate_... --entries 250
   ```

   The script refuses to write when a capture for that season/week already
   exists; pass `--replace` when a second read of the same week is deliberate.
5. **Optional hand-entered extras.** The tiebreaker prompt and the owner's own
   submitted entry are not on the spread board, so they are supplied as a small
   JSON file merged into the capture:

   ```json
   {
     "tiebreaker": {"game_id": "2026_02_...", "prompt": "Predict the total combined score", "submitted_value": 44},
     "submitted_entry": {"entry_id": "entry_...", "picks": ["..."], "best_pick": "..."}
   }
   ```

   passed with `--extra-json extras.json`.

Nothing here scrapes the site: a human or a browser agent supplies the text and
the script does the mechanical part. That is deliberate — the read is the step
that needs a pair of eyes, and everything after it is checkable.

## What the validator refuses

`validate_splash_capture` raises `nfl_ats.data.DataContractError` — the same
error type the nflverse input contracts raise — on any of:

- a capture with no games;
- a season or week that disagrees with what the caller asked for, or a week
  outside 1–22;
- a `game_id` that is not nflverse-shaped `{season}_{week:02d}_{AWAY}_{HOME}`,
  or that disagrees with the game's own `away`/`home` fields;
- duplicate `game_id` values, or a team playing itself;
- a team abbreviation the repository does not recognise (guessing one puts a
  line on the wrong game);
- **any line that is not a half point** — the rule above;
- `away_line` disagreeing with `home_spread`;
- a magnitude beyond 40 points, which is a mis-read of the page rather than a
  line.

The parser is equally unforgiving by design: a block that does not yield
exactly two option rows, two sides that are not exact opposites, an option row
naming a team that is not in the matchup header, a weekday that contradicts the
printed date, or an option row before any matchup header all raise. A silently
mis-parsed line is worse than no capture at all — no capture is visible, a
wrong one is not.

## Freshness

`capture_age(capture, as_of)` and `is_stale(capture, as_of)` let a caller
refuse a board from a previous week. The default horizon is one board cycle
(7 days), because the pool replaces the board every Tuesday at noon: a capture
older than that is a previous week's numbers whatever its filename claims.
`picks_locked(capture, as_of)` answers the separate question of whether the
pool's pick deadline has passed; the deadline recorded on a capture defaults to
the slate's Sunday 16:00 ET, matching the owner's standing
`min(own kickoff, Sunday 16:00 ET)` rule.

## Code

- `src/nfl_ats/splash_lines.py` — dataclasses, loader, validator, board parser,
  freshness helpers.
- `scripts/capture_splash_lines.py` — the capture CLI.
- `tests/test_splash_lines.py` — the contract tests.
- `src/nfl_ats/features.py` — `apply_decision_lines`, the one seam that swaps
  the line, and the three refusals.
- `src/nfl_ats/pool_decision_lines.py` — capture discovery and the manifest
  provenance block.
- `src/nfl_ats/feature_manifest.py` — `decision_lines_block` /
  `decision_line_week`, and the inheritance that carries the block to derived
  tables.
- `tests/test_features.py`, `tests/test_feature_manifest.py`,
  `tests/test_lineage.py` — the wiring tests.
