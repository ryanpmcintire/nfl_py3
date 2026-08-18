# Execution specs for the next sessions (written by Fable, 2026-08-17)

Every outstanding queue item, specified so the executing session makes NO
design decisions. Each spec lists context, exact deliverables, acceptance
commands with expected outputs, and traps. Follow the specs in order:
SPEC-1 unblocks SPEC-4/5; SPEC-3 has a hard calendar deadline.

If a spec turns out to be wrong or ambiguous IN A WAY THAT CHANGES A
METHODOLOGY DECISION (what counts as a look, which window, what gate),
STOP, record the ambiguity in the session notes, and leave the decision to
the owner. Mechanical ambiguities (a renamed column, a moved file) you
resolve yourself and note in the commit message.

## Execution status (updated 2026-08-17, end of the Opus execution session)

| Spec | Status |
|---|---|
| SPEC-1 rotation registry | ✅ done; rule 9 added; two looks recorded through it |
| SPEC-2 postseason snapshots | ✅ done (4 ingests, manifests verified) |
| SPEC-3 A `weekly-run` | ✅ done (`weekly.py`, CLI, tests) |
| SPEC-3 B rehearsal | ✅ **done** — ran end to end, evaluation reproduced `0.5204819277` exactly with the bias columns present, card republished, Pages redeployed; timings in `docs/ops_runbook.md`. Closes the open verification from `opus_session_blockers.md` § 5 |
| SPEC-4 step 1 bias features | ✅ done (9 columns, REG bit-identity proven, resync run) |
| SPEC-4 steps 2-3 | ✅ **done** — `weak_stack` profile + candidate table + the one look on [2020, 2021]: +1.97 points, `probability_positive` 0.8745, verdict `unresolved` (`docs/mod07_stack.md`) |
| SPEC-5 screen | ✅ **done** — [2013, 2015] spent; `sweep_robustness` cleared the 0.75 gate at 0.7955, the other two signals closed (`docs/best_pick_ranker.md`) |
| SPEC-5 opener confirmation | ✅ done running — [2020, 2021] spent on the `player` profile (the deployed one; rationale in `docs/best_pick_ranker.md`). Recorded Top-1 60.0% vs 51.32%, +8.68 points, `probability_positive` 0.865 was originally read as clearing the predeclared 0.75 gate at verdict `confirmed`. **Corrected 2026-08-18 (see `docs/best_pick_ranker.md`):** `sweep_robustness` **tied in 24 of 35 weeks**, so on most weeks it ranked nothing and the recorded top-1 pick was actually chosen by team-name alphabetical order, not the ranker. The tie-break-agnostic delta is **+0.92 points**, honest `probability_positive` **0.536-0.554** — never near the 0.75 gate. The recorded 60.0% sits at the **95.4th percentile** of its own tie-break-luck distribution (flipping 3 of 21 correct picks turns +8.68 into +0.11). Registry verdict is now `unresolved`, not `confirmed`. **The play decision is unchanged**: `sweep_robustness` still picks the Best Pick in 2026 — pool play only, no activation — because its honest tie-agnostic edge is still positive on two independent windows and both alternatives (`calibrated_probability`, `key_number_distance`) measured negative; forced picks mean "unresolved-but-positive-and-least-bad" still plays. Only the *confidence* claimed for the pick changes, not the pick. |

Every spec in this document was executed by 2026-08-17. That does not mean
every conclusion recorded here still holds — see the SPEC-5 opener
confirmation correction immediately above, added 2026-08-18. "Nothing
outstanding" describes execution status, not correctness of every number.

## Global invariants (violating any of these is a failed session)

- **The frozen model's inputs are sacred.** Given the same snapshots,
  regular-season rows of every feature table must be byte-identical before
  and after your change unless a spec explicitly says otherwise. New
  COLUMNS are fine; changed VALUES in existing columns are not. Verify with
  the pattern in `docs/postseason_support.md` (build both ways,
  `assert_frame_equal(check_exact=True)` on the allowlisted columns).
- **A look is one look.** Anything scored on a registry window gets
  recorded (SPEC-1's `record_look`) whatever the result. Never re-score a
  spent window "to check".
- **Training is forward-chained everywhere.** Strictly earlier gamedays
  only. If you find yourself writing a random split, stop.
- **Continuous evidence.** Report `probability_positive` (week-blocked
  bootstrap fraction favoring the candidate), never bare pass/fail.
- **Public pages guardrail** (docs are GitHub Pages): the only market
  number on any public page is the weekly card's single consensus
  `spread_line`. Never per-book data, never archive-derived numbers
  (pending MKT-09). Disclaimers stay on every page.
  `tests/test_public_board.py` enforces; keep it green.
- **Negative results are recorded, not deleted.**

## Environment quirks (Windows, PS 5.1)

- Run Python as `F:\Repos\nfl_py3\.venv\Scripts\python.exe` (or
  `python -m nfl_ats`). Never plain `uv run` — the owner's dashboard can
  lock `nfl-ats.exe`; `uv run --no-sync` is acceptable.
- pytest needs `-p no:cacheprovider --basetemp=<scratchpad>\...`.
- Git commit messages via `git commit -F <file>` (PowerShell mangles
  quotes in inline messages). Multi-line strings to native commands via
  single-quoted here-strings, closing `'@` at column 0.
- A pre-commit hook refreshes HANDOFF.md; a push guard blocks master when
  the tracked publication does not match the active model — the fix is
  `python -m nfl_ats publish-predictions`, then commit, then push.
- Never force-remove git worktrees containing junctions.

## Streamlit/dashboard traps (verified live this session — in code comments too)

- `st.html` strips ALL `<svg>`; charts are pure HTML/CSS (clip-path).
- `st.html` keeps only ONE `<script>` element per block; extra JS rides in
  `theme.theme_sync_script(extra_js=...)`.
- Any tag-like sequence (`<` + letter) anywhere in a script's SOURCE —
  strings and comments included — makes the sanitizer drop that script.
- Wire events by document-level delegation; block scripts execute while
  content is detached.
- Only write `data-theme` on change; unconditional writes pin the renderer.

---

## SPEC-1 — Rotation registry implementation

Rules and rationale: `docs/rotation_registry.md` (binding; read first).
This spec is the implementation contract.

**New files:** `src/nfl_ats/rotation.py`, `registry/rotation_registry.json`
(git-tracked), `tests/test_rotation.py`. **Edit:** `cli.py` (new
subcommand `rotation`), ROADMAP (new row `RWB-17 | 🚧 | Rotation registry`).

**Ledger schema** (`registry/rotation_registry.json`), exactly:

```json
{
  "version": 1,
  "notes": [
    "2018-2025 carries a ~130-150-look multiplicity ledger (ROADMAP RWB-16); windows intersecting it require acknowledges_mined_2018_2025 at declaration and a stated discount in any write-up.",
    "Opener-graded pool is 2020-2025 only (paired Tuesday-opener archive coverage)."
  ],
  "families": {
    "pbp_drive_bundle": {
      "declared_at": "2026-08-13",
      "description": "Raw-PBP/drive aggregate bundle (retroactive entry; see ROADMAP RWB-16).",
      "grade": "nflverse_spread",
      "status": "closed_negative",
      "inherits": [],
      "acknowledges_mined_2018_2025": false,
      "windows": [
        {"seasons": [2013, 2017], "state": "spent",
         "assigned_at": "2026-08-13", "spent_at": "2026-08-13",
         "artifact": "ROADMAP.md#RWB-16",
         "verdict": "closed_negative",
         "probability_positive": null,
         "notes": "-0.08 pts vs base on 1,247 games; margin error resolved worse."}
      ]
    },
    "player_qb_continuity": {
      "declared_at": "2026-08-13",
      "description": "QB + lineup continuity player family (retroactive entry).",
      "grade": "nflverse_spread",
      "status": "closed_negative",
      "inherits": [],
      "acknowledges_mined_2018_2025": false,
      "windows": [
        {"seasons": [2014, 2017], "state": "spent",
         "assigned_at": "2026-08-13", "spent_at": "2026-08-13",
         "artifact": "ROADMAP.md#RWB-16",
         "verdict": "closed_negative",
         "probability_positive": null,
         "notes": "+0.00 pts on 997 games; arms split 88-88; diagnostics worse."}
      ]
    },
    "cfb_role_continuity": {
      "declared_at": "2026-08-17",
      "description": "CFB role-continuity family; closed at the CFB benchmark, no NFL window ever assigned (docs/cfb_role_features.md).",
      "grade": "nflverse_spread",
      "status": "closed_negative",
      "inherits": [],
      "acknowledges_mined_2018_2025": false,
      "windows": []
    }
  }
}
```

**`rotation.py` public API** (exact signatures; frozen dataclasses mirroring
the schema):

```python
GRADE_POOLS: dict[str, tuple[int, int]] = {
    "opener": (2020, 2025),
    "close": (2009, 2025),
    "nflverse_spread": (2009, 2025),
}
DEFAULT_WINDOW_SIZE = {"opener": 2, "close": 3, "nflverse_spread": 3}
MINED_SEASONS = (2018, 2025)

def load_registry(path: Path | None = None) -> Registry            # validates; raises RegistryError
def save_registry(registry: Registry, path: Path | None = None)    # atomic_json; recomputes season_usage
def declare_family(registry, name, *, description, grade,
                   inherits=(), acknowledges_mined_2018_2025=False) -> Registry
def assign_window(registry, family, *, size=None) -> Registry      # earliest-eligible rule from the design doc
def confirmation_split(features: pd.DataFrame, registry, family)
        -> tuple[pd.DataFrame, pd.DataFrame]
    # window = REG rows (regular_season_rows) of exactly the assigned seasons;
    # training = REG rows with gameday strictly before the window's first
    # gameday and non-null result. Raises RegistryError when: no assigned
    # window / window spent / any window season missing from features.
def record_look(registry, family, *, artifact, verdict,
                probability_positive, notes="") -> Registry
```

Validation rules on load (each raises `RegistryError`): unknown top-level
or family fields; a family with >1 `assigned` window; `spent` window
missing artifact/verdict; window seasons outside the family's grade pool;
window intersecting `MINED_SEASONS` on a family without the acknowledgment
flag; overlapping windows within one family+inherits chain.

**CLI** (`nfl-ats rotation ...`):
- `rotation status` — prints every family (name, grade, status, windows
  with state), remaining unspent capacity per grade pool (count of
  eligible default-size blocks not held/spent by ANY family — plus the
  per-family view), and the season-usage table. JSON output like other
  commands.
- `rotation declare --name X --description "..." --grade opener
  [--inherits a,b] [--acknowledge-mined]`
- `rotation assign --name X [--size N]`
- `rotation record --name X --artifact PATH --verdict
  {confirmed,closed_negative,unresolved} --probability-positive F
  [--notes "..."]`

**Tests** (follow house style; ~12): schema round-trip; each validation
rule raises; declare/assign earliest-eligible determinism (opener family
gets [2020,2021]; a second opener family ALSO gets [2020,2021] — per-family
retirement; a family inheriting `player_qb_continuity` skips 2014-2017);
assign refuses a second unspent window; `confirmation_split` forward-chain
boundary exact (training max gameday < window min gameday), REG-only, and
each raise condition; `record_look` marks spent and a re-split raises;
mined-season acknowledgment enforcement.

**Acceptance:** `python -m nfl_ats rotation status` prints the three seeded
families and "opener pool: 3 windows unspent"; full pytest green; ruff and
mypy clean; ledger committed.

> **ADDENDUM 2026-08-17 (Fable):** rule 9 (warm-up eligibility) was added
> to `docs/rotation_registry.md` and `rotation.py` after the SPEC-5
> incident: no window starts before 2013 (`MIN_ELIGIBLE_START_SEASON`);
> the capacity partition starts there too, and `confirmation_split` fails
> closed on an empty training frame.

---

## SPEC-2 — Postseason-inclusive snapshot fetch (before January; ~30 min)

Code shipped 2026-08-17 (`--include-postseason` on the ingest commands;
loaders re-filter REG by default so NOTHING downstream changes).

1. `python -m nfl_ats player-ingest --include-postseason` (seasons default;
   if the command needs explicit seasons, match the current snapshot's
   manifest seasons — read `data/players/raw/<latest>/manifest.json`).
2. `python -m nfl_ats player-value-ingest --include-postseason`
3. `python -m nfl_ats pbp-ingest --include-postseason` (large download).
4. `python -m nfl_ats role-actions-fetch --include-postseason`
5. Verify each new manifest records `"include_postseason": true` and the
   REG-scoped loaders return IDENTICAL row counts to the old snapshots
   (pbp 781,712; injuries 76,784; rosters 646,707; snaps 310,475;
   player-values 291,747; role-actions 224,496 — from the 2026-08-17
   session; if upstream data changed, note the delta, don't force it).
6. Rebuild NOTHING. The postseason-aware read side is a future declared
   change, not part of this task.

---

## SPEC-3 — Weekly ops command + rehearsal (DEADLINE: before Tue Sep 8, 2026)

**Deliverable A — `nfl-ats weekly-run`** (new `src/nfl_ats/weekly.py` +
CLI wiring + tests): one command running the Tuesday sequence in order,
fail-closed, echoing one JSON summary:

1. `ingest` (nflverse snapshot; seasons = existing latest manifest's
   seasons — must include 2026)
2. `build-features` (defaults; postseason included by default already)
3. `build-pbp-features` / `build-player-features` — pass the SAME snapshot
   ids as the current production manifests unless `--refresh-player-data`
   is given (then latest snapshots)
4. `margin-backtest --features data/processed/game_features_player.parquet
   --feature-profile player`
5. `margin-predict --season <S> --week <W> --features ...player.parquet
   --feature-profile player` — `--season/--week` required args of
   weekly-run, passed through
6. Assert the active manifest is SYNCHRONIZED after step 5; abort loudly
   if not.
7. `publish-predictions --with-board` (writes the card, the public site,
   and the CLV ledger; the bare command does NOT write the public site —
   the original spec's wording here was wrong, caught 2026-08-17).

Flags: `--dry-run` (print the plan, run nothing), `--skip-ingest`.
Each step's failure aborts the run with the step name and the underlying
error; no partial publishes (publish only runs after the sync assertion).
Tests: dry-run plan content; step-order; abort-on-desync (monkeypatch a
failing activation); no-publish-on-abort.

**Deliverable B — rehearsal (manual, run it, record it):** once Deliverable
A merges, run `weekly-run --season 2026 --week 1 --skip-ingest` end-to-end.
Expected: evaluation reproduces 0.5204819277 (nothing changed), a fresh
week-1 card, SYNCHRONIZED manifest, Pages redeployed. Record the wall-clock
time per step in `docs/ops_runbook.md` (new doc: the Tuesday timeline —
captures land early Tuesday; weekly-run must complete before the pool's
12:00 ET lock; include the manual fallback: the seven commands in order,
copied from the engine-room crib sheet).

**Trap:** if the owner's dashboard is running it holds `nfl-ats.exe`; use
`python -m nfl_ats` throughout (the spec's commands already do).

---

## SPEC-4 — MOD-07 weak-signal stack (after SPEC-1)

**Hypothesis (predeclared here):** stacking the surviving weak signals —
learned availability, value-weighted injuries, and three peer-reviewed
opener-bias features — onto the frozen player profile improves forced-pick
accuracy against the Tuesday opener.

**Step 1 — bias features** (new family in the canonical build; columns are
ADDITIVE — the REG bit-identity invariant applies to existing columns):

In `features.py`, computed leak-safely from schedules alone, one value per
game (home/away/diff where sided):
- `bias_playoff_holdover_home/away/diff`: 1.0 if week == 1 and the team
  appeared in ANY postseason game (game_type in WC/DIV/CON/SB) in season-1,
  else 0.0. Source: the schedules frame already includes postseason rows.
  (Literature: week-1 playoff holdovers covered 35.6%.)
- `bias_prior_week_ats_home/away/diff`: the team's single previous
  completed game's ats_margin this season (NaN if none). Strictly earlier
  games only — reuse the attach-states earlier-than-lookup pattern, NOT the
  EWM state (the point is single-game recency, distinct from
  state_ats_residual).
- `bias_week2_anchor_home/away/diff`: `bias_prior_week_ats_*` masked to
  week == 2, else 0.0 (the anchoring paper is specifically week 2).
- Register family `"bias"` in `constants.py` FEATURE_FAMILIES; do NOT add
  it to any existing FEATURE_SET (the frozen sets must not change).
- Add a plain-English `FAMILY_PHRASES["bias"]` entry in
  `market_decomposition.py` —
  `test_family_phrases_cover_every_registry_family` enforces this; the
  original spec omitted it (caught 2026-08-17).
- Tests: leak-safety (a team's week-N row unaffected by week-N result),
  playoff-holdover correctness on a synthetic bracket, REG bit-identity of
  all pre-existing columns.
- After merging: rebuild tables + margin-backtest + margin-predict +
  publish (the resync protocol; evaluation must reproduce 0.5204819277
  exactly since the frozen profile ignores new columns).

**Step 2 — candidate table + profile:**
- Build the candidate table: base → pbp → player enrichment WITH
  `--player-value-snapshot` AND the learned-availability rates (the
  existing `build-learned-availability-features` path) → write
  `data/processed/game_features_weak_stack.parquet`.
- New margin profile `weak_stack` in `margin.py`:
  `FEATURE_SETS["full_player_value"] + FEATURE_FAMILIES["bias"]` — the
  existing `full_player_value` composite already equals
  full_player + player_values, so this is the same set as the original
  wording with no duplicated columns (injury columns in this table carry
  learned availability semantics by construction). Add to
  MARGIN_FEATURE_PROFILES.
- Contract-year / friction events are EXCLUDED from v1 (no data source in
  repo; literature ≈ null). Do not build them.

**Step 3 — the look (ONE):**
- `rotation declare --name mod07_weak_signal_stack --grade opener
  --acknowledge-mined --description "..."` then `rotation assign`
  (earliest eligible ⇒ seasons [2020, 2021]).
- Baseline arm: frozen active config (player profile, ridge 10, none) on
  `game_features_player.parquet`. Candidate arm: weak_stack profile,
  ridge 10, none, on the candidate table. Both graded with the EXISTING
  `clv.opener_pick_evaluation` machinery restricted to the window's
  seasons (pass the config override; restrict by filtering the features
  frame to training+window seasons via `rotation.confirmation_split` —
  training frame feeds the fit exactly as the evaluator already does).
- Pair per game on game_id; compute paired accuracy delta and
  probability_positive by week-blocked bootstrap (reuse the
  paired-comparison helpers already shipping probability_positive).
- `rotation record` with the verdict thresholds (predeclared):
  probability_positive ≥ 0.90 → `confirmed` (promotion path: prospective
  2026 scoring as a frozen challenger — NEVER direct activation);
  ≤ 0.10 → `closed_negative`; otherwise `unresolved`.
- Write `docs/mod07_stack.md` recording everything (mirror the style of
  `docs/opener_evaluation.md`), including the mined-2018-2025 discount
  sentence.

**Traps:** the learned-availability injury columns share NAMES with the
fixed-prior columns — the two tables must never be mixed in one run;
`fixed_unavailability` itself is frozen (see its docstring) — the learned
path goes through `availability_rates` input only. The opener archive
pairs ~227-272 games/season; the window yields ~500 paired games — small;
report the interval honestly.

---

## SPEC-5 — Best Pick ranker (after SPEC-1; independent of SPEC-4)

> **REVISED 2026-08-17 (Fable).** The original spec assigned [2009, 2011]
> and promised ~48 scored weeks. That was an authoring error (mine): the
> pool's first block has no history in front of it, so the evaluator's
> 500-game warm-up consumes 2009-2010 whole — 17 scorable weeks, all in
> 2011 — and the calibrated-probability signal, which needs 400 prior
> out-of-sample prediction rows, cannot be computed at all. Opus caught
> this and correctly stopped without spending the window
> (`docs/opus_session_blockers.md`, Issue 1). Resolution: warm-up
> eligibility is now binding rule 9 of `docs/rotation_registry.md`,
> enforced in `rotation.py` — the registry will not offer a block starting
> before 2013. Every number below was reproduced by running the real
> evaluator and calibrator on the real feature table before this revision
> shipped.

**Problem:** one Best Pick per week; our confidence ordering is flat
(weekly top-|residual| 48.6% over 107 weeks). Find a signal that orders
pick quality; the pool pays it directly.

**Candidates (exactly these three, computed per pick from artifacts the
walk-forward evaluator already emits — no player data needed):**
1. `calibrated_probability`: the pick-side cover probability after Platt
   calibration on the chronological stream (calibration.py's existing
   machinery).
2. `key_number_distance`: min distance of the market line from {3, 7}
   minus min distance of the fair line from {3, 7} — positive when our
   number sits on a key number the market misses (reuse
   `key_numbers.py` helpers).
3. `sweep_robustness`: from the line sweep, the width (in points) of the
   interval around the quote where the pick's cover probability stays
   ≥ 0.50 — wide = robust.

**Protocol:**
- Screen (cheap pool): `rotation declare --name best_pick_ranker --grade
  nflverse_spread`; `rotation assign` ⇒ earliest eligible **[2013, 2015]**
  (rule 9; the registry computes this — never pass an override).
- Evaluation stream: run the standard walk-forward evaluator
  (market_residual, BASE profile — player features don't exist pre-2016
  and the ranker doesn't need them) with `start_season=2011,
  end_season=2015`. Verified on the real table: predictions begin 2011
  week 1 (512 completed games precede it, ≥ the 500-game warm-up); 512
  prediction rows precede 2013 week 1 (≥ the 400-row calibration
  requirement; actual per-week calibration histories run 496-1,228 rows);
  the window scores **768 games across 51 weeks, 17 per season**.
  Calibrate the stream with
  `calibrate_cover_prediction_stream(..., evaluation_start_season=2013)`.
- Screen metrics on the window rows (2013-2015) ONLY — the 2011-2012
  stream rows are warm-up plumbing, never evidence. For each candidate
  signal, rank picks within each week; metric = top-1-per-week accuracy
  and Kendall tau between signal rank and pick correctness; paired
  probability_positive of (top-1 accuracy − all-pick accuracy) > 0 by
  week-blocked bootstrap over the 51 weeks.
- **Disclosure (mandatory):** [2013, 2015] sits inside
  `pbp_drive_bundle`'s spent [2013, 2017]. Rule 4 permits this — windows
  retire per-family — but the write-up must state that these seasons were
  previously mined by another family.
- Gate (predeclared): any candidate with probability_positive ≥ 0.75 on
  the screen earns ONE opener-graded confirmation: declare
  `best_pick_ranker_opener` (inherits best_pick_ranker,
  --acknowledge-mined), assign ⇒ [2020, 2021], evaluate the SAME frozen
  signal top-1 at the opener grade. ≥ 0.75 there → use it for Best Pick in
  2026 (a pool-play decision, not a model change — no activation needed).
  No candidate clears the screen → record `closed_negative` for the
  screen window and STOP; do not tune new signals on the same window.
- Machinery: the 2026-08-17 session debugged a runner
  (`best_pick_ranker.py`, flags `--raw-start-season`, `--min-train-games`,
  `--min-calibration-games`) in its session scratchpad; if that copy is
  gone, rebuild it per `docs/best_pick_ranker.md` — it is a thin
  composition over `walk_forward_outcomes`,
  `calibrate_cover_prediction_stream`, and the `key_numbers` helpers.
- Document in `docs/best_pick_ranker.md`.

**Trap:** ranking quality on 3 seasons = 51 top-1 picks — the bootstrap
interval will be wide; `unresolved` is a likely and acceptable verdict.
Never widen the window after seeing results.

> **RESOLVED 2026-08-17 — `player` was chosen and the confirmation ran.**
> The screen ran and `sweep_robustness` cleared the 0.75 gate, so the
> opener confirmation was earned. "Evaluate the SAME frozen signal top-1
> at the opener grade" does not say which **feature profile** generates the
> picks being ranked, and the two readings give different picks, different
> signal values, and different answers:
>
> - **`base`** — replicates the screen exactly (SPEC-5 fixed the screen to
>   `base`, so the signal's definition includes that model). Cleanest as a
>   replication; but the picks it ranks are not the picks we publish.
> - **`player`** — ranks the picks we would actually deploy, which is what
>   "use it for Best Pick in 2026" means in practice, and
>   `opener_pick_evaluation` already defaults to it. But the confirmation
>   then changes two things at once (grade AND model), so it is not a clean
>   replication of what the screen found.
>
> **`player` was chosen**, because the gate's consequence — "use it for
> Best Pick in 2026" — is a choice among the published card's picks, so
> confirming the ranker on picks we would never make answers the wrong
> question. Both profiles were verified to reproduce
> `opener_pick_evaluation`'s weekly fit to `max |diff| = 0.0` on
> [2020, 2021] (466 paired games, 35 weeks) beforehand, as a plumbing check
> that computed no accuracy and took no look. Result and caveats:
> `docs/best_pick_ranker.md` § Opener confirmation.

---

## Deferred — do NOT start without a new spec

Hierarchical pooling / XLG-05 partial pooling (waits for a family that
clears the CFB benchmark); QB-dependence interaction; MKT-09 licensing
audit (blocks publishing anything archive-derived); registry-backed
research views in the dashboard (SPEC-1's `status` JSON is the intended
data source); event-aware close prediction (secondary goals only).
