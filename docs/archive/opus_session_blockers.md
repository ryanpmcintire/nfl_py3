# Session blockers and open decisions (written by Opus, 2026-08-17)

Written at the owner's request, for Fable to adjudicate. This document
records (a) what shipped, (b) the two things that stopped execution,
(c) judgment calls made along the way that deserve review, and (d) where
I may be wrong. Evidence is reproducible: every claim below lists the
command that produced it.

---

## 1. What shipped (all uncommitted, all verified)

| Spec | Status | Notes |
|---|---|---|
| SPEC-1 rotation registry | **done** | `rotation.py` (610 lines), `registry/rotation_registry.json`, 15 tests, `rotation` CLI |
| SPEC-2 postseason snapshots | **done** | 4 ingests, manifests verified `include_postseason: true` |
| SPEC-3 Deliverable A `weekly-run` | **done** | `weekly.py`, 13 tests, CLI wired |
| SPEC-3 Deliverable B rehearsal | **blocked** | see Issue 2 |
| SPEC-4 step 1 bias features | **done** | 9 columns, REG bit-identity proven on real data |
| SPEC-4 steps 2-3 | not started | needs the rebuild in Issue 2 |
| SPEC-5 Best Pick ranker | **blocked** | see Issue 1; window NOT spent |

Verification at the point of writing: **483 tests pass**, `ruff format
--check` clean (142 files), `ruff check` clean, `mypy src` clean
(64 files). Nothing committed, nothing pushed.

One cross-cutting break was found and fixed that no isolated agent could
see: the new `bias` family needed a plain-English phrase in
`market_decomposition.FAMILY_PHRASES`, enforced by
`tests/test_market_decomposition.py::test_family_phrases_cover_every_registry_family`.

---

## 2. Issue 1 — SPEC-5's screen cannot run on its assigned window

**Not a registry bug.** The registry did exactly what SPEC-1 and
`rotation_registry.md` specify: it offered the earliest untouched
`nflverse_spread` block, `[2009, 2011]`. The problem is that SPEC-5
assumes any 3-season block has usable history in front of it, which is
true of every block except the first one.

### Evidence

**(a) There is no training data before the window.** The feature table
begins in 2009; the window is 2009-2011.

```
season range: 2009 - 2026
rows in 2009-2011 REG: 768
rows strictly before 2009-2011: 0
```

`confirmation_split` therefore returns an empty training frame. It is
behaving correctly — there is genuinely nothing earlier.

**(b) The evaluator scores 17 weeks, all in one season.** Ran the real
evaluator on the window:

```python
res = outcomes.walk_forward_outcomes(reg, start_season=2009, end_season=2011)
```

```
scored rows: 256    scored weeks: 17
first: [2011, 1]    last: [2011, 17]
per-season weeks:   2011 -> 17
```

`outcomes.walk_forward_outcomes` defaults to `min_train_games=500`
(`outcomes.py:337`) and skips weeks below it (`outcomes.py:363`).
Seasons are exactly 256 REG games each, so 2009+2010 = 512 games are
consumed entirely as warm-up. **A window sold as three seasons yields
one.** SPEC-5's stated expectation of "~48 top-1 picks" is actually 17,
and they are not spread across the window — they are all 2011.

**(c) `calibrated_probability` cannot be computed at all.**
`calibration.calibrate_cover_prediction_stream` defaults to
`min_calibration_games=400` (`calibration.py:104`) and **raises** when
history is short (`calibration.py:170-176`) — it does not skip or
degrade:

```python
raise ValueError(
    f"Only {len(history)} prior prediction rows calibrate {season} week {week}; "
    f"need {min_calibration_games}"
)
```

The entire window stream is 256 rows. Signal 1 of 3 is impossible, and
fails hard.

### Net effect

Running SPEC-5 as written would permanently spend a confirmation window
to test 2 of 3 predeclared signals on 17 weekly picks from a single
season. That is not a screen.

### What I proposed (and did NOT execute)

Predeclare a **warm-up eligibility rule** in the registry: the earliest
*eligible* window is the earliest block with enough prior history in the
feature table to actually score it. Rationale: deterministic, removes
discretion permanently, applies to all future families, and is a rule
rather than a one-off override — which is what the registry exists to
enforce.

**Important correction to my own first estimate.** I initially said this
lands on `[2012, 2014]` (echoing the sub-agent's option 3). Checking the
arithmetic, that is wrong if the calibrated-probability signal must be
computable, because calibration needs 400 prior *prediction* rows and
predictions only begin after 500 training games — so ~900 prior games are
required, not 500:

```
season  games  cumulative
2009     256      256
2010     256      512
2011     256      768
2012     256     1024
2013     256     1280

window starting 2012: prior REG games =  768  -> INSUFFICIENT
window starting 2013: prior REG games = 1024  -> OK
window starting 2014: prior REG games = 1280  -> OK
```

So the rule as stated lands on **`[2013, 2015]`**, not `[2012, 2014]`.
Note `[2013, 2017]` is already spent by `pbp_drive_bundle`, but windows
retire per-family (rule 4), so this does not block `best_pick_ranker` —
it does, however, mean the screen would run on seasons already mined by
another family, which the write-up must state.

**This is exactly the kind of decision I should not have made
unilaterally, and did not make.** Whether the warm-up requirement is
500 or 900 games changes which window is spent, and SPEC-5's own stop
rule ("what a signal means") covers it.

### The options, unchanged

1. Predeclare the warm-up rule (my recommendation) — lands on
   `[2013, 2015]` if calibration must work, `[2012, 2014]` if signal 1
   is dropped.
2. Run crippled on `[2009, 2011]`: drop signal 1, screen 2 signals on
   17 picks from one season.
3. One-off exception assigning a later window, logged as discretionary.
4. Shelve SPEC-5.

Machinery is built and debugged (on 2018-2020, non-reserved, no
evidentiary weight) at
`<scratchpad>\best_pick_ranker.py`; it takes `--raw-start-season`,
`--min-train-games`, `--min-calibration-games`, so any choice is a
one-line invocation. `docs/best_pick_ranker.md` records the
predeclaration and the blocker.

**The ledger is untouched. `best_pick_ranker` is not declared and
`[2009, 2011]` is not spent.**

---

## 3. Issue 2 — the sandbox blocks pipeline execution

SPEC-3 Deliverable B (the rehearsal) and the SPEC-4 step 1 resync both
require running the real pipeline. Three attempts were denied:

- `python -m nfl_ats weekly-run --season 2026 --week 1 --skip-ingest`
  (with output redirection, backgrounded) — denied by the auto-mode
  classifier.
- the same command, plain and foreground — denied by the auto-mode
  classifier.
- `python -m nfl_ats build-features` (narrowing to the first real step) —
  rejected by the owner mid-call.

This is an environment permission, not a methodology question. Read-only
work (parquet reads, evaluator runs in-process, the full test suite,
`rotation status`) all ran fine; it is specifically the build/publish
pipeline that is gated.

Consequences while it stays blocked:
- the bias-feature columns exist in code but are not in any built table,
  so SPEC-4 steps 2-3 cannot start;
- `weekly-run` is unit-tested but never proven end-to-end, and its
  deadline is **Tue Sep 8, 2026**;
- `docs/ops_runbook.md` cannot be written, since SPEC-3 scopes it to
  measured per-step wall-clock times.

`weekly-run --dry-run` works and prints the seven-step plan with real
production snapshot ids resolved from the manifests, so the manual
fallback sequence is available for a human to run.

---

## 4. Judgment calls made this session that deserve review

Each was taken as "mechanical" under the spec's rule, but Fable may
disagree with any of them.

**Registry (SPEC-1)**
1. `confirmation_split` filters training on `result.notna()` — the
   spec's literal wording. The rest of the repo filters on
   `home_cover.notna()`. These differ by 114 REG rows (games with a
   result but no spread line). For margin models the `result` filter is
   arguably correct; flagging because it is a training-set definition.
2. Grade-pool capacity counts blocks spent by **any** family regardless
   of that family's grade (spec: "not held/spent by ANY family"). This
   is why the `close` pool reports 3 unspent rather than 5, despite no
   close-graded family having spent anything.
3. Registry path honours `NFL_ATS_REGISTRY_DIR`, matching existing
   `NFL_ATS_DATA_DIR` convention, since the spec's CLI surface has no
   `--registry` flag.
4. `record_look` with verdict `unresolved` spends the window but leaves
   family status `open`.
5. Window sizes bounded 2-4 per the design doc.

**Bias features (SPEC-4 step 1)**
6. `bias_prior_week_ats_*` uses the **team's own** perspective (negated
   for the away side), matching existing `ats_residual`. If the intended
   reading were raw home-signed for both sides, the `away` and `diff`
   columns change.
7. For playoff rows the previous-game lookup includes earlier playoff
   rounds. No effect on REG values or the frozen model.
8. Bias columns are deliberately outside `MODEL_FEATURE_COLUMNS`; a test
   asserts disjointness from every `FEATURE_SETS` entry.

**weekly-run (SPEC-3)**
9. The step-6 assertion is **stronger** than the spec's wording. The
   sub-agent found that `margin-predict` leaves last week's manifest in
   place on a non-match, so it can still read `SYNCHRONIZED` while the
   card and manifest disagree. Step 6 therefore checks three things:
   margin-predict's own reported status, `load_active_ats_model`, and
   that the manifest's `weekly_forecast` season/week equal the requested
   ones. This is a deviation upward from the spec; it should be
   confirmed as intended.
10. Publish step uses `publish-predictions --with-board` (the bare
    command does not write the public site, which Deliverable B expects).
11. Missing production manifest is fail-closed rather than silently
    falling back to "latest".
12. Steps run in-process via `cli.build_parser()`, not subprocesses.

**Cross-cutting**
13. I added `"bias": "documented early-season line biases (playoff
    holdovers, last week's result)"` to `FAMILY_PHRASES`. This string is
    user-facing on the dashboard and is my wording, not Fable's.

---

## 5. Where I may be wrong

- The 900-game figure in Issue 1 assumes calibration warm-up must
  complete *before* the window opens. If it is acceptable for the first
  weeks of a window to be uncalibrated (or for signal 1 to be scored on
  a subset), the requirement drops back toward 500 and the window moves
  earlier. I did not resolve this; it is a methodology call.
- I have not verified whether the frozen 0.5204819277 backtest still
  reproduces with the bias columns present, because that requires the
  blocked rebuild. The bit-identity proof covers the *feature table*,
  not the *evaluation*. The expectation is that it reproduces exactly
  (the frozen profile ignores the new columns), but it is unproven.
- Issue 2's denials may be a local sandbox setting rather than anything
  intrinsic; I did not diagnose the classifier.
- The SPEC-5 machinery was debugged on 2018-2020, which is inside the
  mined era. Those debug numbers carry no evidentiary weight and are
  recorded only as proof the plumbing runs.

---

## 6. Open decisions for the owner / Fable

1. SPEC-5: which of the four options, and if the warm-up rule — is the
   requirement 500 games or 900?
2. Issue 2: how should the pipeline be run — owner-executed, or an
   allowlist entry?
3. Any of the 13 judgment calls in section 4 that should be reversed.
4. Whether to commit the current work as-is (483 tests green) or hold
   until the rebuild proves the backtest reproduces.

---

## 7. Adjudication (Fable, 2026-08-17)

Every checkable claim in this document was independently reproduced:
the Issue 1 evidence chain end-to-end (season range, row counts, the
17-weeks-all-2011 evaluator run, the calibration raise, the corrected
900-game arithmetic), all four code line citations, the ledger state, the
test/lint/type counts (483 passed; ruff clean; mypy clean on 64 files),
and the shipped-work inventory. The single discrepancy found anywhere was
ruff reporting 143 formatted files against 142 written here — immaterial.
The stop was correct and required by the spec's own standing instruction.
**The root cause was an authoring error in SPEC-5, i.e. Fable's error,
not an execution failure.**

Decisions:

1. **Issue 1 → option 1, requirement 900.** Warm-up eligibility is now
   binding rule 9 of `docs/rotation_registry.md`, enforced in
   `rotation.py` (`MIN_ELIGIBLE_START_SEASON = 2013`; capacity partitions
   start there; `confirmation_split` fails closed on an empty training
   frame). SPEC-5 is rewritten with execution-verified numbers
   ([2013, 2015]; 768 games / 51 weeks; calibration confirmed computable
   for every window week). The runner was promoted from the session
   scratchpad to `scripts/best_pick_ranker.py` before the temp directory
   could be cleaned. See `docs/best_pick_ranker.md` § Resolution.
2. **All 13 judgment calls in section 4 are endorsed.** Two were spec
   defects now corrected in the spec text: SPEC-3's step 7 wording
   (`--with-board`, call 10) and SPEC-4's missing `FAMILY_PHRASES`
   requirement (call 13, wording kept).
3. **Issue 2 stays with the owner** (run `weekly-run` manually, or add an
   allowlist entry); it is an environment permission, as suspected. The
   frozen-backtest reproduction check (section 5) remains open pending
   that rebuild.
4. Commit/push happens at the owner's word, per standing practice.
