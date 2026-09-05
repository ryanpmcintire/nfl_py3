# NFL ATS repository instructions

## Session startup

- Read `HANDOFF.md`, `README.md`, and the recommended execution order in
  `ROADMAP.md` before proposing substantial work.
- Run `git status --short` and `git log -3 --oneline --decorate`; live Git state
  overrides any snapshot recorded in the handoff.
- Inspect `artifacts/active_ats_model.json` before quoting the active model or
  current historical result. Generated artifacts are local and may be absent in
  a fresh clone.
- Treat `CURRENT_PREDICTIONS.md` as the last deliberately published forecast,
  not necessarily the newest locally generated forecast.
- Ensure `git config --get core.hooksPath` returns `.githooks`. If it does not,
  configure it with `git config --local core.hooksPath .githooks` without asking
  the user to perform setup.
- **Run the capture scheduler every session, without being asked:**
  `.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --once`,
  then `--status` and read it. `--once` runs any capture whose window is open
  and is idempotent, so running it twice costs nothing. `--status` is the part
  that matters: a run of `MISSED` rows means the background scheduler is not
  running and point-in-time captures are being lost permanently — restart it
  (`scripts/start_capture_scheduler.cmd`) and say so in the session report.
  The schedule itself lives in `SCHEDULE` in that file, in version control, on
  purpose. The agent owns this the way it owns handoff refreshes: never ask the
  user to run it, and never hand them a weekly cadence to remember.

## Research invariants

- This is a research and paper-decision project; do not add automated wagering.
- Pregame features must only use information available before the prediction
  timestamp. Add a leakage regression test for every new feature family.
- Evaluate chronologically. Keep validation, selection, calibration, and final
  outer test periods distinct, and preserve prediction-level output.
- Compare additions against the same strong market and simple-model baselines.
  Report uncertainty and season stability, including negative results.
- Never describe the current 52.05% historical forced-pick accuracy as proof of
  a profitable or stable edge. Keep historical accuracy distinct from each
  game's model probability.
- Prediction-safety and evaluator-performance contracts are release-blocking.

### An interval crossing zero is NOT grounds for rejection (binding)

Stated by the project owner repeatedly, and violated repeatedly. It is an
invariant, not a preference.

- **Never discard, close, or decline to build a signal because its confidence
  interval contains zero.** At this evaluator's ~2-point resolution, "no
  significant effect" is the EXPECTED outcome for a real-but-small signal, so
  treating it as a negative silently deletes exactly the signals worth keeping.
- **Only two things justify closing a line of work:** (1) a refuted mechanism —
  wrong sign, or the trait has no split-half reliability, so no sample size
  rescues it; (2) bounded by a positive control — the instrument was PROVEN
  able to detect an effect that size and it was absent. Everything else is
  category 3, unresolved.
- **Every category-3 result must be recorded** with
  `nfl-ats weak-signals record` before any write-up describes it as settled.
  Recording is the default action, not an optional extra.
- **This rule is now enforced in code, not just prose** (added 2026-08-18
  after sessions that never loaded this file kept violating it): a terminal
  classification (`refuted_mechanism` / `bounded_by_control`) or a
  `closed_negative` rotation verdict must name an admissible
  `--closing-ground` (`wrong_sign_resolved`, `no_split_half_reliability`,
  `positive_control_bound`), and `wrong_sign_resolved` is rejected outright
  unless the whole interval sits below zero. If a record command errors, the
  verdict is wrong, not the validator — reclassify as
  `unresolved_below_power`.
- **Subagents never see this file or the session hooks.** Any subagent prompt
  that runs, scores, or adjudicates an experiment MUST include the taxonomy
  above verbatim, and its verdicts must flow through the two record commands,
  never through prose in a doc. A subagent's verdict phrased as "failed
  because the interval contains zero" is a bug in the prompt that spawned it.
- **Report `probability_positive`, never "contains zero".** The binary phrasing
  is what smuggles the rejection back in.
- **Pooling sub-signals into a result that excludes zero is a legitimate,
  transformative finding.** It does not matter that the parts individually
  cross zero — any real effect can be decomposed until its pieces do.
  **Correction, 2026-08-18:** this section previously reported a
  demonstration of three signals at `probability_positive` 0.860 / 0.933 /
  0.917 pooling to +0.724 accuracy points, 95% [+0.056, +1.392], "P+ 0.983",
  sharpening 1.46x over the best single input. Audited later the same day,
  that demonstration does not reproduce from the registry — no subset of
  the current NFL `accuracy_points` entries pools anywhere near +0.724 — and
  its three inputs look like split-half reliabilities mislabeled as
  `probability_positive` (0.933 matches `injury_value_lost`'s reliability,
  0.860 an M6-construct reliability); the pool tool has never emitted a
  "P+" field. Treat that number as unverified, not as a finding.
  The living reference is `nfl-ats weak-signals pool --league nfl
  --effect-units accuracy_points` (read-only; `weak-signals record` is the
  separate recorder that writes to the registry). As of 2026-09-01 (539
  NFL `accuracy_points` signals in the pool; 2026-08-18's read of -0.023
  [-0.073, +0.028] on 107 signals is superseded) it reports a random-effects
  pool of **+0.00168 accuracy points, 95% [-0.01234, +0.01570]**,
  `excludes_zero: false`, sign test 288-of-541 favouring the candidate
  (253 the baseline, p=0.144) — the pile is a coin flip, leaning very
  slightly positive, and is not resolved. The pool now
  includes correlated decompositions of shared windows (see
  `overlap_warnings`, extensive for this batch), so the interval overstates
  precision; the sign-test and per-entry rows are the safer read. Re-run the
  command for the current number rather than quoting a fixed one here.

The one discipline that stays, because it protects the pooled result rather
than gating the inputs: **pooled inputs must be commensurable** — same units,
same scale, same population — and **the family must be declared before the
signs are seen.** A pooled number built from a production quantity plus a
subset cover-rate gap is not a finding; it collapses under the next audit.
The 2026-08-18 session that motivated this rule reported it moving the
pooled estimate from a fragile +0.991 to a robust +0.724 at P+ 0.983; both
figures belong to the unreproducible episode corrected above and are not
independently verified — the lesson (declare commensurability and the
family before signs are seen) stands on its own regardless of the numbers
once attached to it.

### A promotion bar is not a decision bar

The pool is FORCED PICKS: 285 cards must be submitted either way. So declining
a candidate that is 87% likely better is not caution — it is taking the other
side of an 87/13 bet. Predeclared thresholds (e.g. MOD-07's 0.90) govern what
the docs may CLAIM. They must never govern which card is PLAYED; that decision
is expected value, full stop.

**Grade the decision at the OPENER. A close-graded number may never veto a
play.** Added 2026-08-18 after exactly that happened: MOD-07 promotion was
refused on a close-graded comparison (51.57% vs 52.05%) one hour after the rule
above was written. At the opener — the grade the pool actually settles on, and
the project's declared primary goal — the same candidate on the same 1,537
paired games scores **52.83% vs the baseline's 52.50%**, and it was promoted.
The close is the market at its sharpest and systematically understates
pool-relevant edge; using it to reject a candidate inverts the project's stated
priority.

**The failure mode this file keeps catching is not a bad rule, it is a default.**
The reflex on any new number is to look for what is wrong with it, so banning
one justification only produces a different one for the same refusal. When a
result is in hand, state what it implies for the DECISION before stating what
is wrong with it.

### Label how you know it, every time (binding)

Added 2026-08-18 after a session in which the agent misstated a dozen-plus
facts to the owner and each time answered "you caught a real overstatement."
The apology was not the problem. **The problem is that measured, remembered,
inferred and guessed all come out in the same declarative voice**, so the owner
cannot triage and must personally audit every claim. That is the labour this
rule exists to remove.

**Every factual claim carries its provenance, inline, in the same sentence.**
Four tags, and nothing may be stated without one:

- **measured** — you ran it THIS session. Give the command or artifact path.
- **read** — you opened the file just now. Give the path and line.
- **reported** — a subagent or doc says so and you have NOT verified it. Say
  "unverified" out loud. Subagent numbers are claims, not facts.
- **inferred** — your reasoning, prediction, or mechanism. Not evidence. Say
  "I think" or "my guess", and never in the same breath as a measured number.

**Verify before quoting anything that gates a decision.** If a number decides
what gets played, published, deleted or spent, open the source first. Three
separate incidents in one session — the "empty" prospective ledgers that held
16 real rows, the stale `rank-71-of-142` design figure belonging to a retired
profile, and the Best-Pick tie fact stated backwards — were all summaries
repeated without opening the underlying file, and all three would have been
caught by one command.

**Never state a constraint without citing the rule that imposes it.** "We need
more data", "that window is spent", "we must wait for next season" are
prohibitions, and a prohibition needs a source. In the same session the agent
asserted that already-examined seasons "can't give an honest answer anymore"
when rule 4 above says windows retire **per-family, not globally** and rule 6
says a reused window carries a **stated discount, not a ban**. A caution
reported as a wall is a refusal in disguise, and it reliably produces "we
cannot decide yet" — the exact output this file already bans twice.

**Distrust your own summarising adjectives.** "Dud", "dead", "inert",
"narrow", "spent", "biggest" is where every error in that session lived. Give
the number and the interval and let the owner judge; if a one-word verdict is
genuinely needed, it comes after the number, not instead of it.

## The dashboard improves every session (binding)

Added 2026-09-05 after the owner pointed out, again, that "improve the
dashboard" is said in every session and only happens when it is repeated in
anger. The cause was structural: the queue the agent plans from (ROADMAP
open rows, the research queue) is research-shaped, so the dashboard never
gets a lane unless the owner forces one, and even shipped site changes stay
invisible until someone regenerates and deploys the pages.

- **Every session ships at least one visible, reader-facing dashboard
  improvement** (new information on a page, a clearer explanation, a
  populated placeholder), inside the current design system, with a fixture
  test. Dashboard work is a standing lane, not a leftover.
- **Every session ends with the site regenerated and deployed**:
  `nfl-ats publish-board` (and `publish-predictions` when the forecast
  changed), the rendered-page diff reported in the session report, and the
  push made so GitHub Pages rebuilds. A fail-closed publish (stale snapshot,
  drifted curated fingerprint) is fixed in-session, never left for the owner.
- The standing ROADMAP row `UI-20` carries the running candidate list; add to
  it whenever a lane surfaces something a reader would want to see.

## Repository hygiene

- Use Python 3.12 and the locked uv environment.
- Do not commit raw/processed data, model artifacts, credentials, cookies,
  virtual environments, or test output. The tracked Markdown prediction card is
  the deliberate exception for public forecast visibility.
- Preserve unrelated user changes and do not rewrite Git history.
- Do not commit or push unless the user explicitly asks.

## Required verification

Run these after code changes:

```powershell
.\.tools\uv.exe run ruff format --check .
.\.tools\uv.exe run ruff check .
.\.tools\uv.exe run mypy src
.\.tools\uv.exe run pytest
```

## Automatic session handoff

- Update documentation and `ROADMAP.md` when evidence or priorities change.
- If the active weekly forecast changed, run `nfl-ats publish-predictions`.
- The agent owns handoff refreshes. Never ask the user to run the handoff command.
- Refresh `HANDOFF.md` automatically when work is ready to hand off and before
  every commit or push intended for `master`. The tracked pre-commit hook is a
  backstop, not a substitute for agent responsibility.
- Before pushing `master`, run `nfl-ats handoff --check`. If a refresh changes
  the file after a commit, create the required follow-up commit before pushing.
- Report remaining Git changes plus the exact checks run.
