# NFL ATS repository instructions

Rules only. The incident history, dated measurements and corrections that
motivated each rule live in `docs/agents_history.md`; every rule there still
binds. Re-run a command for a current number rather than quoting one from a
doc.

## Session startup

- Read `HANDOFF.md`, `docs/lanes/README.md`, and the lane file the prompt
  names (or the most recently modified lane when the prompt says to
  continue). Read the `## Recommended execution order` section of
  `ROADMAP.md` only when choosing new work. Do not read ROADMAP.md or README.md whole; completed rows
  live in `docs/roadmap_archive.md` and superseded results in
  `docs/research_history.md`, and both are for grep, not for reading.
- Run `git status --short` and `git log -3 --oneline --decorate`; live Git
  state overrides the handoff snapshot.
- Inspect `artifacts/active_ats_model.json` before quoting the active model or
  its historical result. Treat `CURRENT_PREDICTIONS.md` as the last
  deliberately published forecast, not the newest local one.
- Ensure `git config --get core.hooksPath` returns `.githooks`; if not, set it
  with `git config --local core.hooksPath .githooks` without asking.
- **Run the capture scheduler every session, unasked:**
  `.\.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --once`,
  then `--status --brief` and read it. A `MISSED` row means the daemon is not
  running and point-in-time captures are being lost: restart it with
  `scripts/start_capture_scheduler.cmd` and say so in the session report.
  Never hand the user a cadence to remember.
- **A scheduler job is not done until it has run once (binding).** Any job
  added or edited in a session is exercised in that session with
  `capture_scheduler.py --run-job NAME` (`--dry` if it writes a ledger or the
  card) and the `MANUAL-RUN OK` line goes in the session report. A `NEVER RUN`
  row whose first window falls within seven days is treated like `MISSED`:
  exercise it now. A rehearsal that calls a Python function proves nothing
  about the argv the daemon spawns; exercise the argv.

## Research invariants

- Research and paper decisions only; never add automated wagering.
- Pregame features use only information available before the prediction
  timestamp. A leakage guard belongs in the feature builder as a runtime
  assertion (the test-file form is suspended by the moratorium below).
- Evaluate chronologically; keep validation, selection, calibration and outer
  test periods distinct; preserve prediction-level output.
- Compare every addition against the same market and simple-model baselines,
  and report uncertainty and season stability, including negative results.
- Never describe the historical forced-pick accuracy as proof of a profitable
  or stable edge, and keep it distinct from each game's model probability.
- Prediction-safety and evaluator-performance contracts are release-blocking.

### Margins are multimodal, not Gaussian (binding)

Final margins are a discrete distribution with mass at the key numbers (3, 7,
10, 14, 17). Any served cover probability, push probability, flip line or
alternative-line answer is computed against the discrete margin distribution
conditional on the line. A smooth pooled-residual read may run only as a
challenger and must beat the discrete read on the opener grade through the
played card before it is served. "The residual is smooth on average" is a
composition fallacy, not an argument: a cover probability is asked at one
spread, where the mass points decide it. Comparing two pooled-residual
mappings against each other is not a test of this rule.

**No unexplained threshold flips on the played card.** A rule that flips the
model's pick must name a mechanism. An accuracy dip located only at a spread
threshold is a diagnosis to publish and a defect to fix, never a flip to bolt
on.

### An interval crossing zero is NOT grounds for rejection (binding)

- Never discard, close, or decline to build a signal because its interval
  contains zero. At this evaluator's resolution that is the expected outcome
  for a real small signal.
- Only two grounds close a line of work: (1) a refuted mechanism, meaning a
  resolved wrong sign (the whole interval on the wrong side of zero) or zero
  split-half reliability; (2) bounded by a positive control proven able to
  detect an effect that size. Everything else is `unresolved_below_power`.
- Every unresolved result is recorded with `nfl-ats weak-signals record`
  before any write-up describes it as settled. Recording is the default.
- Enforced in code: a terminal classification or a `closed_negative` rotation
  verdict must name an admissible `--closing-ground`, and
  `wrong_sign_resolved` is rejected unless the whole interval sits on the
  wrong side. If a record command errors, the verdict is wrong, not the
  validator. Never weaken the validators.
- Subagents never see this file or the hooks. Any subagent prompt that runs,
  scores or adjudicates an experiment carries this section verbatim, and its
  verdicts flow through `weak-signals record` / `rotation record-look`, never
  through prose.
- Report `probability_positive`, never the binary "contains zero".
- Pooling sub-signals is legitimate when the inputs are commensurable (same
  units, scale, population) and the family is declared before the signs are
  seen. The living reference is `nfl-ats weak-signals pool --league nfl
  --effect-units accuracy_points` (summary by default; `--full` for rows).
  Its `needs_remeasurement` rows are dead heats whose stored value should be
  0.5; flagging them closes nothing.

### One calibrated probability decides every pick (binding, owner, 2026-09-14)

Decisions are never made in isolation. The served side of a game is the
side favoured by ONE probability that combines the model with every
situational signal as a fitted term. Concretely:

- **No rule changes a side on its own.** A line move, the money, a crew, a
  coach, the weather or any other signal enters the model's probability as
  evidence with a fitted weight; it never overrides the model outright. A
  member that can only return `p` or `1 - p` is a flip, not a signal, and
  is inadmissible on the served card. Off switches:
  `LATE_WEEK_FOLLOW_SERVED`, `HANDLE_FOLLOW_SERVED`, `ROOKIE_CREW_SERVED`,
  `OWNER_HELD_MEMBERS`.
- **Every parameter is chosen out of season.** Weights, thresholds and cut
  points are fitted leave-one-season-out and scored on the held-out season;
  the in-sample number is reported beside the out-of-sample number with the
  gap. A constant derived from the games it is scored on never reaches
  `src/`.
- **Probability, not just hit rate.** A served probability is calibrated:
  report a reliability table and log loss or Brier against the market and
  the model-only baselines, not only accuracy points. A combined model is
  small enough to state its coefficients per fold; report their stability.
- **Small splits get a null.** A lopsided record on a small subset
  (shallow 12-7 against deep 7-11 on 37 flips) is compared with a
  permutation or exact null before any sentence about "two populations".
  A "flat" or "uniform" claim carries a slope with an interval.
- **Count the looks.** Every band, cell, arm and continuous fit is a look;
  state the count, name the family, and never present the best cell as the
  finding. Report the decisive-game record (the games where arms differ)
  before the headline effect.
- **Subagents carry this section verbatim** alongside the closing-grounds
  taxonomy whenever they fit, score or adjudicate anything.

### A promotion bar is not a decision bar

Sample sizes here are rarely big enough to clear a p of 0.90 or 0.95. Failing
such a threshold is therefore NOT grounds to reject, fail or close a signal.
That is the whole rule, and it is a rule against one bad rejection reason.

It is **not** a rule that anything above a coin flip gets served. Serving a
rule that moves real picks needs a real reason; a `probability_positive` just
above 0.5 is not one, and neither is a record like 19-18. Research closure and
card serving are two different decisions and the thresholds above govern
neither by themselves. **Grade the decision at the opener**; a close-graded
number may never veto a play. When a result is in hand, state what it implies for
the decision before stating what is wrong with it.

### Label how you know it (binding)

Every factual claim carries its provenance inline: **measured** (ran it this
session; give the command or artifact), **read** (opened the file now; give
path and line), **reported** (a subagent or doc says so and you have not
verified it; say unverified), **inferred** (your reasoning; say so, never in
the same breath as a measured number). Verify before quoting anything that
gates a decision. Never state a constraint without citing the rule that
imposes it. Give the number and the interval before any one-word verdict.

### A flip point is one number, in the adverse direction only (binding)

The board's flip line is how far the line would have to move against the pick
before the card takes the other side: one number, never a range, never a
favourable-direction flip. When no flip is found before the adverse edge, the
cell says the pick holds through that edge. Enforced in `_flip_line` and
`flip_line_text` in `board_content.py`.

## The dashboard improves every session (binding)

- Every session ships at least one visible reader-facing dashboard
  improvement inside the current design system, with a fixture test where an
  existing contract test forces one.
- Every session ends with the site regenerated and deployed: `nfl-ats
  publish-board` (and `publish-predictions` when the forecast changed), the
  rendered-page diff in the session report, and the push so GitHub Pages
  rebuilds. A fail-closed publish is fixed in-session.
- ROADMAP row `UI-20` carries the running candidate list.
- No number on the site may go stale: every percentage a reader sees is read
  from an artifact keyed to the active model, never from a constant, and
  `publish-board` fails closed when a headline number names a different
  model.
- The board is for humans: no snapshot ids, timestamps, hashes, model ids,
  policy slugs, column names, research jargon or compliance boilerplate in
  reader-facing text. Provenance lives in `lineage.json` and
  `explanations.json`; the render-contract tests ban the tokens.

## Test moratorium (binding)

- No new test files and no new test functions, by any session, model tier or
  subagent, until the owner lifts this. A fix ships as the fix; verification
  is running the real command once and reporting its output.
- Existing tests that break on a legitimate change are edited to match or
  deleted, never expanded.
- Two tiers: durable tests live under `tests/` and protect a contract
  (prediction safety, chronology, evaluator arithmetic, board render
  contract, scheduler argv contract, registry closure validation); they never
  pin a research number or exercise a one-off script. Throwaway tests live
  under `tests/scratch/`, gitignored and refused by the pre-commit hook, and
  are never promoted by moving the file.
- The cut of existing test files is an owner decision; propose, do not delete.

### Post-review research-residue sweep (binding, owner, 2026-09-19)

After every code review, check that no redundant tests, assumed behaviors, or
assertions that only needed to happen during research (not at runtime) remain.
Redundant tests duplicate contract coverage, pin a research number, or exercise
a one-off script. Assumed behaviors are mocks, stubs, or fixture constants that
assert what was assumed instead of reading the artifact the production path
reads. Research-only assertions are runtime checks that validated a research
step and now duplicate (or fight) the fail-closed guards on the served path;
their rationale belongs in the commit message, ROADMAP row, or a doc, not in
`src/`. Sweep findings are proposed, never deleted outright: the cut of
existing test files stays an owner decision.

## No code comments (binding)

No `#` comments and no docstrings of any length in any `.py`, `.ps1`, `.sh`
or `.cmd` file; pragmas (`# noqa`, `# type:`, `# pragma`, the shebang) are the
only exception. Enforced by `.claude/hooks/guard_comments.py` on every
Edit/Write and by `scripts/strip_comments.py --check` in the pre-commit hook.
Every subagent prompt says "no code comments". Rationale goes in the commit
message, the ROADMAP row or a doc. CLI help text is a string literal passed
to argparse, never `__doc__`.

## Lanes and clearing (binding, owner, 2026-09-12)

The owner clears the conversation as often as possible; the agent's job is
to make that free. Every turn re-sends the whole transcript, so undistilled
tool output is the cost and the lane file is the cure.

- One file per task under `docs/lanes/` (format in its README): Goal, State,
  Tried, Next, Open, under one page. Update it when a unit of work completes,
  before saying the work is done. Finished lanes move to `docs/lanes/done/`.
- **Every response ends with one line, the clear verdict**, enforced by the
  Stop hook: `CLEAR OK - <what is saved and where>` when a fresh session
  could continue from the lane file, memory and `HANDOFF.md` alone; or
  `HOLD - <reason>` when the next step depends on state that exists only in
  this conversation (a debugging loop mid-cycle, an answer awaited from a
  running task, a decision the owner is about to make on numbers just
  shown). HOLD is the exception and names what would be lost.
- Anything the owner has had to explain twice is written to memory or a lane
  file, never carried in the conversation.
- Discovery that would dump files or logs into the main context goes to a
  subagent or an opencode lane with a context packet; only the conclusion
  comes back.

## Token discipline

- Never read ROADMAP.md, README.md, `docs/roadmap_archive.md`,
  `docs/research_history.md` or `docs/agents_history.md` whole; grep for the
  row or heading you need.
- Prefer the summary form of every command that has one: `weak-signals pool`
  without `--full`, `capture_scheduler.py --status --brief`, `pytest -q`,
  and pipe ruff and mypy through `Select-Object -Last 5`.
- Delegate multi-file searches and mechanical edits to subagents with a
  context packet (files, functions, line numbers, the exact verification
  command); keep the conclusion, not the file dumps. Subagent reports are
  claims until verified by running the command.

## Repository hygiene

- Python 3.12 and the locked uv environment.
- Never commit raw or processed data, model artifacts, credentials, cookies,
  virtual environments or test output; the tracked Markdown prediction card
  is the deliberate exception.
- Preserve unrelated user changes; never rewrite Git history.
- Do not commit or push unless the user explicitly asks.

## Required verification

Run after code changes:

```powershell
.\.tools\uv.exe run --no-sync ruff format --check . 2>&1 | Select-Object -Last 5
.\.tools\uv.exe run --no-sync ruff check . 2>&1 | Select-Object -Last 5
.\.tools\uv.exe run --no-sync mypy src 2>&1 | Select-Object -Last 5
.\.tools\uv.exe run --no-sync pytest -q 2>&1 | Select-Object -Last 15
```

## Automatic session handoff

- Update documentation and `ROADMAP.md` when evidence or priorities change.
- If the active weekly forecast changed, run `nfl-ats publish-predictions`.
- The agent owns handoff refreshes. Never ask the user to run the handoff command.
  Refresh `HANDOFF.md` when work is ready to hand off and before every commit
  or push intended for `master`; the pre-commit hook is a backstop.
- Before pushing `master`, run `nfl-ats handoff --check`; if a refresh changes
  the file after a commit, make the follow-up commit before pushing.
- Report remaining Git changes plus the exact checks run.
