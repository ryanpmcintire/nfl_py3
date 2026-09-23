# NFL ATS repository instructions

This file is the repository's normative agent policy. `docs/agents_history.md`
is historical rationale, not an additional source of requirements. Conditional
commands and session procedures live in `docs/agent_workflow.md`.

## Scope and ownership

- Complete authorized work, preserve unrelated changes, and never rewrite Git
  history.
- The primary orchestrator owns operational jobs, publication, handoff refresh,
  commits, and pushes. Subagents perform bounded delegated tasks and do none of
  those actions unless the task explicitly assigns one.
- Read-only audits, harness maintenance, instruction hygiene, and subagent tasks
  do not trigger operational jobs, dashboard work, publication, or handoff.
- Delegate with a small context packet naming files or functions, scope,
  constraints, and verification. Every coding packet says `no code comments`.
- In an explicit backlog session, the primary orchestrator keeps available
  subagents on concrete bounded useful work while progressing its own task.
  Refill completed assignments within the current bounded backlog batch while
  independent authorized work remains. Finish verification, commit, push, and a
  short handoff at a clear stopping point before starting a fresh thread. Never
  create automatic or open-ended goals.
- Do not hardcode changing repository counts, current model names, or agent
  tiers in general policy.

## Research invariants

- Research and paper decisions only; never add automated wagering.
- Pregame features use only information available before the prediction
  timestamp. Enforce leakage in the feature builder at runtime.
- Evaluate chronologically. Keep validation, selection, calibration, and outer
  test periods distinct, and preserve prediction-level output.
- Compare every addition against the same market and simple-model baselines.
  Report uncertainty and season stability, including negative results.
- Never present historical forced-pick accuracy as proof of a profitable or
  stable edge. Keep it distinct from each game's model probability.
- Prediction-safety and evaluator-performance contracts are release-blocking.

### Margins are multimodal

Final margins are discrete, with mass at key numbers 3, 7, 10, 14, and 17.
Every served cover probability, push probability, flip line, or alternative-line
answer uses the discrete margin distribution conditional on the line. A smooth
pooled-residual read is only a challenger and must beat the discrete read on the
opener grade through the played card before being served. Comparing two pooled
residual mappings does not test this requirement.

No rule may flip the model's pick without naming a mechanism. A dip located
only at a spread threshold is a diagnosis to publish and repair, never a flip
to bolt on.

### An interval crossing zero is not grounds for rejection

- Never discard, close, or decline to build a signal because its interval
  contains zero. At this evaluator's resolution, that is expected for a real
  small signal.
- Only two grounds close a line of work: a refuted mechanism, meaning a resolved
  wrong sign with the whole interval on the wrong side of zero or zero
  split-half reliability; or a positive control proven able to detect an effect
  of that size. Everything else is `unresolved_below_power`.
- Record every unresolved result with `nfl-ats weak-signals record` before a
  write-up calls it settled. A terminal classification or `closed_negative`
  rotation verdict must name an admissible `--closing-ground`.
  `wrong_sign_resolved` requires the whole interval on the wrong side. If the
  record command errors, the verdict is wrong; never weaken the validator.
- Report `probability_positive`, never the binary phrase "contains zero."
- Pool sub-signals only when inputs are commensurable in units, scale, and
  population and the family was declared before signs were seen. The living
  reference is `nfl-ats weak-signals pool --league nfl --effect-units
  accuracy_points`; use summary output by default and `--full` for rows. A
  `needs_remeasurement` row is a dead heat whose stored value should be 0.5;
  it closes nothing.

### One calibrated probability decides every pick

- One calibrated probability combines the model with every situational signal
  as a fitted term and selects the served side. No rule changes a side alone.
  A member returning only `p` or `1 - p` is an inadmissible flip. Off switches
  are `LATE_WEEK_FOLLOW_SERVED`, `HANDLE_FOLLOW_SERVED`,
  `ROOKIE_CREW_SERVED`, and `OWNER_HELD_MEMBERS`.
- Choose every parameter out of season. Fit weights, thresholds, and cut points
  leave-one-season-out and score the held-out season. Report in-sample and
  out-of-sample results together with their gap. A constant derived from games
  it is scored on never reaches `src/`.
- Report calibration with a reliability table and log loss or Brier score
  against market and model-only baselines, not hit rate alone. State combined
  model coefficients per fold and their stability.
- Compare small lopsided splits with a permutation or exact null before claiming
  two populations. A flat or uniform claim includes a slope and interval.
- Count every band, cell, arm, and continuous fit as a look; state the count and
  family, do not present the best cell as the finding, and report the record on
  decisive games before the headline effect.

Any subagent that fits, scores, or adjudicates research must first read the
research rules above. Its context packet explicitly states that zero crossing
does not close a signal and that one fitted probability selects the side. Its
verdicts go through `weak-signals record` or `rotation record-look`, and the
primary orchestrator verifies every decision-gating claim and command.

### Promotion, reporting, and flip lines

- A promotion bar is not a decision bar. Failure to reach 0.90 or 0.95 is not
  grounds to reject or close a signal; probability just above 0.5 and a record
  such as 19-18 are not sufficient reasons to serve it. Research closure and
  card serving are separate decisions. Grade decisions at the opener; a
  close-graded number never vetoes a play. State what a result implies for the
  decision before criticizing it.
- Label factual claims that can change a research, serving, release, or
  operational decision inline as **measured** (command or artifact from this
  session), **read** (current file and line), **reported** (unverified source),
  or **inferred** (reasoning, separate from measured numbers). Give the number
  and interval before a verdict, and cite the rule imposing a gating constraint.
  Mundane workflow statements need no provenance label.
- A flip line is one number in the adverse direction: how far the line must move
  against the pick before the card changes side. Never give a range or a
  favorable-direction flip. If none exists before the adverse edge, say the
  pick holds through that edge. `_flip_line` and `flip_line_text` in
  `board_content.py` enforce this.

## Dashboard and publication

- A task that changes the dashboard or forecast includes a visible
  reader-facing improvement in the current design system. Add fixture coverage
  only when an existing contract test requires it. `ROADMAP.md` row `UI-20`
  holds the candidate list.
- After such a task, the primary orchestrator regenerates and publishes the
  affected site output, reviews the rendered-page diff, fixes fail-closed
  publication in-session, and pushes so GitHub Pages rebuilds.
- Every displayed percentage comes from an artifact keyed to the active model.
  `publish-board` fails closed if a headline number names another model.
- Reader-facing text contains no snapshot IDs, timestamps, hashes, model IDs,
  policy slugs, column names, research jargon, or compliance boilerplate.
  Provenance belongs in `lineage.json` and `explanations.json`.

## Test moratorium

- Add no test files or test functions until the owner lifts the moratorium. Ship
  a fix as the fix and verify it by running the real command once.
- Edit existing tests when a legitimate change breaks them; do not expand them.
  Durable tests under `tests/` protect prediction safety, chronology, evaluator
  arithmetic, board rendering, scheduler argv, and registry closure. They never
  pin a research number or exercise a one-off script. Files under
  `tests/scratch/` remain ignored and are never promoted by moving them.
- After code review, propose removal of redundant contract coverage, research
  number assertions, one-off script tests, mocks or fixtures that assert an
  assumption instead of reading the production artifact, and research-only
  runtime assertions duplicated by served-path guards. The owner decides which
  existing tests to delete.

## Code and repository hygiene

- No `#` comments or docstrings of any length in `.py`, `.ps1`, `.sh`, or `.cmd`.
  Only shebangs and pragmas such as `# noqa`, `# type:`, and `# pragma` are
  exempt. Put rationale in a commit message, roadmap row, or document. CLI help
  is an argparse string, never `__doc__`.
- Use Python 3.12 and the locked uv environment. Follow
  `docs/agent_workflow.md` for Windows and Linux environment separation.
- Never commit raw or processed data, model artifacts, credentials, cookies,
  virtual environments, or test output. The tracked Markdown prediction card
  is the deliberate exception.
- Search narrowly from the named file or owner module, summarize long output in
  a temporary log, and treat truncation as unread context rather than absence.

## Lanes and completion

- Keep one task file under `docs/lanes/` with Goal, State, Tried, Next, and Open,
  under one page; move completed lanes to `docs/lanes/done/`. Put guidance the
  owner has repeated into policy or the lane instead of conversation history.
- Only the final response ends with `CLEAR OK - <what is saved and where>` or
  `HOLD - <conversation-only state that would be lost>`. `HOLD` reports fragile
  state; it never pauses or cancels otherwise authorized work.
- A progress or status note is not a stopping point. Continue authorized work
  through review, applicable verification, and the repository completion
  workflow while useful work remains.
- The primary orchestrator follows `docs/agent_workflow.md` for conditional
  handoff, commit, and push. Standing authorization applies at verified clear
  stopping points; do not ask again.
