# Backlog fleet 2026-09-29

## Goal
Clear the backlog and add accretive, non-overfit accuracy work with a fleet of
at least seven parallel workers, committing and pushing at each verified return.

## State
- Codex `--no-daemon exec` pilot ran headless (no console window in
  `.tmp/popup-trace/windows.jsonl`), but the Claude Code auto-mode classifier
  denied launching Codex workers ("Create Unsafe Agents"). Decision: run the
  lanes as Claude Code subagents (the fleet default per memory `agent-fleet`),
  six concurrent, refilling each slot on return.
- Wave 1 packets: `tests/scratch/codex/*.md` (preamble + odds-fri, ui20,
  ideation, st-study, tiebreaker, xlg09, mod22).

- Owner wants Codex workers (2026-09-29). Decision: the four Claude subagents
  already running (ui20, ideation, st-study, mod22) finish so their work is not
  discarded; every new lane launches as `codex --no-daemon exec` once the allow
  rule exists. No new Claude subagents.
- Uncommitted returns: tiebreaker (docs/tiebreaker_total_study.md), XLG-09 unit 6
  (docs/xlg09_unit6.md, four record commands pending).

- Owner added allow rule `Bash(codex --no-daemon exec:*)`. Codex wave 1 running
  (launch must start with `codex --no-daemon exec`; prompt = preamble + packet in
  `tests/scratch/codex/<lane>.prompt.md`, stdin redirect): ui20, st-study,
  lead66, lead67, lead68, lead69, lead72. Outputs `<lane>.final.md`.
- Committed: odds window fix; XLG-09 unit 6, MOD-22 unit 4, tiebreaker study,
  LEAD-66..73 rows, records 7428-7430.

- Committed: LEAD-68, ST study, LEAD-72 (all unresolved), UI-20 tiebreaker
  header + count wording (published). Running: lead66/67/69/70/71/73,
  ideation2, tbfix.
- Bug: margin-predict-challenger re-points active weekly_forecast to 194213Z after
  publish; tiebreaker panel shows "Not available yet". Lane
  docs/lanes/tiebreaker-forecast-pointer.md (tbfix worker).

- Queue for refills: LEAD-75..81 (ROADMAP Phase 12), LEAD-69 unit 2 after
  LEAD-73 unit 2. MOD-22 unit 5 reuses unit 4's timestamp filter; if its result
  lands before mod22fix, rerun it on the repaired filter before recording.
- Verify lane reopened MOD-22 unit 4 (guessed report times); record replaced
  as unresolved. XLG-09 unit 6 and ST lag re-derive exactly.

## Tried
- `codex --no-daemon exec` launcher: blocked by permission classifier.

## Next
- Launch wave 1; on each return verify, run record commands serially, commit,
  push, refill from ideation rows.

## Open
- Codex workers need an owner Bash allow rule for `codex --no-daemon exec`.
