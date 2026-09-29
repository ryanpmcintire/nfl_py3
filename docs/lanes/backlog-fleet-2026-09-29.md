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

## Tried
- `codex --no-daemon exec` launcher: blocked by permission classifier.

## Next
- Launch wave 1; on each return verify, run record commands serially, commit,
  push, refill from ideation rows.

## Open
- Codex workers need an owner Bash allow rule for `codex --no-daemon exec`.
