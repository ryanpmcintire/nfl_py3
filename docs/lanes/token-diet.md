# token-diet

## Goal

Cut what a session pays before doing any work, and make clearing between
units of work cheap enough to do after almost every completed unit.

## State

Pushed as `d57ac7e` on master (2026-09-12): `weak-signals pool` summary by
default (`--full` for rows), `capture_scheduler.py --status --brief`,
AGENTS.md rewritten to rules only (old text in `docs/agents_history.md`),
235 done ROADMAP rows and wave logs in `docs/roadmap_archive.md`, README
history in `docs/research_history.md`. Local only, because `.claude/` is
gitignored: the per-turn hook shortened to 1 KB, a Stop hook
(`.claude/hooks/clear_verdict.py`) that requires the clear verdict line, and
a status line (`.claude/hooks/statusline.py`) showing context use and the
active lane. Memory pruned: 27 old session logs folded into one archive.

## Tried

- Paid opencode models: 401 "No payment method"; the six `-free` models and
  `big-pickle` all worked. Report paths must be inside the repo
  (`tests/scratch/lanes/`), opencode refuses writes outside it.
- Line-level conservation check on the ROADMAP split: 0 of 2,135 lines lost.

## Next

Measure the first few cleared sessions: fixed startup cost from the first
assistant usage in the transcript, and whether the lane file was enough to
continue without re-explaining. If a re-explanation happens, the missing
fact goes into memory or the lane file, not into the conversation.

## Open

- ROADMAP.md is still 325 KB because MOD-17, MOD-18 and UI-20 hold over
  15 KB each of owner prose; trimming them needs a go-ahead.
- Whether to track `.claude/hooks/` and `.claude/settings.json` so the hooks
  and status line follow the repo to other machines.
