# Working-tree safety for concurrent agent sessions (ENG-31)

Added 2026-09-04 after a measured incident: with many subagents editing one
working tree, a single agent ran `git stash` and then `git stash pop`, which
failed on a conflict and reverted other agents' uncommitted edits to
`src/nfl_ats/cli.py`, `publishing.py`, `weekly.py`, `provenance.py`,
`pyproject.toml` and 24 test files. Every edit was re-applied by hand, but the
`nfl-ats preflight` registration was only recovered from `stash@{0}` because
the coordinator diffed the stash against the tree.

## Rules

1. **Never run `git stash`, `git checkout -- <path>`, `git restore`,
   `git reset --hard`, `git clean -f`, or `git worktree remove --force` from an
   agent session.** None of them recovers anything that cannot be recovered by
   editing the file, and all of them can destroy another session's work.
   `git stash list` and `git stash show` are read-only and stay allowed.
2. **Never rewrite a shared module with a whole-file write.** Use anchored,
   additive edits; re-read the file immediately before each edit; if your edit
   is missing when you re-read, another session removed it, so re-apply it in
   place rather than restoring from git.
3. **Keep concurrency small and files disjoint.** The owner's rule is about
   three subagents at a time on non-overlapping files.
   `.claude/settings.local.json` sets `CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS`
   to 3 to enforce it mechanically.
4. **Use an out-of-repo `--basetemp`** for pytest (see
   `docs/verification_tiers.md`), never a shared default temp directory, when
   other sessions may be running tests.

## Enforcement

`.claude/settings.json` registers a `PreToolUse` hook on `Bash|PowerShell`
that runs `.claude/hooks/guard_git.py`. The script reads the tool call's
command and returns a deny decision when one of the rule-1 commands appears in
command position (start of a line or after a shell separator), naming the
match and this document. It is a harness hook, so it applies to subagents as
well as the main session. A denied command is not an error in the guard: it
means the intended change must be made by editing files.

Limits, stated plainly:

- The whole `.claude/` directory is gitignored in this repo, so the hook and
  its script live only on the machine where sessions run. A fresh clone must
  recreate them (copy the two files from this machine or from this doc's
  history).
- The hook sees only commands issued through the Bash and PowerShell tools. A
  human at a terminal, a script that shells out to git, or git hidden inside
  another interpreter is not covered.
- The git pre-commit hook in `.githooks/` cannot intercept stash or checkout
  because git provides no hook for them.
- First-day calibration: the initial pattern matched the words anywhere in the
  command text and denied a heredoc that was writing this very document. The
  pattern is now anchored to command position.
