# Wave 2 — mechanical `docs/archive/` move (`docs-archive-triage`)

Executes the mechanical ARCHIVE subset of the wave-1 triage
`reports/wave1/hyg-docs-rot.md` (recovered from commit `2b8cebe` on branch
`swarm/hyg-docs-rot`; **read** this session — the file was not present in this
worktree's `reports/wave1/`). Branch: `swarm/docs-archive-triage`. Nothing was
deleted and no prose content was rewritten; the only edits to file bodies are
path-pointer updates required by the move ("fixing relative links").

## Provenance summary

Every claim below is **measured** (command run this session) or **read**
(file opened this session) unless tagged otherwise.

## Rows executed (7 moves via `git mv`, measured: `git status --short`)

Per the §E/§F table of `hyg-docs-rot.md`, verdict **ARCHIVE**, no CORRECT/
MERGE qualifier:

| Old path | New path | Table reason |
|---|---|---|
| `docs/idea_ledger.md` | `docs/archive/idea_ledger.md` | Session note (2026-08-22 consolidated ledger). |
| `docs/literature_leads_20260821.md` | `docs/archive/literature_leads_20260821.md` | Dated literature-mining session output. |
| `docs/opus_execution_specs.md` | `docs/archive/opus_execution_specs.md` | Agent execution specs (2026-08-17). |
| `docs/opus_session_blockers.md` | `docs/archive/opus_session_blockers.md` | Agent session blockers (2026-08-17). |
| `docs/data_source_scout_v2.md` | `docs/archive/data_source_scout_v2.md` | Superseded by scout v5. |
| `docs/data_source_scout_v3.md` | `docs/archive/data_source_scout_v3.md` | Superseded by scout v5. |
| `docs/data_source_scout_v4.md` | `docs/archive/data_source_scout_v4.md` | Superseded by scout v5. |

`docs/data_source_scout_v5.md` stays put per the same table (**KEEP**,
current milestone).

## Row deliberately skipped (needs judgment)

- `docs/revisit_list.md` — table verdict is "**CORRECT + ARCHIVE**". My task
  says to skip any row marked correct (judgment required: D2-retraction note +
  PageRank family status fix must land before or alongside archiving). Left in
  `docs/revisit_list.md` untouched (**measured**: `git status --short` shows no
  change to it).

## Link fixes applied

**Outbound:** the 7 moved files contain zero markdown-style `]()` links
(**measured**: `grep -oE '\]\([^)]+\)'` over all 7 → empty); their references
are root-relative `` `docs/X.md` `` prose paths, which resolve identically from
the new location. No outbound changes were needed beyond the inbound-style
rewrites below.

**Inbound + cross-references between moved files:** rewrote
`docs/<name>.md` → `docs/archive/<name>.md` for exactly the 7 moved names in
35 tracked files (**measured**: rewrite file list + post-check grep returning
zero remaining old-path references outside `reports/`):

- 4 of the moved files themselves (cross-references to sibling moved files,
  e.g. scout v4 → v2/v3, idea ledger → literature leads).
- 16 kept `docs/*.md` files.
- `ROADMAP.md` (3 references).
- 13 `scripts/*.py` + `src/nfl_ats/experiment_runner.py` (17 comment/
  docstring path mentions total).

Line-length safety for the `.py` rewrites (+8 chars each): verified no line
exceeds the ruff `line-length = 100` limit before applying (**measured**).
No test references any moved filename (**measured**: grep over `tests/`
returns nothing).

## Gates

Docs-plus-comment change; all four gates run because `src/`/`scripts/` comment
strings were touched:

```bash
/f/Repos/nfl_py3/.venv/Scripts/ruff.exe format --check .
/f/Repos/nfl_py3/.venv/Scripts/ruff.exe check .
/f/Repos/nfl_py3/.venv/Scripts/mypy.exe src
PYTHONPATH="$WT/src" /f/Repos/nfl_py3/.venv/Scripts/python.exe -m pytest -q \
    --basetemp="C:/Users/Ryan/AppData/Local/Temp/nflats-swarm-basetemp"
```

Results recorded below after execution.

## Not done here (deferred to judgment agents, unverified-by-design)

- All **CORRECT** rows of the triage table (broken-link typo fixes, stale
  verdict updates, `modeling.md` missing-artifact flag).
- The skipped `revisit_list.md` correct+archive row.
- Cross-link additions within KEEP clusters (best-pick/weather/injury families).
