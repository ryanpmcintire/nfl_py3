# Environment preflight (ENG-02)

`nfl-ats preflight` is a read-only check that answers one question before any
research command runs: is the local machine set up correctly, independent of
whether research data has been rebuilt yet? It never writes Git config, never
fetches from a network source, and never mutates `data/`, `artifacts/`, or
`registry/`. The only filesystem writes it performs are writability probes: a
uniquely named temporary file is created and immediately removed.

Implementation: `src/nfl_ats/preflight.py` (pure, testable functions) plus a
thin CLI handler in `src/nfl_ats/cli.py` (`_cmd_preflight`). Tests:
`tests/test_preflight.py`.

## Usage

```powershell
.\.tools\uv.exe run --no-sync nfl-ats preflight
.\.tools\uv.exe run --no-sync nfl-ats preflight --json
.\.tools\uv.exe run --no-sync nfl-ats preflight --strict
```

## What is checked, and why it is categorized that way

Every check carries a `category` in `{environment, research_data,
configuration}`. The categories exist specifically so a fresh clone with no
local data does not look broken, while a genuinely broken local toolchain
never looks fine.

### `environment` — local tooling that blocks all work if broken

- **Python interpreter version** — the running interpreter is compared
  against `pyproject.toml`'s `[project.requires-python]` (falls back to a
  hardcoded `>=3.12,<3.14` if `pyproject.toml` is missing or unparsable).
- **uv executable available** — resolved from `.tools/uv.exe`, `.tools/uv`,
  then `PATH`, and confirmed runnable via `uv --version`.
- **uv cache accessible** — `uv cache dir` is queried (no network call) and
  the resulting directory is probed for writability. If the cache directory
  does not exist yet (uv creates it lazily on first use), the nearest
  existing ancestor is probed instead and the row is `warn`, not `fail`.
- **git executable available** — via `PATH`.
- **git hooksPath configuration** — reads (never sets) `core.hooksPath` and
  compares it against the repo's required value, `.githooks` (see
  `AGENTS.md`, "Session startup"). This is the same setting the pre-commit
  hook depends on for the automatic `HANDOFF.md` refresh.
- **artifacts / data / registry directory writable** — each of the three
  configurable roots (`NFL_ATS_ARTIFACTS_DIR`, `NFL_ATS_DATA_DIR`,
  `NFL_ATS_REGISTRY_DIR`, defaulting to `artifacts/`, `data/`, `registry/`)
  is probed for writability. A missing directory is `warn` (it will be
  created on first write by the existing `atomic_*` helpers in
  `nfl_ats/io.py`); a directory that exists but rejects the write probe is
  `fail`.

Any `fail` in this category means real commands will not run correctly
regardless of what research data is present, so it is always exit-blocking.

### `configuration` — source-policy inputs

- **Source policy API keys** (`THE_ODDS_API_KEY`, `CFBD_API_KEY`) — presence
  or absence is reported; **values are never read into the report**. Absence
  is `warn`, not `fail`, since most commands do not need live network
  sources. A `fail` in this category is reserved for cases where a required
  piece of configuration is actively broken rather than merely absent (the
  current check set only ever reports `ok`/`warn` here, but the category is
  exit-blocking by the same rule as `environment` if that ever changes).
- **Directory overrides** (`NFL_ATS_DATA_DIR`, `NFL_ATS_ARTIFACTS_DIR`,
  `NFL_ATS_REGISTRY_DIR`) — reports whether each is set and to what path.
  Paths are not secrets, so they are shown; API key values never are.

### `research_data` — local artifacts a fresh clone legitimately lacks

Reuses the exact inventory rendered in `HANDOFF.md`'s "Local reproducibility
inventory" section (`nfl_ats.handoff._local_inventory`, imported rather than
duplicated so the two views cannot drift): canonical/PBP/player/player-value
feature tables, the participation source snapshot, participation-rating and
learned-availability research features, the frozen player-model selection,
the participation-rating and learned-availability experiment artifacts, and
the active model manifest. Presence is a simple `Path.is_file()` check — no
content/integrity validation.

**This category only ever reports `ok` or `warn`, never `fail`.** A fresh
clone with no local artifacts is a legitimate, common, and expected state,
not an error. Rebuilding these is normal research work, not preflight's job.

## Exit-code rule

- **Default:** exit code is non-zero only if some `environment` or
  `configuration` row has status `fail`. Missing research data (`warn` rows
  in `research_data`) is reported but never fails the command on its own.
- **`--strict`:** exit code is also non-zero if *any* row anywhere is not
  `ok` — including missing research data. Use this in a context (e.g. a
  release gate) where an incomplete local artifact set should itself block.

## What this deliberately does not do

- It does not configure `core.hooksPath`, install `uv`, or create any
  directory. `AGENTS.md`'s session-startup step (auto-configuring
  `core.hooksPath` when it drifts) is a *separate*, explicitly mutating
  action owned by the agent's session startup, not by this read-only check.
- It does not validate the *contents* of research artifacts (schema,
  row counts, freshness) — only presence. That is the job of `nfl-ats
  doctor` (runtime/data health snapshot) and the various data-contract
  smoke tests (`FND-10`).
- It does not fetch anything over the network, including to validate that a
  present API key actually works.
