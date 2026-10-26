# Dependency security update 2026-10-26

## Goal

Close the 23 open Dependabot alerts on `uv.lock` (1 critical, 14 high).

## State

- **Measured:** alerts covered three packages: pypdf (<6.21.2), certifi
  (<2026.10.19, emSign root CA compromise, critical), urllib3 (<2.8.0).
- `pyproject.toml` pypdf floor raised to `>=6.21.2`; `uv lock
  --upgrade-package` for the three moved only them: pypdf 6.16.2 -> 6.21.2,
  certifi 2026.7.22 -> 2026.10.23, urllib3 2.7.0 -> 2.8.0. Windows `.venv`
  synced with `uv sync --locked`; HTTPS request through requests succeeded.
- **Measured:** `pytest -m "not full"` 1637 passed, 0 failed; the 144
  collection errors are all PermissionError on ACL-locked leftovers under
  `tests/scratch/ui20_render/`. `mypy src` clean (254 files).

## Tried

- `scripts/verify_fast.py` stops at `ruff format --check .` (211 files
  unformatted, pre-existing, no Python touched here), so pytest and mypy were
  run directly.

## Next

- Second capture host (`docs/offsite_backup.md`) receives code by scp, not git;
  it keeps the old certifi/urllib3 until `uv.lock` is copied and
  `./.tools/uv sync --no-dev` runs there.

## Open

- Pre-existing format drift across 211 files blocks the fast tier.
