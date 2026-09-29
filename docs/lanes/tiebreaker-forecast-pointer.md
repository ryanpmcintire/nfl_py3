# Tiebreaker forecast pointer

## Goal
Keep prospective challenger predictions from replacing the served weekly forecast; document restoration of the published week 4 artifact for the orchestrator.

## State
Fix verified in `src/nfl_ats/weekly.py:239` and activation caller `src/nfl_ats/cli_commands/prediction.py:161,177,385,909`. `--no-activate` skips `activate_matching_ats_model` and leaves challenger metadata UNLINKED; the regular prediction still activates. Existing fixture `tests/test_prediction_safety.py:516` supplies the new parser default after a legitimate failure; no tests added. No artifact writes, publication, or Git mutations.

## Protocol
Declared before inspection: maintenance only; preserve the board staleness guard and add an explicit challenger opt-out. No research population, target, terms, folds, metrics, fits, or scores; looks = 0, intervals not applicable. Verification: Ruff, mypy, requested existing pytest selection, and weekly-run dry-run.

## Tried
- **Read:** the unconditional activation call wrote the pointer through `active_model.py:113-171`; `board_content.py:3411` correctly rejects mismatched tiebreaker provenance. That guard is unchanged.
- **Measured:** `ruff check --no-cache` and `ruff format --check --no-cache` pass for all three touched Python files; `mypy src` passes (253 files); parser activation is True for the card and False with the flag. All Python commands used `.tools/uv.exe run --no-sync` and a writable temporary UV cache.
- **Measured:** `nfl-ats weekly-run --season 2026 --week 4 --dry-run` exits 0, reports `published=false`, and includes `--no-activate` only on the challenger. Log: `$TEMP/nfl-tiebreaker-weekly-dry-run.json`.
- **Measured:** `pytest -q -n 2 --basetemp <fresh temporary directory> -k "weekly or margin or active"`: 121 passed, 16 fixture warnings, 12.24 seconds. Initial default-temp attempt failed before collection with WinError 5; the first completed selection found the missing fixture default (fixed above). Log: `$TEMP/nfl-tiebreaker-pytest.log`. `git diff --check` passes.
- **Measured:** read-only restoration preflight confirms model `b578fbea1c5c706f`, evaluation `margins/20260929T192312Z`, tiebreaker NO 25–ATL 23, and current pointer ending `194213Z` versus the card's `192403Z`.

## Next
Orchestrator reviews the changes and runs this bash-compatible restoration command (prepared, NOT executed), then handles board publication:

```bash
.tools/uv.exe run --no-sync python - <<'PY'
import json
from pathlib import Path
from nfl_ats.active_model import _matching_evaluation, activate_matching_ats_model, load_active_ats_model
root = Path("artifacts")
forecast = root / "margin_predictions/2026-week-04-20260929T192403Z"
metadata = json.loads((forecast / "metadata.json").read_text(encoding="utf-8"))
tie = json.loads((forecast / "tiebreaker.json").read_text(encoding="utf-8"))
active = load_active_ats_model(root)
assert active is not None
assert active["model_id"] == metadata["active_model_id"] == tie["model_id"]
assert active["weekly_forecast"]["season"] == metadata["season"] == tie["season"] == 2026
assert active["weekly_forecast"]["week"] == metadata["week"] == tie["week"] == 4
assert tie["forecast_artifact"] == forecast.relative_to(root).as_posix()
assert _matching_evaluation(root, metadata) == root / metadata["historical_evaluation"]["artifact"]
restored = activate_matching_ats_model(root, forecast, metadata)
assert restored is not None and restored["model_id"] == tie["model_id"]
assert restored["weekly_forecast"]["artifact"] == tie["forecast_artifact"]
print(json.dumps(restored["weekly_forecast"], indent=2))
PY
```

## Open
Live repair remains orchestrator-owned; restoration rewrites only the active model manifest through its existing activation function and does not rebuild predictions. No unresolved code or verification failures.

## Record commands
None: no research result or registry decision is produced by this maintenance task.
