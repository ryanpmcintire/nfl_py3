"""ENG-13: reproducible run replay.

A read-only command that answers "does this recorded run still reproduce"
without refetching anything or touching a production artifact. It consumes
either of the two manifest shapes this repository already writes:

* an ENG-01 lock-day package ``manifest.json``
  (``nfl_ats.lockday_package``, ``kind == "lockday_decision_package"``); or
* a forecast artifact's ``metadata.json`` -- the file ``margin-predict``
  writes beside ``predictions.csv`` -- identified by carrying a
  ``provenance`` block (``nfl_ats.provenance.artifact_provenance``).

For either shape, :func:`replay_manifest` performs four checks, in order:

1. **Digests.** Every source/feature-table/forecast/card digest the manifest
   actually recorded is recomputed from disk and compared. For a lock-day
   package this reuses :func:`nfl_ats.lockday_package.verify_package`
   verbatim -- the same verifier ``scripts/lockday_package_verify.py`` calls
   -- rather than re-implementing digest comparison. A bare forecast
   ``metadata.json`` records fewer digests (only ``provenance.feature_table``
   and ``provenance.uv_lock_sha256``); those are checked the same way, reusing
   :func:`nfl_ats.lockday_package.hash_entry` for the hashing itself.
2. **Environment.** The manifest's recorded environment block (if any) is
   diffed against :func:`nfl_ats.environment_report.environment_report` run
   now, via :func:`nfl_ats.environment_report.compare_environment`, which
   classifies every differing field as reproducibility-affecting or cosmetic.
3. **Code revision.** The manifest's recorded git revision and dirty flag are
   compared against ``git rev-parse HEAD`` / working-tree status now. This is
   reported for context; per this repository's CLI exit-code contract (see
   ``scripts/replay_run.py``) a revision mismatch alone does not fail replay
   -- replaying an old run from a newer checkout is an expected use, not an
   error.
4. **Recompute (optional, default on).** If digests verify, the forecast is
   regenerated in-process for the manifest's own season/week -- the same
   ``score_outcome_week`` call ``margin-predict`` makes, driven by the
   configuration the manifest itself recorded -- into a caller-supplied
   TEMPORARY ``output_root`` (never a production artifact directory). The
   regenerated predictions table is compared column-by-column against the
   recorded ``predictions.csv``, and a small mechanically-derivable subset of
   metadata (games/methods/game_type) is compared against the recorded
   metadata. Timestamps, ``provenance``, and ``environment`` are excluded by
   design -- they are expected to differ on any later replay and are already
   covered by checks 2-3 above.

Guarantees, by construction:

* **Never fetches.** Every read is local disk (the manifest, the feature
  table, ``uv.lock``, a `git` subprocess call). No network I/O anywhere in
  this module.
* **Never writes outside ``output_root``.** The only writes this module ever
  performs are ``regenerated_predictions.csv`` and ``regenerated_metadata.json``
  inside the caller-supplied ``output_root``, and only when recompute actually
  runs. Nothing under ``artifacts/`` or ``registry/`` is ever touched.
* **Never touches ledgers.** Recompute calls ``score_outcome_week`` directly --
  the pure scoring function -- never ``weekly-run``, never a recorder, never
  ``--record-decisions``.

This module reports rather than gates: per this repository's binding research
invariant, a difference alone is context, not an automatic verdict. The CLI
(``scripts/replay_run.py``) turns :attr:`ReplayReport.ok` into an exit code;
this module just measures.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.environment_report import compare_environment, environment_report
from nfl_ats.io import atomic_csv, atomic_json
from nfl_ats.lockday_package import (
    MANIFEST_FILENAME,
    PACKAGE_KIND,
    hash_entry,
    verify_package,
)
from nfl_ats.outcomes import score_outcome_week
from nfl_ats.provenance import git_state

KIND_LOCKDAY_PACKAGE = "lockday_decision_package"
KIND_FORECAST_METADATA = "forecast_metadata"
KIND_UNKNOWN = "unknown"

REPLAY_SCHEMA_VERSION = 1

#: A forecast generator: ``(feature_table, configuration) -> predictions``.
#: Overridable so tests can replay against a tiny synthetic forecast without
#: fitting a real model, and so a future caller can plug in a different
#: scoring entry point without editing this module.
GenerateForecast = Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame]


# ---------------------------------------------------------------------------
# report shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayReport:
    """The result of one :func:`replay_manifest` call."""

    manifest_path: str
    manifest_kind: str
    season: int | None
    week: int | None
    digest_verification: dict[str, Any]
    git_revision: dict[str, Any]
    environment_comparison: dict[str, Any]
    recompute: dict[str, Any] | None
    notes: tuple[str, ...]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "manifest_path": self.manifest_path,
            "manifest_kind": self.manifest_kind,
            "season": self.season,
            "week": self.week,
            "digest_verification": self.digest_verification,
            "git_revision": self.git_revision,
            "environment_comparison": self.environment_comparison,
            "recompute": self.recompute,
            "notes": list(self.notes),
            "ok": self.ok,
        }


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _default_repo_root() -> Path:
    # src/nfl_ats/run_replay.py -> parents[2] is the repo root, the same
    # pattern nfl_ats.environment_report._default_project_root() uses.
    return Path(__file__).resolve().parents[2]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _detect_kind(manifest: Mapping[str, Any]) -> str:
    if manifest.get("kind") == PACKAGE_KIND:
        return KIND_LOCKDAY_PACKAGE
    if isinstance(manifest.get("provenance"), Mapping):
        return KIND_FORECAST_METADATA
    return KIND_UNKNOWN


def _load_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    """Accept a lock-day package folder, a forecast artifact folder, or a
    manifest/metadata file directly."""

    candidate = path
    if candidate.is_dir():
        lockday_candidate = candidate / MANIFEST_FILENAME
        metadata_candidate = candidate / "metadata.json"
        if lockday_candidate.is_file():
            candidate = lockday_candidate
        elif metadata_candidate.is_file():
            candidate = metadata_candidate
        else:
            raise FileNotFoundError(
                f"{path} is a directory with neither {MANIFEST_FILENAME} nor metadata.json"
            )
    if not candidate.is_file():
        raise FileNotFoundError(f"Manifest not found: {candidate}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{candidate} does not contain a JSON object")
    return payload, candidate


# ---------------------------------------------------------------------------
# digest verification
# ---------------------------------------------------------------------------


def _forecast_metadata_digest_entries(
    manifest: Mapping[str, Any], repo_root: Path
) -> list[dict[str, Any]]:
    """The digests a bare forecast ``metadata.json`` actually records.

    Narrower than a lock-day package by construction: ``margin-predict`` on
    its own only pins ``provenance.feature_table`` and
    ``provenance.uv_lock_sha256``. It never hashes its own output files (that
    is what the ENG-01 package adds), so there is nothing recorded to verify
    for the predictions/card side of this manifest shape.
    """

    entries: list[dict[str, Any]] = []
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        return entries
    feature_table = provenance.get("feature_table")
    if isinstance(feature_table, Mapping):
        path = feature_table.get("path")
        digest = feature_table.get("sha256")
        if path and digest:
            entries.append({"role": "feature_table", "path": str(path), "sha256": str(digest)})
    uv_lock_sha256 = provenance.get("uv_lock_sha256")
    if uv_lock_sha256:
        entries.append(
            {
                "role": "uv_lock",
                "path": str(repo_root / "uv.lock"),
                "sha256": str(uv_lock_sha256),
            }
        )
    return entries


def _verify_digest_entries(entries: Sequence[Mapping[str, Any]], repo_root: Path) -> dict[str, Any]:
    """Recompute each entry's digest via :func:`hash_entry` (ENG-01) and compare.

    Deliberately reuses ``hash_entry`` for the hashing itself rather than
    re-implementing sha256 comparison -- the same reuse
    :func:`nfl_ats.lockday_package.verify_package` already embodies for the
    lock-day package shape.
    """

    verified: list[str] = []
    changed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for entry in entries:
        role = str(entry.get("role", ""))
        expected = entry.get("sha256")
        path = Path(str(entry.get("path", "")))
        described = {"role": role, "path": str(path)}
        if not path.is_file():
            missing.append(described)
            continue
        actual_entry = hash_entry(role, path, repo_root=repo_root)
        actual = actual_entry.get("sha256")
        if actual == expected:
            verified.append(str(path))
        else:
            changed.append({**described, "expected": expected, "actual": actual})
    return {
        "kind": "forecast_metadata_digests",
        "files_checked": len(entries),
        "files_verified": len(verified),
        "changed": changed,
        "missing": missing,
        "unhashed": [],
        "ok": not changed and not missing,
    }


# ---------------------------------------------------------------------------
# git revision + environment
# ---------------------------------------------------------------------------


def _git_revision_check(code: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    recorded_revision = code.get("revision")
    recorded_dirty = code.get("dirty")
    current = git_state(repo_root)
    current_revision = current.get("revision")
    return {
        "recorded_revision": recorded_revision,
        "current_revision": current_revision,
        "revision_match": bool(recorded_revision) and recorded_revision == current_revision,
        "recorded_dirty": recorded_dirty,
        "current_dirty": current.get("dirty"),
    }


def _environment_comparison(recorded_env: Any, repo_root: Path) -> dict[str, Any]:
    if not isinstance(recorded_env, Mapping):
        return {
            "available": False,
            "reason": "manifest carries no environment block",
            "differs": False,
            "reproducibility_affecting": False,
            "fields": {},
            "reproducibility_affecting_fields": [],
            "cosmetic_fields": [],
        }
    current_env = environment_report(project_root=repo_root)
    diff = compare_environment(dict(recorded_env), current_env)
    return {"available": True, **diff}


# ---------------------------------------------------------------------------
# recompute
# ---------------------------------------------------------------------------


#: Digest roles whose mismatch alone must never block recompute. ``uv_lock``
#: pins the *dependency environment*, not the model's actual input (the
#: feature table) -- a lockfile that has moved since the manifest was
#: written is the common case for replaying any run older than the newest
#: ``uv sync``, and refusing recompute over it would make this command
#: unable to ever demonstrate recompute against a real, slightly-aged
#: artifact. A drifted ``uv_lock`` is still a real, reported finding (see
#: ``digest_verification`` and ``notes``) and it still keeps
#: :attr:`ReplayReport.ok` false -- it just does not veto the recompute
#: check specifically.
_DIGEST_ROLES_THAT_DO_NOT_BLOCK_RECOMPUTE = frozenset({"uv_lock"})


def _digest_blocks_recompute(digest_verification: Mapping[str, Any]) -> bool:
    for key in ("changed", "missing"):
        for entry in digest_verification.get(key) or ():
            if (
                isinstance(entry, Mapping)
                and entry.get("role") not in _DIGEST_ROLES_THAT_DO_NOT_BLOCK_RECOMPUTE
            ):
                return True
    return False


def _skipped_recompute(reason: str) -> dict[str, Any]:
    return {
        "attempted": False,
        "reason": reason,
        "configuration": None,
        "output_root": None,
        "predictions_comparison": None,
        "metadata_comparison": None,
        "match": None,
    }


def _configuration_and_feature_path(
    provenance: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, Path | None]:
    configuration = provenance.get("configuration")
    feature_table = provenance.get("feature_table")
    feature_table_path: Path | None = None
    if isinstance(feature_table, Mapping):
        raw_path = feature_table.get("path")
        if raw_path:
            feature_table_path = Path(str(raw_path))
    return (configuration if isinstance(configuration, Mapping) else None, feature_table_path)


def _recompute_inputs(
    manifest: Mapping[str, Any], kind: str, manifest_file: Path
) -> tuple[Mapping[str, Any] | None, Path | None, Mapping[str, Any]]:
    """``(configuration, feature_table_path, recorded_metadata)`` for recompute.

    For a bare forecast ``metadata.json`` these come straight off its own
    ``provenance`` block. For a lock-day package there is no separate
    ``margin-predict`` configuration section -- but ``recorders.steps`` stores
    every executed step's own printed JSON **verbatim** (see
    ``nfl_ats.lockday_package``'s module docstring), and the ``margin-predict``
    step's output IS a forecast metadata dict of exactly this shape. Reading
    it from there is using the manifest's own recorded evidence, not guessing
    at command-line flags.
    """

    if kind == KIND_FORECAST_METADATA:
        provenance = manifest.get("provenance")
        if not isinstance(provenance, Mapping):
            return None, None, manifest
        configuration, feature_table_path = _configuration_and_feature_path(provenance)
        return configuration, feature_table_path, manifest

    if kind == KIND_LOCKDAY_PACKAGE:
        recorders = manifest.get("recorders")
        steps = recorders.get("steps") if isinstance(recorders, Mapping) else None
        step = steps.get("margin-predict") if isinstance(steps, Mapping) else None
        output = step.get("output") if isinstance(step, Mapping) else None
        if not isinstance(output, Mapping):
            return None, None, {}
        provenance = output.get("provenance")
        if not isinstance(provenance, Mapping):
            return None, None, output
        configuration, feature_table_path = _configuration_and_feature_path(provenance)
        return configuration, feature_table_path, output

    return None, None, {}


def _recorded_predictions_path(
    manifest: Mapping[str, Any], kind: str, manifest_file: Path
) -> Path | None:
    if kind == KIND_FORECAST_METADATA:
        candidate = manifest_file.parent / "predictions.csv"
        return candidate if candidate.is_file() else None
    if kind == KIND_LOCKDAY_PACKAGE:
        outputs = manifest.get("outputs")
        forecast = outputs.get("forecast") if isinstance(outputs, Mapping) else None
        directory = forecast.get("directory") if isinstance(forecast, Mapping) else None
        if directory:
            candidate = Path(str(directory)) / "predictions.csv"
            if candidate.is_file():
                return candidate
    return None


def _default_generate_forecast(
    feature_table: pd.DataFrame, configuration: Mapping[str, Any]
) -> pd.DataFrame:
    """The real forecast generator: the same ``score_outcome_week`` call
    ``_cmd_margin_predict`` makes, driven entirely by the configuration the
    manifest itself recorded. Pure: no I/O beyond reading ``feature_table``,
    which the caller already loaded from a digest-verified path."""

    return score_outcome_week(
        feature_table,
        season=int(configuration["season"]),
        week=int(configuration["week"]),
        regressor=str(configuration.get("regressor", "ridge")),
        min_edge=float(configuration.get("min_edge", 0.02)),
        min_train_games=int(configuration.get("min_train_games", DEFAULT_MIN_TRAIN_GAMES)),
        feature_profile=configuration.get("feature_profile", "base"),
        ridge_alpha=float(configuration.get("ridge_alpha", 10.0)),
        probability_method=configuration.get("probability_method", "gaussian"),
    )


def _derive_metadata_from_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """The mechanically-derivable subset of forecast metadata a regenerated
    predictions table can answer on its own, with no re-run of the rest of
    the ``margin-predict`` pipeline (safety validation, card lineage,
    active-model linking) -- all of which either touch registries/ledgers or
    require inputs replay is not given."""

    games = int(predictions["game_id"].nunique()) if "game_id" in predictions.columns else None
    methods = (
        sorted(str(value) for value in predictions["method"].unique())
        if "method" in predictions.columns
        else []
    )
    game_type = None
    if "game_type" in predictions.columns and not predictions.empty:
        game_type = str(predictions["game_type"].iloc[0])
    return {"games": games, "methods": methods, "game_type": game_type}


def _compare_metadata(
    recorded_metadata: Mapping[str, Any], regenerated_predictions: pd.DataFrame
) -> dict[str, Any]:
    derived = _derive_metadata_from_predictions(regenerated_predictions)
    fields: dict[str, Any] = {}
    match = True
    for key, regenerated_value in derived.items():
        recorded_value = recorded_metadata.get(key)
        equal = recorded_value == regenerated_value
        fields[key] = {
            "recorded": recorded_value,
            "regenerated": regenerated_value,
            "equal": bool(equal),
        }
        if not equal:
            match = False
    return {
        "fields": fields,
        "match": bool(match),
        "scope": (
            "mechanically derivable subset only (games/methods/game_type); timestamps, "
            "provenance, and environment are excluded by design -- see the module docstring"
        ),
    }


def _compare_predictions(
    recorded: pd.DataFrame,
    regenerated: pd.DataFrame,
    *,
    key_columns: Sequence[str] = ("game_id", "method"),
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Per-column equality between a recorded and a regenerated predictions
    table. Numeric columns compare within ``tolerance`` (BLAS/thread
    nondeterminism can move the low bits of a float even for identical code
    and data, per ``nfl_ats.environment_report``'s own documented rationale);
    every other column compares as text."""

    sort_columns = [
        column
        for column in key_columns
        if column in recorded.columns and column in regenerated.columns
    ]
    if sort_columns:
        recorded = recorded.sort_values(sort_columns).reset_index(drop=True)
        regenerated = regenerated.sort_values(sort_columns).reset_index(drop=True)
    else:
        recorded = recorded.reset_index(drop=True)
        regenerated = regenerated.reset_index(drop=True)

    recorded_columns = set(recorded.columns)
    regenerated_columns = set(regenerated.columns)
    common_columns = sorted(recorded_columns & regenerated_columns)
    only_recorded = sorted(recorded_columns - regenerated_columns)
    only_regenerated = sorted(regenerated_columns - recorded_columns)
    row_count_match = len(recorded) == len(regenerated)

    columns_report: dict[str, Any] = {}
    columns_match = True
    if row_count_match:
        for column in common_columns:
            left = recorded[column]
            right = regenerated[column]
            max_diff: float | None
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                left_values = left.astype(float)
                right_values = right.astype(float)
                both_nan = left_values.isna() & right_values.isna()
                diff = (left_values - right_values).abs()
                equal_mask = diff.le(tolerance) | both_nan
                remaining = diff[~both_nan]
                max_diff = float(remaining.max()) if not remaining.empty else 0.0
            else:
                left_text = left.astype(str)
                right_text = right.astype(str)
                equal_mask = (left_text == right_text) | (left.isna() & right.isna())
                max_diff = None
            equal = bool(equal_mask.all())
            columns_report[column] = {
                "equal": equal,
                "mismatched_rows": int((~equal_mask).sum()),
                "max_abs_diff": max_diff,
            }
            if not equal:
                columns_match = False
    else:
        columns_match = False

    match = row_count_match and columns_match and not only_recorded and not only_regenerated
    return {
        "row_count_recorded": len(recorded),
        "row_count_regenerated": len(regenerated),
        "row_count_match": bool(row_count_match),
        "columns_only_in_recorded": only_recorded,
        "columns_only_in_regenerated": only_regenerated,
        "columns": columns_report,
        "match": bool(match),
    }


def _run_recompute(
    manifest: Mapping[str, Any],
    kind: str,
    manifest_file: Path,
    *,
    output_root: Path,
    generate_forecast: GenerateForecast,
) -> dict[str, Any]:
    configuration, feature_table_path, recorded_metadata = _recompute_inputs(
        manifest, kind, manifest_file
    )
    if configuration is None or feature_table_path is None:
        return _skipped_recompute(
            "manifest does not carry enough information to recompute (no "
            "provenance.configuration / feature_table.path found for this manifest kind)"
        )
    if not feature_table_path.is_file():
        return _skipped_recompute(f"feature table not found on disk: {feature_table_path}")

    recorded_predictions_path = _recorded_predictions_path(manifest, kind, manifest_file)
    if recorded_predictions_path is None:
        return _skipped_recompute("recorded predictions.csv not found alongside the manifest")

    feature_table = pd.read_parquet(feature_table_path)
    regenerated = generate_forecast(feature_table, configuration)

    output_root.mkdir(parents=True, exist_ok=True)
    regenerated_path = output_root / "regenerated_predictions.csv"
    atomic_csv(regenerated, regenerated_path)
    # Round-trip the regenerated frame through the same CSV encoding the
    # recorded predictions.csv went through, so the comparison is apples to
    # apples rather than in-memory-float vs. csv-parsed-float.
    regenerated_for_compare = pd.read_csv(regenerated_path)
    recorded = pd.read_csv(recorded_predictions_path)

    predictions_comparison = _compare_predictions(recorded, regenerated_for_compare)
    metadata_comparison = _compare_metadata(recorded_metadata, regenerated)
    atomic_json(
        _derive_metadata_from_predictions(regenerated), output_root / "regenerated_metadata.json"
    )

    return {
        "attempted": True,
        "reason": None,
        "configuration": dict(configuration),
        "output_root": str(output_root),
        "recorded_predictions_path": str(recorded_predictions_path),
        "regenerated_predictions_path": str(regenerated_path),
        "predictions_comparison": predictions_comparison,
        "metadata_comparison": metadata_comparison,
        "match": bool(predictions_comparison["match"] and metadata_comparison["match"]),
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def replay_manifest(
    manifest_path: Path,
    *,
    output_root: Path,
    recompute: bool = True,
    repo_root: Path | None = None,
    generate_forecast: GenerateForecast | None = None,
) -> ReplayReport:
    """Replay one recorded run: verify digests, environment, code revision,
    and (optionally) recomputed outputs. Read-only except for
    ``output_root``, which recompute writes into and nothing else ever does.
    """

    root = (repo_root or _default_repo_root()).resolve()
    manifest, manifest_file = _load_manifest(Path(manifest_path))
    kind = _detect_kind(manifest)
    notes: list[str] = []

    season = _optional_int(manifest.get("season"))
    week = _optional_int(manifest.get("week"))

    code: Mapping[str, Any]
    recorded_env: Any
    if kind == KIND_LOCKDAY_PACKAGE:
        digest_verification = dict(verify_package(manifest_file, repo_root=root))
        raw_code = manifest.get("code")
        code = raw_code if isinstance(raw_code, Mapping) else {}
        recorded_env = manifest.get("environment")
    elif kind == KIND_FORECAST_METADATA:
        entries = _forecast_metadata_digest_entries(manifest, root)
        digest_verification = _verify_digest_entries(entries, root)
        notes.append(
            "forecast metadata.json records fewer digests than a lock-day package: only "
            "feature_table and uv_lock are checked here; this manifest format records no "
            "card or forecast-directory digest."
        )
        provenance = manifest.get("provenance")
        raw_code = provenance.get("code") if isinstance(provenance, Mapping) else None
        code = raw_code if isinstance(raw_code, Mapping) else {}
        recorded_env = manifest.get("environment")
        if not isinstance(recorded_env, Mapping) and isinstance(provenance, Mapping):
            recorded_env = provenance.get("environment")
    else:
        notes.append(
            f"Unrecognized manifest shape at {manifest_file}: expected a lock-day package "
            f"(kind={PACKAGE_KIND!r}) or a forecast metadata.json carrying a 'provenance' block."
        )
        digest_verification = {
            "ok": False,
            "reason": "unrecognized manifest kind",
            "files_checked": 0,
            "files_verified": 0,
            "changed": [],
            "missing": [],
            "unhashed": [],
        }
        code = {}
        recorded_env = None

    digest_ok = bool(digest_verification.get("ok"))

    git_revision = _git_revision_check(code, root)
    if git_revision["recorded_revision"] and not git_revision["revision_match"]:
        notes.append(
            "git revision mismatch: recorded "
            f"{git_revision['recorded_revision']} vs current {git_revision['current_revision']} "
            "(reported for context; a revision mismatch alone does not fail replay -- see "
            "the CLI exit-code contract in scripts/replay_run.py)"
        )

    environment_comparison = _environment_comparison(recorded_env, root)
    if not environment_comparison["available"]:
        notes.append(
            "No environment block recorded in this manifest; environment comparison skipped."
        )
    env_ok = not bool(environment_comparison.get("reproducibility_affecting"))

    recompute_report: dict[str, Any] | None = None
    if recompute:
        if kind == KIND_UNKNOWN:
            recompute_report = _skipped_recompute("unrecognized manifest kind")
        elif _digest_blocks_recompute(digest_verification):
            recompute_report = _skipped_recompute(
                "digest verification failed for a recomputed input; refusing to recompute "
                "from unverified inputs"
            )
        else:
            recompute_report = _run_recompute(
                manifest,
                kind,
                manifest_file,
                output_root=Path(output_root),
                generate_forecast=generate_forecast or _default_generate_forecast,
            )
        if not recompute_report.get("attempted") and recompute_report.get("reason"):
            notes.append(f"recompute skipped: {recompute_report['reason']}")

    recompute_ok = True
    if recompute_report is not None and recompute_report.get("attempted"):
        recompute_ok = bool(recompute_report.get("match"))

    ok = digest_ok and env_ok and recompute_ok

    return ReplayReport(
        manifest_path=str(manifest_file),
        manifest_kind=kind,
        season=season,
        week=week,
        digest_verification=digest_verification,
        git_revision=git_revision,
        environment_comparison=environment_comparison,
        recompute=recompute_report,
        notes=tuple(notes),
        ok=bool(ok),
    )


__all__ = [
    "KIND_FORECAST_METADATA",
    "KIND_LOCKDAY_PACKAGE",
    "KIND_UNKNOWN",
    "REPLAY_SCHEMA_VERSION",
    "GenerateForecast",
    "ReplayReport",
    "replay_manifest",
]
