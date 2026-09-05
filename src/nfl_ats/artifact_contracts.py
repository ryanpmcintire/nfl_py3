"""ENG-09: explicit schema/builder-version contracts for generated artifacts.

ROADMAP Phase 13's definition of done: "Give feature tables, forecasts,
cards, and ledgers explicit schema and builder versions plus compatibility
checks, and refuse incompatible combinations before fitting or publishing."

This module is a SECOND, coarser version axis layered on top of what already
exists, not a replacement for it:

* Feature-table families already carry their own fine-grained version
  constants recorded directly in the manifest (``pbp_feature_version``,
  ``player_feature_version``, ``qb_feature_version`` -- see
  :mod:`nfl_ats.pbp`, :mod:`nfl_ats.players`, :mod:`nfl_ats.quarterbacks`).
* Card provenance already has :mod:`nfl_ats.lineage` (``schema_version``,
  ``builder_version`` on ``CardLineage``), which answers "where did this
  decision-bearing field come from" at the *field* level.

What was missing is an *artifact-kind* level contract: a small, uniform
``{"kind", "schema_version", "builder_version", "builder_module"}`` block
every artifact kind carries, plus one function that looks at two (or three)
of those blocks together and says whether they are safe to combine --
refusing a genuine version MISMATCH while never treating an artifact that
predates this module (no block at all) as an error. That distinction is the
whole point: an absent/unknown version on an artifact nobody has ever
version-stamped is ``legacy_unversioned`` (a warning, so existing local
artifacts keep working); a mismatch between two versions that are BOTH
present is the hard failure.

Stamped artifacts carry the contract block under the single top-level key
:data:`CONTRACT_KEY` (``"artifact_contract"``), never as flat top-level
keys, specifically so stamping never collides with a pre-existing
``schema_version`` key an artifact may already use for something else (for
example ``lockday_package.build_manifest``'s own package ``schema_version``,
which predates this module and means something narrower: that package
format's own version, not this artifact-kind contract).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nfl_ats.features import BUILDER_VERSION as FEATURES_BUILDER_VERSION

#: Top-level key a stamped artifact's contract block lives under.
CONTRACT_KEY = "artifact_contract"

KIND_FEATURE_TABLE = "feature_table"
KIND_FORECAST = "forecast"
KIND_CARD = "card"
KIND_DECISION_LEDGER = "decision_ledger"
KIND_PICK_REVISION_LEDGER = "pick_revision_ledger"
KIND_LOCKDAY_PACKAGE = "lockday_package"

#: Builder-version sources for kinds that do not already own a module
#: constant of their own. ``feature_table`` reuses
#: ``nfl_ats.features.BUILDER_VERSION`` because that module owns the base
#: feature-table build path every other family enriches; the kinds below
#: have no single existing owner, so this module is that owner, bumped here
#: (and only here) when this contract layer's own construction rules change
#: for that kind.
FORECAST_BUILDER_VERSION = "v1"
CARD_BUILDER_VERSION = "v1"
DECISION_LEDGER_BUILDER_VERSION = "v1"
PICK_REVISION_LEDGER_BUILDER_VERSION = "v1"
LOCKDAY_PACKAGE_BUILDER_VERSION = "v1"

# ---------------------------------------------------------------------------
# Ledger required-column lists.
#
# Read 2026-09-04 directly from the source of truth -- copied, not imported,
# because nfl_ats.clv and nfl_ats.pick_refresh both import
# nfl_ats.prediction_safety (for validate_three_way_split), and
# prediction_safety imports THIS module for CompatibilityReport: an eager
# top-level `from nfl_ats.clv import PAPER_DECISION_COLUMNS` here would close
# that loop into a real circular import. tests/test_artifact_contracts.py
# imports both source tuples directly (tests sit outside the cycle) and
# asserts they still equal the copies below, so drift is caught mechanically
# rather than trusted to stay in sync by hand.
# ---------------------------------------------------------------------------

#: Mirrors ``nfl_ats.clv.PAPER_DECISION_COLUMNS``.
_DECISION_LEDGER_COLUMNS: tuple[str, ...] = (
    "recorded_at_utc",
    "forecast_artifact",
    "forecast_created_at_utc",
    "model_id",
    "method",
    "decision_policy_id",
    "decision_policy_fingerprint",
    "game_id",
    "season",
    "week",
    "kickoff",
    "away_team",
    "home_team",
    "model_pick_side",
    "pre_arrest_pick_side",
    "former_policy_pick_side",
    "pick_side",
    "coach_fade_flip",
    "division_revenge_flip",
    "player_arrests_flip",
    "spread_gap_zone_flip",
    "composed_overlay_flip",
    "player_arrests_home_flag",
    "player_arrests_away_flag",
    "player_arrests_snapshot_id",
    "player_arrests_snapshot_fetched_at_utc",
    "player_arrests_safe_index_sha256",
    "schedule_snapshot_id",
    "schedule_parquet_sha256",
    "bet_side",
    "decision_home_spread",
    "edge",
    "is_best_pick",
)

#: Mirrors ``nfl_ats.pick_refresh.PICK_REVISION_COLUMNS``.
_PICK_REVISION_LEDGER_COLUMNS: tuple[str, ...] = (
    "revision_recorded_at_utc",
    "refresh_run_id",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "decision_home_spread",
    "original_recorded_at_utc",
    "previous_pick_side",
    "previous_home_cover_probability",
    "new_pick_side",
    "new_home_cover_probability",
    "decision_policy_id",
    "decision_policy_fingerprint",
    "coach_fade_flip",
    "division_revenge_flip",
    "player_arrests_flip",
    "spread_gap_zone_flip",
    "composed_overlay_flip",
    "player_arrests_snapshot_id",
    "player_arrests_safe_index_sha256",
    "movement_policy",
    "movement_delta",
    "movement_pick_side",
    "model_only_pick_side",
    "model_id",
    "feature_table_sha256",
    "reason",
    "trigger_type",
    "trigger_source",
    "trigger_observed_at_utc",
)


@dataclass(frozen=True)
class ArtifactKindSpec:
    """The registered contract for one artifact kind."""

    kind: str
    schema_version: int
    builder_version: str
    builder_module: str
    required: tuple[str, ...]
    description: str


ARTIFACT_KINDS: dict[str, ArtifactKindSpec] = {
    KIND_FEATURE_TABLE: ArtifactKindSpec(
        kind=KIND_FEATURE_TABLE,
        schema_version=1,
        builder_version=FEATURES_BUILDER_VERSION,
        builder_module="nfl_ats.features",
        required=("built_at_utc", "rows", "destination"),
        description="A data/processed/*.parquet feature table and its *.manifest.json sibling.",
    ),
    KIND_FORECAST: ArtifactKindSpec(
        kind=KIND_FORECAST,
        schema_version=1,
        builder_version=FORECAST_BUILDER_VERSION,
        builder_module="nfl_ats.cli",
        required=("created_at_utc", "season", "week"),
        description="A margin-predict/predict weekly forecast's metadata.json.",
    ),
    KIND_CARD: ArtifactKindSpec(
        kind=KIND_CARD,
        schema_version=1,
        builder_version=CARD_BUILDER_VERSION,
        builder_module="nfl_ats.publishing",
        required=("model_id", "season", "week"),
        description="The published CURRENT_PREDICTIONS card's publish-predictions summary.",
    ),
    KIND_DECISION_LEDGER: ArtifactKindSpec(
        kind=KIND_DECISION_LEDGER,
        schema_version=1,
        builder_version=DECISION_LEDGER_BUILDER_VERSION,
        builder_module="nfl_ats.clv",
        required=_DECISION_LEDGER_COLUMNS,
        description="nfl_ats.clv.paper_decision_ledger_path's decisions.parquet.",
    ),
    KIND_PICK_REVISION_LEDGER: ArtifactKindSpec(
        kind=KIND_PICK_REVISION_LEDGER,
        schema_version=1,
        builder_version=PICK_REVISION_LEDGER_BUILDER_VERSION,
        builder_module="nfl_ats.pick_refresh",
        required=_PICK_REVISION_LEDGER_COLUMNS,
        description="nfl_ats.pick_refresh.pick_revision_ledger_path's pick_revisions.parquet.",
    ),
    KIND_LOCKDAY_PACKAGE: ArtifactKindSpec(
        kind=KIND_LOCKDAY_PACKAGE,
        schema_version=1,
        builder_version=LOCKDAY_PACKAGE_BUILDER_VERSION,
        builder_module="nfl_ats.lockday_package",
        required=("kind", "schema_version", "season", "week", "created_at_utc"),
        description=(
            "nfl_ats.lockday_package.build_manifest's package manifest.json. Not wired to "
            "stamp() -- it already owns a schema_version field with different, narrower "
            "semantics (the package format's own version); this registry entry exists so "
            "check_ledger/read_contract can still describe it uniformly."
        ),
    ),
}


class ArtifactContractError(ValueError):
    """An artifact-contract check found a hard failure that must block the caller."""


def _spec(kind: str) -> ArtifactKindSpec:
    try:
        return ARTIFACT_KINDS[kind]
    except KeyError:
        raise ArtifactContractError(
            f"Unknown artifact kind {kind!r}; register it in "
            "nfl_ats.artifact_contracts.ARTIFACT_KINDS"
        ) from None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def stamp(kind: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``metadata`` with an ``artifact_contract`` block added.

    Additive and pure: ``metadata`` itself is never mutated, and every
    existing key is preserved untouched. Raises :class:`ArtifactContractError`
    for an unregistered ``kind`` rather than silently stamping nonsense.
    """

    spec = _spec(kind)
    stamped = dict(metadata)
    stamped[CONTRACT_KEY] = {
        "kind": spec.kind,
        "schema_version": spec.schema_version,
        "builder_version": spec.builder_version,
        "builder_module": spec.builder_module,
    }
    return stamped


@dataclass(frozen=True)
class ArtifactContract:
    """One artifact's contract block, or the explicit absence of one."""

    kind: str | None
    schema_version: int | None
    builder_version: str | None
    builder_module: str | None
    #: True when the source artifact carries no ``artifact_contract`` block at
    #: all -- i.e. it predates this module. Never conflated with "present but
    #: reports null fields": those are two different failure shapes, and only
    #: this one is a warning rather than a mismatch.
    legacy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "builder_version": self.builder_version,
            "builder_module": self.builder_module,
            "legacy": self.legacy,
        }


_LEGACY_CONTRACT = ArtifactContract(
    kind=None, schema_version=None, builder_version=None, builder_module=None, legacy=True
)


def read_contract(path_or_metadata: Path | str | Mapping[str, Any]) -> ArtifactContract:
    """Read an artifact's ``artifact_contract`` block from a path or an in-memory mapping.

    Returns the explicit ``legacy_unversioned`` shape (never raises) when the
    block is absent, so callers can read any artifact -- stamped or not --
    through one function.
    """

    if isinstance(path_or_metadata, Mapping):
        payload: Mapping[str, Any] = path_or_metadata
    else:
        text = Path(path_or_metadata).read_text(encoding="utf-8")
        payload = json.loads(text)
    block = payload.get(CONTRACT_KEY)
    if not isinstance(block, Mapping):
        return _LEGACY_CONTRACT
    return ArtifactContract(
        kind=_optional_str(block.get("kind")),
        schema_version=_optional_int(block.get("schema_version")),
        builder_version=_optional_str(block.get("builder_version")),
        builder_module=_optional_str(block.get("builder_module")),
        legacy=False,
    )


CODE_LEGACY_UNVERSIONED = "legacy_unversioned"
CODE_VERSION_MISMATCH = "version_mismatch"
CODE_UNKNOWN_FORECAST_SCHEMA = "unknown_forecast_schema"
CODE_MISSING_COLUMNS = "missing_columns"

SEVERITY_WARNING = "warning"
SEVERITY_HARD_FAILURE = "hard_failure"


@dataclass(frozen=True)
class CompatibilityIssue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class CompatibilityReport:
    """The result of one :func:`check_compatible` or :func:`check_ledger` call."""

    issues: tuple[CompatibilityIssue, ...] = ()

    @property
    def hard_failures(self) -> tuple[CompatibilityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == SEVERITY_HARD_FAILURE)

    @property
    def warnings(self) -> tuple[CompatibilityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == SEVERITY_WARNING)

    @property
    def compatible(self) -> bool:
        """``True`` iff nothing here rises above a warning.

        Deliberately NOT named after the crossing-zero rule this repository
        already bans (an interval containing zero is never grounds to
        reject) -- this is a different, narrower question: whether two
        artifacts' EXPLICIT version stamps contradict each other. A
        ``legacy_unversioned`` warning is compatible; a ``version_mismatch``
        is not.
        """

        return not self.hard_failures

    def refuse_if_incompatible(self, *, action: str) -> None:
        """Raise :class:`ArtifactContractError` when a hard failure is present."""

        failures = self.hard_failures
        if not failures:
            return
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in failures)
        raise ArtifactContractError(f"Refusing to {action}: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def check_compatible(
    model_manifest: Mapping[str, Any] | None,
    feature_table_manifest: Mapping[str, Any] | None,
    forecast_metadata: Mapping[str, Any] | None = None,
) -> CompatibilityReport:
    """Whether a model, the feature table it would use, and (optionally) a forecast agree.

    Three independent checks, each fail-soft into ``legacy_unversioned``
    rather than fail-hard, EXCEPT the one case this exists to catch -- two
    version stamps that are both present and disagree:

    1. ``feature_table_manifest``'s stamped ``schema_version``/``builder_version``
       vs. the same fields recorded on ``model_manifest`` (the active model's
       own record of what it was fit on -- see
       ``nfl_ats.active_model.activate_matching_ats_model``). Absent on
       either side -> ``legacy_unversioned`` warning. Present on both and
       different -> ``version_mismatch`` hard failure.
    2. ``forecast_metadata``'s stamped contract, when supplied. Absent ->
       ``legacy_unversioned`` warning. Present but not the schema version
       this code recognizes -> ``unknown_forecast_schema`` hard failure.

    ``model_manifest=None`` (no active model yet) skips check 1 entirely --
    there is nothing to compare against, which is not the same as a mismatch.
    """

    issues: list[CompatibilityIssue] = []
    table_contract = read_contract(feature_table_manifest or {})

    if model_manifest is not None:
        recorded_schema = model_manifest.get("feature_table_schema_version")
        recorded_builder = model_manifest.get("feature_table_builder_version")
        model_versioned = recorded_schema is not None or recorded_builder is not None
        if not model_versioned or table_contract.legacy:
            issues.append(
                CompatibilityIssue(
                    SEVERITY_WARNING,
                    CODE_LEGACY_UNVERSIONED,
                    "active model manifest or feature-table manifest carries no "
                    "artifact_contract version to compare (legacy artifact)",
                )
            )
        else:
            if recorded_schema is not None and recorded_schema != table_contract.schema_version:
                issues.append(
                    CompatibilityIssue(
                        SEVERITY_HARD_FAILURE,
                        CODE_VERSION_MISMATCH,
                        f"feature table schema_version {table_contract.schema_version!r} does not "
                        f"match the version {recorded_schema!r} the active model was fit on",
                    )
                )
            if recorded_builder is not None and recorded_builder != table_contract.builder_version:
                issues.append(
                    CompatibilityIssue(
                        SEVERITY_HARD_FAILURE,
                        CODE_VERSION_MISMATCH,
                        f"feature table builder_version {table_contract.builder_version!r} "
                        f"does not match the version {recorded_builder!r} the active model "
                        "was fit on",
                    )
                )

    if forecast_metadata is not None:
        forecast_contract = read_contract(forecast_metadata)
        if forecast_contract.legacy:
            issues.append(
                CompatibilityIssue(
                    SEVERITY_WARNING,
                    CODE_LEGACY_UNVERSIONED,
                    "forecast metadata carries no artifact_contract block (legacy artifact)",
                )
            )
        else:
            expected = ARTIFACT_KINDS[KIND_FORECAST].schema_version
            if forecast_contract.schema_version != expected:
                issues.append(
                    CompatibilityIssue(
                        SEVERITY_HARD_FAILURE,
                        CODE_UNKNOWN_FORECAST_SCHEMA,
                        f"forecast schema_version {forecast_contract.schema_version!r} is not the "
                        f"recognized version {expected!r}",
                    )
                )

    return CompatibilityReport(issues=tuple(issues))


def check_ledger(kind: str, columns: Iterable[str]) -> CompatibilityReport:
    """Whether a ledger frame's columns satisfy ``kind``'s required-column contract.

    Missing columns are always a hard failure -- there is no legacy-warning
    case here, because both ledger loaders (``nfl_ats.clv.load_paper_decisions``,
    ``nfl_ats.pick_refresh.load_pick_revisions``) already backfill defaults for
    columns older artifacts lack before this check would ever see them; a
    frame that still lacks a required column has a schema problem, not an
    age problem.
    """

    spec = _spec(kind)
    present = set(columns)
    missing = sorted(set(spec.required).difference(present))
    if not missing:
        return CompatibilityReport(issues=())
    return CompatibilityReport(
        issues=(
            CompatibilityIssue(
                SEVERITY_HARD_FAILURE,
                CODE_MISSING_COLUMNS,
                f"{kind} ledger is missing required columns: {', '.join(missing)}",
            ),
        )
    )


__all__ = [
    "ARTIFACT_KINDS",
    "CODE_LEGACY_UNVERSIONED",
    "CODE_MISSING_COLUMNS",
    "CODE_UNKNOWN_FORECAST_SCHEMA",
    "CODE_VERSION_MISMATCH",
    "CONTRACT_KEY",
    "KIND_CARD",
    "KIND_DECISION_LEDGER",
    "KIND_FEATURE_TABLE",
    "KIND_FORECAST",
    "KIND_LOCKDAY_PACKAGE",
    "KIND_PICK_REVISION_LEDGER",
    "SEVERITY_HARD_FAILURE",
    "SEVERITY_WARNING",
    "ArtifactContract",
    "ArtifactContractError",
    "ArtifactKindSpec",
    "CompatibilityIssue",
    "CompatibilityReport",
    "check_compatible",
    "check_ledger",
    "read_contract",
    "stamp",
]
