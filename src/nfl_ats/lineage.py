"""End-to-end lineage from a published card field back to its source (ENG-16).

A weekly card is a small number of decisions wrapped in a lot of display.
When a pick is questioned months later the answerable question is not "what
did the model say" -- ``recommendations.csv`` already answers that -- but
"what did that number *see*, and when could it have seen it".  This module
emits that answer as a machine-readable artifact (``lineage.json``, written
next to the forecast) so the prediction-safety contract can refuse to publish
a card whose decision-bearing fields cannot say where they came from.

Decision-bearing, defined
-------------------------
A card field is **decision-bearing** when changing it would change what gets
submitted to the pool.  Concretely, and this list is the definition rather
than a summary of one:

1. :data:`FIELD_PICK` -- the side actually played.
2. :data:`FIELD_MODEL_PROBABILITY` -- the model probability the pick is read
   from, and the number the confidence ordering uses.
3. :data:`FIELD_MARKET_LINE` -- the market line the pick is expressed and
   graded against.
4. One ``overlay:<member_id>`` field for **each overlay that fired**.  An
   overlay that did not fire changed nothing, so it has nothing to justify;
   an overlay that flipped a pick is as decision-bearing as the pick.
5. One ``tiebreaker:<input>`` field per tiebreaker input, because the pool's
   tiebreaker score is a submitted number too.
6. One ``model_input:<family>`` field per feature family the fitted model
   actually consumed.  These are the pick's ingredients; a family whose
   snapshot nobody recorded is exactly the gap this module exists to surface.

Everything else on the card -- matchup text, formatted dates, cosmetic ranks
-- is display.  Display fields may carry ``"lineage": null`` provided they
carry an explicit ``reason``; silence is not permitted anywhere.

What "complete" means
---------------------
:func:`validate_card_lineage` (re-exported through
:mod:`nfl_ats.prediction_safety`, so it is release-blocking alongside the
existing checks) requires, for every decision-bearing field:

* a record exists, with non-empty ``feature_family``, ``builder_version``,
  ``builder_module`` and ``effective_timestamp``;
* ``source_snapshot`` is present, **or** ``unknown_source_reason`` explains in
  words why it is not recordable today.  An unrecorded provenance has to be
  declared, not merely absent -- that is what keeps
  ``docs/feature_lineage.md``'s honest gap list from quietly growing;
* ``effective_timestamp <= prediction_timestamp``.  This is the project's
  pregame-information invariant ("features may only use information available
  before the prediction timestamp") restated at the lineage layer, where it is
  checkable from the artifact alone rather than from the builder's intent.

``effective_timestamp`` is the as-of cutoff the feature used when the builder
records one.  Most builders do not, so the field carries a companion
:attr:`LineageRecord.effective_timestamp_basis` naming what was actually
available: a recorded cutoff (``"declared"`` / ``"training_cutoff"``), the
capture instant of the source snapshot (``"source_capture"``), or the feature
table's build time (``"feature_table_build"``) -- the tightest provable upper
bound on an as-of that nobody wrote down.  Reporting an upper bound as though
it were the cutoff would be exactly the kind of unlabelled claim this
repository already bans.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.constants import FEATURE_FAMILIES
from nfl_ats.feature_manifest import SOURCE_SNAPSHOTS_KEY
from nfl_ats.features import BUILDER_VERSION as FEATURES_BUILDER_VERSION
from nfl_ats.io import atomic_json
from nfl_ats.market_observation import MARKET_OBSERVED_AT_COLUMN
from nfl_ats.pbp import PBP_FEATURE_VERSION
from nfl_ats.players import PLAYER_FEATURE_VERSION
from nfl_ats.quarterbacks import QB_FEATURE_VERSION

LINEAGE_SCHEMA_VERSION = 1

#: Version of *this* module's record-construction rules.  Bumping it means a
#: previously emitted ``lineage.json`` was built by different logic.
BUILDER_VERSION = "v1"

LINEAGE_FILENAME = "lineage.json"

FIELD_PICK = "pick"
FIELD_MODEL_PROBABILITY = "model_probability"
FIELD_MARKET_LINE = "market_line"

OVERLAY_FIELD_PREFIX = "overlay:"
TIEBREAKER_FIELD_PREFIX = "tiebreaker:"
MODEL_INPUT_FIELD_PREFIX = "model_input:"

#: Decision-bearing fields that must exist on every card, whatever else did or
#: did not fire.  Overlay and tiebreaker fields are conditional by nature (an
#: overlay that never fired has nothing to justify) and so are not listed here;
#: when present they are still validated in full.
REQUIRED_DECISION_BEARING_FIELDS: tuple[str, ...] = (
    FIELD_PICK,
    FIELD_MODEL_PROBABILITY,
    FIELD_MARKET_LINE,
)

TIMESTAMP_BASES = frozenset(
    ("declared", "training_cutoff", "source_capture", "feature_table_build")
)


class LineageError(ValueError):
    """A card's lineage is missing, incomplete, or violates the cutoff rule."""


def is_decision_bearing(card_field: str) -> bool:
    """Whether ``card_field`` changes what actually gets submitted to the pool."""

    return card_field in REQUIRED_DECISION_BEARING_FIELDS or card_field.startswith(
        (OVERLAY_FIELD_PREFIX, TIEBREAKER_FIELD_PREFIX, MODEL_INPUT_FIELD_PREFIX)
    )


# ---------------------------------------------------------------------------
# Small conversions
# ---------------------------------------------------------------------------


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_utc(value: Any) -> datetime | None:
    """Best-effort UTC datetime for a timestamp of unknown flavour."""

    if value is None:
        return None
    if isinstance(value, datetime):
        instant = value
    else:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if parsed is None or not isinstance(parsed, pd.Timestamp) or pd.isna(parsed):
            return None
        instant = parsed.to_pydatetime()
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return instant.astimezone(UTC)


def _iso(instant: datetime) -> str:
    return instant.astimezone(UTC).isoformat()


#: ENG-23: per-game columns a forecast frame may carry for the
#: ``player_injuries`` family -- see the identical mechanism for
#: :data:`nfl_ats.market_observation.MARKET_OBSERVED_AT_COLUMN` on the
#: ``market_line`` record in :func:`build_card_lineage`.
INJURY_OBSERVED_AT_COLUMNS: tuple[str, ...] = ("home_injury_observed_at", "away_injury_observed_at")


def _frame_observed_at(forecast: pd.DataFrame, columns: Sequence[str]) -> datetime | None:
    """The LATEST non-null instant across ``columns`` present in ``forecast``.

    A single lineage record covers every row of the card, so the latest (not
    earliest) per-row observation is the only value that is still a valid
    upper bound for all of them: if the newest is still ``<=
    prediction_timestamp``, every row's is too. Returns ``None`` when none of
    ``columns`` exist or all values are null -- callers keep their existing
    fallback in that case, so a frame built before ENG-23 (no such columns)
    is unaffected.
    """

    present = [column for column in columns if column in forecast.columns]
    if not present:
        return None
    combined = pd.concat([forecast[column] for column in present], ignore_index=True)
    parsed = pd.to_datetime(combined, errors="coerce", utc=True).dropna()
    if parsed.empty:
        return None
    return as_utc(parsed.max())


def parse_snapshot_capture(snapshot_id: str | None) -> str | None:
    """Capture instant encoded in an ``nfl_ats.snapshots`` id (``%Y%m%dT%H%M%SZ``).

    Snapshot directories are named for the UTC instant they were written, so
    the id *is* the capture timestamp -- and it is the only capture field that
    is uniform across sources (the manifests themselves variously call it
    ``fetched_at_utc``, ``created_at_utc``, ``captured_at_utc``,
    ``observed_at_utc``, ``generated_at_utc`` or ``retrieved_at_utc``).
    Returns ``None`` for ids that do not follow the convention rather than
    guessing.
    """

    if not snapshot_id:
        return None
    try:
        instant = datetime.strptime(snapshot_id.strip(), "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None
    return instant.replace(tzinfo=UTC).isoformat()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineageRecord:
    """One card field's path back to a source snapshot and a builder."""

    card_field: str
    feature_family: str
    #: Snapshot directory name, snapshot id, or artifact identity -- whatever
    #: names the immutable source that was read.  ``None`` is permitted only
    #: when ``unknown_source_reason`` says why.
    source_snapshot: str | None
    source_captured_at: str | None
    effective_timestamp: str
    builder_version: str
    builder_module: str
    effective_timestamp_basis: str = "declared"
    unknown_source_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_field": self.card_field,
            "feature_family": self.feature_family,
            "source_snapshot": self.source_snapshot,
            "source_captured_at": self.source_captured_at,
            "effective_timestamp": self.effective_timestamp,
            "effective_timestamp_basis": self.effective_timestamp_basis,
            "builder_version": self.builder_version,
            "builder_module": self.builder_module,
            "unknown_source_reason": self.unknown_source_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LineageRecord:
        return cls(
            card_field=str(payload["card_field"]),
            feature_family=str(payload["feature_family"]),
            source_snapshot=_optional_text(payload.get("source_snapshot")),
            source_captured_at=_optional_text(payload.get("source_captured_at")),
            effective_timestamp=str(payload["effective_timestamp"]),
            effective_timestamp_basis=str(payload.get("effective_timestamp_basis", "declared")),
            builder_version=str(payload["builder_version"]),
            builder_module=str(payload["builder_module"]),
            unknown_source_reason=_optional_text(payload.get("unknown_source_reason")),
        )


@dataclass(frozen=True)
class CardLineageEntry:
    """One card field: a lineage record, or an explicitly reasoned absence."""

    card_field: str
    decision_bearing: bool
    lineage: LineageRecord | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_field": self.card_field,
            "decision_bearing": self.decision_bearing,
            "lineage": None if self.lineage is None else self.lineage.to_dict(),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CardLineageEntry:
        record = payload.get("lineage")
        return cls(
            card_field=str(payload["card_field"]),
            decision_bearing=bool(payload["decision_bearing"]),
            lineage=None if record is None else LineageRecord.from_dict(record),
            reason=_optional_text(payload.get("reason")),
        )


@dataclass(frozen=True)
class CardLineage:
    """Every field of one weekly card, with its provenance or its excuse."""

    prediction_timestamp: str
    entries: tuple[CardLineageEntry, ...]
    season: int | None = None
    week: int | None = None
    schema_version: int = LINEAGE_SCHEMA_VERSION
    builder_version: str = BUILDER_VERSION
    generated_at_utc: str = ""
    forecast_artifact: str | None = None
    model_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "builder_version": self.builder_version,
            "generated_at_utc": self.generated_at_utc,
            "prediction_timestamp": self.prediction_timestamp,
            "season": self.season,
            "week": self.week,
            "forecast_artifact": self.forecast_artifact,
            "model_id": self.model_id,
            "decision_bearing_fields": list(self.decision_bearing_fields()),
            "fields": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CardLineage:
        return cls(
            prediction_timestamp=str(payload["prediction_timestamp"]),
            entries=tuple(CardLineageEntry.from_dict(entry) for entry in payload.get("fields", ())),
            season=_optional_int(payload.get("season")),
            week=_optional_int(payload.get("week")),
            schema_version=int(payload.get("schema_version", LINEAGE_SCHEMA_VERSION)),
            builder_version=str(payload.get("builder_version", BUILDER_VERSION)),
            generated_at_utc=str(payload.get("generated_at_utc", "")),
            forecast_artifact=_optional_text(payload.get("forecast_artifact")),
            model_id=_optional_text(payload.get("model_id")),
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> CardLineage:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return cls.from_dict(json.loads(text))

    def decision_bearing_fields(self) -> tuple[str, ...]:
        return tuple(entry.card_field for entry in self.entries if entry.decision_bearing)

    def records(self) -> tuple[LineageRecord, ...]:
        return tuple(entry.lineage for entry in self.entries if entry.lineage is not None)

    def field(self, card_field: str) -> CardLineageEntry | None:
        for entry in self.entries:
            if entry.card_field == card_field:
                return entry
        return None

    def with_entries(self, entries: Iterable[CardLineageEntry]) -> CardLineage:
        """Return a copy with ``entries`` appended.

        Used by the publish path, which only learns which overlays fired after
        the forecast artifact has already been written.
        """

        return replace(self, entries=self.entries + tuple(entries))


# ---------------------------------------------------------------------------
# Feature family -> builder / snapshot mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyBuilder:
    """Which module builds a feature family, and where its source is recorded."""

    builder_module: str
    builder_version: str
    #: Key in the feature-table manifest naming the immutable source snapshot.
    manifest_snapshot_key: str | None = None
    unknown_source_reason: str | None = None


NFLVERSE_SNAPSHOT_KEY = "source_snapshot"

#: Why a base-table family usually cannot name its snapshot: the derived
#: manifests (``game_features_weak_stack.manifest.json`` and every other
#: enrichment step) record ``source_features`` -- a *path* to the parquet they
#: enriched -- but do not propagate the nflverse ``source_snapshot`` id that
#: the base ``game_features`` build recorded.  Read 2026-09-04 from
#: ``data/processed/*.manifest.json``: only ``game_features.manifest.json``
#: carries ``source_snapshot``.
BASE_SNAPSHOT_UNRECORDED = (
    "derived feature-table manifests record source_features (a path) but do not "
    "propagate the base nflverse source_snapshot id; see docs/feature_lineage.md"
)

_FEATURES_BUILDER = FamilyBuilder(
    "nfl_ats.features",
    FEATURES_BUILDER_VERSION,
    NFLVERSE_SNAPSHOT_KEY,
    BASE_SNAPSHOT_UNRECORDED,
)
_PBP_BUILDER = FamilyBuilder("nfl_ats.pbp", PBP_FEATURE_VERSION, "source_pbp_snapshot")

FAMILY_BUILDERS: dict[str, FamilyBuilder] = {
    "market": _FEATURES_BUILDER,
    "context": _FEATURES_BUILDER,
    "elo": _FEATURES_BUILDER,
    "experience": _FEATURES_BUILDER,
    "offense": _FEATURES_BUILDER,
    "results": _FEATURES_BUILDER,
    "defense": _FEATURES_BUILDER,
    "graph": _FEATURES_BUILDER,
    "schedule_rating": _FEATURES_BUILDER,
    "bias": _FEATURES_BUILDER,
    "surface_switch": _FEATURES_BUILDER,
    "pbp": _PBP_BUILDER,
    "pbp_opponent_adjusted": _PBP_BUILDER,
    "drive": _PBP_BUILDER,
    "quarterback": FamilyBuilder("nfl_ats.quarterbacks", QB_FEATURE_VERSION, "source_pbp_snapshot"),
    "quarterback_depth": FamilyBuilder(
        "nfl_ats.quarterbacks", QB_FEATURE_VERSION, "source_depth_snapshot"
    ),
    "player_qb": FamilyBuilder("nfl_ats.players", PLAYER_FEATURE_VERSION, "source_depth_snapshot"),
    "player_injuries": FamilyBuilder(
        "nfl_ats.players", PLAYER_FEATURE_VERSION, "source_player_snapshot"
    ),
    "player_continuity": FamilyBuilder(
        "nfl_ats.players", PLAYER_FEATURE_VERSION, "source_player_snapshot"
    ),
    "roster_returning_snaps": FamilyBuilder(
        "nfl_ats.players", PLAYER_FEATURE_VERSION, "source_player_snapshot"
    ),
    "player_values": FamilyBuilder(
        "nfl_ats.players", PLAYER_FEATURE_VERSION, "source_player_value_snapshot"
    ),
    "player_values_js_prior": FamilyBuilder(
        "nfl_ats.players", PLAYER_FEATURE_VERSION, "source_player_value_snapshot"
    ),
    "player_participation_values": FamilyBuilder(
        "nfl_ats.players", PLAYER_FEATURE_VERSION, "source_participation_snapshot"
    ),
    "travel_geometry": FamilyBuilder(
        "nfl_ats.travel_geometry",
        FEATURES_BUILDER_VERSION,
        None,
        "travel geometry is derived from static stadium coordinates plus the "
        "schedules table; it captures no snapshot of its own",
    ),
    "rest_context": FamilyBuilder(
        "nfl_ats.rest_context",
        FEATURES_BUILDER_VERSION,
        NFLVERSE_SNAPSHOT_KEY,
        BASE_SNAPSHOT_UNRECORDED,
    ),
    "forecast_weather": FamilyBuilder(
        "nfl_ats.forecast_weather_features",
        FEATURES_BUILDER_VERSION,
        None,
        "forecast-weather columns come from the kickoff-nearest archive under "
        "data/raw/forecast_archive/; the archive id is not recorded in the "
        "feature-table manifest",
    ),
    "observed_weather": FamilyBuilder(
        "nfl_ats.forecast_weather_features",
        FEATURES_BUILDER_VERSION,
        NFLVERSE_SNAPSHOT_KEY,
        "observed weather is the schedules table's own temp/wind fields; "
        + BASE_SNAPSHOT_UNRECORDED,
    ),
}

DEFAULT_FAMILY_BUILDER = FamilyBuilder(
    "nfl_ats.features",
    FEATURES_BUILDER_VERSION,
    None,
    "feature family has no entry in nfl_ats.lineage.FAMILY_BUILDERS; register it "
    "there to record its builder module and source snapshot",
)

#: Families whose builder version the feature-table manifest already records.
#: Preferring the manifest means the record reports what actually built the
#: table rather than whatever the importing process happens to have on disk --
#: the production weak-stack table says ``player_feature_version:
#: "v3-availability-v1"`` while ``players.PLAYER_FEATURE_VERSION`` is ``"v2"``.
MANIFEST_VERSION_KEYS: dict[str, str] = {
    "pbp": "pbp_feature_version",
    "pbp_opponent_adjusted": "pbp_feature_version",
    "drive": "pbp_feature_version",
    "player_qb": "player_feature_version",
    "player_injuries": "player_feature_version",
    "player_continuity": "player_feature_version",
    "roster_returning_snaps": "player_feature_version",
    "player_values": "player_feature_version",
    "player_values_js_prior": "player_feature_version",
    "player_participation_values": "player_feature_version",
}


def family_builder(family: str) -> FamilyBuilder:
    """The registered builder for ``family``, or an explicitly unknown default."""

    return FAMILY_BUILDERS.get(family, DEFAULT_FAMILY_BUILDER)


def families_for_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Feature families covering ``columns``, in :data:`FEATURE_FAMILIES` order.

    A column belonging to no declared family is reported under the synthetic
    family ``"unassigned"`` rather than dropped: an input the model consumed
    that no family claims is precisely the thing a lineage record should
    expose, not hide.
    """

    wanted = set(columns)
    families: list[str] = []
    claimed: set[str] = set()
    for family, family_columns in FEATURE_FAMILIES.items():
        overlap = wanted.intersection(family_columns)
        if overlap:
            families.append(family)
            claimed.update(overlap)
    if wanted.difference(claimed):
        families.append("unassigned")
    return tuple(families)


# ---------------------------------------------------------------------------
# Non-model sources supplied by the caller
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlaySource:
    """One overlay that actually fired, and what it read to decide that."""

    member_id: str
    builder_module: str
    builder_version: str
    effective_timestamp: str
    source_snapshot: str | None = None
    source_captured_at: str | None = None
    effective_timestamp_basis: str = "source_capture"
    unknown_source_reason: str | None = None
    flipped_game_ids: tuple[str, ...] = ()

    def record(self) -> LineageRecord:
        return LineageRecord(
            card_field=f"{OVERLAY_FIELD_PREFIX}{self.member_id}",
            feature_family=f"overlay/{self.member_id}",
            source_snapshot=self.source_snapshot,
            source_captured_at=self.source_captured_at,
            effective_timestamp=self.effective_timestamp,
            effective_timestamp_basis=self.effective_timestamp_basis,
            builder_version=self.builder_version,
            builder_module=self.builder_module,
            unknown_source_reason=self.unknown_source_reason,
        )


@dataclass(frozen=True)
class TiebreakerSource:
    """One input to the pool tiebreaker guess (market consensus, model view...)."""

    input_name: str
    builder_module: str
    builder_version: str
    effective_timestamp: str
    source_snapshot: str | None = None
    source_captured_at: str | None = None
    effective_timestamp_basis: str = "source_capture"
    unknown_source_reason: str | None = None

    def record(self) -> LineageRecord:
        return LineageRecord(
            card_field=f"{TIEBREAKER_FIELD_PREFIX}{self.input_name}",
            feature_family=f"tiebreaker/{self.input_name}",
            source_snapshot=self.source_snapshot,
            source_captured_at=self.source_captured_at,
            effective_timestamp=self.effective_timestamp,
            effective_timestamp_basis=self.effective_timestamp_basis,
            builder_version=self.builder_version,
            builder_module=self.builder_module,
            unknown_source_reason=self.unknown_source_reason,
        )


#: Overlay members whose decision reads the point-in-time arrest snapshot.
#: Every other member of the played policy reads only the schedules table and
#: the incoming card, both already covered by the model-input records.
ARREST_SNAPSHOT_MEMBERS = frozenset({"player_arrests_back_side_policy"})


def overlay_sources_from_composition(
    result: Any, *, fallback_effective_timestamp: str
) -> tuple[OverlaySource, ...]:
    """Adapt a ``four_overlay_composition`` result into overlay lineage inputs.

    Duck-typed on purpose: the composition result is a heavy object owned by
    the played-policy module, and lineage should not become a reason that
    module cannot change.  Only members that actually flipped a game are
    returned -- a member that changed nothing is not decision-bearing.
    """

    arrest_snapshot = _optional_text(getattr(result, "arrest_snapshot_id", None))
    captured = as_utc(getattr(result, "arrest_snapshot_fetched_at_utc", None))
    arrest_captured = None if captured is None else _iso(captured)
    fingerprint = str(getattr(result, "policy_fingerprint", "")) or BUILDER_VERSION
    sources: list[OverlaySource] = []
    for member in getattr(result, "members", ()):
        flipped = tuple(str(game) for game in getattr(member, "flipped_game_ids", ()))
        if not flipped:
            continue
        member_id = str(getattr(member, "member_id", "unknown"))
        implementation = str(getattr(member, "implementation", "unknown"))
        module = implementation.rsplit(".", 1)[0] if "." in implementation else implementation
        reads_arrests = member_id in ARREST_SNAPSHOT_MEMBERS and arrest_captured is not None
        sources.append(
            OverlaySource(
                member_id=member_id,
                builder_module=module,
                builder_version=fingerprint[:16],
                effective_timestamp=(
                    arrest_captured
                    if reads_arrests and arrest_captured is not None
                    else fallback_effective_timestamp
                ),
                source_snapshot=arrest_snapshot if reads_arrests else None,
                source_captured_at=arrest_captured if reads_arrests else None,
                effective_timestamp_basis=(
                    "source_capture" if reads_arrests else "feature_table_build"
                ),
                unknown_source_reason=(
                    None
                    if reads_arrests
                    else (
                        "overlay member reads the schedules table and the incoming card, "
                        "both already covered by the model_input records; it captures no "
                        "snapshot of its own"
                    )
                ),
                flipped_game_ids=flipped,
            )
        )
    return tuple(sources)


# ---------------------------------------------------------------------------
# Building the card lineage
# ---------------------------------------------------------------------------


def feature_table_manifest(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """The feature-builder manifest embedded in a forecast's provenance block."""

    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        return {}
    feature_table = provenance.get("feature_table")
    if not isinstance(feature_table, Mapping):
        return {}
    manifest = feature_table.get("manifest")
    return manifest if isinstance(manifest, Mapping) else {}


def _feature_table_identifier(metadata: Mapping[str, Any]) -> str | None:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    feature_table = provenance.get("feature_table")
    if not isinstance(feature_table, Mapping):
        return None
    digest = _optional_text(feature_table.get("sha256"))
    if digest is not None:
        return f"feature_table:sha256:{digest}"
    return _optional_text(feature_table.get("path"))


def _training_cutoff(forecast: pd.DataFrame) -> datetime | None:
    if "train_max_gameday" not in forecast.columns:
        return None
    values = pd.to_datetime(forecast["train_max_gameday"], errors="coerce", utc=True).dropna()
    if values.empty:
        return None
    return as_utc(values.max())


def _decision_basis(forecast: pd.DataFrame, manifest: Mapping[str, Any]) -> tuple[datetime, str]:
    """The tightest defensible as-of for the card's own decision fields.

    A pick sees both the fitted model (bounded by the training cutoff) and the
    target week's own feature row (bounded by when the feature table was
    built), so the honest bound is the later of the two.
    """

    built = as_utc(manifest.get("built_at_utc"))
    trained = _training_cutoff(forecast)
    if built is not None and trained is not None:
        return (built, "feature_table_build") if built >= trained else (trained, "training_cutoff")
    if built is not None:
        return built, "feature_table_build"
    if trained is not None:
        return trained, "training_cutoff"
    return datetime.now(UTC), "declared"


def _inherited_snapshot(
    manifest_snapshot_key: str | None, manifest: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    """(snapshot_id, captured_at) an ENG-22 ``source_snapshots`` block names.

    ``manifest_snapshot_key`` (e.g. ``"source_snapshot"``,
    ``"source_pbp_snapshot"``) is the same key
    :data:`FamilyBuilder.manifest_snapshot_key` already looks for directly on
    the manifest; derived manifests that predate ENG-22, or a legacy
    manifest with no ``source_snapshots`` block at all, simply have nothing
    under this key, and the caller keeps falling through to the digest
    fallback exactly as before ENG-22 existed.
    """

    if manifest_snapshot_key is None:
        return None, None
    block = manifest.get(SOURCE_SNAPSHOTS_KEY)
    if not isinstance(block, Mapping):
        return None, None
    entry = block.get(manifest_snapshot_key)
    if not isinstance(entry, Mapping):
        return None, None
    snapshot_id = _optional_text(entry.get("snapshot_id"))
    if snapshot_id is None:
        return None, None
    return snapshot_id, _optional_text(entry.get("captured_at"))


def _family_record(
    family: str,
    *,
    manifest: Mapping[str, Any],
    feature_table_id: str | None,
    basis_instant: datetime,
    basis_name: str,
    frame_captured_at: datetime | None = None,
) -> LineageRecord:
    builder = family_builder(family)
    snapshot: str | None = None
    captured: str | None = None
    if builder.manifest_snapshot_key is not None:
        snapshot = _optional_text(manifest.get(builder.manifest_snapshot_key))
        if snapshot is not None:
            captured = parse_snapshot_capture(snapshot)
        else:
            # ENG-22: this manifest carries no direct key for the family
            # (true of every derived feature table for the base nflverse
            # source_snapshot -- see inherit_source_snapshots), but it may
            # have inherited one transitively. Prefer that real snapshot id
            # over the feature-table digest fallback below; a legacy
            # manifest with no source_snapshots block simply gets (None,
            # None) here and falls through exactly as it did before ENG-22.
            snapshot, captured = _inherited_snapshot(builder.manifest_snapshot_key, manifest)

    if frame_captured_at is not None:
        # ENG-23: the frame's own per-game observed-at (e.g. INJURY_OBSERVED_AT_COLUMNS
        # on player_injuries) is a real, game-level capture instant -- prefer it over
        # the whole-table manifest value above for source_captured_at/effective_timestamp
        # only; source_snapshot (WHICH snapshot) is unaffected either way.
        captured = _iso(frame_captured_at)

    captured_instant = as_utc(captured)
    if captured_instant is not None:
        effective_instant, effective_basis = captured_instant, "source_capture"
    else:
        effective_instant, effective_basis = basis_instant, basis_name

    reason: str | None = None
    if snapshot is None:
        reason = builder.unknown_source_reason or (
            f"feature family {family!r} records no source snapshot in the feature-table manifest"
        )
        snapshot = feature_table_id
        if snapshot is not None:
            reason = (
                f"{reason}; the record falls back to the feature-table identity, which pins "
                "the bytes the model read but not the upstream capture"
            )

    manifest_version = _optional_text(manifest.get(MANIFEST_VERSION_KEYS.get(family, "")))
    return LineageRecord(
        card_field=f"{MODEL_INPUT_FIELD_PREFIX}{family}",
        feature_family=family,
        source_snapshot=snapshot,
        source_captured_at=captured,
        effective_timestamp=_iso(effective_instant),
        effective_timestamp_basis=effective_basis,
        builder_version=manifest_version or builder.builder_version,
        builder_module=builder.builder_module,
        unknown_source_reason=reason,
    )


#: Display-only fields the published Markdown card renders (see
#: ``publishing._published_card``), each with the reason it carries no lineage.
#: Passed to :func:`build_card_lineage` so a reader of ``lineage.json`` sees the
#: whole card, not only the parts that happen to be traceable.
PUBLISHED_DISPLAY_FIELDS: dict[str, str] = {
    "Date": "formatted from the card's own gameday column; introduces no new source",
    "Matchup": "formatted from home_team/away_team, already covered by model_input:market",
    "ATS prediction": "rendering of pick and market_line, both of which carry their own lineage",
    "Decision score": (
        "rendering of model_probability from the picked side's perspective; introduces "
        "no independent source"
    ),
}


def build_card_lineage(
    forecast: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    active_model: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] = (),
    prediction_timestamp: Any = None,
    overlay_sources: Sequence[OverlaySource] = (),
    tiebreaker_sources: Sequence[TiebreakerSource] = (),
    display_fields: Mapping[str, str] | None = None,
    generated_at: datetime | None = None,
) -> CardLineage:
    """Build lineage for every decision-bearing field of one weekly card.

    ``metadata`` is the forecast artifact's ``metadata.json`` payload; its
    ``provenance.feature_table.manifest`` block supplies the source snapshots
    and builder versions.  ``active_model`` is ``active_ats_model.json`` when
    the card has been activated.  ``feature_columns`` is the model's own input
    contract -- pass ``margin.margin_feature_columns(target, profile)`` -- so
    the emitted families are the ones the fit actually consumed rather than
    every column that happens to sit in the table.

    ENG-23: when ``forecast`` carries :data:`nfl_ats.market_observation.MARKET_OBSERVED_AT_COLUMN`
    or :data:`INJURY_OBSERVED_AT_COLUMNS`, the ``market_line`` and
    ``model_input:player_injuries`` records use the latest non-null value
    across those columns as ``source_captured_at`` / ``effective_timestamp``
    in place of the whole-table manifest fallback -- a real per-card capture
    instant instead of an upper bound nobody wrote down. A frame without
    those columns (every frame built before ENG-23) validates exactly as it
    did before.
    """

    manifest = feature_table_manifest(metadata)
    feature_table_id = _feature_table_identifier(metadata)
    basis_instant, basis_name = _decision_basis(forecast, manifest)

    prediction_instant = as_utc(prediction_timestamp) or as_utc(metadata.get("created_at_utc"))
    if prediction_instant is None:
        prediction_instant = as_utc(generated_at) or datetime.now(UTC)

    model_identity = _optional_text(
        (active_model or {}).get("model_id") or metadata.get("active_model_id")
    )
    decision_version = model_identity or _optional_text(metadata.get("feature_profile"))
    decision_module = "nfl_ats.outcomes" if "method" in forecast.columns else "nfl_ats.backtest"

    def decision_record(card_field: str, family: str) -> LineageRecord:
        return LineageRecord(
            card_field=card_field,
            feature_family=family,
            source_snapshot=feature_table_id,
            source_captured_at=_optional_text(manifest.get("built_at_utc")),
            effective_timestamp=_iso(basis_instant),
            effective_timestamp_basis=basis_name,
            builder_version=decision_version or BUILDER_VERSION,
            builder_module=decision_module,
            unknown_source_reason=(
                None
                if feature_table_id is not None
                else "forecast metadata carries no provenance.feature_table identity"
            ),
        )

    market_builder = family_builder("market")
    market_snapshot = _optional_text(manifest.get(NFLVERSE_SNAPSHOT_KEY))
    market_captured = parse_snapshot_capture(market_snapshot)
    if market_snapshot is None:
        # ENG-22: same inheritance preference as _family_record -- a derived
        # manifest that never recorded source_snapshot directly may still
        # name it via an inherited source_snapshots block.
        market_snapshot, market_captured = _inherited_snapshot(NFLVERSE_SNAPSHOT_KEY, manifest)
    # ENG-23: the frame's own market_observed_at_utc (the point-in-time odds
    # capture's observation instant, joined by nfl_ats.market_observation) is
    # a real per-card capture instant -- prefer it over the manifest-derived
    # value above for captured_at/effective_timestamp only; source_snapshot
    # (WHICH snapshot) is unaffected, same rule as _family_record below.
    market_frame_captured = _frame_observed_at(forecast, (MARKET_OBSERVED_AT_COLUMN,))
    if market_frame_captured is not None:
        market_captured = _iso(market_frame_captured)
    market_instant = as_utc(market_captured)
    market_record = LineageRecord(
        card_field=FIELD_MARKET_LINE,
        feature_family="market",
        source_snapshot=market_snapshot or feature_table_id,
        source_captured_at=market_captured,
        effective_timestamp=_iso(market_instant or basis_instant),
        effective_timestamp_basis=("source_capture" if market_instant is not None else basis_name),
        builder_version=market_builder.builder_version,
        builder_module=market_builder.builder_module,
        unknown_source_reason=(None if market_snapshot is not None else BASE_SNAPSHOT_UNRECORDED),
    )

    entries: list[CardLineageEntry] = [
        CardLineageEntry(FIELD_PICK, True, decision_record(FIELD_PICK, "model_decision")),
        CardLineageEntry(
            FIELD_MODEL_PROBABILITY,
            True,
            decision_record(FIELD_MODEL_PROBABILITY, "model_probability"),
        ),
        CardLineageEntry(FIELD_MARKET_LINE, True, market_record),
    ]
    # ENG-23: same per-family override as the market_line record above, for
    # the one other family with a per-game observed-at column on the frame.
    injury_frame_captured = _frame_observed_at(forecast, INJURY_OBSERVED_AT_COLUMNS)
    entries.extend(
        CardLineageEntry(
            f"{MODEL_INPUT_FIELD_PREFIX}{family}",
            True,
            _family_record(
                family,
                manifest=manifest,
                feature_table_id=feature_table_id,
                basis_instant=basis_instant,
                basis_name=basis_name,
                frame_captured_at=(injury_frame_captured if family == "player_injuries" else None),
            ),
        )
        for family in families_for_columns(feature_columns)
    )
    entries.extend(
        CardLineageEntry(f"{OVERLAY_FIELD_PREFIX}{overlay.member_id}", True, overlay.record())
        for overlay in overlay_sources
    )
    entries.extend(
        CardLineageEntry(
            f"{TIEBREAKER_FIELD_PREFIX}{tiebreaker.input_name}", True, tiebreaker.record()
        )
        for tiebreaker in tiebreaker_sources
    )
    entries.extend(
        CardLineageEntry(card_field, False, None, reason)
        for card_field, reason in (display_fields or {}).items()
    )

    weekly_forecast = (active_model or {}).get("weekly_forecast")
    return CardLineage(
        prediction_timestamp=_iso(prediction_instant),
        entries=tuple(entries),
        season=_optional_int(metadata.get("season")),
        week=_optional_int(metadata.get("week")),
        generated_at_utc=_iso(generated_at or datetime.now(UTC)),
        forecast_artifact=(
            _optional_text(weekly_forecast.get("artifact"))
            if isinstance(weekly_forecast, Mapping)
            else None
        ),
        model_id=model_identity,
    )


def extend_card_lineage_for_publication(
    lineage: CardLineage,
    *,
    overlay_sources: Sequence[OverlaySource] = (),
    tiebreaker_sources: Sequence[TiebreakerSource] = (),
    prediction_timestamp: Any = None,
    generated_at: datetime | None = None,
) -> CardLineage:
    """Extend a forecast's own lineage with what publish time learns (ENG-24).

    ``margin-predict``/``predict`` write ``lineage.json`` before the four-member
    overlay policy and the pool tiebreaker guess ever run (see
    ``docs/feature_lineage.md`` gap items 4-5), so neither can appear in that
    file. ``nfl_ats.publishing`` reads it back (or builds an equivalent fresh
    one when it is absent), adapts the overlay result and the tiebreaker guess
    into :class:`OverlaySource`/:class:`TiebreakerSource` records the same way
    :func:`build_card_lineage` would have, and calls this function to produce
    the PLAYED card's own lineage -- a distinct object/file from the
    forecast's, written beside the published card rather than overwritten
    into the forecast artifact.

    ``prediction_timestamp`` defaults to keeping the base lineage's own value
    unchanged (a no-op extension). Pass the publish instant when the new
    records were captured after the original forecast was built -- the normal
    case, since an arrest snapshot or market quote read fresh at publish time
    postdates the forecast's own cutoff -- so the pregame-information check in
    :func:`validate_card_lineage` compares every record, old and new, against
    the moment the PLAYED card was actually decided. The cutoff only ever
    moves LATER than the base's own: a caller-supplied instant earlier than
    ``lineage.prediction_timestamp`` is ignored rather than applied, because
    shrinking the cutoff could turn an already-valid base record (built,
    honestly, at its own "now" when neither a manifest nor a training cutoff
    was available) into a manufactured leak.
    """

    entries = [
        CardLineageEntry(f"{OVERLAY_FIELD_PREFIX}{overlay.member_id}", True, overlay.record())
        for overlay in overlay_sources
    ]
    entries.extend(
        CardLineageEntry(
            f"{TIEBREAKER_FIELD_PREFIX}{tiebreaker.input_name}", True, tiebreaker.record()
        )
        for tiebreaker in tiebreaker_sources
    )
    extended = lineage.with_entries(entries)
    updates: dict[str, Any] = {"generated_at_utc": _iso(generated_at or datetime.now(UTC))}
    resolved_prediction_instant = as_utc(prediction_timestamp)
    current_instant = as_utc(lineage.prediction_timestamp)
    if resolved_prediction_instant is not None and (
        current_instant is None or resolved_prediction_instant > current_instant
    ):
        updates["prediction_timestamp"] = _iso(resolved_prediction_instant)
    return replace(extended, **updates)


# ---------------------------------------------------------------------------
# Validation -- the release-blocking half
# ---------------------------------------------------------------------------

LINEAGE_CHECKS: tuple[str, ...] = (
    "lineage_schema",
    "lineage_required_fields",
    "lineage_completeness",
    "lineage_effective_timestamp",
)


def _incomplete_reason(entry: CardLineageEntry) -> str | None:
    record = entry.lineage
    if record is None:
        if entry.decision_bearing:
            return "decision-bearing field carries no lineage record"
        if not entry.reason:
            return "display field carries neither a lineage record nor a reason"
        return None
    for name, value in (
        ("feature_family", record.feature_family),
        ("builder_version", record.builder_version),
        ("builder_module", record.builder_module),
        ("effective_timestamp", record.effective_timestamp),
    ):
        if not str(value).strip():
            return f"{name} is empty"
    if record.source_snapshot is None and not (record.unknown_source_reason or "").strip():
        return "source_snapshot is absent and no unknown_source_reason explains why"
    if record.effective_timestamp_basis not in TIMESTAMP_BASES:
        return f"effective_timestamp_basis {record.effective_timestamp_basis!r} is not recognized"
    return None


def validate_card_lineage(
    lineage: CardLineage,
    *,
    prediction_timestamp: Any = None,
    required_fields: Sequence[str] = REQUIRED_DECISION_BEARING_FIELDS,
) -> tuple[str, ...]:
    """Fail closed when a card cannot say where its decisions came from.

    Raises :class:`LineageError` naming every offending field.  Returns the
    names of the checks that passed, matching the shape
    :mod:`nfl_ats.prediction_safety` already reports.
    """

    if lineage.schema_version != LINEAGE_SCHEMA_VERSION:
        raise LineageError(
            f"unsupported lineage schema version {lineage.schema_version} "
            f"(expected {LINEAGE_SCHEMA_VERSION})"
        )

    seen = {entry.card_field: entry for entry in lineage.entries}
    missing = sorted(
        field
        for field in required_fields
        if field not in seen or seen[field].lineage is None or not seen[field].decision_bearing
    )
    if missing:
        raise LineageError("lineage is missing decision-bearing fields: " + ", ".join(missing))

    incomplete = {
        entry.card_field: reason
        for entry in lineage.entries
        if (reason := _incomplete_reason(entry)) is not None
    }
    if incomplete:
        detail = ", ".join(f"{field} ({reason})" for field, reason in sorted(incomplete.items()))
        raise LineageError(f"lineage is incomplete for: {detail}")

    cutoff = as_utc(prediction_timestamp) or as_utc(lineage.prediction_timestamp)
    if cutoff is None:
        raise LineageError(
            f"lineage prediction_timestamp {lineage.prediction_timestamp!r} is unparseable"
        )
    unparseable: list[str] = []
    leaking: list[str] = []
    for entry in lineage.entries:
        if entry.lineage is None:
            continue
        effective = as_utc(entry.lineage.effective_timestamp)
        if effective is None:
            unparseable.append(entry.card_field)
        elif effective > cutoff:
            leaking.append(
                f"{entry.card_field} (effective {entry.lineage.effective_timestamp} "
                f"> prediction {_iso(cutoff)})"
            )
    if unparseable:
        raise LineageError(
            "lineage effective_timestamp is unparseable for: " + ", ".join(sorted(unparseable))
        )
    if leaking:
        raise LineageError(
            "lineage effective_timestamp is after the prediction timestamp for: "
            + ", ".join(sorted(leaking))
        )
    return LINEAGE_CHECKS


def write_card_lineage(lineage: CardLineage, directory: Path) -> Path:
    """Write ``lineage.json`` into a forecast artifact directory, atomically."""

    destination = Path(directory) / LINEAGE_FILENAME
    atomic_json(lineage.to_dict(), destination)
    return destination


def read_card_lineage(directory: Path) -> CardLineage:
    """Read ``lineage.json`` back out of a forecast artifact directory."""

    return CardLineage.from_json((Path(directory) / LINEAGE_FILENAME).read_text(encoding="utf-8"))


__all__ = [
    "ARREST_SNAPSHOT_MEMBERS",
    "BASE_SNAPSHOT_UNRECORDED",
    "BUILDER_VERSION",
    "DEFAULT_FAMILY_BUILDER",
    "FAMILY_BUILDERS",
    "FIELD_MARKET_LINE",
    "FIELD_MODEL_PROBABILITY",
    "FIELD_PICK",
    "INJURY_OBSERVED_AT_COLUMNS",
    "LINEAGE_CHECKS",
    "LINEAGE_FILENAME",
    "LINEAGE_SCHEMA_VERSION",
    "MANIFEST_VERSION_KEYS",
    "MODEL_INPUT_FIELD_PREFIX",
    "NFLVERSE_SNAPSHOT_KEY",
    "OVERLAY_FIELD_PREFIX",
    "PUBLISHED_DISPLAY_FIELDS",
    "REQUIRED_DECISION_BEARING_FIELDS",
    "TIEBREAKER_FIELD_PREFIX",
    "TIMESTAMP_BASES",
    "CardLineage",
    "CardLineageEntry",
    "FamilyBuilder",
    "LineageError",
    "LineageRecord",
    "OverlaySource",
    "TiebreakerSource",
    "as_utc",
    "build_card_lineage",
    "extend_card_lineage_for_publication",
    "families_for_columns",
    "family_builder",
    "feature_table_manifest",
    "is_decision_bearing",
    "overlay_sources_from_composition",
    "parse_snapshot_capture",
    "read_card_lineage",
    "validate_card_lineage",
    "write_card_lineage",
]
