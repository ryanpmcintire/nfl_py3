from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.constants import FEATURE_FAMILIES
from nfl_ats.feature_manifest import SOURCE_SNAPSHOTS_KEY, decision_line_week
from nfl_ats.features import BUILDER_VERSION as FEATURES_BUILDER_VERSION
from nfl_ats.io import atomic_json
from nfl_ats.market_observation import MARKET_OBSERVED_AT_COLUMN
from nfl_ats.pbp import PBP_FEATURE_VERSION
from nfl_ats.players import PLAYER_FEATURE_VERSION
from nfl_ats.quarterbacks import QB_FEATURE_VERSION

LINEAGE_SCHEMA_VERSION = 1

BUILDER_VERSION = "v1"

LINEAGE_FILENAME = "lineage.json"

FIELD_PICK = "pick"
FIELD_MODEL_PROBABILITY = "model_probability"
FIELD_MARKET_LINE = "market_line"

OVERLAY_FIELD_PREFIX = "overlay:"
TIEBREAKER_FIELD_PREFIX = "tiebreaker:"
MODEL_INPUT_FIELD_PREFIX = "model_input:"

REQUIRED_DECISION_BEARING_FIELDS: tuple[str, ...] = (
    FIELD_PICK,
    FIELD_MODEL_PROBABILITY,
    FIELD_MARKET_LINE,
)

TIMESTAMP_BASES = frozenset(
    ("declared", "training_cutoff", "source_capture", "feature_table_build")
)


class LineageError(ValueError):
    pass


def is_decision_bearing(card_field: str) -> bool:

    return card_field in REQUIRED_DECISION_BEARING_FIELDS or card_field.startswith(
        (OVERLAY_FIELD_PREFIX, TIEBREAKER_FIELD_PREFIX, MODEL_INPUT_FIELD_PREFIX)
    )


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


INJURY_OBSERVED_AT_COLUMNS: tuple[str, ...] = ("home_injury_observed_at", "away_injury_observed_at")


def _frame_observed_at(forecast: pd.DataFrame, columns: Sequence[str]) -> datetime | None:

    present = [column for column in columns if column in forecast.columns]
    if not present:
        return None
    combined = pd.concat([forecast[column] for column in present], ignore_index=True)
    parsed = pd.to_datetime(combined, errors="coerce", utc=True).dropna()
    if parsed.empty:
        return None
    return as_utc(parsed.max())


def parse_snapshot_capture(snapshot_id: str | None) -> str | None:

    if not snapshot_id:
        return None
    try:
        instant = datetime.strptime(snapshot_id.strip(), "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None
    return instant.replace(tzinfo=UTC).isoformat()


@dataclass(frozen=True)
class LineageRecord:
    card_field: str
    feature_family: str
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

        return replace(self, entries=self.entries + tuple(entries))


@dataclass(frozen=True)
class FamilyBuilder:
    builder_module: str
    builder_version: str
    manifest_snapshot_key: str | None = None
    unknown_source_reason: str | None = None


NFLVERSE_SNAPSHOT_KEY = "source_snapshot"

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

    return FAMILY_BUILDERS.get(family, DEFAULT_FAMILY_BUILDER)


def families_for_columns(columns: Iterable[str]) -> tuple[str, ...]:

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


@dataclass(frozen=True)
class OverlaySource:
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


ARREST_SNAPSHOT_MEMBERS = frozenset({"player_arrests_back_side_policy"})


def overlay_sources_from_composition(
    result: Any, *, fallback_effective_timestamp: str
) -> tuple[OverlaySource, ...]:

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


def feature_table_manifest(metadata: Mapping[str, Any]) -> Mapping[str, Any]:

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
            snapshot, captured = _inherited_snapshot(builder.manifest_snapshot_key, manifest)

    if frame_captured_at is not None:
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


PUBLISHED_DISPLAY_FIELDS: dict[str, str] = {
    "Date": "formatted from the card's own gameday column; introduces no new source",
    "Matchup": "formatted from home_team/away_team, already covered by model_input:market",
    "ATS prediction": "rendering of pick and market_line, both of which carry their own lineage",
    "Decision score": (
        "rendering of model_probability from the picked side's perspective, calibrated by "
        "the served displayed-confidence cells (docs/displayed_confidence.md); the cells "
        "come from the opener evaluation matched to the active model, so it introduces no "
        "independent source"
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
        market_snapshot, market_captured = _inherited_snapshot(NFLVERSE_SNAPSHOT_KEY, manifest)
    market_frame_captured = _frame_observed_at(forecast, (MARKET_OBSERVED_AT_COLUMN,))
    if market_frame_captured is not None:
        market_captured = _iso(market_frame_captured)
    decision_line = decision_line_week(
        manifest, _optional_int(metadata.get("season")), _optional_int(metadata.get("week"))
    )
    if decision_line is not None:
        market_snapshot = _optional_text(decision_line.get("capture_id")) or market_snapshot
        market_captured = _optional_text(decision_line.get("captured_at_utc")) or market_captured
        market_builder = FamilyBuilder(
            _optional_text(decision_line.get("builder_module")) or market_builder.builder_module,
            _optional_text(decision_line.get("builder_version")) or market_builder.builder_version,
        )
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

    destination = Path(directory) / LINEAGE_FILENAME
    atomic_json(lineage.to_dict(), destination)
    return destination


def read_card_lineage(directory: Path) -> CardLineage:

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
