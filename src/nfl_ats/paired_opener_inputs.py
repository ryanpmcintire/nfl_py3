from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.calibration import RESIDUAL_SMOOTHING_METHODS, ResidualSmoothingMethod
from nfl_ats.clv import load_decision_quotes_with_sources
from nfl_ats.constants import MIN_FITTABLE_TRAIN_GAMES
from nfl_ats.data import DataContractError
from nfl_ats.home_side_location import fit_home_side_offsets
from nfl_ats.margin import (
    MarginFeatureProfile,
    MarginModel,
    fit_margin_model,
    margin_feature_columns,
)
from nfl_ats.mass_preserving_lattice import (
    BAND_HALF_WIDTH,
    MIN_BAND_GAMES,
    DiscretePushReader,
    serve_discrete_three_way,
)
from nfl_ats.modeling import regular_season_rows
from nfl_ats.paired_opener_prices import PairedPriceExtraction, extract_paired_opener_prices
from nfl_ats.provenance import sha256_file
from nfl_ats.single_book_opener import validate_opener_line_input_manifest

PAIRED_OPENER_INPUT_PROFILES: tuple[MarginFeatureProfile, ...] = ("base", "weak_stack")


@dataclass(frozen=True)
class PairedOpenerInputConfig:
    regressor: str = "ridge"
    feature_profile: MarginFeatureProfile = "base"
    ridge_alpha: float = 10.0
    distribution_fraction: float = 0.20
    min_distribution_rows: int = 10
    random_state: int = 42
    probability_method: ResidualSmoothingMethod = "ecdf"
    lattice_half_width: float = BAND_HALF_WIDTH
    lattice_min_band_games: int = MIN_BAND_GAMES


@dataclass(frozen=True)
class PairedOpenerInputs:
    fanduel: pd.DataFrame
    consensus: pd.DataFrame
    exclusions: pd.DataFrame
    folds: pd.DataFrame
    provenance: dict[str, Any]
    output_paths: dict[str, str]


@dataclass(frozen=True)
class _AuthenticatedArm:
    lines: pd.DataFrame
    manifest: dict[str, Any]
    manifest_sha256: str
    line_path: Path
    line_sha256: str
    market_root: Path
    quotes: pd.DataFrame
    raw_sources: list[dict[str, Any]]


def _read_manifest_locator(path: Path) -> tuple[Path, Path]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise DataContractError(f"Input manifest does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        market_root = Path(payload["market_selection"]["market_root"])
        line_path = Path(payload["line_series"]["artifact_path"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DataContractError(f"Input manifest locator is unreadable: {resolved}") from exc
    if not market_root.is_absolute() or not line_path.is_absolute():
        raise DataContractError("Input manifest locator paths must be absolute")
    return market_root, line_path


def _authenticate_arm(manifest_path: Path, feature_path: Path) -> _AuthenticatedArm:
    market_root, line_path = _read_manifest_locator(manifest_path)
    lines, manifest, manifest_sha = validate_opener_line_input_manifest(
        manifest_path=manifest_path,
        feature_path=feature_path,
        market_root=market_root,
        line_series_path=line_path,
    )
    selection = cast(dict[str, Any], manifest["market_selection"])
    quotes, sources = load_decision_quotes_with_sources(
        market_root,
        capture_kind=str(selection["capture_kind"]),
        labels=cast(list[str], selection["labels"]),
        seasons=cast(list[int] | None, selection["seasons"]),
    )
    recorded_sources = cast(list[dict[str, Any]], selection["raw_sources"])
    line_record = cast(dict[str, Any], manifest["line_series"])
    line_sha256 = str(line_record["artifact_sha256"])
    if sources != recorded_sources:
        raise DataContractError("Loaded raw market sources differ from the validated manifest")
    if sha256_file(line_path) != line_sha256:
        raise DataContractError("Line artifact changed after manifest validation")
    return _AuthenticatedArm(
        lines=lines,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        line_path=line_path.resolve(),
        line_sha256=line_sha256,
        market_root=market_root.resolve(),
        quotes=quotes,
        raw_sources=sources,
    )


def _validate_arm_identity(fd: _AuthenticatedArm, consensus: _AuthenticatedArm) -> None:
    fd_feature = cast(dict[str, Any], fd.manifest["feature_file"])
    consensus_feature = cast(dict[str, Any], consensus.manifest["feature_file"])
    if fd_feature != consensus_feature:
        raise DataContractError("Paired manifests do not bind the same feature artifact")
    if fd.raw_sources != consensus.raw_sources:
        raise DataContractError("Paired manifests do not bind the same raw market source set")
    fd_line = cast(dict[str, Any], fd.manifest["line_series"])
    consensus_line = cast(dict[str, Any], consensus.manifest["line_series"])
    if (
        fd_line["series"] != "book"
        or fd_line["resolved_book"] != "fanduel"
        or consensus_line["series"] != "halfpoint_median"
    ):
        raise DataContractError("Paired manifests must bind FanDuel and half-point median lines")


def _load_authenticated_features(path: Path, expected_sha256: str) -> pd.DataFrame:
    resolved = path.resolve()
    before = sha256_file(resolved)
    if before != expected_sha256:
        raise DataContractError("Feature file SHA-256 differs from paired manifests")
    frame = pd.read_parquet(resolved)
    if sha256_file(resolved) != before:
        raise DataContractError("Feature artifact changed while it was being loaded")
    return frame


def _completed_seasons(values: Sequence[int]) -> tuple[int, ...]:
    if not values or any(isinstance(value, (bool, np.bool_)) for value in values):
        raise DataContractError("Completed seasons must be a nonempty integer sequence")
    seasons = tuple(sorted({int(value) for value in values}))
    if len(seasons) != len(values) or any(int(value) != value for value in values):
        raise DataContractError("Completed seasons must be unique integers")
    return seasons


def _validate_config(config: PairedOpenerInputConfig, min_training_games: int) -> None:
    if min_training_games < MIN_FITTABLE_TRAIN_GAMES:
        raise DataContractError(f"min_training_games must be at least {MIN_FITTABLE_TRAIN_GAMES}")
    if config.feature_profile not in PAIRED_OPENER_INPUT_PROFILES:
        raise DataContractError(
            "paired opener inputs support only line-invariant feature profiles: "
            + ", ".join(PAIRED_OPENER_INPUT_PROFILES)
        )
    if config.regressor not in {"ridge", "hgb"}:
        raise DataContractError(f"Unsupported margin regressor: {config.regressor}")
    if config.probability_method not in RESIDUAL_SMOOTHING_METHODS:
        raise DataContractError(
            f"Unsupported residual probability method: {config.probability_method}"
        )
    if not math.isfinite(config.ridge_alpha) or config.ridge_alpha <= 0.0:
        raise DataContractError("ridge_alpha must be finite and positive")
    if not 0.10 <= config.distribution_fraction < 0.5:
        raise DataContractError("distribution_fraction must be in [0.10, 0.5)")
    if config.min_distribution_rows < 1:
        raise DataContractError("min_distribution_rows must be positive")
    if not math.isfinite(config.lattice_half_width) or config.lattice_half_width <= 0.0:
        raise DataContractError("lattice_half_width must be finite and positive")
    if config.lattice_min_band_games < 1:
        raise DataContractError("lattice_min_band_games must be positive")


def _normalize_features(
    features: pd.DataFrame, feature_profile: MarginFeatureProfile
) -> pd.DataFrame:
    required = {
        "game_id",
        "season",
        "week",
        "gameday",
        "result",
        "ats_margin",
        "spread_line",
        *margin_feature_columns("market_residual", feature_profile),
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise DataContractError(f"Feature artifact is missing columns: {', '.join(missing)}")
    frame = regular_season_rows(features).copy()
    frame["game_id"] = frame["game_id"].astype("string").str.strip()
    if frame["game_id"].isna().any() or frame["game_id"].eq("").any():
        raise DataContractError("Feature game_id values must be nonempty")
    for column in ("season", "week"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise DataContractError(f"Feature {column} values must be finite integers")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise DataContractError(f"Feature {column} values must be integers")
        frame[column] = numeric.astype(int)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    if frame.duplicated(["season", "week", "game_id"]).any():
        raise DataContractError("Feature artifact has duplicate game keys")
    return frame.sort_values(["season", "week", "game_id"], kind="stable").reset_index(drop=True)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or bool(pd.isna(value)):
        return None
    return value


def _frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.copy()
    ordered.columns = ordered.columns.astype(str)
    ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    records = [
        {str(column): _canonical_value(value) for column, value in row.items()}
        for row in ordered.to_dict(orient="records")
    ]
    rows = sorted(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for record in records
    )
    payload = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _price_exclusions(
    arm: str, extraction: PairedPriceExtraction, seasons: tuple[int, ...]
) -> pd.DataFrame:
    frame = extraction.exclusions.loc[extraction.exclusions["season"].isin(seasons)].copy()
    frame.insert(0, "arm", arm)
    return frame


def _population(
    features: pd.DataFrame,
    fd: PairedPriceExtraction,
    consensus: PairedPriceExtraction,
    seasons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["season", "week", "game_id"]
    fd_lines = fd.prices.loc[fd.prices["season"].isin(seasons)].copy()
    consensus_lines = consensus.prices.loc[consensus.prices["season"].isin(seasons)].copy()
    candidates = pd.concat(
        [fd_lines[keys], consensus_lines[keys]], ignore_index=True
    ).drop_duplicates(keys)
    candidates = candidates.sort_values(keys, kind="stable").reset_index(drop=True)
    fd_keys = set(map(tuple, fd_lines[keys].itertuples(index=False, name=None)))
    consensus_keys = set(map(tuple, consensus_lines[keys].itertuples(index=False, name=None)))
    feature_keys = set(map(tuple, features[keys].itertuples(index=False, name=None)))
    completed_keys = set(
        map(
            tuple,
            features.loc[features["result"].notna(), keys].itertuples(index=False, name=None),
        )
    )
    exclusions: list[dict[str, Any]] = []
    eligible: list[tuple[Any, ...]] = []
    for key in candidates.itertuples(index=False, name=None):
        reasons: list[tuple[str, str]] = []
        if key not in fd_keys:
            reasons.append(("fanduel", "missing_authenticated_price"))
        if key not in consensus_keys:
            reasons.append(("consensus", "missing_authenticated_price"))
        if key not in feature_keys:
            reasons.append(("paired", "missing_feature_row"))
        elif key not in completed_keys:
            reasons.append(("paired", "missing_completed_outcome"))
        if reasons:
            for arm, reason in reasons:
                exclusions.append(
                    {
                        "arm": arm,
                        "season": int(key[0]),
                        "week": int(key[1]),
                        "game_id": str(key[2]),
                        "line": np.nan,
                        "bookmaker_key": None,
                        "scope": "population",
                        "reason": reason,
                    }
                )
        else:
            eligible.append(key)
    eligible_frame = pd.DataFrame(eligible, columns=keys)
    if eligible_frame.empty:
        raise DataContractError("No completed games have authenticated paired opener prices")
    paired = eligible_frame.merge(features, on=keys, how="left", validate="one_to_one")
    paired = paired.merge(
        fd_lines[[*keys, "line", "market_probability"]].rename(
            columns={"line": "fd_line", "market_probability": "fd_market_probability"}
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    paired = paired.merge(
        consensus_lines[[*keys, "line", "market_probability"]].rename(
            columns={
                "line": "consensus_line",
                "market_probability": "consensus_market_probability",
            }
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    exclusion_frame = pd.DataFrame(
        exclusions,
        columns=[
            "arm",
            "season",
            "week",
            "game_id",
            "line",
            "bookmaker_key",
            "scope",
            "reason",
        ],
    )
    return paired.sort_values(keys, kind="stable").reset_index(drop=True), exclusion_frame


def _prior_lattice_pool(
    features: pd.DataFrame, lines: pd.DataFrame, target_season: int, arm: str
) -> pd.DataFrame:
    keys = ["season", "week", "game_id"]
    prior = features.loc[
        features["season"].lt(target_season) & features["result"].notna(),
        [*keys, "gameday", "result"],
    ]
    historical_lines = lines.loc[lines["season"].lt(target_season), [*keys, "home_spread"]]
    pool = prior.merge(historical_lines, on=keys, how="inner", validate="one_to_one")
    if pool.empty:
        raise DataContractError(
            f"{arm} has no authenticated prior-season line support before {target_season}"
        )
    pool["home_spread"] = pd.to_numeric(pool["home_spread"], errors="coerce")
    pool["result"] = pd.to_numeric(pool["result"], errors="coerce")
    if pool[["home_spread", "result"]].isna().any().any():
        raise DataContractError(f"{arm} prior-season lattice support is nonnumeric")
    return pool.sort_values(keys, kind="stable").reset_index(drop=True)


def _quote_ids(extraction: PairedPriceExtraction, game_id: str) -> list[dict[str, Any]]:
    rows = extraction.book_prices.loc[extraction.book_prices["game_id"].astype(str).eq(game_id)]
    selected = rows[
        ["bookmaker_key", "provider_event_id", "snapshot_timestamp_utc", "observed_at_utc"]
    ].sort_values(["bookmaker_key", "provider_event_id"], kind="stable")
    return [
        {str(column): _canonical_value(value) for column, value in row.items()}
        for row in selected.to_dict(orient="records")
    ]


def _score_arm(
    arm: str,
    season_rows: pd.DataFrame,
    line_column: str,
    market_column: str,
    model: MarginModel,
    reader: DiscretePushReader,
    archive: pd.DataFrame,
    extraction: PairedPriceExtraction,
    fold_provenance: Mapping[str, Any],
    probability_method: ResidualSmoothingMethod,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    games = season_rows.copy()
    games["spread_line"] = pd.to_numeric(games[line_column], errors="raise").astype(float)
    base = model.predict(games, probability_method=probability_method)
    offsets: npt.NDArray[np.float64]
    if archive.empty:
        offsets = np.zeros(len(games), dtype=float)
        offset_rows = 0
        offset_seasons: list[int] = []
    else:
        fitted = fit_home_side_offsets(archive)
        offsets = fitted.offset_for(games["spread_line"]).fillna(0.0).to_numpy(dtype=float)
        offset_rows = len(archive)
        offset_seasons = sorted(archive["season"].astype(int).unique().tolist())
    served = model.predict(
        games,
        probability_method=probability_method,
        center_offset=offsets,
    )
    model_only = serve_discrete_three_way(
        base,
        games,
        reader,
        residuals=model.residuals,
        probability_method=probability_method,
    )
    served_discrete = serve_discrete_three_way(
        served,
        games,
        reader,
        residuals=model.residuals,
        probability_method=probability_method,
    )
    three_way = served_discrete[
        ["home_cover_probability_excluding_push", "push_probability", "home_loss_probability"]
    ].sum(axis=1)
    if not np.allclose(three_way.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-12):
        raise DataContractError(f"{arm} served probabilities do not sum to one")
    probability_values = np.column_stack(
        [
            served_discrete["home_cover_probability"].to_numpy(dtype=float),
            served_discrete["push_probability"].to_numpy(dtype=float),
            model_only["home_cover_probability"].to_numpy(dtype=float),
            pd.to_numeric(games[market_column], errors="raise").to_numpy(dtype=float),
        ]
    )
    if (
        not np.isfinite(probability_values).all()
        or not ((probability_values >= 0.0) & (probability_values <= 1.0)).all()
    ):
        raise DataContractError(f"{arm} contains invalid probability components")
    line = games["spread_line"].to_numpy(dtype=float)
    result = pd.to_numeric(games["result"], errors="raise").to_numpy(dtype=float)
    ats_result = result - line
    outcome = np.where(
        np.isclose(ats_result, 0.0, rtol=0.0, atol=1e-9),
        np.nan,
        ats_result > 0.0,
    )
    rows: list[dict[str, Any]] = []
    for index, game_id in enumerate(games["game_id"].astype(str).tolist()):
        row_provenance = {
            **dict(fold_provenance),
            "arm": arm,
            "line_source": str(extraction.provenance["line_provenance"].get("line_source", arm)),
            "market_quotes": _quote_ids(extraction, game_id),
            "offset_training_rows": offset_rows,
            "offset_training_seasons": offset_seasons,
        }
        rows.append(
            {
                "season": int(games["season"].iloc[index]),
                "week": int(games["week"].iloc[index]),
                "game_id": game_id,
                "line": float(line[index]),
                "outcome": outcome[index],
                "raw_probability": float(served_discrete["home_cover_probability"].iloc[index]),
                "push_probability": float(served_discrete["push_probability"].iloc[index]),
                "model_only_probability": float(model_only["home_cover_probability"].iloc[index]),
                "market_probability": float(games[market_column].iloc[index]),
                "provenance": json.dumps(
                    row_provenance,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )
    scored = pd.DataFrame(rows)
    next_archive = pd.DataFrame(
        {
            "game_id": games["game_id"].astype(str).to_numpy(),
            "season": games["season"].astype(int).to_numpy(),
            "week": games["week"].astype(int).to_numpy(),
            "spread_line": line,
            "point_incumbent": base["predicted_margin"].to_numpy(dtype=float),
            "result": result,
        }
    )
    fold = {
        **dict(fold_provenance),
        "arm": arm,
        "offset_training_rows": offset_rows,
        "offset_training_seasons": offset_seasons,
    }
    return scored, next_archive, fold


def _write_outputs(
    root: Path,
    fanduel: pd.DataFrame,
    consensus: pd.DataFrame,
    exclusions: pd.DataFrame,
    folds: pd.DataFrame,
    provenance: dict[str, Any],
) -> dict[str, str]:
    resolved = root.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    frames = {
        "fanduel": fanduel,
        "consensus": consensus,
        "exclusions": exclusions,
        "folds": folds,
    }
    paths: dict[str, str] = {}
    for name, frame in frames.items():
        path = resolved / f"{name}.parquet"
        temporary = resolved / f".{name}.parquet.tmp"
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        paths[name] = str(path)
    provenance["output_files"] = {
        name: {"path": path, "sha256": sha256_file(Path(path))}
        for name, path in sorted(paths.items())
    }
    manifest_path = resolved / "manifest.json"
    temporary_manifest = resolved / ".manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    paths["manifest"] = str(manifest_path)
    return paths


def build_paired_opener_inputs(
    feature_path: Path,
    fd_manifest_path: Path,
    consensus_manifest_path: Path,
    completed_seasons: Sequence[int],
    config: PairedOpenerInputConfig,
    min_training_games: int,
    output_root: Path | None = None,
) -> PairedOpenerInputs:
    seasons = _completed_seasons(completed_seasons)
    _validate_config(config, min_training_games)
    fd_arm = _authenticate_arm(fd_manifest_path, feature_path)
    consensus_arm = _authenticate_arm(consensus_manifest_path, feature_path)
    _validate_arm_identity(fd_arm, consensus_arm)
    feature_record = cast(dict[str, Any], fd_arm.manifest["feature_file"])
    features = _normalize_features(
        _load_authenticated_features(feature_path, str(feature_record["sha256"])),
        config.feature_profile,
    )
    fd_prices = extract_paired_opener_prices(
        fd_arm.lines,
        fd_arm.quotes,
        method="fanduel",
        bookmaker_key="fanduel",
        line_provenance={
            "manifest_sha256": fd_arm.manifest_sha256,
            "line_artifact_sha256": fd_arm.line_sha256,
            "line_source": "book:fanduel",
        },
        quote_provenance={"raw_sources": fd_arm.raw_sources},
    )
    consensus_prices = extract_paired_opener_prices(
        consensus_arm.lines,
        consensus_arm.quotes,
        method="consensus",
        line_provenance={
            "manifest_sha256": consensus_arm.manifest_sha256,
            "line_artifact_sha256": consensus_arm.line_sha256,
            "line_source": "halfpoint_median",
        },
        quote_provenance={"raw_sources": consensus_arm.raw_sources},
    )
    requested_seasons = set(seasons)
    maximum_season = max(seasons)
    candidate_seasons = tuple(
        sorted(
            {
                int(value)
                for value in pd.concat(
                    [fd_arm.lines["season"], consensus_arm.lines["season"]],
                    ignore_index=True,
                ).tolist()
                if int(value) <= maximum_season
            }
        )
    )
    if not candidate_seasons:
        raise DataContractError("Paired manifests contain no seasons at or before the request")
    population, population_exclusions = _population(
        features, fd_prices, consensus_prices, candidate_seasons
    )
    population_seasons = set(population["season"].astype(int).unique().tolist())
    missing_requested = sorted(requested_seasons - population_seasons)
    if missing_requested:
        raise DataContractError(
            f"Completed seasons have no eligible paired games: {missing_requested}"
        )
    exclusions = pd.concat(
        [
            _price_exclusions("fanduel", fd_prices, candidate_seasons),
            _price_exclusions("consensus", consensus_prices, candidate_seasons),
            population_exclusions,
        ],
        ignore_index=True,
    ).sort_values(
        ["season", "week", "game_id", "arm", "scope", "reason"],
        kind="stable",
        na_position="first",
    )
    scored: dict[str, list[pd.DataFrame]] = {"fanduel": [], "consensus": []}
    archive_columns = [
        "game_id",
        "season",
        "week",
        "spread_line",
        "point_incumbent",
        "result",
    ]
    archives = {
        "fanduel": pd.DataFrame(columns=archive_columns),
        "consensus": pd.DataFrame(columns=archive_columns),
    }
    arm_specs = (
        (
            "fanduel",
            fd_arm,
            fd_prices,
            "fd_line",
            "fd_market_probability",
        ),
        (
            "consensus",
            consensus_arm,
            consensus_prices,
            "consensus_line",
            "consensus_market_probability",
        ),
    )
    fold_rows: list[dict[str, Any]] = []
    skipped_warmups: list[dict[str, Any]] = []
    for season in sorted(population_seasons):
        target = population.loc[population["season"].eq(season)].copy()
        requested = season in requested_seasons
        training = features.loc[features["season"].lt(season) & features["result"].notna()].copy()
        if len(training) < min_training_games:
            reason = (
                f"Season {season} has {len(training)} prior completed games; "
                f"{min_training_games} required"
            )
            if requested:
                raise DataContractError(reason)
            skipped_warmups.append({"season": season, "reason": reason})
            continue
        training_max = pd.Timestamp(training["gameday"].max())
        target_min = pd.Timestamp(target["gameday"].min())
        if training_max >= target_min:
            raise DataContractError(f"Season {season} training does not precede its target games")
        pools: dict[str, pd.DataFrame] = {}
        pool_maxes: dict[str, pd.Timestamp] = {}
        support_errors: list[str] = []
        for arm, authenticated, _, _, _ in arm_specs:
            try:
                pool = _prior_lattice_pool(features, authenticated.lines, season, arm)
            except DataContractError as exc:
                if "has no authenticated prior-season line support" not in str(exc):
                    raise
                support_errors.append(str(exc))
                continue
            pool_max = pd.Timestamp(pool["gameday"].max())
            if pool_max >= target_min:
                raise DataContractError(
                    f"{arm} season {season} lattice support does not precede target games"
                )
            pools[arm] = pool
            pool_maxes[arm] = pool_max
        if support_errors:
            reason = "; ".join(support_errors)
            if requested:
                raise DataContractError(reason)
            skipped_warmups.append({"season": season, "reason": reason})
            continue
        model = fit_margin_model(
            training,
            target="market_residual",
            model_name=config.regressor,
            distribution_fraction=config.distribution_fraction,
            min_distribution_rows=config.min_distribution_rows,
            random_state=config.random_state,
            feature_profile=config.feature_profile,
            ridge_alpha=config.ridge_alpha,
        )
        training_seasons = sorted(training["season"].astype(int).unique().tolist())
        for arm, authenticated, extraction, line_column, market_column in arm_specs:
            pool = pools[arm]
            pool_max = pool_maxes[arm]
            reader = DiscretePushReader(
                lines=pool["home_spread"].to_numpy(dtype=float),
                margins=np.rint(pool["result"].to_numpy(dtype=float)),
                half_width=config.lattice_half_width,
                min_band_games=config.lattice_min_band_games,
                prior_rows=len(pool),
                max_gameday=pool_max.date().isoformat(),
            )
            fold_provenance = {
                "schema": "paired_opener_inputs.row.v1",
                "feature_sha256": str(feature_record["sha256"]),
                "manifest_sha256": authenticated.manifest_sha256,
                "line_artifact_sha256": authenticated.line_sha256,
                "raw_source_hashes": [
                    {
                        "manifest_sha256": source["manifest_sha256"],
                        "quotes_sha256": source["quotes_sha256"],
                    }
                    for source in authenticated.raw_sources
                ],
                "model_config": asdict(config),
                "target_season": season,
                "training_seasons": training_seasons,
                "training_rows": len(training),
                "training_max_gameday": training_max.date().isoformat(),
                "lattice_rows": len(pool),
                "lattice_seasons": sorted(pool["season"].astype(int).unique().tolist()),
                "lattice_max_gameday": pool_max.date().isoformat(),
            }
            arm_scored, next_archive, fold = _score_arm(
                arm,
                target,
                line_column,
                market_column,
                model,
                reader,
                archives[arm],
                extraction,
                fold_provenance,
                config.probability_method,
            )
            if requested:
                scored[arm].append(arm_scored)
            archives[arm] = pd.concat([archives[arm], next_archive], ignore_index=True)
            fold_rows.append(fold)
    keys = ["season", "week", "game_id"]
    fanduel = (
        pd.concat(scored["fanduel"], ignore_index=True)
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
    consensus = (
        pd.concat(scored["consensus"], ignore_index=True)
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
    if fanduel[keys].to_dict("records") != consensus[keys].to_dict("records"):
        raise DataContractError("Produced arms do not have identical ordered game keys")
    folds = (
        pd.DataFrame(fold_rows)
        .sort_values(["target_season", "arm"], kind="stable")
        .reset_index(drop=True)
    )
    provenance = {
        "schema": "paired_opener_inputs.v1",
        "feature_path": str(feature_path.resolve()),
        "feature_sha256": str(feature_record["sha256"]),
        "fanduel_manifest_path": str(fd_manifest_path.resolve()),
        "fanduel_manifest_sha256": fd_arm.manifest_sha256,
        "consensus_manifest_path": str(consensus_manifest_path.resolve()),
        "consensus_manifest_sha256": consensus_arm.manifest_sha256,
        "completed_seasons": list(seasons),
        "warmup_seasons": sorted(
            {int(row["target_season"]) for row in fold_rows} - requested_seasons
        ),
        "skipped_warmups": skipped_warmups,
        "config": asdict(config),
        "min_training_games": min_training_games,
        "raw_sources": fd_arm.raw_sources,
        "fanduel_price_provenance": fd_prices.provenance,
        "consensus_price_provenance": consensus_prices.provenance,
        "exclusion_counts": {
            str(reason): int(count)
            for reason, count in exclusions["reason"].value_counts().sort_index().items()
        },
        "fanduel_sha256": _frame_hash(fanduel),
        "consensus_sha256": _frame_hash(consensus),
        "exclusions_sha256": _frame_hash(exclusions),
        "folds_sha256": _frame_hash(folds),
    }
    paths = (
        {}
        if output_root is None
        else _write_outputs(
            output_root,
            fanduel,
            consensus,
            exclusions,
            folds,
            provenance,
        )
    )
    return PairedOpenerInputs(
        fanduel=fanduel,
        consensus=consensus,
        exclusions=exclusions,
        folds=folds,
        provenance=provenance,
        output_paths=paths,
    )


__all__ = [
    "PAIRED_OPENER_INPUT_PROFILES",
    "PairedOpenerInputConfig",
    "PairedOpenerInputs",
    "build_paired_opener_inputs",
]
