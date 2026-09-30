from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

BASE = Path("artifacts/loso_upstream/20260929T235904133782Z")
FEATURES = Path("data/processed/game_features_weak_stack.parquet")
QUOTES = Path("tests/scratch/codex/lead89_unit1/stage_quotes.parquet")
OUTPUT = Path("tests/scratch/codex/lead89_unit2")
LANE = Path("docs/lanes/lead89.md")
STAGES = ("tuesday", "thursday", "sunday")
OUTER = (2023, 2024, 2025)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def inspect_sources() -> tuple[pd.DataFrame, dict]:
    metadata_path = BASE / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prediction_path = BASE / "predictions.parquet"
    prediction_hash = digest(prediction_path)
    feature_hash = digest(FEATURES)
    require(prediction_hash == metadata["predictions_sha256"], "Prediction digest mismatch")
    require(feature_hash == metadata["feature_table_sha256"], "Feature digest mismatch")
    require(feature_hash == metadata["source_hashes"][FEATURES.as_posix()], "Feature provenance")
    injury_terms = [name for name in metadata["margin_features"] if "injury" in name]
    require(bool(injury_terms), "Reassess the clock gate: this model has no injury inputs")
    overlap_columns = [
        "same_season_training_rows",
        "upstream_same_season_training_rows",
        "location_same_season_training_rows",
        "discrete_same_season_training_rows",
        "protection_same_season_training_rows",
    ]
    training_columns = ["base_training_seasons", "upstream_training_seasons"]
    keys = ["game_id", "season", "week"]
    base = read_frame(
        prediction_path,
        [*keys, "held_out_season", "tue_open_home_spread", *overlap_columns, *training_columns],
    )
    require(not base.game_id.duplicated().any(), "Duplicate upstream games")
    require(base.season.eq(base.held_out_season).all(), "Wrong held-out season")
    require(base[overlap_columns].eq(0).all().all(), "Training season overlap")
    for column in training_columns:
        require(
            all(
                str(int(season)) not in str(training).split(",")
                for season, training in zip(base.season, base[column], strict=True)
            ),
            f"Training season overlap in {column}",
        )
    for season in OUTER:
        fold = metadata["calibration_folds"][str(season)]
        require(season not in fold["calibration_training_seasons"], "Calibration overlap")
        require(fold["same_season_training_rows"] == 0, "Calibration overlap count")
    opener_path = Path(metadata["opener_source"]) / "per_game.parquet"
    opener_hash = digest(opener_path)
    require(opener_hash == metadata["source_hashes"][opener_path.as_posix()], "Opener digest")
    opener = read_frame(opener_path, [*keys, "tue_open_home_spread"])
    joined = base.merge(opener, on=keys, how="outer", validate="one_to_one", indicator=True)
    require(joined._merge.eq("both").all(), "Incomplete opener coverage")
    require(
        joined.tue_open_home_spread_x.eq(joined.tue_open_home_spread_y).all(),
        "Different frozen opener",
    )
    observation_columns = [
        f"{side}_injury_{suffix}"
        for side in ("home", "away")
        for suffix in ("observed_at", "observed_at_basis", "observed_at_is_proxy")
    ]
    features = read_frame(FEATURES, ["game_id", *observation_columns])
    require(not features.game_id.duplicated().any(), "Duplicate feature games")
    quote_columns = [
        *keys,
        "stage",
        "decision_at",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
        "commence_time_utc",
        "kickoff",
    ]
    quotes = read_frame(QUOTES, quote_columns)
    for column in quote_columns[4:]:
        quotes[column] = pd.to_datetime(quotes[column], utc=True)
    require(quotes.observed_at_utc.le(quotes.decision_at).all(), "Future stage quote")
    require(quotes.bookmaker_last_update_utc.le(quotes.observed_at_utc).all(), "Future book update")
    known_market = quotes.market_last_update_utc.notna()
    require(
        quotes.loc[known_market, "market_last_update_utc"]
        .le(quotes.loc[known_market, "observed_at_utc"])
        .all(),
        "Future market update",
    )
    require(quotes.decision_at.lt(quotes.kickoff).all(), "Stage after schedule kickoff")
    require(quotes.decision_at.lt(quotes.commence_time_utc).all(), "Stage after quote kickoff")
    clocks = quotes[[*keys, "stage", "decision_at"]].drop_duplicates()
    require(not clocks.duplicated([*keys, "stage"]).any(), "Ambiguous stage clocks")
    audit = clocks.merge(base[keys], on=keys, validate="many_to_one", how="left", indicator=True)
    require(audit._merge.eq("both").all(), "Quote outside the pinned population")
    audit = audit.drop(columns="_merge").merge(
        features, on="game_id", how="left", validate="many_to_one"
    )
    for side in ("home", "away"):
        observed = pd.to_datetime(audit[f"{side}_injury_observed_at"], utc=True)
        audit[f"{side}_injury_observed_at"] = observed
        audit[f"{side}_late"] = observed.gt(audit.decision_at)
        audit[f"{side}_missing"] = observed.isna()
    audit["late"] = audit.home_late | audit.away_late
    audit["missing"] = audit.home_missing | audit.away_missing
    audit["available"] = ~(audit.late | audit.missing)
    require(
        set(base.game_id) == set(audit.loc[audit.stage.eq("tuesday"), "game_id"]),
        "Tuesday coverage",
    )
    summary = {
        "status": "blocked_missing_stage_timed_upstream_forecasts",
        "upstream_season_gate": "passed",
        "frozen_opener_gate": "passed",
        "quote_clock_gate": "passed",
        "outcome_columns_read": [],
        "fits": 0,
        "scored_replays": 0,
        "outcome_looks": 0,
        "planned_looks": 717,
        "population_games": len(base),
        "population_weeks": len(base[["season", "week"]].drop_duplicates()),
        "quote_rows": len(quotes),
        "injury_margin_terms": injury_terms,
        "source_sha256": {
            str(path): value
            for path, value in [
                (prediction_path, prediction_hash),
                (metadata_path, digest(metadata_path)),
                (FEATURES, feature_hash),
                (opener_path, opener_hash),
                (QUOTES, digest(QUOTES)),
                (Path(__file__).relative_to(Path.cwd()), digest(Path(__file__))),
            ]
        },
    }
    return audit, summary


def census(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for population, frame in [
        ("2020-2025", audit),
        ("outer 2023-2025", audit.loc[audit.season.isin(OUTER)]),
    ]:
        for stage in STAGES:
            group = frame.loc[frame.stage.eq(stage)]
            eligible_weeks = group.groupby(["season", "week"]).available.any()
            rows.append(
                {
                    "population": population,
                    "stage": stage,
                    "game_stage_rows": len(group),
                    "late": int(group.late.sum()),
                    "missing": int(group.missing.sum()),
                    "unavailable_union": int((~group.available).sum()),
                    "available": int(group.available.sum()),
                    "weeks_with_available_game": int(eligible_weeks.sum()),
                    "weeks": len(eligible_weeks),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    protocol = LANE.read_text(encoding="utf-8-sig")
    require("Unit-2 amendment saved before scoring" in protocol, "Missing saved amendment")
    require(BASE.as_posix() in protocol and "717 looks" in protocol, "Protocol mismatch")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT / "protocol_before_run.md"
    protocol_path.write_text(protocol, encoding="utf-8")
    with threadpool_limits(limits=1):
        audit, summary = inspect_sources()
        counts = census(audit)
        if audit.available.all():
            summary["status"] = "injury_clock_passed_remaining_stage_audit_required"
        audit.to_parquet(OUTPUT / "input_clock_audit.parquet", index=False)
        counts.to_csv(OUTPUT / "stage_counts.csv", index=False)
        audit.groupby(["season", "stage"])[["late", "missing", "available"]].agg(
            ["sum", "size"]
        ).to_csv(OUTPUT / "season_counts.csv")
        summary["protocol_sha256"] = digest(protocol_path)
        summary["stage_counts"] = counts.to_dict(orient="records")
        (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "scored_replays", "outcome_looks")}))
    print(counts.to_string(index=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
