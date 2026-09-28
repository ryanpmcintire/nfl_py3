from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.market_quote_proof import (
    resolve_exact_market_quote_proofs,
    verify_exact_market_quote_sources,
)
from nfl_ats.provenance import sha256_file
from nfl_ats.public_betting_live import SITE_TEAM_ALIASES
from nfl_ats.public_betting_proof import (
    load_verified_public_snapshots,
    verify_public_snapshot_sources,
)
from nfl_ats.published_picks import published_picks_path
from nfl_ats.published_probability_history import load_frozen_probability_history

ALIAS: dict[str, str] = {**SITE_TEAM_ALIASES, "JAC": "JAX"}


def norm(team: object) -> str:
    text = str(team).strip().upper()
    return ALIAS.get(text, text)


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def valid_hash(value: object) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def load_decisions(artifacts_root: Path, season: int) -> tuple[pd.DataFrame, str]:
    path = artifacts_root / "clv_ledger" / "decisions.parquet"
    if not path.is_file():
        return pd.DataFrame(), ""
    frame = pd.read_parquet(path)
    required = {
        "game_id",
        "season",
        "recorded_at_utc",
        "forecast_created_at_utc",
        "forecast_artifact",
        "decision_home_spread",
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame(), sha256_file(path)
    selected = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(int(season))].copy()
    return selected, sha256_file(path)


def binding_row(
    game: Any,
    *,
    artifacts_root: Path,
    decisions: pd.DataFrame,
    decisions_sha256: str,
) -> dict[str, object]:
    game_id = str(game.game_id)
    publication = pd.Timestamp(game.published_at_utc)
    original_line = float(game.decision_home_spread)
    output: dict[str, object] = {
        "game_id": game_id,
        "published_at_utc": publication,
        "decision_home_spread": original_line,
        "binding_status": "unavailable",
        "binding_reason": "unresolved",
        "raw_status": "unavailable",
        "raw_reason": "unresolved",
        "market_status": "unavailable",
        "market_reason": "unresolved",
        "binding_source": None,
        "binding_recorded_at_utc": pd.NaT,
        "forecast_created_at_utc": pd.NaT,
        "forecast_artifact": None,
        "recommendations_sha256": None,
        "metadata_sha256": None,
        "decision_ledger_sha256": decisions_sha256 or None,
        "market_observed_at_utc": pd.NaT,
        "home_spread_odds": np.nan,
        "away_spread_odds": np.nan,
        "raw_home_probability": np.nan,
    }
    direct_values = {
        "artifact": getattr(game, "baseline_forecast_artifact", pd.NA),
        "created": getattr(game, "baseline_created_at_utc", pd.NaT),
        "line": getattr(game, "baseline_home_spread", pd.NA),
        "recommendations_sha256": getattr(game, "baseline_recommendations_sha256", pd.NA),
        "metadata_sha256": getattr(game, "baseline_metadata_sha256", pd.NA),
    }
    direct_present = any(not pd.isna(value) for value in direct_values.values())
    if direct_present:
        if any(pd.isna(value) for value in direct_values.values()):
            output["binding_reason"] = "incomplete_published_binding"
            return output
        direct_artifact = direct_values["artifact"]
        artifact = Path(str(direct_artifact))
        binding_recorded = publication
        forecast_created = pd.to_datetime(direct_values["created"], utc=True, errors="coerce")
        binding_line = float(direct_values["line"])
        expected_recommendations_sha256 = str(direct_values["recommendations_sha256"])
        expected_metadata_sha256 = str(direct_values["metadata_sha256"])
        binding_source = "published_pick"
    else:
        if decisions.empty:
            output["binding_reason"] = "missing_decision_ledger"
            return output
        matched = decisions.loc[decisions["game_id"].astype(str).eq(game_id)]
        if len(matched) != 1:
            output["binding_reason"] = f"legacy_decision_rows_{len(matched)}"
            return output
        decision = matched.iloc[0]
        artifact = Path(str(decision["forecast_artifact"]))
        binding_recorded = pd.to_datetime(decision["recorded_at_utc"], utc=True, errors="coerce")
        forecast_created = pd.to_datetime(
            decision["forecast_created_at_utc"], utc=True, errors="coerce"
        )
        binding_line = pd.to_numeric(decision["decision_home_spread"], errors="coerce")
        expected_recommendations_sha256 = ""
        expected_metadata_sha256 = ""
        binding_source = "legacy_decision_ledger"
    output["binding_source"] = binding_source
    output["binding_recorded_at_utc"] = binding_recorded
    output["forecast_created_at_utc"] = forecast_created
    output["forecast_artifact"] = path_text(artifact)
    if artifact.is_absolute() or ".." in artifact.parts:
        output["binding_reason"] = "invalid_forecast_artifact_path"
        return output
    recommendations_path = artifacts_root / artifact / "recommendations.csv"
    metadata_path = artifacts_root / artifact / "metadata.json"
    if not recommendations_path.is_file() or not metadata_path.is_file():
        output["binding_reason"] = "missing_forecast_files"
        return output
    recommendations_sha256 = sha256_file(recommendations_path)
    metadata_sha256 = sha256_file(metadata_path)
    output["recommendations_sha256"] = recommendations_sha256
    output["metadata_sha256"] = metadata_sha256
    if expected_recommendations_sha256 and (
        not valid_hash(expected_recommendations_sha256)
        or expected_recommendations_sha256 != recommendations_sha256
    ):
        output["binding_reason"] = "recommendations_hash_mismatch"
        return output
    if expected_metadata_sha256 and (
        not valid_hash(expected_metadata_sha256) or expected_metadata_sha256 != metadata_sha256
    ):
        output["binding_reason"] = "metadata_hash_mismatch"
        return output
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_created = pd.to_datetime(metadata.get("created_at_utc"), utc=True, errors="coerce")
    if pd.isna(binding_recorded) or pd.isna(forecast_created):
        output["binding_reason"] = "invalid_binding_time"
        return output
    if pd.isna(metadata_created) or metadata_created != forecast_created:
        output["binding_reason"] = "metadata_time_mismatch"
        return output
    if binding_recorded > publication or forecast_created > publication:
        output["binding_reason"] = "binding_or_forecast_after_publication"
        return output
    if forecast_created > binding_recorded:
        output["binding_reason"] = "forecast_after_binding"
        return output
    if not np.isfinite(binding_line) or not np.isclose(
        float(binding_line), original_line, atol=1e-10, rtol=0.0
    ):
        output["binding_reason"] = "binding_line_mismatch"
        return output
    forecasts = pd.read_csv(recommendations_path)
    required = {
        "game_id",
        "away_team",
        "home_team",
        "market_spread",
        "home_cover_probability",
        "home_spread_odds",
        "away_spread_odds",
        "market_observed_at_utc",
    }
    if not required.issubset(forecasts.columns):
        output["binding_reason"] = "forecast_columns_missing"
        return output
    quote = forecasts.loc[forecasts["game_id"].astype(str).eq(game_id)]
    if len(quote) != 1:
        output["binding_reason"] = f"forecast_rows_{len(quote)}"
        return output
    quote = quote.iloc[0]
    if norm(quote["away_team"]) != norm(game.away_team) or norm(quote["home_team"]) != norm(
        game.home_team
    ):
        output["binding_reason"] = "forecast_identity_mismatch"
        return output
    quote_line = pd.to_numeric(quote["market_spread"], errors="coerce")
    if not np.isfinite(quote_line) or not np.isclose(
        float(quote_line), original_line, atol=1e-10, rtol=0.0
    ):
        output["binding_reason"] = "forecast_line_mismatch"
        return output
    probability = pd.to_numeric(quote["home_cover_probability"], errors="coerce")
    if not np.isfinite(probability) or not 0.0 <= float(probability) <= 1.0:
        output["binding_reason"] = "invalid_raw_probability"
        return output
    output["raw_home_probability"] = float(probability)
    output["binding_status"] = "ready"
    output["binding_reason"] = "ready_before_publication_same_line"
    output["raw_status"] = "ready"
    output["raw_reason"] = "ready_before_publication_same_line"
    observed = pd.to_datetime(quote["market_observed_at_utc"], utc=True, errors="coerce")
    output["market_observed_at_utc"] = observed
    if pd.isna(observed):
        output["market_reason"] = "quote_time_missing"
        return output
    if observed > publication:
        output["market_reason"] = "quote_after_publication"
        return output
    if observed > forecast_created:
        output["market_reason"] = "quote_after_forecast_creation_potential_leakage"
        return output
    home_odds = pd.to_numeric(quote["home_spread_odds"], errors="coerce")
    away_odds = pd.to_numeric(quote["away_spread_odds"], errors="coerce")
    if (
        not np.isfinite([home_odds, away_odds]).all()
        or float(home_odds) == 0.0
        or float(away_odds) == 0.0
    ):
        output["market_reason"] = "invalid_market_odds"
        return output
    output["home_spread_odds"] = float(home_odds)
    output["away_spread_odds"] = float(away_odds)
    output["market_status"] = "ready"
    output["market_reason"] = "ready_before_forecast_creation_same_line"
    return output


def load_public_snapshots(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots, issues = load_verified_public_snapshots(data_root, repo_root=Path("."))
    if not snapshots.empty:
        snapshots["away_n"] = snapshots["away_team"].map(norm)
        snapshots["home_n"] = snapshots["home_team"].map(norm)
    return snapshots, issues


def public_row(game: Any, snapshots: pd.DataFrame) -> dict[str, object]:
    output: dict[str, object] = {
        "public_status": "unavailable",
        "public_reason": "unresolved",
        "public_split": np.nan,
        "public_captured_at_utc": pd.NaT,
        "public_source": None,
        "public_artifact": None,
        "public_artifact_sha256": None,
        "public_manifest_artifact": None,
        "public_manifest_sha256": None,
        "public_raw_html_artifact": None,
        "public_raw_html_sha256": None,
        "public_parser_artifact": None,
        "public_parser_sha256": None,
        "public_identity_reparsed": None,
        "public_identity_basis": None,
        "public_observation_time_basis": None,
        "public_chronology_verified": False,
        "public_home_line": np.nan,
        "public_away_line": np.nan,
        "public_home_odds": np.nan,
        "public_away_odds": np.nan,
        "public_inspected_source_hashes": "{}",
        "public_line_status": "unavailable_source_omits_line",
    }
    if snapshots.empty:
        output["public_reason"] = "no_public_snapshots"
        return output
    candidates = snapshots.loc[
        pd.to_numeric(snapshots["season"], errors="coerce").eq(int(game.season))
        & pd.to_numeric(snapshots["week"], errors="coerce").eq(int(game.week))
        & snapshots["away_n"].eq(norm(game.away_team))
        & snapshots["home_n"].eq(norm(game.home_team))
        & snapshots["has_any_public_data"].fillna(False).astype(bool)
        & snapshots["capture_ts"].le(pd.Timestamp(game.published_at_utc))
    ].copy()
    if candidates.empty:
        output["public_reason"] = "no_pair_capture_before_publication"
        return output
    latest_at = candidates["capture_ts"].max()
    latest = candidates.loc[candidates["capture_ts"].eq(latest_at)]
    if len(latest) != 1:
        output["public_reason"] = f"ambiguous_latest_capture_{len(latest)}"
        return output
    row = latest.iloc[0]
    output["public_captured_at_utc"] = row["capture_ts"]
    output["public_source"] = row["source"]
    output["public_artifact"] = row["public_artifact"]
    output["public_artifact_sha256"] = row["public_artifact_sha256"]
    for column in (
        "public_manifest_artifact",
        "public_manifest_sha256",
        "public_raw_html_artifact",
        "public_raw_html_sha256",
        "public_parser_artifact",
        "public_parser_sha256",
        "public_identity_reparsed",
        "public_observation_time_basis",
        "public_chronology_verified",
        "public_inspected_source_hashes",
    ):
        output[column] = row[column]
    output["public_identity_basis"] = row["team_side_basis"]
    output["public_home_line"] = row.get("spread_home_line")
    output["public_away_line"] = row.get("spread_away_line")
    output["public_home_odds"] = row.get("spread_home_odds")
    output["public_away_odds"] = row.get("spread_away_odds")
    output["public_line_status"] = "captured_raw_preserved_not_market_baseline"
    if pd.isna(row["start_time_utc"]) or pd.Timestamp(row["start_time_utc"]) != pd.Timestamp(
        game.kickoff
    ):
        output["public_reason"] = "kickoff_mismatch"
        return output
    if not bool(row["public_chronology_verified"]):
        output["public_reason"] = "request_started_time_not_completion_verified"
        return output
    source = str(row["source"]).strip()
    if not source or source in {"nan", "<NA>"}:
        output["public_reason"] = "missing_public_source"
        return output
    share = pd.to_numeric(row["spread_home_bet_pct"], errors="coerce")
    if not np.isfinite(share) or not 0.0 <= float(share) <= 100.0:
        output["public_reason"] = "invalid_public_share"
        return output
    output["public_split"] = float(share) / 100.0
    output["public_status"] = "ready"
    output["public_reason"] = "ready_before_publication"
    return output


def pool_rows(
    *,
    data_root: Path,
    frozen: pd.DataFrame,
    bindings: pd.DataFrame,
    snapshots: pd.DataFrame,
    weeks: tuple[int, ...],
) -> pd.DataFrame:
    if frozen.empty:
        raise ValueError("No frozen published rows are available for the pool audit")
    season = int(frozen["season"].iloc[0])
    rows = []
    binding_by_game = bindings.set_index("game_id", drop=False)
    for week in weeks:
        field_path = (
            data_root / "splash" / "field" / f"{season}_week{week:02d}_field_distribution.tsv"
        )
        if not field_path.is_file():
            raise FileNotFoundError(field_path)
        field_sha256 = sha256_file(field_path)
        field = pd.read_csv(field_path, sep="\t", comment="#")
        field["away_n"] = field["away"].map(norm)
        field["home_n"] = field["home"].map(norm)
        field["team_n"] = field["team"].map(norm)
        for (away, home), group in field.groupby(["away_n", "home_n"], sort=False):
            home_rows = group.loc[group["team_n"].eq(home)]
            away_rows = group.loc[group["team_n"].eq(away)]
            if len(home_rows) != 1 or len(away_rows) != 1:
                raise ValueError(f"Invalid field rows for {away}@{home}")
            home_line = float(home_rows.iloc[0]["line"])
            away_line = float(away_rows.iloc[0]["line"])
            selected = frozen.loc[
                pd.to_numeric(frozen["week"], errors="coerce").eq(week)
                & frozen["away_team"].map(norm).eq(away)
                & frozen["home_team"].map(norm).eq(home)
            ]
            base: dict[str, object] = {
                "season": season,
                "week": week,
                "away_team": away,
                "home_team": home,
                "field_home_line": home_line,
                "field_away_line": away_line,
                "field_artifact": path_text(field_path),
                "field_artifact_sha256": field_sha256,
                "pool_status": "unavailable",
                "pool_reason": "unresolved",
            }
            if len(selected) != 1:
                base["game_id"] = None
                base["pool_reason"] = f"frozen_rows_{len(selected)}"
                rows.append(base)
                continue
            game = selected.iloc[0]
            game_id = str(game["game_id"])
            base.update(game.to_dict())
            line_ready = np.isclose(
                home_line, -float(game["decision_home_spread"]), atol=1e-10, rtol=0.0
            ) and np.isclose(home_line, -away_line, atol=1e-10, rtol=0.0)
            public = public_row(game, snapshots)
            base.update(public)
            if game_id not in binding_by_game.index:
                base["pool_reason"] = "missing_binding_audit"
                rows.append(base)
                continue
            binding = binding_by_game.loc[game_id]
            if isinstance(binding, pd.DataFrame):
                base["pool_reason"] = "duplicate_binding_audit"
                rows.append(base)
                continue
            base.update(binding.to_dict())
            reasons = []
            if not line_ready:
                reasons.append("field_line_mismatch")
            if base["public_status"] != "ready":
                reasons.append(str(base["public_reason"]))
            if base["market_status"] != "ready":
                reasons.append(str(base["market_reason"]))
            if reasons:
                base["pool_reason"] = ";".join(reasons)
            else:
                base["pool_status"] = "ready"
                base["pool_reason"] = "ready_full_declared_input"
            rows.append(base)
    return pd.DataFrame(rows)


def counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit frozen prospective and pool inputs without fitting models"
    )
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--pool-weeks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    frozen = load_frozen_probability_history(
        args.artifacts_root, season=args.season, now=datetime.now(UTC)
    )
    decisions, decisions_sha256 = load_decisions(args.artifacts_root, args.season)
    bindings = pd.DataFrame(
        [
            binding_row(
                game,
                artifacts_root=args.artifacts_root,
                decisions=decisions,
                decisions_sha256=decisions_sha256,
            )
            for game in frozen.itertuples(index=False)
        ]
    )
    unresolved = bindings["binding_status"].ne("ready")
    bindings.loc[unresolved, "raw_reason"] = bindings.loc[unresolved, "binding_reason"]
    quote_proofs = resolve_exact_market_quote_proofs(args.data_root / "market" / "raw", bindings)
    bindings = bindings.merge(quote_proofs, on="game_id", validate="one_to_one")
    bindings["market_status"] = bindings["exact_market_status"]
    bindings["market_reason"] = bindings["exact_market_reason"]
    bindings.loc[unresolved, "market_status"] = "unavailable"
    bindings.loc[unresolved, "market_reason"] = bindings.loc[unresolved, "binding_reason"]
    published_path = published_picks_path(args.artifacts_root)
    bindings["published_picks_sha256"] = (
        sha256_file(published_path) if published_path.is_file() else None
    )
    snapshots, public_issues = load_public_snapshots(args.data_root)
    pool = pool_rows(
        data_root=args.data_root,
        frozen=frozen,
        bindings=bindings,
        snapshots=snapshots,
        weeks=tuple(args.pool_weeks),
    )
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "season": args.season,
        "prospective_declared_rows": len(bindings),
        "prospective_binding_status": counts(bindings, "binding_status"),
        "prospective_raw_status": counts(bindings, "raw_status"),
        "prospective_market_status": counts(bindings, "market_status"),
        "pool_declared_rows": len(pool),
        "pool_status": counts(pool, "pool_status"),
        "pool_public_status": counts(pool, "public_status"),
        "pool_public_reason": counts(pool, "public_reason"),
        "public_snapshot_verification_issues": (
            counts(public_issues, "public_verification_reason") if not public_issues.empty else {}
        ),
        "public_capture_eligibility": (
            "response_received capture_ts <= exact frozen published_at_utc; legacy "
            "request-start timestamps are retained but unavailable"
        ),
        "baseline_eligibility": (
            "raw forecast and binding timestamps <= exact frozen published_at_utc; market proof "
            "additionally requires a completed immutable snapshot with the same game, opposing "
            "lines, both prices at one bookmaker, and observation <= forecast creation; hashes "
            "recorded"
        ),
    }
    verify_exact_market_quote_sources(bindings)
    verify_public_snapshot_sources(snapshots)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        bindings.to_csv(args.out / "prospective_readiness.csv", index=False)
        pool.to_csv(args.out / "pool_readiness.csv", index=False)
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
