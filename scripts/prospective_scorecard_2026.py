from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.io import atomic_json, run_id
from nfl_ats.prospective_scoring import dedicated_challenger_settlement_arm
from nfl_ats.published_picks import locked_best_pick
from nfl_ats.published_probability_history import load_frozen_probability_history
from nfl_ats.settlement import Arm, LedgerSpec, grade_ledger, results_artifact_path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEASON = 2026


def implied_probability(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0)),
        index=odds.index,
    )


def brier(p: pd.Series, y: pd.Series) -> float:
    return float(np.mean((p.to_numpy() - y.to_numpy()) ** 2))


def log_loss(p: pd.Series, y: pd.Series) -> float:
    clipped = np.clip(p.to_numpy(), 1e-6, 1 - 1e-6)
    yv = y.to_numpy()
    return float(-np.mean(yv * np.log(clipped) + (1 - yv) * np.log(1 - clipped)))


def load_forecast_probabilities(clv_ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for forecast_artifact in sorted(clv_ledger["forecast_artifact"].dropna().unique()):
        directory = ARTIFACTS / forecast_artifact
        csv_path = directory / "recommendations.csv"
        metadata_path = directory / "metadata.json"
        if not csv_path.is_file() or not metadata_path.is_file():
            raise DataContractError(f"Recorded baseline forecast is unavailable: {directory}")
        frame = pd.read_csv(
            csv_path,
            usecols=[
                "game_id",
                "spread_line",
                "home_cover_probability_excluding_push",
                "home_spread_odds",
                "away_spread_odds",
            ],
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame["forecast_artifact"] = forecast_artifact
        frame["baseline_created_at_utc"] = metadata["created_at_utc"]
        frame["baseline_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        rows.append(frame.rename(columns={"spread_line": "baseline_home_spread"}))
    if not rows:
        raise DataContractError("No recorded raw-model and market baselines are available")
    return pd.concat(rows, ignore_index=True)


def build_served_frame() -> pd.DataFrame:
    instant = datetime.now(UTC)
    frozen = load_frozen_probability_history(ARTIFACTS, season=SEASON, now=instant)
    if frozen.empty:
        raise DataContractError("No immutable frozen published probabilities are available")
    results = pd.read_parquet(results_artifact_path(ARTIFACTS))
    spec = LedgerSpec(
        key="published_picks",
        relative_path="clv_ledger/published_picks.parquet",
        arms=(Arm("played", "pick_side"),),
        order_column="published_at_utc",
        deadline_columns=("pick_deadline_utc", "kickoff"),
    )
    graded = grade_ledger(frozen, results, spec, graded_at=instant)
    if set(graded["game_id"]) != set(frozen["game_id"]):
        raise DataContractError("Settlement did not retain every eligible frozen decision")
    served = frozen.merge(
        graded[["game_id", "result", "settle_margin", "outcome"]],
        on="game_id",
        validate="one_to_one",
    )
    ledger = pd.read_parquet(ARTIFACTS / "clv_ledger" / "decisions.parquet")
    ledger = ledger.loc[
        ledger["season"].eq(SEASON),
        ["game_id", "forecast_artifact", "recorded_at_utc", "forecast_created_at_utc"],
    ]
    served = served.merge(ledger, on="game_id", how="left", validate="one_to_one")
    forecasts = load_forecast_probabilities(ledger)
    served = served.merge(
        forecasts,
        on=["game_id", "forecast_artifact"],
        how="left",
        validate="one_to_one",
    )
    baseline_time = pd.to_datetime(served["baseline_created_at_utc"], utc=True, errors="coerce")
    binding_time = pd.to_datetime(served["recorded_at_utc"], utc=True, errors="coerce")
    recorded_forecast_time = pd.to_datetime(
        served["forecast_created_at_utc"], utc=True, errors="coerce"
    )
    published_time = pd.to_datetime(served["published_at_utc"], utc=True, errors="coerce")
    deadlines = pd.to_datetime(served["pick_deadline_utc"], utc=True, errors="coerce")
    baseline_line = pd.to_numeric(served["baseline_home_spread"], errors="coerce")
    original_line = pd.to_numeric(served["decision_home_spread"], errors="coerce")
    aligned = baseline_time.notna() & binding_time.notna() & deadlines.notna()
    aligned &= baseline_time.eq(recorded_forecast_time) & baseline_time.le(binding_time)
    aligned &= baseline_time.le(published_time) & binding_time.le(published_time)
    aligned &= baseline_time.lt(deadlines) & binding_time.lt(deadlines)
    aligned &= np.isclose(baseline_line, original_line, rtol=0, atol=1e-10)
    served["baseline_status"] = np.where(
        aligned, "recorded_before_publication_same_line", "unavailable_before_publication_same_line"
    )
    home_market = implied_probability(served["home_spread_odds"])
    away_market = implied_probability(served["away_spread_odds"])
    served["market_home_probability"] = home_market / (home_market + away_market)
    is_home = served["pick_side"].eq("HOME")
    raw_home = pd.to_numeric(served["home_cover_probability_excluding_push"], errors="coerce")
    served["p_raw"] = raw_home.where(is_home, 1.0 - raw_home)
    served["p_market"] = served["market_home_probability"].where(
        is_home, 1.0 - served["market_home_probability"]
    )
    served["p_neutral"] = 0.5
    for column in ("p_served", "p_raw", "p_market", "p_neutral"):
        values = pd.to_numeric(served[column], errors="coerce")
        required = (
            aligned if column in {"p_raw", "p_market"} else pd.Series(True, index=served.index)
        )
        if not (np.isfinite(values[required]) & values[required].between(0, 1)).all():
            raise DataContractError(f"Incomplete or invalid matched probability: {column}")
    served.loc[~aligned, ["p_raw", "p_market", "market_home_probability"]] = np.nan
    served["y"] = served["outcome"].map({"won": 1.0, "lost": 0.0})
    return served


def record_row(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["outcome"].value_counts()
    return {
        "wins": int(counts.get("won", 0)),
        "losses": int(counts.get("lost", 0)),
        "pushes": int(counts.get("pushed", 0)),
        "pending": int(counts.get("pending", 0)),
    }


def probability_block(frame: pd.DataFrame) -> dict[str, object]:
    decisive = frame.dropna(subset=["y"])
    n = len(decisive)
    if n == 0:
        return {"n": 0}
    y = decisive["y"]
    result: dict[str, object] = {
        "n": n,
        "served_brier": brier(decisive["p_served"], y),
        "served_log_loss": log_loss(decisive["p_served"], y),
        "coinflip_brier": brier(pd.Series(0.5, index=decisive.index), y),
        "coinflip_log_loss": log_loss(pd.Series(0.5, index=decisive.index), y),
    }
    matched = decisive.dropna(subset=["p_raw", "p_market"])
    result["matched_baseline_comparison"] = {
        "n": len(matched),
        "unavailable_games": decisive.loc[
            decisive["p_raw"].isna() | decisive["p_market"].isna(), "game_id"
        ].tolist(),
        "arms": {
            arm: {
                "brier": brier(matched[f"p_{arm}"], matched["y"]) if len(matched) else None,
                "log_loss": log_loss(matched[f"p_{arm}"], matched["y"]) if len(matched) else None,
            }
            for arm in ("served", "raw", "market", "neutral")
        },
    }
    return result


def reliability_table(frame: pd.DataFrame) -> list[dict[str, object]]:
    decisive = frame.dropna(subset=["y"])
    bounds = (0.0, 0.5, 0.55, 0.6, 0.65, 1.0)
    rows = []
    for arm in ("served", "raw", "market", "neutral"):
        probabilities = decisive[f"p_{arm}"]
        for lower, upper in pairwise(bounds):
            mask = probabilities.ge(lower) & (
                probabilities.le(upper) if upper == 1 else probabilities.lt(upper)
            )
            group = decisive.loc[mask]
            rows.append(
                {
                    "arm": arm,
                    "lower": lower,
                    "upper": upper,
                    "games": len(group),
                    "mean_probability": float(probabilities.loc[mask].mean())
                    if len(group)
                    else None,
                    "observed_cover_rate": float(group["y"].mean()) if len(group) else None,
                }
            )
    return rows


def best_pick_record(served: pd.DataFrame, decisive_weeks: list[int]) -> dict[str, object]:
    out: dict[str, object] = {}
    best_ids = []
    missing = []
    for week in sorted(served["week"].unique()):
        game_id = locked_best_pick(ARTIFACTS, season=SEASON, week=int(week), now=datetime.now(UTC))
        if game_id is None or game_id not in set(served["game_id"]):
            missing.append(int(week))
            continue
        best_ids.append(game_id)
        out[str(int(week))] = record_row(served.loc[served["game_id"].eq(game_id)])
    best = served.loc[served["game_id"].isin(best_ids)]
    out["to_date"] = record_row(best.loc[best["week"].isin(decisive_weeks)])
    out["all_recorded"] = record_row(best)
    out["missing_weeks"] = missing
    return out


def challenger_paired_records(
    served: pd.DataFrame, decisive_weeks: list[int]
) -> list[dict[str, object]]:
    registry = json.loads((ARTIFACTS / "prospective" / "challengers.json").read_text())
    active_ids = sorted(
        {
            str(entry["challenger_id"])
            for entry in registry["challengers"]
            if entry.get("status") == "ACTIVE_PROSPECTIVE"
        }
    )
    graded = pd.read_parquet(ARTIFACTS / "settlement" / "graded_decisions.parquet")
    generic_frame = graded.loc[
        graded["ledger"].eq("challenger_decisions")
        & graded["season"].eq(SEASON)
        & graded["week"].isin(decisive_weeks)
    ]
    served_outcome = served.loc[
        served["week"].isin(decisive_weeks), ["game_id", "outcome", "decision_home_spread"]
    ].rename(columns={"outcome": "served_outcome", "decision_home_spread": "served_home_spread"})
    rows: list[dict[str, object]] = []
    for challenger_id in active_ids:
        generic = generic_frame.loc[generic_frame["arm"].eq(challenger_id)]
        binding = dedicated_challenger_settlement_arm(challenger_id)
        dedicated = pd.DataFrame()
        dedicated_source: str | None = None
        if binding is not None:
            spec, arm = binding
            dedicated_source = f"{spec.key}:{arm.label}"
            dedicated = graded.loc[
                graded["ledger"].eq(spec.key)
                & graded["arm"].eq(arm.label)
                & graded["season"].eq(SEASON)
                & graded["week"].isin(decisive_weeks)
            ]
        if not generic.empty and not dedicated.empty:
            raise DataContractError(
                f"Challenger {challenger_id!r} has both generic and dedicated graded rows"
            )
        entrant = generic if not generic.empty else dedicated
        source = "challenger_decisions" if not generic.empty else dedicated_source
        if entrant.empty:
            rows.append(
                {
                    "challenger_id": challenger_id,
                    "status": "unsupported" if binding is None else "missing_scored_rows",
                    "source": source,
                }
            )
            continue
        current = grade_ledger(
            entrant,
            pd.read_parquet(results_artifact_path(ARTIFACTS)),
            LedgerSpec(
                key="challenger_freshness_check",
                relative_path="settlement/graded_decisions.parquet",
                arms=(Arm("played", "pick_side"),),
                order_column="recorded_at_utc",
            ),
            graded_at=datetime.now(UTC),
        )
        checked = entrant.merge(
            current, on="game_id", suffixes=("_saved", "_current"), validate="one_to_one"
        )
        if len(checked) != len(entrant):
            raise DataContractError(f"Challenger {challenger_id!r} has invalid graded sides")
        consistent = checked["outcome_saved"].eq(checked["outcome_current"])
        for column in ("result", "settle_margin"):
            consistent &= np.isclose(
                pd.to_numeric(checked[f"{column}_saved"], errors="coerce"),
                pd.to_numeric(checked[f"{column}_current"], errors="coerce"),
                rtol=0,
                atol=1e-10,
                equal_nan=True,
            )
        if not consistent.all():
            raise DataContractError(
                f"Challenger {challenger_id!r} settlement is stale or inconsistent"
            )
        paired = entrant.merge(served_outcome, on="game_id", how="inner", validate="one_to_one")
        paired = paired.loc[
            paired["outcome"].isin(["won", "lost"]) & paired["served_outcome"].isin(["won", "lost"])
        ]
        line_match = np.isclose(
            pd.to_numeric(paired["decision_home_spread"], errors="coerce"),
            pd.to_numeric(paired["served_home_spread"], errors="coerce"),
            rtol=0,
            atol=1e-10,
        )
        if not line_match.all():
            rows.append(
                {
                    "challenger_id": challenger_id,
                    "status": "grading_line_mismatch",
                    "source": source,
                    "mismatched_game_ids": paired.loc[~line_match, "game_id"].astype(str).tolist(),
                }
            )
            continue
        n = len(paired)
        if n == 0:
            rows.append(
                {
                    "challenger_id": challenger_id,
                    "status": "no_decisive_paired_rows",
                    "source": source,
                }
            )
            continue
        challenger_better = int(
            (paired["outcome"].eq("won") & paired["served_outcome"].eq("lost")).sum()
        )
        served_better = int(
            (paired["outcome"].eq("lost") & paired["served_outcome"].eq("won")).sum()
        )
        ties = n - challenger_better - served_better
        rows.append(
            {
                "challenger_id": challenger_id,
                "status": "scored",
                "source": source,
                "n_decisive_paired": n,
                "challenger_beats_served": challenger_better,
                "served_beats_challenger": served_better,
                "ties": ties,
            }
        )
    return rows


def main() -> None:
    source_paths = [
        ARTIFACTS / "clv_ledger" / "published_picks.parquet",
        ARTIFACTS / "clv_ledger" / "decisions.parquet",
        results_artifact_path(ARTIFACTS),
        ARTIFACTS / "settlement" / "graded_decisions.parquet",
        ARTIFACTS / "prospective" / "challengers.json",
        ROOT / "src" / "nfl_ats" / "published_probability_history.py",
        Path(__file__),
    ]
    ledger = pd.read_parquet(ARTIFACTS / "clv_ledger" / "decisions.parquet")
    for artifact in sorted(
        ledger.loc[ledger["season"].eq(SEASON), "forecast_artifact"].dropna().unique()
    ):
        source_paths.extend(
            [ARTIFACTS / artifact / "recommendations.csv", ARTIFACTS / artifact / "metadata.json"]
        )
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    served = build_served_frame()
    decisive_weeks = sorted(
        int(week)
        for week in served.loc[served["outcome"].isin(["won", "lost"]), "week"].dropna().unique()
    )
    served_weeks = {}
    for week, group in served.groupby("week"):
        served_weeks[str(int(cast(Any, week)))] = {
            "record": record_row(group),
            "probability": probability_block(group),
        }
    to_date = served.loc[served["week"].isin(decisive_weeks)]
    decisive_count = int(to_date["outcome"].isin(["won", "lost"]).sum())
    summary = {
        "season": SEASON,
        "decisive_weeks_graded": decisive_weeks,
        "note": (
            f"{len(decisive_weeks)} graded weeks (n={decisive_count} decisive games) "
            "are shown below; descriptive records alone do not establish a serving decision."
        ),
        "served_card": {
            **served_weeks,
            "to_date": {
                "record": record_row(to_date),
                "probability": probability_block(to_date),
            },
        },
        "best_pick": best_pick_record(served, decisive_weeks),
        "challenger_paired_vs_served": challenger_paired_records(served, decisive_weeks),
    }
    summary["reliability"] = reliability_table(to_date)
    summary["probability_event"] = "the frozen played side covers at its recorded original line"
    summary["research_status"] = "descriptive_only_no_selection_or_closure"
    summary["baseline_scope"] = (
        "Archived paper-recorded raw-model and quoted-market baselines at the same line, "
        "created and recorded no later than frozen publication. Unavailable bindings stay "
        "visible and are excluded from all arms of the matched comparison. This is not "
        "an isolated estimate of the served fitted terms."
    )
    summary["baseline_coverage"] = served["baseline_status"].value_counts().to_dict()
    summary["look_accounting"] = {
        "family": "prospective_scorecard_frozen_2026_v1",
        "probability_metric_cells": 12 * (len(served_weeks) + 1),
        "reliability_cells": len(summary["reliability"]),
        "paired_challenger_cells": len(summary["challenger_paired_vs_served"]),
        "record_cells": len(served_weeks) + 1 + len(served_weeks) + 2,
        "all_cells_reported": True,
    }
    after_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    if source_hashes != after_hashes:
        raise DataContractError("Prospective scorecard inputs changed during measurement")
    summary["sources"] = source_hashes
    output_dir = ARTIFACTS / "prospective_scorecard" / run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    served.to_csv(output_dir / "predictions.csv", index=False)
    atomic_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
