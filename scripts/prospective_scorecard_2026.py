from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.io import atomic_json, run_id

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEASON = 2026
DECISIVE_WEEKS = (1, 2)


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
        csv_path = ARTIFACTS / forecast_artifact / "recommendations.csv"
        if not csv_path.is_file():
            continue
        frame = pd.read_csv(
            csv_path,
            usecols=[
                "game_id",
                "home_cover_probability_excluding_push",
                "home_spread_odds",
                "away_spread_odds",
            ],
        )
        frame["forecast_artifact"] = forecast_artifact
        rows.append(frame)
    if not rows:
        return pd.DataFrame(
            columns=[
                "game_id",
                "home_cover_probability_excluding_push",
                "home_spread_odds",
                "away_spread_odds",
                "forecast_artifact",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def latest_revision_probability(pick_revisions: pd.DataFrame) -> pd.Series:
    revised = pick_revisions.loc[pick_revisions["season"].eq(SEASON)].copy()
    if revised.empty:
        return pd.Series(dtype=float)
    revised = revised.sort_values("revision_recorded_at_utc")
    revised = revised.drop_duplicates("game_id", keep="last")
    return revised.set_index("game_id")["new_home_cover_probability"].astype(float)


def build_served_frame() -> pd.DataFrame:
    graded = pd.read_parquet(ARTIFACTS / "settlement" / "graded_decisions.parquet")
    served = graded.loc[
        graded["ledger"].eq("paper_decisions")
        & graded["arm"].eq("played")
        & graded["season"].eq(SEASON)
    ].copy()
    clv_ledger = pd.read_parquet(ARTIFACTS / "clv_ledger" / "decisions.parquet")
    clv_ledger = clv_ledger.loc[
        clv_ledger["season"].eq(SEASON), ["game_id", "forecast_artifact"]
    ]
    served = served.merge(clv_ledger, on="game_id", how="left")
    forecasts = load_forecast_probabilities(clv_ledger)
    served = served.merge(forecasts, on=["game_id", "forecast_artifact"], how="left")
    pick_revisions = pd.read_parquet(ARTIFACTS / "prospective" / "pick_revisions.parquet")
    revision_probability = latest_revision_probability(pick_revisions)
    served["recorded_home_probability"] = served["game_id"].map(revision_probability)
    served["recorded_home_probability"] = served["recorded_home_probability"].fillna(
        served["home_cover_probability_excluding_push"]
    )
    home_market = implied_probability(served["home_spread_odds"])
    away_market = implied_probability(served["away_spread_odds"])
    fair_home_market = home_market / (home_market + away_market)
    served["market_home_probability"] = fair_home_market
    is_home = served["pick_side"].eq("HOME")
    served["p_served"] = served["recorded_home_probability"].where(
        is_home, 1.0 - served["recorded_home_probability"]
    )
    served["p_market"] = served["market_home_probability"].where(
        is_home, 1.0 - served["market_home_probability"]
    )
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
    n = int(len(decisive))
    if n == 0:
        return {"n": 0}
    y = decisive["y"]
    return {
        "n": n,
        "served_brier": brier(decisive["p_served"], y),
        "served_log_loss": log_loss(decisive["p_served"], y),
        "coinflip_brier": brier(pd.Series(0.5, index=decisive.index), y),
        "coinflip_log_loss": log_loss(pd.Series(0.5, index=decisive.index), y),
        "market_brier": brier(decisive["p_market"], y),
        "market_log_loss": log_loss(decisive["p_market"], y),
    }


def best_pick_record() -> dict[str, dict[str, int]]:
    graded = pd.read_parquet(ARTIFACTS / "settlement" / "graded_decisions.parquet")
    best = graded.loc[
        graded["ledger"].eq("paper_decisions")
        & graded["arm"].eq("best_pick")
        & graded["season"].eq(SEASON)
    ]
    out: dict[str, dict[str, int]] = {}
    for week, group in best.groupby("week"):
        out[str(int(week))] = record_row(group)
    out["to_date"] = record_row(best.loc[best["week"].isin(DECISIVE_WEEKS)])
    out["all_recorded"] = record_row(best)
    return out


def challenger_paired_records(served: pd.DataFrame) -> list[dict[str, object]]:
    registry = json.loads((ARTIFACTS / "prospective" / "challengers.json").read_text())
    active_ids = sorted(
        {
            str(entry["challenger_id"])
            for entry in registry["challengers"]
            if entry.get("status") == "ACTIVE_PROSPECTIVE"
        }
    )
    graded = pd.read_parquet(ARTIFACTS / "settlement" / "graded_decisions.parquet")
    challenger_frame = graded.loc[
        graded["ledger"].eq("challenger_decisions")
        & graded["season"].eq(SEASON)
        & graded["week"].isin(DECISIVE_WEEKS)
    ]
    served_outcome = served.loc[
        served["week"].isin(DECISIVE_WEEKS), ["game_id", "outcome"]
    ].rename(columns={"outcome": "served_outcome"})
    rows: list[dict[str, object]] = []
    for challenger_id in active_ids:
        entrant = challenger_frame.loc[challenger_frame["arm"].eq(challenger_id)]
        if entrant.empty:
            continue
        paired = entrant.merge(served_outcome, on="game_id", how="inner")
        paired = paired.loc[
            paired["outcome"].isin(["won", "lost"])
            & paired["served_outcome"].isin(["won", "lost"])
        ]
        n = int(len(paired))
        if n == 0:
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
                "n_decisive_paired": n,
                "challenger_beats_served": challenger_better,
                "served_beats_challenger": served_better,
                "ties": ties,
            }
        )
    return rows


def main() -> None:
    served = build_served_frame()
    served_weeks = {}
    for week, group in served.groupby("week"):
        served_weeks[str(int(week))] = {
            "record": record_row(group),
            "probability": probability_block(group),
        }
    to_date = served.loc[served["week"].isin(DECISIVE_WEEKS)]
    summary = {
        "season": SEASON,
        "decisive_weeks_graded": list(DECISIVE_WEEKS),
        "note": (
            "Two graded weeks (n=32 decisive games) is far too few to decide anything; "
            "figures below are reported, not a verdict."
        ),
        "served_card": {
            **served_weeks,
            "to_date": {
                "record": record_row(to_date),
                "probability": probability_block(to_date),
            },
        },
        "best_pick": best_pick_record(),
        "challenger_paired_vs_served": challenger_paired_records(served),
    }
    output_dir = ARTIFACTS / "prospective_scorecard" / run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
