import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.data import DataContractError
from nfl_ats.public_betting_live import SITE_TEAM_ALIASES
from nfl_ats.published_probability_history import load_frozen_probability_history

ALIAS: dict[str, str] = {**SITE_TEAM_ALIASES, "JAC": "JAX"}
ENTRANTS = 249
SIM_SAMPLES = 2000
SIM_SEED = 20260923
BOOT_REPS = 10000
BOOT_SEED = 20260923


def norm(team):
    return ALIAS.get(team, team)


def load_field_share(path):
    if not path.is_file():
        raise DataContractError(f"Missing declared field-share input: {path}")
    raw = pd.read_csv(path, sep="\t", comment="#")
    raw["away_n"] = raw["away"].map(norm)
    raw["home_n"] = raw["home"].map(norm)
    raw["team_n"] = raw["team"].map(norm)
    rows = []
    for (away, home), grp in raw.groupby(["away_n", "home_n"]):
        home_rows = grp.loc[grp["team_n"] == home]
        away_rows = grp.loc[grp["team_n"] == away]
        if len(home_rows) != 1 or len(away_rows) != 1:
            raise DataContractError(f"Invalid field-share sides for {away}@{home}")
        home_row = home_rows.iloc[0]
        away_row = away_rows.iloc[0]
        home_picks = float(home_row["picks"])
        away_picks = float(away_row["picks"])
        total = home_picks + away_picks
        if not np.isfinite(total) or total <= 0.0:
            raise DataContractError(f"Invalid field-share total for {away}@{home}")
        field_home_line = float(home_row["line"])
        field_away_line = float(away_row["line"])
        if not np.isfinite([field_home_line, field_away_line]).all() or not np.isclose(
            field_home_line, -field_away_line, atol=1e-9, rtol=0.0
        ):
            raise DataContractError(f"Inconsistent field-share lines for {away}@{home}")
        home_result = str(home_row["result"]).strip().upper()
        away_result = str(away_row["result"]).strip().upper()
        if (home_result, away_result) == ("W", "L"):
            home_covered = True
            outcome_graded = True
        elif (home_result, away_result) == ("L", "W"):
            home_covered = False
            outcome_graded = True
        elif home_result in {"P", "T"} and away_result in {"P", "T"}:
            home_covered = pd.NA
            outcome_graded = False
        else:
            raise DataContractError(f"Inconsistent field-share result for {away}@{home}")
        rows.append(
            {
                "away": away,
                "home": home,
                "home_share": home_picks / total,
                "field_entries": total,
                "field_home_line": field_home_line,
                "field_away_line": field_away_line,
                "home_covered": home_covered,
                "outcome_graded": outcome_graded,
            }
        )
    return pd.DataFrame(rows)


def require_columns(frame, columns, label):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataContractError(f"{label} is missing columns: {', '.join(missing)}")


def load_market_quotes(artifacts_root, served):
    ledger_path = artifacts_root / "clv_ledger" / "decisions.parquet"
    if not ledger_path.is_file():
        raise DataContractError(f"Missing immutable decision ledger: {ledger_path}")
    ledger = pd.read_parquet(ledger_path)
    ledger_columns = (
        "game_id",
        "season",
        "week",
        "recorded_at_utc",
        "forecast_created_at_utc",
        "forecast_artifact",
        "decision_home_spread",
    )
    require_columns(ledger, ledger_columns, "Immutable decision ledger")
    ledger = ledger.loc[
        pd.to_numeric(ledger["season"], errors="coerce").eq(int(served["season"].iloc[0]))
        & pd.to_numeric(ledger["week"], errors="coerce").eq(int(served["week"].iloc[0]))
    ]
    rows = []
    for game in served.itertuples(index=False):
        decision = ledger.loc[ledger["game_id"].astype(str).eq(str(game.game_id))]
        if len(decision) != 1:
            raise DataContractError(
                "Expected one immutable market decision for "
                f"game_id={game.game_id}, found {len(decision)}"
            )
        decision = decision.iloc[0]
        deadline = pd.Timestamp(game.pick_deadline_utc)
        recorded_at = pd.to_datetime(decision["recorded_at_utc"], utc=True, errors="coerce")
        forecast_at = pd.to_datetime(decision["forecast_created_at_utc"], utc=True, errors="coerce")
        if pd.isna(recorded_at) or pd.isna(forecast_at):
            raise DataContractError(f"Invalid market lineage time for game_id={game.game_id}")
        if recorded_at > deadline or forecast_at > deadline:
            raise DataContractError(f"Market lineage is after deadline for game_id={game.game_id}")
        decision_line = float(decision["decision_home_spread"])
        if not np.isfinite(decision_line) or not np.isclose(
            decision_line, float(game.decision_home_spread), atol=1e-9, rtol=0.0
        ):
            raise DataContractError(f"Immutable decision line mismatch for game_id={game.game_id}")
        artifact = Path(str(decision["forecast_artifact"]))
        if artifact.is_absolute() or ".." in artifact.parts:
            raise DataContractError(f"Invalid forecast artifact for game_id={game.game_id}")
        quote_path = artifacts_root / artifact / "recommendations.csv"
        if not quote_path.is_file():
            raise DataContractError(f"Missing immutable market quote: {quote_path}")
        quotes = pd.read_csv(quote_path)
        quote_columns = (
            "game_id",
            "away_team",
            "home_team",
            "market_spread",
            "home_spread_odds",
            "away_spread_odds",
            "market_observed_at_utc",
        )
        require_columns(quotes, quote_columns, f"Market quote {quote_path}")
        quote = quotes.loc[quotes["game_id"].astype(str).eq(str(game.game_id))]
        if len(quote) != 1:
            raise DataContractError(
                f"Expected one matching quote for game_id={game.game_id}, found {len(quote)}"
            )
        quote = quote.iloc[0]
        if norm(str(quote["away_team"])) != norm(str(game.away_team)) or norm(
            str(quote["home_team"])
        ) != norm(str(game.home_team)):
            raise DataContractError(f"Market quote identity mismatch for game_id={game.game_id}")
        quote_line = float(quote["market_spread"])
        if not np.isfinite(quote_line) or not np.isclose(
            quote_line, float(game.decision_home_spread), atol=1e-9, rtol=0.0
        ):
            raise DataContractError(f"Market quote line mismatch for game_id={game.game_id}")
        observed_at = pd.to_datetime(quote["market_observed_at_utc"], utc=True, errors="coerce")
        if pd.isna(observed_at) or observed_at > deadline:
            raise DataContractError(f"Market quote is after deadline for game_id={game.game_id}")
        home_odds = float(quote["home_spread_odds"])
        away_odds = float(quote["away_spread_odds"])
        if not np.isfinite([home_odds, away_odds]).all() or home_odds == 0 or away_odds == 0:
            raise DataContractError(f"Invalid market odds for game_id={game.game_id}")
        rows.append(
            {
                "game_id": str(game.game_id),
                "home_spread_odds": home_odds,
                "away_spread_odds": away_odds,
                "market_quote_observed_at_utc": observed_at,
                "market_quote_recorded_at_utc": recorded_at,
                "market_quote_forecast_created_at_utc": forecast_at,
                "market_quote_artifact": str(artifact / "recommendations.csv").replace("\\", "/"),
            }
        )
    return pd.DataFrame(rows)


def implied_prob(odds):
    odds = odds.astype(float)
    return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def load_public_split(data_root, served):
    paths = sorted((data_root / "raw" / "public_betting_live").glob("*/index.parquet"))
    if not paths:
        raise DataContractError("No immutable public-split artifacts were found")
    frames = []
    public_columns = (
        "capture_ts",
        "source",
        "season",
        "week",
        "away_team",
        "home_team",
        "start_time_utc",
        "spread_home_bet_pct",
        "spread_away_bet_pct",
        "has_any_public_data",
    )
    for path in paths:
        frame = pd.read_parquet(path)
        require_columns(frame, public_columns, f"Public-split artifact {path}")
        frame = frame.loc[
            pd.to_numeric(frame["season"], errors="coerce").eq(int(served["season"].iloc[0]))
            & pd.to_numeric(frame["week"], errors="coerce").eq(int(served["week"].iloc[0]))
        ].copy()
        if frame.empty:
            continue
        frame["public_split_artifact"] = str(path).replace("\\", "/")
        frame["capture_ts"] = pd.to_datetime(frame["capture_ts"], utc=True, errors="coerce")
        frame["start_time_utc"] = pd.to_datetime(frame["start_time_utc"], utc=True, errors="coerce")
        frames.append(frame)
    if not frames:
        raise DataContractError("No public-split artifacts match the declared season and week")
    public = pd.concat(frames, ignore_index=True)
    public["away_n"] = public["away_team"].map(norm)
    public["home_n"] = public["home_team"].map(norm)
    rows = []
    missing = []
    for game in served.itertuples(index=False):
        pair = {norm(str(game.away_team)), norm(str(game.home_team))}
        candidates = public.loc[
            public["away_n"].isin(pair)
            & public["home_n"].isin(pair)
            & public["has_any_public_data"].fillna(False).astype(bool)
            & public["capture_ts"].le(pd.Timestamp(game.pick_deadline_utc))
        ].copy()
        if candidates.empty:
            missing.append(str(game.game_id))
            continue
        latest_at = candidates["capture_ts"].max()
        latest = candidates.loc[candidates["capture_ts"].eq(latest_at)]
        if len(latest) != 1:
            raise DataContractError(
                f"Ambiguous public split at the deadline for game_id={game.game_id}"
            )
        row = latest.iloc[0]
        if pd.isna(row["capture_ts"]) or pd.isna(row["start_time_utc"]):
            raise DataContractError(f"Invalid public-split time for game_id={game.game_id}")
        if pd.Timestamp(row["start_time_utc"]) != pd.Timestamp(game.kickoff):
            raise DataContractError(f"Public-split kickoff mismatch for game_id={game.game_id}")
        source = str(row["source"]).strip()
        if not source or source == "<NA>":
            raise DataContractError(f"Missing public-split source for game_id={game.game_id}")
        if row["home_n"] == norm(str(game.home_team)):
            share = float(row["spread_home_bet_pct"])
        elif row["away_n"] == norm(str(game.home_team)):
            share = float(row["spread_away_bet_pct"])
        else:
            raise DataContractError(f"Public-split identity mismatch for game_id={game.game_id}")
        if not np.isfinite(share) or not 0.0 <= share <= 100.0:
            raise DataContractError(f"Invalid public split for game_id={game.game_id}")
        rows.append(
            {
                "game_id": str(game.game_id),
                "public_split": share / 100.0,
                "public_split_captured_at_utc": pd.Timestamp(row["capture_ts"]),
                "public_split_source": source,
                "public_split_artifact": str(row["public_split_artifact"]),
            }
        )
    if missing:
        raise DataContractError(
            "Missing deadline-eligible public split for declared games: " + ", ".join(missing)
        )
    return pd.DataFrame(rows)


def build_week(field_path, data_root, artifacts_root, season, week):
    field = load_field_share(field_path)
    served = load_frozen_probability_history(artifacts_root, season=season)
    served = served.loc[pd.to_numeric(served["week"], errors="coerce").eq(int(week))].copy()
    if len(served) != len(field):
        raise DataContractError(
            f"Declared week {week} population has {len(field)} games but frozen "
            f"history has {len(served)}"
        )
    served["away"] = served["away_team"].map(norm)
    served["home"] = served["home_team"].map(norm)
    merged = field.merge(served, on=["away", "home"], how="left", validate="one_to_one")
    if merged["game_id"].isna().any():
        missing = [
            f"{row.away}@{row.home}"
            for row in merged.loc[merged["game_id"].isna(), ["away", "home"]].itertuples(
                index=False
            )
        ]
        raise DataContractError("Missing frozen published games: " + ", ".join(missing))
    matching_line = np.isclose(
        merged["field_home_line"].to_numpy(dtype=float),
        -merged["decision_home_spread"].to_numpy(dtype=float),
        atol=1e-9,
        rtol=0.0,
    )
    if not matching_line.all():
        bad = merged.loc[~matching_line, "game_id"].astype(str).tolist()
        raise DataContractError("Field-share line mismatch for games: " + ", ".join(bad))
    quotes = load_market_quotes(artifacts_root, served)
    public = load_public_split(data_root, served)
    merged = merged.merge(quotes, on="game_id", how="left", validate="one_to_one")
    merged = merged.merge(public, on="game_id", how="left", validate="one_to_one")
    required = (
        "public_split",
        "home_spread_odds",
        "away_spread_odds",
        "p_served",
        "recorded_home_probability",
        "home_share",
    )
    if merged[list(required)].isna().any().any():
        raise DataContractError(f"Week {week} has missing declared research inputs")
    merged["market_home_implied"] = implied_prob(merged["home_spread_odds"])
    merged["market_away_implied"] = implied_prob(merged["away_spread_odds"])
    merged["market_vig_free_home"] = merged["market_home_implied"] / (
        merged["market_home_implied"] + merged["market_away_implied"]
    )
    merged["season"] = season
    merged["week"] = week
    return merged


def mae(pred, actual):
    return float(np.abs(pred - actual).mean())


def pearson_ci(pred, actual, reps, seed):
    n = len(pred)
    r = float(np.corrcoef(pred, actual)[0, 1])
    rng = np.random.default_rng(seed)
    boots = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        p, a = pred[idx], actual[idx]
        if np.std(p) == 0 or np.std(a) == 0:
            boots[i] = r
        else:
            boots[i] = np.corrcoef(p, a)[0, 1]
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    prob_positive = float((boots > 0.0).mean())
    return r, [lo, hi], prob_positive


def fit_ols(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coef


def predict_ols(coef, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    pred = X1 @ coef
    return np.clip(pred, 0.02, 0.98)


def field_scores(home_covered_sim, field_share, entrants, generator):
    samples, games = home_covered_sim.shape
    q = np.where(home_covered_sim, field_share[None, :], 1.0 - field_share[None, :])
    p = np.broadcast_to(q[:, :, None], (samples, games, entrants))
    draws = generator.random((samples, games, entrants)) < p
    return draws.sum(axis=1).astype(float)


def simulate_expected_rank(pick_prob, pick_home, field_share, entrants, samples, seed):
    generator = np.random.default_rng(seed)
    our_correct = generator.random((samples, len(pick_prob))) < pick_prob[None, :]
    ours = our_correct.sum(axis=1).astype(float)
    home_covered_sim = np.where(pick_home[None, :], our_correct, ~our_correct)
    rivals = field_scores(home_covered_sim, field_share, entrants, generator)
    beaten = (rivals > ours[:, None]).sum(axis=1)
    return float((beaten + 1).mean())


def greedy_card(pick_prob, pick_home, field_share, entrants, samples, seed):
    cur_prob = pick_prob.copy()
    cur_home = pick_home.copy()
    cur_rank = simulate_expected_rank(cur_prob, cur_home, field_share, entrants, samples, seed)
    flips = []
    improved = True
    while improved:
        improved = False
        best_idx = -1
        best_rank = cur_rank
        for idx in range(len(pick_prob)):
            if idx in flips:
                continue
            trial_prob = cur_prob.copy()
            trial_home = cur_home.copy()
            trial_prob[idx] = 1.0 - trial_prob[idx]
            trial_home[idx] = ~trial_home[idx]
            rank = simulate_expected_rank(
                trial_prob, trial_home, field_share, entrants, samples, seed
            )
            if rank < best_rank:
                best_rank = rank
                best_idx = idx
        if best_idx >= 0:
            cur_prob[best_idx] = 1.0 - cur_prob[best_idx]
            cur_home[best_idx] = ~cur_home[best_idx]
            flips.append(best_idx)
            cur_rank = best_rank
            improved = True
    return cur_home, cur_rank, flips


def rival_tail(q, score):
    pmf = np.zeros(len(q) + 1)
    pmf[0] = 1.0
    for filled, qi in enumerate(q):
        nxt = np.zeros(len(q) + 1)
        nxt[: filled + 1] += pmf[: filled + 1] * (1.0 - qi)
        nxt[1 : filled + 2] += pmf[: filled + 1] * qi
        pmf = nxt
    total = pmf.sum()
    return float(pmf[int(score) + 1 :].sum() / total) if score < len(q) else 0.0


def realised_rank(pick_home, home_covered, field_share, entrants):
    correct = pick_home == home_covered
    ours = int(correct.sum())
    q = np.where(home_covered, field_share, 1.0 - field_share)
    return 1.0 + entrants * rival_tail(q, ours)


def main():
    parser = argparse.ArgumentParser(
        description="Fit a field-share model and replay POOL-01 with it"
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "pool_field_share")
    args = parser.parse_args()

    data_root = Path("data")
    artifacts_root = Path("artifacts")
    week1 = build_week(
        data_root / "splash" / "field" / "2026_week01_field_distribution.tsv",
        data_root,
        artifacts_root,
        2026,
        1,
    )
    week2 = build_week(
        data_root / "splash" / "field" / "2026_week02_field_distribution.tsv",
        data_root,
        artifacts_root,
        2026,
        2,
    )
    both = pd.concat([week1, week2], ignore_index=True)
    if len(both) != 32:
        raise DataContractError(
            f"Declared Week 1-2 population must contain 32 games, found {len(both)}"
        )

    actual = both["home_share"].to_numpy(dtype=float)
    proxies = {
        "public_split": both["public_split"].to_numpy(dtype=float),
        "market_vig_free_home": both["market_vig_free_home"].to_numpy(dtype=float),
        "served_home_cover_probability": both["recorded_home_probability"].to_numpy(dtype=float),
    }
    proxy_report = {}
    for name, pred in proxies.items():
        r, ci, pp = pearson_ci(pred, actual, BOOT_REPS, BOOT_SEED)
        proxy_report[name] = {
            "mae_share_points": mae(pred, actual),
            "pearson_r": r,
            "bootstrap_ci_95": ci,
            "probability_positive": pp,
            "n": len(pred),
        }

    folds = {}
    fitted_share = np.full(len(both), np.nan)
    train = both.loc[both["week"].eq(1)]
    test = both.loc[both["week"].eq(2)]
    X_train = train[["public_split", "market_vig_free_home"]].to_numpy(dtype=float)
    y_train = train["home_share"].to_numpy(dtype=float)
    coef = fit_ols(y_train, X_train)
    X_test = test[["public_split", "market_vig_free_home"]].to_numpy(dtype=float)
    pred_test = predict_ols(coef, X_test)
    fitted_share[test.index.to_numpy()] = pred_test
    y_test = test["home_share"].to_numpy(dtype=float)
    folds["trained_on_week_1_scored_on_week_2"] = {
        "coefficients_intercept_public_split_market": [float(c) for c in coef],
        "mae_share_points": mae(pred_test, y_test),
        "pearson_r": float(np.corrcoef(pred_test, y_test)[0, 1]) if len(y_test) > 1 else None,
    }
    both["fitted_field_share"] = fitted_share

    weekly_rows = []
    for (season, week), grp in both.groupby(["season", "week"]):
        grp = grp.reset_index(drop=True)
        if grp["fitted_field_share"].isna().any():
            weekly_rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "games": len(grp),
                    "status": "unavailable_no_prior_training_week",
                    "graded_games": int(grp["outcome_graded"].sum()),
                    "n_flips": None,
                    "served_realised_rank": None,
                    "optimal_realised_rank": None,
                    "realised_rank_gain": None,
                    "flip_games": [],
                }
            )
            continue
        pick_home_served = grp["pick_side"].eq("HOME").to_numpy(dtype=bool)
        pick_prob_served = grp["p_served"].to_numpy(dtype=float)
        real_share = grp["home_share"].to_numpy(dtype=float)
        fitted = grp["fitted_field_share"].to_numpy(dtype=float)
        outcome_graded = grp["outcome_graded"].to_numpy(dtype=bool)
        home_covered = grp["home_covered"].fillna(False).to_numpy(dtype=bool)

        opt_home, _opt_pregame_rank, flips = greedy_card(
            pick_prob_served, pick_home_served, fitted, ENTRANTS, SIM_SAMPLES, SIM_SEED
        )
        served_realised = realised_rank(
            pick_home_served[outcome_graded],
            home_covered[outcome_graded],
            real_share[outcome_graded],
            ENTRANTS,
        )
        opt_realised = realised_rank(
            opt_home[outcome_graded],
            home_covered[outcome_graded],
            real_share[outcome_graded],
            ENTRANTS,
        )

        flip_games = [
            {
                "away": grp.loc[i, "away"],
                "home": grp.loc[i, "home"],
                "served_side": "HOME" if pick_home_served[i] else "AWAY",
                "optimal_side": "HOME" if opt_home[i] else "AWAY",
                "home_covered": bool(home_covered[i]) if outcome_graded[i] else None,
                "served_pick_correct": (
                    bool(pick_home_served[i] == home_covered[i]) if outcome_graded[i] else None
                ),
                "optimal_pick_correct": (
                    bool(opt_home[i] == home_covered[i]) if outcome_graded[i] else None
                ),
                "fitted_field_share_home": float(fitted[i]),
                "real_field_share_home": float(real_share[i]),
            }
            for i in flips
        ]
        weekly_rows.append(
            {
                "season": int(season),
                "week": int(week),
                "games": len(grp),
                "status": "chronological_out_of_sample",
                "graded_games": int(outcome_graded.sum()),
                "n_flips": len(flips),
                "served_realised_rank": served_realised,
                "optimal_realised_rank": opt_realised,
                "realised_rank_gain": served_realised - opt_realised,
                "flip_games": flip_games,
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    both.drop(columns=["home_spread_odds", "away_spread_odds"], errors="ignore").to_csv(
        run_dir / "games.csv", index=False
    )
    results = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "n_games": len(both),
        "entrants": ENTRANTS,
        "proxy_vs_real_field_share": proxy_report,
        "field_share_model": {
            "form": "home_share ~ intercept + public_split + market_vig_free_home",
            "chronological_folds": folds,
        },
        "rank_replay_with_fitted_field": weekly_rows,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
