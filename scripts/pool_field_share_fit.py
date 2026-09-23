import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.public_betting_live import SITE_TEAM_ALIASES

ALIAS: dict[str, str] = {**SITE_TEAM_ALIASES, "JAC": "JAX"}
ENTRANTS = 249
SIM_SAMPLES = 2000
SIM_SEED = 20260923
BOOT_REPS = 10000
BOOT_SEED = 20260923


def norm(team):
    return ALIAS.get(team, team)


def load_field_share(path):
    raw = pd.read_csv(path, sep="\t", comment="#")
    raw["away_n"] = raw["away"].map(norm)
    raw["home_n"] = raw["home"].map(norm)
    raw["team_n"] = raw["team"].map(norm)
    rows = []
    for (away, home), grp in raw.groupby(["away_n", "home_n"]):
        home_row = grp.loc[grp["team_n"] == home].iloc[0]
        away_row = grp.loc[grp["team_n"] == away].iloc[0]
        home_picks = float(home_row["picks"])
        away_picks = float(away_row["picks"])
        total = home_picks + away_picks
        home_covered = home_row["result"] == "W"
        rows.append(
            {
                "away": away,
                "home": home,
                "home_share": home_picks / total,
                "field_entries": total,
                "home_covered": bool(home_covered),
            }
        )
    return pd.DataFrame(rows)


def load_margin_dirs(root, week_tag):
    out = []
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith(week_tag):
            ts = d.name[len(week_tag) :]
            dt = datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            out.append((dt, d))
    return sorted(out)


def load_served(root, week_tag):
    frames = []
    for dt, d in load_margin_dirs(root, week_tag):
        f = d / "recommendations.csv"
        if not f.exists():
            continue
        cols = [
            "game_id",
            "kickoff",
            "home_team",
            "away_team",
            "home_cover_probability",
            "home_spread_odds",
            "away_spread_odds",
        ]
        df = pd.read_csv(f, usecols=cols)
        df["snapshot_ts"] = dt
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["kickoff"] = pd.to_datetime(all_df["kickoff"], utc=True)
    pregame = all_df.loc[all_df["snapshot_ts"] < all_df["kickoff"]]
    latest = pregame.sort_values("snapshot_ts").groupby("game_id").tail(1).reset_index(drop=True)
    latest["away"] = latest["away_team"].map(norm)
    latest["home"] = latest["home_team"].map(norm)
    return latest


def implied_prob(odds):
    odds = odds.astype(float)
    return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def load_public_split(path, pairs):
    bet = pd.read_parquet(path)
    bet = bet.loc[bet["has_any_public_data"]].copy()
    bet["away_n"] = bet["away_team"].map(norm)
    bet["home_n"] = bet["home_team"].map(norm)
    out = {}
    for away, home in pairs:
        pair = {away, home}
        match = bet.loc[bet["away_n"].isin(pair) & bet["home_n"].isin(pair)]
        if match.empty:
            continue
        row = match.iloc[0]
        share = row["spread_home_bet_pct"] if row["home_n"] == home else row["spread_away_bet_pct"]
        out[(away, home)] = float(share) / 100.0
    return out


def build_week(field_path, betting_path, margin_root, week_tag, season, week):
    field = load_field_share(field_path)
    served = load_served(margin_root, week_tag)
    merged = field.merge(served, on=["away", "home"], how="left")
    public = load_public_split(betting_path, list(zip(field["away"], field["home"], strict=False)))
    merged["public_split"] = [public.get((a, h), np.nan) for a, h in zip(merged["away"], merged["home"], strict=False)]
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
            rank = simulate_expected_rank(trial_prob, trial_home, field_share, entrants, samples, seed)
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
    parser = argparse.ArgumentParser(description="Fit a field-share model and replay POOL-01 with it")
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "pool_field_share")
    args = parser.parse_args()

    data_root = Path("data")
    margin_root = Path("artifacts") / "margin_predictions"
    week1 = build_week(
        data_root / "splash" / "field" / "2026_week01_field_distribution.tsv",
        data_root / "raw" / "public_betting_live" / "20260913T162822Z" / "index.parquet",
        margin_root,
        "2026-week-01-",
        2026,
        1,
    )
    week2 = build_week(
        data_root / "splash" / "field" / "2026_week02_field_distribution.tsv",
        data_root / "raw" / "public_betting_live" / "20260920T161833Z" / "index.parquet",
        margin_root,
        "2026-week-02-",
        2026,
        2,
    )
    both = pd.concat([week1, week2], ignore_index=True)
    both = both.dropna(
        subset=["public_split", "market_vig_free_home", "home_cover_probability", "home_share"]
    ).reset_index(drop=True)

    actual = both["home_share"].to_numpy(dtype=float)
    proxies = {
        "public_split": both["public_split"].to_numpy(dtype=float),
        "market_vig_free_home": both["market_vig_free_home"].to_numpy(dtype=float),
        "served_home_cover_probability": both["home_cover_probability"].to_numpy(dtype=float),
    }
    proxy_report = {}
    for name, pred in proxies.items():
        r, ci, pp = pearson_ci(pred, actual, BOOT_REPS, BOOT_SEED)
        proxy_report[name] = {
            "mae_share_points": mae(pred, actual),
            "pearson_r": r,
            "bootstrap_ci_95": ci,
            "probability_positive": pp,
            "n": int(len(pred)),
        }

    folds = {}
    fitted_share = np.full(len(both), np.nan)
    for held_week in (1, 2):
        train = both.loc[both["week"] != held_week]
        test = both.loc[both["week"] == held_week]
        X_train = train[["public_split", "market_vig_free_home"]].to_numpy(dtype=float)
        y_train = train["home_share"].to_numpy(dtype=float)
        coef = fit_ols(y_train, X_train)
        X_test = test[["public_split", "market_vig_free_home"]].to_numpy(dtype=float)
        pred_test = predict_ols(coef, X_test)
        fitted_share[test.index.to_numpy()] = pred_test
        y_test = test["home_share"].to_numpy(dtype=float)
        folds[f"trained_on_week_{3 - held_week}_scored_on_week_{held_week}"] = {
            "coefficients_intercept_public_split_market": [float(c) for c in coef],
            "mae_share_points": mae(pred_test, y_test),
            "pearson_r": float(np.corrcoef(pred_test, y_test)[0, 1]) if len(y_test) > 1 else None,
        }
    both["fitted_field_share"] = fitted_share

    weekly_rows = []
    for (season, week), grp in both.groupby(["season", "week"]):
        grp = grp.reset_index(drop=True)
        pick_home_served = grp["home_cover_probability"].to_numpy(dtype=float) >= 0.5
        pick_prob_served = np.where(
            pick_home_served,
            grp["home_cover_probability"].to_numpy(dtype=float),
            1.0 - grp["home_cover_probability"].to_numpy(dtype=float),
        )
        real_share = grp["home_share"].to_numpy(dtype=float)
        fitted = grp["fitted_field_share"].to_numpy(dtype=float)
        home_covered = grp["home_covered"].to_numpy(dtype=bool)

        opt_home, opt_pregame_rank, flips = greedy_card(
            pick_prob_served, pick_home_served, fitted, ENTRANTS, SIM_SAMPLES, SIM_SEED
        )
        served_realised = realised_rank(pick_home_served, home_covered, real_share, ENTRANTS)
        opt_realised = realised_rank(opt_home, home_covered, real_share, ENTRANTS)

        flip_games = [
            {
                "away": grp.loc[i, "away"],
                "home": grp.loc[i, "home"],
                "served_side": "HOME" if pick_home_served[i] else "AWAY",
                "optimal_side": "HOME" if opt_home[i] else "AWAY",
                "home_covered": bool(home_covered[i]),
                "served_pick_correct": bool(pick_home_served[i] == home_covered[i]),
                "optimal_pick_correct": bool(opt_home[i] == home_covered[i]),
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
        "n_games": int(len(both)),
        "entrants": ENTRANTS,
        "proxy_vs_real_field_share": proxy_report,
        "field_share_model": {
            "form": "home_share ~ intercept + public_split + market_vig_free_home",
            "folds_weeks_as_folds": folds,
        },
        "rank_replay_with_fitted_field": weekly_rows,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
