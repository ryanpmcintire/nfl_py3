import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability_fit import build_fit_population
from nfl_ats.pool import FieldModel, PoolFormat, build_entry, deviate, simulate_pool_finish

ENTRANTS = 100
SIM_SAMPLES = 4000
SIM_SEED = 20260916
BOOT_REPS = 10000
BOOT_SEED = 20260917


def served_home_probability(pop, fold_coefficients):
    probs = np.full(len(pop), np.nan)
    for season, coef in fold_coefficients.items():
        mask = pop["season"].astype(int).eq(int(season)).to_numpy()
        sub = pop.loc[mask]
        z = (
            float(coef["intercept"])
            + float(coef["model_logit"]) * sub["model_logit"].astype(float)
            + float(coef["composition_flag_sum"]) * sub["composition_flag_sum"].astype(float)
            + float(coef["market_move_toward_home"]) * sub["market_move_toward_home"].astype(float)
            + float(coef["market_move_available"]) * sub["market_move_available"].astype(float)
        )
        probs[mask] = 1.0 / (1.0 + np.exp(-z.to_numpy()))
    return probs


def latest_pregame_spread_pct(pop, snapshot_path):
    games = pop[["game_id", "season", "away_team", "home_team"]].copy()
    gf = pd.read_parquet(
        Path("data") / "processed" / "game_features.parquet",
        columns=["game_id", "kickoff"],
    )
    games = games.merge(gf, on="game_id", how="left")
    games["kickoff"] = pd.to_datetime(games["kickoff"], utc=True, errors="coerce")
    bet = pd.read_parquet(snapshot_path)
    bet = bet.loc[bet["has_any_public_data"]].copy()
    bet["capture_ts"] = pd.to_datetime(bet["capture_ts"], utc=True, errors="coerce")
    joined = games.merge(bet, on=["season", "away_team", "home_team"], how="inner")
    joined = joined.loc[joined["capture_ts"] < joined["kickoff"]]
    joined = joined.sort_values("capture_ts").groupby("game_id").tail(1)
    pct = pd.to_numeric(joined["spread_home_bet_pct"], errors="coerce")
    joined = joined.loc[pct.notna()]
    return dict(
        zip(
            joined["game_id"].astype(str),
            pd.to_numeric(joined["spread_home_bet_pct"], errors="coerce") / 100.0,
            strict=False,
        )
    )


def rival_tail(q, score):
    pmf = np.zeros(len(q) + 1)
    pmf[0] = 1.0
    for filled, qi in enumerate(q):
        nxt = np.zeros(len(q) + 1)
        nxt[: filled + 1] += pmf[: filled + 1] * (1.0 - qi)
        nxt[1 : filled + 2] += pmf[: filled + 1] * qi
        pmf = nxt
    total = pmf.sum()
    gt = float(pmf[int(score) + 1 :].sum() / total) if score < len(q) else 0.0
    return gt


def realised_expected_rank(pick_home, home_covered, public_home, lean, entrants):
    pick_home = np.asarray(pick_home, dtype=bool)
    home_covered = np.asarray(home_covered, dtype=bool)
    public_home = np.asarray(public_home, dtype=bool)
    ours = int((pick_home == home_covered).sum())
    public_won = public_home == home_covered
    q = np.where(public_won, lean, 1.0 - lean)
    return 1.0 + entrants * rival_tail(q, ours), ours


def week_entry(pick_prob, on_public, fmt):
    return build_entry(
        fmt,
        cover_probability=np.asarray(pick_prob, dtype=float),
        public_agreement=np.asarray(on_public, dtype=bool),
        best_pick_game=np.array([-1]),
    )


def greedy_rank_card(pick_prob, on_public, field, fmt):
    current = week_entry(pick_prob, on_public, fmt)
    current_rank = simulate_pool_finish(current, field, fmt, samples=SIM_SAMPLES, seed=SIM_SEED)[
        "expected_rank"
    ]
    flips = []
    improved = True
    while improved:
        improved = False
        best_idx = -1
        for idx in range(len(pick_prob)):
            if idx in flips:
                continue
            trial = [*flips, idx]
            entry = current
            entry = deviate(week_entry(pick_prob, on_public, fmt), np.array(trial))
            rank = simulate_pool_finish(entry, field, fmt, samples=SIM_SAMPLES, seed=SIM_SEED)[
                "expected_rank"
            ]
            if rank < current_rank:
                current_rank = rank
                best_idx = idx
        if best_idx >= 0:
            flips.append(best_idx)
            improved = True
    if not flips:
        return current, current_rank, flips
    final = deviate(week_entry(pick_prob, on_public, fmt), np.array(flips))
    final_rank = simulate_pool_finish(final, field, fmt, samples=SIM_SAMPLES, seed=SIM_SEED)[
        "expected_rank"
    ]
    return final, final_rank, flips


def main():
    parser = argparse.ArgumentParser(
        description="Replay rank-optimal vs served ATS cards over past weeks"
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "pool_rank_replay")
    parser.add_argument(
        "--betting",
        type=Path,
        default=Path("data") / "raw" / "public_betting" / "20260820T111148Z" / "index.parquet",
    )
    args = parser.parse_args()

    artifacts_root = Path("artifacts")
    data_root = Path("data")
    pop, _ = build_fit_population(artifacts_root, data_root)
    pointer = json.loads((artifacts_root / "active_pick_probability.json").read_text())
    meta = json.loads((artifacts_root / str(pointer["artifact"]) / "metadata.json").read_text())
    pop = pop.copy()
    pop["served_home_probability"] = served_home_probability(pop, meta["fold_coefficients"])
    pop = pop.loc[pop["served_home_probability"].notna()].reset_index(drop=True)
    pop["served_pick_home"] = pop["served_home_probability"] >= 0.5
    pop["served_pick_prob"] = np.where(
        pop["served_pick_home"],
        pop["served_home_probability"],
        1.0 - pop["served_home_probability"],
    )

    spread_home_frac = latest_pregame_spread_pct(pop, args.betting)
    have = np.array([spread_home_frac.get(str(g), np.nan) for g in pop["game_id"]])
    fitted = have[~np.isnan(have)]
    lean = float(np.mean(np.maximum(fitted, 1.0 - fitted)))
    fav_home = pd.to_numeric(pop["tue_open_home_spread"], errors="coerce").fillna(0.0) < 0.0
    public_home = np.where(~np.isnan(have), have >= 0.5, fav_home.to_numpy(dtype=bool))
    pop["public_home"] = public_home
    pop["field_fitted"] = ~np.isnan(have)

    field = FieldModel(entrants=ENTRANTS, public_lean=lean)
    rows = []
    for (season, week), grp in pop.groupby(["season", "week"]):
        grp = grp.sort_values("game_id").reset_index(drop=True)
        n = len(grp)
        fmt = PoolFormat(weekly_games=(n,), best_pick_bonus=0.0, best_pick_penalty=0.0)
        served_home = grp["served_pick_home"].to_numpy(dtype=bool)
        pick_prob = grp["served_pick_prob"].to_numpy(dtype=float)
        on_public = served_home == grp["public_home"].to_numpy(dtype=bool)
        served_entry = week_entry(pick_prob, on_public, fmt)
        served_sim = simulate_pool_finish(
            served_entry, field, fmt, samples=SIM_SAMPLES, seed=SIM_SEED
        )
        _opt_entry, opt_sim_rank, flips = greedy_rank_card(pick_prob, on_public, field, fmt)
        opt_home = served_home.copy()
        opt_home[np.array(flips, dtype=int)] = ~opt_home[np.array(flips, dtype=int)]
        covered = grp["home_covered"].astype(float).gt(0.5).to_numpy(dtype=bool)
        pub = grp["public_home"].to_numpy(dtype=bool)
        served_rank, served_score = realised_expected_rank(
            served_home, covered, pub, lean, ENTRANTS
        )
        opt_rank, opt_score = realised_expected_rank(opt_home, covered, pub, lean, ENTRANTS)
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "games": n,
                "games_field_fitted": int(grp["field_fitted"].sum()),
                "served_expected_rank_sim": float(served_sim["expected_rank"]),
                "optimal_expected_rank_sim": float(opt_sim_rank),
                "n_flips": len(flips),
                "served_score": served_score,
                "optimal_score": opt_score,
                "served_realised_rank": served_rank,
                "optimal_realised_rank": opt_rank,
                "rank_gain": served_rank - opt_rank,
            }
        )
    weekly = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)
    gains = weekly["rank_gain"].to_numpy(dtype=float)
    seasons = weekly["season"].to_numpy(dtype=int)
    uniq = sorted(int(s) for s in np.unique(seasons))
    rng = np.random.default_rng(BOOT_SEED)
    reps = np.empty(BOOT_REPS)
    for rep in range(BOOT_REPS):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([gains[seasons == s] for s in pick])
        reps[rep] = float(vals.mean())
    mean_gain = float(gains.mean())
    interval = [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))]
    prob_positive = float((reps > 0.0).mean())
    dec = weekly.loc[weekly["n_flips"] > 0]
    dec_gains = dec["rank_gain"].to_numpy(dtype=float)
    better = int((dec_gains > 0).sum())
    tied = int((dec_gains == 0).sum())
    worse = int((dec_gains < 0).sum())
    per_season = [
        {
            "season": int(s),
            "weeks": int((seasons == s).sum()),
            "mean_rank_gain": float(gains[seasons == s].mean()),
        }
        for s in uniq
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.out / "weekly.csv", index=False)
    results = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "entrants": ENTRANTS,
        "public_lean": lean,
        "field_games_fitted": int(pop["field_fitted"].sum()),
        "field_games_total": len(pop),
        "pool_published_picks_available": False,
        "sim_samples": SIM_SAMPLES,
        "sim_seed": SIM_SEED,
        "weeks_replayed": len(weekly),
        "variants_compared": 1,
        "mean_rank_gain": mean_gain,
        "season_block_bootstrap_interval_95": interval,
        "probability_positive": prob_positive,
        "decisive_weeks": len(dec),
        "decisive_record": {"better": better, "tied": tied, "worse": worse},
        "decisive_mean_rank_gain": float(dec_gains.mean()) if len(dec) else 0.0,
        "per_season": per_season,
        "served_total_score": int(weekly["served_score"].sum()),
        "optimal_total_score": int(weekly["optimal_score"].sum()),
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
