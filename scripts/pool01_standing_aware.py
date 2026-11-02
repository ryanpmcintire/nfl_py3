import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pool_rank_replay import latest_pregame_spread_pct, served_home_probability  # noqa: E402

from nfl_ats.pick_probability_fit import build_fit_population  # noqa: E402
from nfl_ats.pool import FieldModel, PoolFormat, _field_scores  # noqa: E402

BOOT_REPS = 10000
BOOT_SEED = 20261102
SEED = 20261102


def season_replay(grp_by_week, field, samples, bonus, penalty, seed):
    weeks = sorted(grp_by_week)
    sizes = tuple(len(grp_by_week[w]) for w in weeks)
    rng = np.random.default_rng(seed)
    served_idx = {}
    cover = {}
    pubwon = {}
    prob = {}
    on_public = {}
    for w in weeks:
        g = grp_by_week[w]
        served_idx[w] = int(np.argmax(g["served_pick_prob"].to_numpy()))
        sh = g["served_pick_home"].to_numpy(bool)
        hc = g["home_covered"].astype(float).gt(0.5).to_numpy(bool)
        ph = g["public_home"].to_numpy(bool)
        cover[w] = sh == hc
        pubwon[w] = ph == hc
        prob[w] = g["served_pick_prob"].to_numpy(float)
        on_public[w] = sh == ph

    def bp_score(w, idx):
        return bonus if cover[w][idx] else -penalty

    base_path = dict(served_idx)
    chosen = dict(served_idx)
    rows = []
    for t, w in enumerate(weeks):
        past = weeks[:t]
        fut = weeks[t:]
        ours_past = sum(int(cover[v].sum()) + bp_score(v, chosen[v]) for v in past)
        if past:
            fmt_p = PoolFormat(
                weekly_games=tuple(len(grp_by_week[v]) for v in past),
                best_pick_bonus=bonus,
                best_pick_penalty=penalty,
            )
            pw = np.tile(np.concatenate([pubwon[v] for v in past]), (samples, 1))
            field_p = _field_scores(pw, fmt_p, field, rng)
        else:
            field_p = np.zeros((samples, field.entrants))
        fmt_f = PoolFormat(
            weekly_games=tuple(len(grp_by_week[v]) for v in fut),
            best_pick_bonus=bonus,
            best_pick_penalty=penalty,
        )
        cov_f = {v: rng.random((samples, len(prob[v]))) < prob[v] for v in fut}
        cov_cat = np.concatenate([cov_f[v] for v in fut], axis=1)
        pub_f = np.concatenate(
            [np.where(on_public[v], cov_f[v], ~cov_f[v]) for v in fut], axis=1
        )
        field_f = _field_scores(pub_f, fmt_f, field, rng)
        field_total = field_p + field_f
        ours_f_cover = cov_cat.sum(axis=1).astype(float)
        later = np.zeros(samples)
        for v in fut[1:]:
            hit = cov_f[v][:, served_idx[v]]
            later += np.where(hit, bonus, -penalty)
        ranks = []
        for idx in range(len(prob[w])):
            hit = cov_f[w][:, idx]
            ours = ours_past + ours_f_cover + later + np.where(hit, bonus, -penalty)
            beaten = (field_total > ours[:, None]).sum(axis=1)
            ranks.append(float((beaten + 1).mean()))
        ranks = np.array(ranks)
        base_rank = ranks[served_idx[w]]
        pick = served_idx[w]
        if float(ranks.min()) < base_rank:
            pick = int(np.argmin(ranks))
        chosen[w] = pick
        rows.append(
            {
                "week": int(w),
                "games": len(prob[w]),
                "served_best_pick_prob": float(prob[w][served_idx[w]]),
                "aware_best_pick_prob": float(prob[w][pick]),
                "differs": bool(pick != served_idx[w]),
                "aware_on_public_side": bool(on_public[w][pick]),
                "served_on_public_side": bool(on_public[w][served_idx[w]]),
                "sim_rank_served": float(base_rank),
                "sim_rank_aware": float(ranks[pick]),
                "served_bp_realised": float(bp_score(w, served_idx[w])),
                "aware_bp_realised": float(bp_score(w, pick)),
                "served_bp_covered": bool(cover[w][served_idx[w]]),
                "aware_bp_covered": bool(cover[w][pick]),
                "weeks_remaining_after": len(fut) - 1,
            }
        )
    fmt_all = PoolFormat(
        weekly_games=sizes, best_pick_bonus=bonus, best_pick_penalty=penalty
    )
    pw_all = np.tile(np.concatenate([pubwon[v] for v in weeks]), (samples * 2, 1))
    final_field = _field_scores(pw_all, fmt_all, field, np.random.default_rng(seed + 1))
    cover_total = sum(int(cover[v].sum()) for v in weeks)
    out = {}
    for name, path in (("served", base_path), ("aware", chosen)):
        sc = cover_total + sum(bp_score(v, path[v]) for v in weeks)
        beaten = (final_field > sc).sum(axis=1)
        out[name] = {"score": float(sc), "rank": float((beaten + 1).mean())}
    return rows, out


def main():
    parser = argparse.ArgumentParser(
        description="Standing-aware Best Pick choice replay for POOL-01"
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "pool_rank_replay")
    parser.add_argument(
        "--betting",
        type=Path,
        default=Path("data") / "raw" / "public_betting" / "20260820T111148Z" / "index.parquet",
    )
    parser.add_argument("--entrants", type=int, required=True)
    parser.add_argument("--bonus", type=float, required=True)
    parser.add_argument("--penalty", type=float, required=True)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--tag", default="standing_aware")
    args = parser.parse_args()

    artifacts_root = Path("artifacts")
    pop, _ = build_fit_population(artifacts_root, Path("data"))
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
    share = latest_pregame_spread_pct(pop, args.betting)
    have = np.array([share.get(str(g), np.nan) for g in pop["game_id"]])
    fitted = have[~np.isnan(have)]
    lean = float(np.mean(np.maximum(fitted, 1.0 - fitted)))
    fav_home = pd.to_numeric(pop["tue_open_home_spread"], errors="coerce").fillna(0.0) < 0.0
    pop["public_home"] = np.where(~np.isnan(have), have >= 0.5, fav_home.to_numpy(bool))
    field = FieldModel(entrants=args.entrants, public_lean=lean)

    all_rows = []
    season_rows = []
    for season, sg in pop.groupby("season"):
        by_week = {
            int(w): g.sort_values("game_id").reset_index(drop=True)
            for w, g in sg.groupby("week")
        }
        rows, fin = season_replay(
            by_week, field, args.samples, args.bonus, args.penalty, SEED + int(season)
        )
        for r in rows:
            r["season"] = int(season)
        all_rows.extend(rows)
        season_rows.append(
            {
                "season": int(season),
                "weeks": len(rows),
                "served_final_rank": fin["served"]["rank"],
                "aware_final_rank": fin["aware"]["rank"],
                "rank_gain": fin["served"]["rank"] - fin["aware"]["rank"],
                "score_gain": fin["aware"]["score"] - fin["served"]["score"],
            }
        )
    weekly = pd.DataFrame(all_rows)
    seasons = pd.DataFrame(season_rows)
    gains = seasons["rank_gain"].to_numpy(float)
    rng = np.random.default_rng(BOOT_SEED)
    reps = np.array(
        [gains[rng.integers(0, len(gains), len(gains))].mean() for _ in range(BOOT_REPS)]
    )
    dec = weekly.loc[weekly["differs"]]
    weeks_total = int(seasons["weeks"].sum())
    pts = (weekly["aware_bp_realised"] - weekly["served_bp_realised"]).to_numpy(float)
    pts_reps = np.array(
        [pts[rng.integers(0, len(pts), len(pts))].mean() for _ in range(BOOT_REPS)]
    )
    results = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pool_parameters_status": "entrants, Best Pick bonus and penalty are unknown (no captured pool observables); supplied as sensitivity arguments, not measured",
        "entrants": args.entrants,
        "best_pick_bonus": args.bonus,
        "best_pick_penalty": args.penalty,
        "public_lean_from_splits": lean,
        "field_games_with_splits": int((~np.isnan(have)).sum()),
        "games": len(pop),
        "samples": args.samples,
        "side_flips": 0,
        "weeks": weeks_total,
        "decisive_weeks": int(len(dec)),
        "decisive_record": {
            "aware_pick_covered_served_missed": int(
                (dec["aware_bp_covered"] & ~dec["served_bp_covered"]).sum()
            ),
            "same_outcome": int((dec["aware_bp_covered"] == dec["served_bp_covered"]).sum()),
            "served_pick_covered_aware_missed": int(
                (~dec["aware_bp_covered"] & dec["served_bp_covered"]).sum()
            ),
        },
        "decisive_aware_on_public_side_share": float(dec["aware_on_public_side"].mean())
        if len(dec)
        else None,
        "decisive_mean_probability_given_up": float(
            (dec["served_best_pick_prob"] - dec["aware_best_pick_prob"]).mean()
        )
        if len(dec)
        else None,
        "mean_final_rank_gain_per_season": float(gains.mean()),
        "final_rank_gain_interval_95": [
            float(np.quantile(reps, 0.025)),
            float(np.quantile(reps, 0.975)),
        ],
        "probability_positive_rank": float((reps > 0).mean()),
        "rank_gain_per_week_equivalent": float(gains.sum() / weeks_total),
        "pool_points_gain_per_week": float(pts.mean()),
        "pool_points_interval_95": [
            float(np.quantile(pts_reps, 0.025)),
            float(np.quantile(pts_reps, 0.975)),
        ],
        "probability_positive_points": float((pts_reps > 0).mean()),
        "per_season": season_rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.out / f"{args.tag}_weekly.csv", index=False)
    (args.out / f"{args.tag}_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
