import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pool_rank_replay import latest_pregame_spread_pct, served_home_probability

from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.pick_probability_fit import build_fit_population
from nfl_ats.pool import (
    Entry,
    FieldModel,
    PoolFormat,
    _field_scores,
    deviate,
    simulate_pool_finish,
)

KEY_SCALE = 1000.0
BOOT_REPS = 10000
BOOT_SEED = 20261103


def share_given_key(field_key, ours_key):
    top = field_key.max(axis=1)
    level = (field_key == ours_key[:, None]).sum(axis=1)
    return np.where(ours_key >= top, 1.0 / (1.0 + level), 0.0)


def field_bp(public_won_by_week, lean, entrants, rng):
    total = np.zeros((public_won_by_week[0].shape[0], entrants))
    for pw in public_won_by_week:
        rate = np.where(pw, lean, 1.0 - lean).mean(axis=1)
        total += (rng.random(total.shape) < rate[:, None]).astype(float)
    return total


def season_replay(by_week, field, samples, seed):
    weeks = sorted(by_week)
    rng = np.random.default_rng(seed)
    served = {}
    cover = {}
    pubwon = {}
    prob = {}
    onpub = {}
    for w in weeks:
        g = by_week[w]
        served[w] = int(np.argmax(g["served_pick_prob"].to_numpy()))
        sh = g["served_pick_home"].to_numpy(bool)
        hc = g["home_covered"].astype(float).gt(0.5).to_numpy(bool)
        ph = g["public_home"].to_numpy(bool)
        cover[w] = sh == hc
        pubwon[w] = ph == hc
        prob[w] = g["served_pick_prob"].to_numpy(float)
        onpub[w] = sh == ph
    chosen = dict(served)
    rows = []
    for t, w in enumerate(weeks):
        past = weeks[:t]
        fut = weeks[t:]
        pts_past = float(sum(cover[v].sum() for v in past))
        bp_past = float(sum(cover[v][chosen[v]] for v in past))
        if past:
            fmt_p = PoolFormat(
                weekly_games=tuple(len(by_week[v]) for v in past),
                best_pick_bonus=0.0,
                best_pick_penalty=0.0,
            )
            pw = np.tile(np.concatenate([pubwon[v] for v in past]), (samples, 1))
            fpts_p = _field_scores(pw, fmt_p, field, rng)
            fbp_p = field_bp(
                [np.tile(pubwon[v], (samples, 1)) for v in past],
                field.public_lean,
                field.entrants,
                rng,
            )
        else:
            fpts_p = np.zeros((samples, field.entrants))
            fbp_p = np.zeros((samples, field.entrants))
        fmt_f = PoolFormat(
            weekly_games=tuple(len(by_week[v]) for v in fut),
            best_pick_bonus=0.0,
            best_pick_penalty=0.0,
        )
        cov_f = {v: rng.random((samples, len(prob[v]))) < prob[v] for v in fut}
        pub_by_week = [np.where(onpub[v], cov_f[v], ~cov_f[v]) for v in fut]
        fpts_f = _field_scores(np.concatenate(pub_by_week, axis=1), fmt_f, field, rng)
        fbp_f = field_bp(pub_by_week, field.public_lean, field.entrants, rng)
        field_key = (fpts_p + fpts_f) * KEY_SCALE + fbp_p + fbp_f
        ours_pts = pts_past + np.concatenate([cov_f[v] for v in fut], axis=1).sum(axis=1)
        later = np.zeros(samples)
        for v in fut[1:]:
            later += cov_f[v][:, served[v]]
        base_key = ours_pts * KEY_SCALE + bp_past + later
        p1 = share_given_key(field_key, base_key + 1.0)
        p0 = share_given_key(field_key, base_key)
        value = np.array(
            [float(np.where(cov_f[w][:, i], p1, p0).mean()) for i in range(len(prob[w]))]
        )
        pick = served[w]
        if float(value.max()) > value[served[w]]:
            pick = int(np.argmax(value))
        chosen[w] = pick
        rows.append(
            {
                "week": int(w),
                "games": len(prob[w]),
                "differs": bool(pick != served[w]),
                "sim_pfirst_served": float(value[served[w]]),
                "sim_pfirst_aware": float(value[pick]),
                "served_prob": float(prob[w][served[w]]),
                "aware_prob": float(prob[w][pick]),
                "served_covered": bool(cover[w][served[w]]),
                "aware_covered": bool(cover[w][pick]),
                "aware_on_public_side": bool(onpub[w][pick]),
            }
        )
    sizes = tuple(len(by_week[w]) for w in weeks)
    fmt_all = PoolFormat(weekly_games=sizes, best_pick_bonus=0.0, best_pick_penalty=0.0)
    frng = np.random.default_rng(seed + 1)
    pw_all = np.tile(np.concatenate([pubwon[v] for v in weeks]), (samples * 2, 1))
    fk = _field_scores(pw_all, fmt_all, field, frng) * KEY_SCALE + field_bp(
        [np.tile(pubwon[v], (samples * 2, 1)) for v in weeks],
        field.public_lean,
        field.entrants,
        frng,
    )
    pts = float(sum(cover[v].sum() for v in weeks))
    out = {}
    for name, path in (("served", served), ("aware", chosen)):
        bp = float(sum(cover[v][path[v]] for v in weeks))
        out[name] = {
            "points": pts,
            "bp_wins": bp,
            "p_first": float(
                share_given_key(fk, np.full(fk.shape[0], pts * KEY_SCALE + bp)).mean()
            ),
        }
    return rows, out


def draws_summary(values, rng):
    v = np.asarray(values, dtype=float)
    reps = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(BOOT_REPS)])
    return {
        "mean": float(v.mean()),
        "interval_95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
        "probability_positive": float(probability_positive_from_draws(reps)),
    }


def pol05(entrants, lean, accuracy, agreement, paid, samples, seed):
    fmt = PoolFormat(weekly_games=(15,) * 18, best_pick_bonus=0.0, best_pick_penalty=0.0)
    field = FieldModel(entrants=entrants - 1, public_lean=lean)
    rng = np.random.default_rng(seed)
    pub = rng.random(fmt.games) < agreement
    base = Entry(
        cover_probability=np.full(fmt.games, accuracy),
        on_public_side=pub,
        best_pick_index=np.full(fmt.weeks, -1),
    )
    rows = {}
    for name, entry in (
        ("served_card", base),
        ("flip_10_public_sides", deviate(base, np.flatnonzero(pub)[:10])),
        ("flip_25_public_sides", deviate(base, np.flatnonzero(pub)[:25])),
        (
            "plus_two_accuracy_points",
            Entry(np.full(fmt.games, accuracy + 0.02), pub, base.best_pick_index),
        ),
    ):
        r = simulate_pool_finish(entry, field, fmt, samples=samples, seed=seed, prize_places=paid)
        rows[name] = {
            k: r[k]
            for k in (
                "probability_first",
                "probability_outright",
                "expected_score",
                "expected_rank",
            )
        }
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Standing-aware replay and contest utility on measured pool observables"
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "pool_rank_replay")
    parser.add_argument(
        "--betting",
        type=Path,
        default=Path("data") / "raw" / "public_betting" / "20260820T111148Z" / "index.parquet",
    )
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--tag", default="observed")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    roots = sorted((Path("data") / "pool_observables").glob("*/observations.json"))
    records = [json.loads(p.read_text(encoding="utf-8")) for p in roots]
    obs = [r for r in records if r.get("kind") == "field"][-1]["observation"]
    entries = int(obs["entries"])
    paid = int(obs["paid_places"])

    fdf = pd.read_csv(Path("data/splash/field/2026_romana_empire_w01-07.csv"))
    npick = fdf.n_away + fdf.n_home
    measured_lean = float(np.maximum(fdf.n_away, fdf.n_home).sum() / npick.sum())
    agree = float((fdf.our_side == fdf.field_major_side).mean())

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
    split_lean = float(np.mean(np.maximum(fitted, 1.0 - fitted)))
    fav_home = pd.to_numeric(pop["tue_open_home_spread"], errors="coerce").fillna(0.0) < 0.0
    pop["public_home"] = np.where(~np.isnan(have), have >= 0.5, fav_home.to_numpy(bool))
    hit = float((pop["served_pick_home"] == pop["home_covered"].astype(float).gt(0.5)).mean())

    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "observables": {
            "entries": entries,
            "paid_places": paid,
            "best_pick_bonus": 0.0,
            "measured_field_lean_2026_w1_7": measured_lean,
            "split_lean_pre_lock": split_lean,
            "our_agreement_with_field_major_side_2026_w1_7": agree,
            "served_hit_rate_all_games": hit,
        },
        "replay": {},
    }
    rng = np.random.default_rng(BOOT_SEED)
    for lean_name, lean in (("measured_lean", measured_lean), ("split_lean", split_lean)):
        field = FieldModel(entrants=entries - 1, public_lean=lean)
        seasons = []
        allrows = []
        for season, sg in pop.groupby("season"):
            by_week = {
                int(w): g.sort_values("game_id").reset_index(drop=True)
                for w, g in sg.groupby("week")
            }
            rows, fin = season_replay(by_week, field, args.samples, 20261103 + int(season))
            for r in rows:
                r["season"] = int(season)
            allrows.extend(rows)
            seasons.append(
                {
                    "season": int(season),
                    "served_p_first": fin["served"]["p_first"],
                    "aware_p_first": fin["aware"]["p_first"],
                    "gain": fin["aware"]["p_first"] - fin["served"]["p_first"],
                    "served_bp": fin["served"]["bp_wins"],
                    "aware_bp": fin["aware"]["bp_wins"],
                    "points": fin["served"]["points"],
                }
            )
        wk = pd.DataFrame(allrows)
        dec = wk.loc[wk["differs"]]
        sdf = pd.DataFrame(seasons)
        result["replay"][lean_name] = {
            "lean": lean,
            "weeks": len(wk),
            "decisive_weeks": len(dec),
            "decisive_aware_covered_served_missed": int(
                (dec["aware_covered"] & ~dec["served_covered"]).sum()
            ),
            "decisive_served_covered_aware_missed": int(
                (~dec["aware_covered"] & dec["served_covered"]).sum()
            ),
            "decisive_same": int((dec["aware_covered"] == dec["served_covered"]).sum()),
            "mean_probability_given_up": float((dec["served_prob"] - dec["aware_prob"]).mean())
            if len(dec)
            else None,
            "sim_pfirst_week_start_served_mean": float(wk["sim_pfirst_served"].mean()),
            "sim_pfirst_week_start_aware_mean": float(wk["sim_pfirst_aware"].mean()),
            "realised_pfirst_gain_per_season": draws_summary(sdf["gain"], rng),
            "per_season": seasons,
        }
        wk.to_csv(args.out / f"{args.tag}_{lean_name}_weekly.csv", index=False)

    result["pol05"] = {
        "headline": pol05(entries, measured_lean, hit, agree, paid, 40000, 20261103),
        "split_lean": pol05(entries, split_lean, hit, agree, paid, 40000, 20261103),
    }
    (args.out / f"{args.tag}_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
