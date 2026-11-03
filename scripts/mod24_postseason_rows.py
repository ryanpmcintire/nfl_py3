import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod24_ftn_turnover_luck as ftn

from nfl_ats import clv, margin

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "mod24_postseason_rows"
SCORE_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)


def keep_all_rows(frame):
    return frame


def stage_run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    served = pd.read_parquet(ftn.SERVED_FEATURES)
    post = served.loc[~served["game_type"].eq("REG") & served["ats_margin"].notna()]
    print("postseason labelled rows", len(post), flush=True)
    base = ftn.decisive(ftn.run_eval(served, ftn.BASE_COLUMNS, "mod24_post_base"))
    original = clv.regular_season_rows
    clv.regular_season_rows = keep_all_rows
    try:
        arm = ftn.decisive(ftn.run_eval(served, ftn.BASE_COLUMNS, "mod24_post_arm"))
    finally:
        clv.regular_season_rows = original
    print("scored rows base", len(base), "arm", len(arm), flush=True)
    results = {}
    a = arm.loc[arm["season"].isin(SCORE_SEASONS)]
    b = base.loc[base["season"].isin(SCORE_SEASONS)]
    results["postseason_rows_2020_2025"] = ftn.compare("postseason rows v base", a, b)
    print("base", ftn.record(b), "arm", ftn.record(a), flush=True)
    (OUT_DIR / "report.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=1, default=str), flush=True)


def stage_replicate():
    import mod24_u1 as u1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arm_path = OUT_DIR / "arm_2011_2019.parquet"
    base_path = OUT_DIR / "base_2011_2019.parquet"
    if arm_path.exists() and base_path.exists():
        arm = pd.read_parquet(arm_path)
        base = pd.read_parquet(base_path)
    else:
        features = pd.read_parquet(u1.u3.FEATURES)
        pop = u1.proxy_population(features)

        def spec(season):
            return (u1.FULL, u1.BASE_ALPHA)

        base = u1.proxy_eval(features, pop, spec)
        original = u1.regular_season_rows
        original_fit = margin.regular_season_rows
        u1.regular_season_rows = keep_all_rows
        margin.regular_season_rows = keep_all_rows
        try:
            arm = u1.proxy_eval(features, pop, spec)
        finally:
            u1.regular_season_rows = original
            margin.regular_season_rows = original_fit
        arm.to_parquet(arm_path)
        base.to_parquet(base_path)
    j = arm.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    j["d_ll"] = j["ll_b"] - j["ll"]
    j["d_brier"] = j["brier_b"] - j["brier"]
    j["block"] = j["season"].astype(str) + "-" + j["week"].astype(str)
    out = {
        "n": len(j),
        "arm_record": f"{int(j['correct'].sum())}-{len(j) - int(j['correct'].sum())}",
        "base_record": f"{int(j['correct_b'].sum())}-{len(j) - int(j['correct_b'].sum())}",
        "flips": int((j["pick_home"] != j["pick_home_b"]).sum()),
    }
    for column in ("d_acc", "d_ll", "d_brier"):
        out[column] = {"mean": float(j[column].mean()), **u1.boot(j, column, ["block"])}
    seasons = {}
    for season, g in j.groupby("season"):
        seasons[int(season)] = {
            "n": len(g),
            "arm": int(g["correct"].sum()),
            "base": int(g["correct_b"].sum()),
            "d_acc": float(100.0 * (g["correct"] - g["correct_b"]).mean()),
            "d_ll": float((g["ll"] - g["ll_b"]).mean()),
        }
    out["per_season"] = seasons
    flipped = j.loc[j["pick_home"] != j["pick_home_b"]]
    out["flip_arm_wins"] = int(flipped["correct"].sum())
    out["flip_base_wins"] = int(flipped["correct_b"].sum())
    (OUT_DIR / "report_2011_2019.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str), flush=True)


def cached(name, builder):
    path = OUT_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame = builder()
    frame.to_parquet(path)
    return frame


def stage_refit_2020_2025():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    served = pd.read_parquet(ftn.SERVED_FEATURES)
    post_ids = set(served.loc[~served["game_type"].eq("REG"), "game_id"].astype(str))
    results = {}

    def run(tag, patch_fit, drop_post_stream):
        original_clv = clv.regular_season_rows
        original_fit = margin.regular_season_rows
        original_prior = clv.prior_rows_before
        clv.regular_season_rows = keep_all_rows
        if patch_fit:
            margin.regular_season_rows = keep_all_rows
        if drop_post_stream:

            def prior_reg(stream, season, week):
                rows = original_prior(stream, season, week)
                return rows.loc[~rows["game_id"].astype(str).isin(post_ids)]

            clv.prior_rows_before = prior_reg
        try:
            return ftn.decisive(ftn.run_eval(served, ftn.BASE_COLUMNS, tag))
        finally:
            clv.regular_season_rows = original_clv
            margin.regular_season_rows = original_fit
            clv.prior_rows_before = original_prior

    base = cached(
        "base_2020_2025",
        lambda: ftn.decisive(ftn.run_eval(served, ftn.BASE_COLUMNS, "mod24_post_base")),
    )
    true_arm = cached("arm_fit_2020_2025", lambda: run("mod24_post_fit_arm", True, False))
    clv_only = cached("arm_clvonly_2020_2025", lambda: run("mod24_post_clvonly", False, False))
    clv_nostream = cached(
        "arm_clvonly_nostream_2020_2025", lambda: run("mod24_post_clvonly_ns", False, True)
    )
    for name, arm in (
        ("fit_true_arm", true_arm),
        ("clv_only", clv_only),
        ("clv_only_postseason_dropped_from_offset_stream", clv_nostream),
    ):
        a = arm.loc[arm["season"].isin(SCORE_SEASONS)]
        b = base.loc[base["season"].isin(SCORE_SEASONS)]
        results[name] = ftn.compare(name, a, b)
        print(name, ftn.record(b), ftn.record(a), results[name]["flips"], flush=True)
    results["scored_week_max_clv_only"] = int(clv_only["week"].max())
    results["scored_week_max_base"] = int(base["week"].max())
    results["clv_only_postseason_games_scored"] = int(
        clv_only.index.astype(str).isin(post_ids).sum()
    )
    flips = clv_only.join(base, rsuffix="_b", how="inner")
    flips = flips.loc[
        flips["correct_at_open_probability_rule"] != flips["correct_at_open_probability_rule_b"]
    ]
    results["clv_only_flips_by_season"] = {
        int(k): int(v) for k, v in flips.groupby("season").size().items()
    }
    (OUT_DIR / "report_2020_2025_fit.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=1, default=str), flush=True)


def main():
    if sys.argv[1:] == ["run"]:
        stage_run()
    elif sys.argv[1:] == ["refit_2020_2025"]:
        stage_refit_2020_2025()
    elif sys.argv[1:] == ["replicate_2011_2019"]:
        stage_replicate()
    else:
        raise SystemExit("usage: mod24_postseason_rows.py run|refit_2020_2025|replicate_2011_2019")


if __name__ == "__main__":
    main()
