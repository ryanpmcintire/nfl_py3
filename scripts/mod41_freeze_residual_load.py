import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod41_cumulative_workload as m41
import mod41_unit2_components as u2
from per07_first_game_fitted_term import fit

FIT_LAST = 2025
TARGET_SEASON = 2026
COLS = ["logit_s", "s_resid"]
OUTPATH = (
    m41.REPO / "artifacts" / "forward_terms" / "mod41_residual_load" / "frozen_coefficients.json"
)


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def latest_pbp(root):
    return sorted(p for p in Path(root).iterdir() if p.is_dir())[-1]


def season_team_games(pbp_dir, season):
    cols = ["game_id", "season_type", "posteam", "defteam", "play_type", "qb_kneel", "qb_spike"]
    p = pd.read_parquet(Path(pbp_dir) / f"season={season}" / "plays.parquet", columns=cols)
    p = p[(p.season_type == "REG") & p.play_type.isin(["pass", "run"])]
    p = p[(p.qb_kneel != 1) & (p.qb_spike != 1) & p.posteam.notna() & p.defteam.notna()]
    run = p.groupby(["game_id", "posteam"]).size().rename("run").rename_axis(["game_id", "team"])
    dfn = (
        p.groupby(["game_id", "defteam"]).size().rename("defended").rename_axis(["game_id", "team"])
    )
    t = pd.concat([run, dfn], axis=1).fillna(0).reset_index()
    t["total"] = t["run"] + t["defended"]
    t["week"] = t["game_id"].str.split("_").str[1].astype(int)
    return t.sort_values(["team", "week"]).reset_index(drop=True)


def team_state(t):
    g = t.groupby("team")
    n = g.cumcount() + 1
    last = pd.DataFrame({"team": t["team"], "week": t["week"], "n_played": n})
    roll = g["total"].transform(lambda s: s.rolling(m41.WINDOW).mean())
    norm = g["total"].transform(lambda s: s.expanding().mean())
    last["load_total"] = roll - norm
    last["own_td"] = g["run"].transform(lambda s: s.expanding().mean())
    last["opp3"] = g["defended"].transform(lambda s: s.rolling(m41.WINDOW).mean())
    last["ok"] = last["n_played"] >= m41.WINDOW + 1
    return last.sort_values("week").groupby("team").tail(1).set_index("team")


def main():
    d, beta = u2.build()
    po = d.loc[d.season <= FIT_LAST].reset_index(drop=True)
    assert int(po.season.max()) <= FIT_LAST
    mod = fit(po, COLS)
    _, co = u2.loso(po, COLS)
    resid_folds = {s: v[1] for s, v in co.items()}
    pbp = latest_pbp(m41.REPO / "data" / "pbp" / "raw")
    body = {
        "term": "mod41_residual_recent_workload",
        "term_definition": "home minus away of [last-3-game mean of plays run plus plays defended minus the team-season mean of the earlier games] minus the linear projection of that difference on the home-minus-away season mean of plays run and the home-minus-away last-3-game mean of plays defended; zero when either team has fewer than 4 prior games",  # noqa: E501
        "b_logit_per_unit": float(mod.coef_[0][1]),
        "served_logit_coefficient": float(mod.coef_[0][0]),
        "intercept": float(mod.intercept_[0]),
        "decomposition_beta_const_own_opp": beta,
        "loso_resid_coefficient_by_heldout_season": resid_folds,
        "loso_resid_coefficient_min": float(min(resid_folds.values())),
        "loso_resid_coefficient_max": float(max(resid_folds.values())),
        "fit_seasons": [int(po.season.min()), int(po.season.max())],
        "fit_games": len(po),
        "fit_usable_games": int(po["usable"].sum()),
        "fit_base": "served logit of opener-evaluation per_game home cover probability at open; opener grade; unusable early-season rows carry zero",  # noqa: E501
        "workload_window_games": m41.WINDOW,
        "fit_pbp_snapshot": str(pbp.name),
        "fit_pbp_latest_season_used": FIT_LAST,
        "data_cutoff": "2020-2025 regular-season games with a recorded opener",
        "served_per_game_sha256": sha_file(u2.SERVED),
        "fit_script": "scripts/mod41_freeze_residual_load.py",
        "unit2_result": "artifacts/mod41/unit2_components.json",
        "target_season": TARGET_SEASON,
        "no_2026_outcomes": True,
    }
    body["coefficient_hash"] = sha_obj(
        {k: body[k] for k in ("b_logit_per_unit", "intercept", "decomposition_beta_const_own_opp")}
    )
    body["created_at_utc"] = datetime.now(UTC).isoformat()
    if OUTPATH.exists():
        raise SystemExit(f"already frozen: {OUTPATH}")
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPATH.write_text(json.dumps(body, indent=1))
    print(json.dumps({k: v for k, v in body.items() if k != "term_definition"}, indent=1))


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
