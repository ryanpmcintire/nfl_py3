import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod41_cumulative_workload as m41
import mod41_unit2_components as u2
from per07_first_game_fitted_term import fit

FIT_LAST = 2025
TARGET_SEASON = 2026
COLS = ["logit_s", "s_resid"]
PICK = m41.REPO / "artifacts" / "pick_probability" / "20261103T195411Z" / "per_game.parquet"
SERVED_COLUMN = "out_of_season_home_probability"
OUTPATH = (
    m41.REPO
    / "artifacts"
    / "forward_terms"
    / "mod41_residual_load_served"
    / "frozen_coefficients.json"
)


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    d, beta = u2.build()
    pk = pd.read_parquet(PICK, columns=["game_id", SERVED_COLUMN])
    d = d.merge(pk, on="game_id", how="left")
    assert int(d[SERVED_COLUMN].isna().sum()) == 0
    ps = d[SERVED_COLUMN].clip(1e-6, 1 - 1e-6)
    d["logit_s"] = np.log(ps / (1 - ps))
    po = d.loc[d.season <= FIT_LAST].reset_index(drop=True)
    assert int(po.season.max()) <= FIT_LAST
    mod = fit(po, COLS)
    _, co = u2.loso(po, COLS)
    resid_folds = {s: v[1] for s, v in co.items()}
    served_folds = {s: v[0] for s, v in co.items()}
    body = {
        "term": "mod41_residual_recent_workload_served",
        "term_definition": "home minus away of [last-3-game mean of plays run plus plays defended minus the team-season mean of the earlier games] minus the linear projection of that difference on the home-minus-away season mean of plays run and the home-minus-away last-3-game mean of plays defended; zero when either team has fewer than 4 prior games; added to the logit of the served home probability",  # noqa: E501
        "b_logit_per_unit": float(mod.coef_[0][1]),
        "served_logit_coefficient": float(mod.coef_[0][0]),
        "intercept": float(mod.intercept_[0]),
        "decomposition_beta_const_own_opp": beta,
        "loso_resid_coefficient_by_heldout_season": resid_folds,
        "loso_resid_coefficient_min": float(min(resid_folds.values())),
        "loso_resid_coefficient_max": float(max(resid_folds.values())),
        "loso_served_logit_coefficient_by_heldout_season": served_folds,
        "loso_served_logit_coefficient_min": float(min(served_folds.values())),
        "loso_served_logit_coefficient_max": float(max(served_folds.values())),
        "fit_seasons": [int(po.season.min()), int(po.season.max())],
        "fit_games": len(po),
        "fit_usable_games": int(po["usable"].sum()),
        "fit_base": f"logit of {SERVED_COLUMN} from the pick_probability artifact; opener grade; unusable early-season rows carry zero residual",  # noqa: E501
        "base_artifact": str(PICK.relative_to(m41.REPO)),
        "base_artifact_sha256": sha_file(PICK),
        "workload_window_games": m41.WINDOW,
        "fit_pbp_latest_season_used": FIT_LAST,
        "data_cutoff": "2020-2025 regular-season games with a recorded opener",
        "fit_script": "scripts/mod41_freeze_residual_load_served.py",
        "unit6_result": "artifacts/mod41/unit6_served_base.json",
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
