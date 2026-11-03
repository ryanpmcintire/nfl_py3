import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from per07_first_game_fitted_term import (  # noqa: E402
    GAME_FEATURES,
    block_bootstrap,
    fit,
    metrics,
    prepare,
    summarize,
)
from playcaller_change_screen import BOOTSTRAP_SEED  # noqa: E402

from nfl_ats.bye_edge_fade_overlay import bye_edge_flag_by_game  # noqa: E402
from nfl_ats.post_bye_new_playcaller_back_overlay import (  # noqa: E402
    NEW_PLAYCALLER_TENURE_YEARS,
    load_coordinator_history,
    oc_tenure_by_team_season,
)

OUT = REPO / "artifacts" / "lead28" / "post_bye_fitted_interaction.json"
import per07_first_game_fitted_term as _per07  # noqa: E402

DRAWS = 4000
_per07.BOOTSTRAP_SAMPLES = DRAWS
PERMUTATIONS = DRAWS


def build() -> pd.DataFrame:
    d = prepare()
    sched = pd.read_parquet(
        GAME_FEATURES,
        columns=["game_id", "season", "game_type", "gameday", "home_team", "away_team"],
    )
    bye = bye_edge_flag_by_game(sched)
    ten = oc_tenure_by_team_season(load_coordinator_history(REPO / "data"))
    ten["new_oc"] = ten["oc_tenure"].isin(NEW_PLAYCALLER_TENURE_YEARS)
    ten["known"] = ten["oc_tenure"].ne("unknown")
    d = d.merge(bye[["game_id", "home_off_bye", "away_off_bye"]], on="game_id", how="left")
    for side in ("home", "away"):
        t = ten.rename(columns={"team": f"{side}_team"})[
            ["season", f"{side}_team", "new_oc", "known"]
        ].rename(columns={"new_oc": f"{side}_new", "known": f"{side}_known"})
        d = (
            d.merge(sched[["game_id", "home_team", "away_team"]], on="game_id", how="left")
            if side == "home"
            else d
        )
        d = d.merge(t, on=["season", f"{side}_team"], how="left")
    for c in ("home_off_bye", "away_off_bye", "home_new", "away_new", "home_known", "away_known"):
        d[c] = d[c].fillna(False).astype(bool)
    d["flag_h"] = (d["home_off_bye"] & d["home_new"]).astype(float)
    d["flag_a"] = (d["away_off_bye"] & d["away_new"]).astype(float)
    d["s_flag"] = d["flag_h"] - d["flag_a"]
    d["s_bye"] = d["home_off_bye"].astype(float) - d["away_off_bye"].astype(float)
    d["s_new"] = d["home_new"].astype(float) - d["away_new"].astype(float)
    d["s_int"] = d["s_flag"]
    return d


def loso(d, cols):
    p = pd.Series(np.nan, index=d.index)
    co = {}
    for season in sorted(d["season"].unique()):
        tr = d.loc[d["season"].ne(season)]
        te = d["season"].eq(season)
        m = fit(tr, cols)
        p[te] = m.predict_proba(d.loc[te, cols])[:, 1]
        co[int(season)] = [float(x) for x in m.coef_[0]]
    return p, co


def analyze(d, cols, name):
    base_p, _ = loso(d, ["logit_p"])
    term_p, co = loso(d, cols)
    sc = d.assign(p_base=base_p, p_term=term_p)

    def gain(f):
        return (
            ((f["p_term"] >= 0.5).astype(int) == f["y"]).mean()
            - ((f["p_base"] >= 0.5).astype(int) == f["y"]).mean()
        ) * 100.0

    def llg(f):
        return metrics(f, "p_base")["log_loss"] - metrics(f, "p_term")["log_loss"]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    full = fit(d, cols)
    ins = d.assign(
        p_base=fit(d, ["logit_p"]).predict_proba(d[["logit_p"]])[:, 1],
        p_term=full.predict_proba(d[cols])[:, 1],
    )
    dec = sc.loc[(sc["p_term"] >= 0.5) != (sc["p_base"] >= 0.5)]
    key = cols[-1]
    coef_obs = full.coef_[0][-1]
    flagged = d[key].ne(0)
    rng2 = np.random.default_rng(BOOTSTRAP_SEED + 1)
    null = np.empty(PERMUTATIONS)
    groups = d.groupby("block").indices
    for i in range(PERMUTATIONS):
        shuffled = d[key].to_numpy().copy()
        for idx in groups.values():
            shuffled[idx] = rng2.permutation(shuffled[idx])
        null[i] = fit(d.assign(**{key: shuffled}), cols).coef_[0][-1]
    return {
        "name": name,
        "cols": cols,
        "games": len(d),
        "flagged_games": int(flagged.sum()),
        "flagged_record_baseline_pick": [
            int(((sc.loc[flagged, "p_base"] >= 0.5).astype(int) == sc.loc[flagged, "y"]).sum()),
            int(flagged.sum()),
        ],
        "decisive_pick_changed_oos": len(dec),
        "decisive_term_correct": int(((dec["p_term"] >= 0.5).astype(int) == dec["y"]).sum()),
        "decisive_base_correct": int(((dec["p_base"] >= 0.5).astype(int) == dec["y"]).sum()),
        "baseline_oos": metrics(sc, "p_base"),
        "term_oos": metrics(sc, "p_term"),
        "oos_accuracy_points_gain": summarize(block_bootstrap(sc, gain, rng), float(gain(sc))),
        "oos_log_loss_gain": summarize(block_bootstrap(sc, llg, rng), float(llg(sc))),
        "in_sample_accuracy_points_gain": float(gain(ins)),
        "in_sample_log_loss_gain": float(llg(ins)),
        "key_coef_full": summarize(
            block_bootstrap(d, lambda f: fit(f, cols).coef_[0][-1], rng), float(coef_obs)
        ),
        "key_coef_permutation_p_two_sided": float((np.abs(null) >= abs(coef_obs)).mean()),
        "coef_per_heldout_season": co,
    }


def main():
    d = build()
    d["s_bye_x_new"] = d["s_flag"]
    out = {
        "family": "lead28_post_bye_x_new_oc_fitted_interaction",
        "grade": "opener",
        "looks": 2,
        "cell_counts": {
            "post_bye_team_games": int(d["home_off_bye"].sum() + d["away_off_bye"].sum()),
            "flagged_team_games": int(d["flag_h"].sum() + d["flag_a"].sum()),
        },
        "flag_term": analyze(d, ["logit_p", "s_flag"], "flag_term"),
        "interaction_with_main_effects": analyze(
            d, ["logit_p", "s_bye", "s_new", "s_flag"], "interaction_with_main_effects"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
