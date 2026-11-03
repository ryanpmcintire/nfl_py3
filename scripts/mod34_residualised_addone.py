# ruff: noqa: E501
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
U8 = Path(__file__).resolve().parent / "mod23_unit8_span.py"
NS = {"__file__": str(U8), "__name__": "mod23_unit8_span_import"}
exec(compile(U8.read_text().rsplit("\nmain()", 1)[0], str(U8), "exec"), NS)

from nfl_ats import clv  # noqa: E402

u3 = NS["u3"]
HARNESS = NS["HARNESS"]
FULL = tuple(NS["FULL"])
BASE_ALPHA = NS["BASE_ALPHA"]
BASE5 = NS["BASE5"]
compare = NS["compare"]
boot_season = NS["boot_season"]
OUT = u3.ROOT / "artifacts" / "mod34_residualised_addone"
U13 = u3.ROOT / "artifacts" / "mod23_unit13_matchup"
U14 = u3.ROOT / "artifacts" / "mod23_unit14_drive"
SEASONS = range(2020, 2026)


def arm_columns(features):
    prods = {
        "A_pass_rush": (
            ("home_off_pass_epa_per_play", "away_def_pass_epa_per_play"),
            ("away_off_pass_epa_per_play", "home_def_pass_epa_per_play"),
            ("home_off_rush_epa_per_play", "away_def_rush_epa_per_play"),
            ("away_off_rush_epa_per_play", "home_def_rush_epa_per_play"),
        ),
        "B_total": (
            ("home_off_epa_per_play", "away_def_epa_per_play"),
            ("away_off_epa_per_play", "home_def_epa_per_play"),
        ),
    }
    f = features.copy()
    arms = {}
    for tag, pairs in prods.items():
        cols = []
        for a, b in pairs:
            name = f"x13_{a}__{b}"
            f[name] = pd.to_numeric(f[a], errors="coerce") * pd.to_numeric(f[b], errors="coerce")
            cols.append(name)
        arms[tag] = (tuple(cols), U13 / f"{tag}_run.parquet")
    for tag, prefix in (("D_drive", "diff_drive_"), ("P_playquality", "diff_pbp_")):
        cols = tuple(
            c
            for c in features.columns
            if c.startswith(prefix) and not c.startswith("diff_pbp_matchup_") and c not in FULL
        )
        for c in cols:
            f[c] = pd.to_numeric(f[c], errors="coerce")
        arms[tag] = (cols, U14 / f"{tag}_run.parquet")
    import mod29_hour_rest as m29

    r = m29.team_rows()
    piv = r.pivot_table(index="game_id", columns="side", values="rest_hours", aggfunc="first")
    f["mod29_hours_diff"] = f["game_id"].map(piv["home"] - piv["away"])
    arms["H_hours"] = (("mod29_hours_diff",), None)
    return f, arms


def design(frame):
    x = frame.loc[:, list(FULL)].apply(pd.to_numeric, errors="coerce")
    return x


def residualise(f, cols, season):
    reg = f.loc[f["game_type"].eq("REG") & f["ats_margin"].notna()]
    train = reg.loc[reg["season"].lt(season)]
    xtr = design(train)
    keep = [c for c in xtr.columns if xtr[c].notna().any()]
    med = xtr[keep].median()
    xtr = xtr[keep].fillna(med)
    mu = xtr.mean()
    sd = xtr.std().replace(0.0, np.nan)
    ok = sd.notna()
    keep = [c for c in keep if ok[c]]
    xtr = ((xtr[keep] - mu[keep]) / sd[keep]).to_numpy()
    xall = ((design(f)[keep].fillna(med[keep]) - mu[keep]) / sd[keep]).to_numpy()
    a_tr = np.column_stack([np.ones(len(xtr)), xtr])
    a_all = np.column_stack([np.ones(len(xall)), xall])
    out = f.copy()
    shares = {}
    for c in cols:
        y = pd.to_numeric(train[c], errors="coerce").to_numpy()
        m = ~np.isnan(y)
        beta = np.linalg.lstsq(a_tr[m], y[m], rcond=None)[0]
        res_tr = y[m] - a_tr[m] @ beta
        shares[c] = float(res_tr.var() / y[m].var()) if y[m].var() > 0 else float("nan")
        full = pd.to_numeric(f[c], errors="coerce").to_numpy() - a_all @ beta
        out[c] = full
    return out, shares


def band_flips(run, base):
    j = run.join(base[["p", "pick_home", "correct"]], rsuffix="_b", how="inner")
    j = j.loc[j["season"].ge(2020)]
    dist = (j["p_b"] - 0.5).abs()
    q = pd.qcut(dist, 3, labels=["coin_flip", "mid", "lopsided"])
    flip = j["pick_home"] != j["pick_home_b"]
    out = {}
    for lab in ("coin_flip", "mid", "lopsided"):
        m = q.eq(lab)
        out[lab] = {
            "n": int(m.sum()),
            "flips": int((flip & m).sum()),
            "flip_W": int((flip & m & j["correct"].eq(1.0)).sum()),
            "flip_L": int((flip & m & j["correct"].eq(0.0)).sum()),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    features = pd.read_parquet(u3.FEATURES)
    base = pd.read_parquet(BASE5)
    base = base.loc[base["season"].ge(2020)]
    if (int(base["correct"].sum()), len(base)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    f, arms = arm_columns(features)
    only = sys.argv[1:]
    results = {}
    for tag, (cols, raw_path) in arms.items():
        if only and tag not in only:
            continue
        if raw_path is not None and raw_path.exists():
            raw = pd.read_parquet(raw_path)
        else:
            raw = HARNESS["true_eval"](f, FULL + cols, BASE_ALPHA)
            raw = raw.loc[raw["season"].ge(2020)]
        parts = []
        shares = {}
        for s in SEASONS:
            fs, sh = residualise(f, cols, s)
            shares[s] = sh
            run = HARNESS["true_eval"](fs, FULL + cols, BASE_ALPHA)
            parts.append(run.loc[run["season"].eq(s)])
            print(tag, s, flush=True)
        res = pd.concat(parts)
        res.to_parquet(OUT / f"{tag}_resid_run.parquet")
        cr = compare(res.loc[res.index.isin(base.index)], base)["tuesday_2020_2025"]
        cw = compare(raw.loc[raw.index.isin(base.index)], base)["tuesday_2020_2025"]
        j = res.join(raw[["correct"]], rsuffix="_raw", how="inner")
        j["d_acc"] = 100.0 * (j["correct"] - j["correct_raw"])
        results[tag] = {
            "n_cols": len(cols),
            "resid_vs_base": cr,
            "raw_vs_base": cw,
            "resid_vs_raw_d_acc": float(j["d_acc"].mean()),
            "resid_vs_raw_boot": boot_season(j, "d_acc"),
            "band_raw": band_flips(raw, base),
            "band_resid": band_flips(res, base),
            "variance_share_outside_span": shares,
        }
        for k in ("raw_vs_base", "resid_vs_base"):
            c = results[tag][k]
            print(tag, k, c["record"], c["record_base"], c["d_acc"], c["d_acc_season_boot"]["pp"], c["flips"], c["flips_W_L"], c["ll_arm"], c["ll_base"], c["brier_arm"], c["brier_base"], flush=True)
        print(tag, "bands", json.dumps(results[tag]["band_raw"]), json.dumps(results[tag]["band_resid"]), flush=True)
        (OUT / f"report_{tag}.json").write_text(json.dumps(results[tag], indent=1, default=str))


main()
