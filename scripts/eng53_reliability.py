import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ART / "eng53"
SEED = 53
N_BOOT = 4000
N_PERM = 5000
LEVEL = 0.95
TOLERANCE = 0.02
MIN_SEASON_GAMES = 6
UNITS = OUT / "units"

COLS = [
    "trait",
    "key",
    "league",
    "era",
    "metric_kind",
    "n_groups",
    "n_games",
    "oe_r",
    "oe_lo",
    "oe_hi",
    "sb",
    "oe_perm_p",
    "tt_r",
    "tt_lo",
    "tt_hi",
    "tt_pairs",
    "tt_perm_p",
    "shrink_used",
    "pick_term",
    "decisive",
    "ll_gain",
    "ll_p_pos",
    "shared_r",
    "shared_lo",
    "shared_hi",
    "shared_p",
    "shared_n",
    "unit_ref",
    "unit_groups",
    "u_oe_r",
    "u_oe_lo",
    "u_oe_hi",
    "u_sb",
    "u_oe_p",
    "u_tt_r",
    "u_tt_lo",
    "u_tt_hi",
    "u_tt_pairs",
    "flag",
    "status",
    "source",
]


def jload(p):
    return json.loads((ART / p).read_text())


def blank():
    return dict.fromkeys(COLS, "")


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pair_estimate(frame, xcol, ycol, rng):
    d = frame[[xcol, ycol, "person"]].dropna()
    x = d[xcol].to_numpy()
    y = d[ycol].to_numpy()
    r = corr(x, y)
    groups = {k: g[[xcol, ycol]].to_numpy() for k, g in d.groupby("person")}
    keys = list(groups)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        m = np.vstack([groups[keys[i]] for i in pick])
        boots.append(corr(m[:, 0], m[:, 1]))
    lo, hi = np.nanquantile(boots, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    null = np.array([corr(x, rng.permutation(y)) for _ in range(N_PERM)])
    p = float((np.sum(np.abs(null) >= abs(r)) + 1) / (N_PERM + 1))
    return r, float(lo), float(hi), p, len(d)


def rel_row(trait, key, league, era, rec, source, n_games, shrink):
    r = blank()
    r.update(
        trait=trait,
        key=key,
        league=league,
        era=era,
        metric_kind="split-half and season t to t+1",
        source=source,
        status="reported only",
        shrink_used=shrink,
        pick_term="no",
    )
    r["n_groups"] = rec.get("n_spells", rec.get("n_units", ""))
    r["n_games"] = n_games if n_games is not None else rec.get("n_games", "")
    r["oe_r"] = rec.get("odd_even_r", rec.get("r_half", ""))
    ci = rec.get("odd_even_ci95")
    if ci is None and "sb_ci" in rec:
        ci = [x / (2 - x) for x in rec["sb_ci"]]
    if ci:
        r["oe_lo"], r["oe_hi"] = ci
    r["sb"] = rec.get("spearman_brown", rec.get("split_half_sb", rec.get("sb", "")))
    r["oe_perm_p"] = rec.get(
        "odd_even_perm_p",
        rec.get("r_perm_p", rec.get("split_half_p", rec.get("perm_p", ""))),
    )
    r["tt_r"] = rec.get("season_t_t1_r", rec.get("t_t1_r", ""))
    ci = rec.get("season_t_t1_ci95", rec.get("t_t1_ci95"))
    if ci:
        r["tt_lo"], r["tt_hi"] = ci
    r["tt_pairs"] = rec.get("season_pairs", rec.get("t_t1_pairs", rec.get("n_season_pairs", "")))
    r["tt_perm_p"] = rec.get("season_p_signflip", rec.get("season_p", ""))
    return r


def set_pick(row, won, lost, d, pp):
    row.update(pick_term="yes", decisive=f"{won}-{lost}", ll_gain=d, ll_p_pos=pp)


def mod40(trait, era, n, r, sb, p, dec, dll, pp):
    row = blank()
    row.update(
        trait=trait,
        key="team-season, odd/even REG weeks",
        league="nfl",
        era=era,
        metric_kind="split-half only",
        n_groups=n,
        oe_r=r,
        sb=sb,
        oe_perm_p=p,
        status="reported only",
        source="MOD-40 lane",
        shrink_used="pre-game same-season shrunk; factor not in lane",
        pick_term="yes",
        decisive=dec,
        ll_gain=dll,
        ll_p_pos=pp,
    )
    return row


def tag(row, ref):
    row["unit_ref"] = ref
    return row


def build_reported():
    rows = []
    h = jload("sim17/halftime_adjustment.json")["metrics"]
    u3 = jload("sim17/unit3_heldout.json")["part2"]["arms"]
    for side, nm, arm, sh in (
        ("off_res", "halftime adjustment, offense", "served+f_off", "0.14"),
        ("def_res", "halftime adjustment, defense", "served+f_def", "0.33"),
    ):
        row = rel_row(
            nm,
            "hc spell (>=16 games)",
            "nfl",
            "2009-2025",
            h[side],
            "SIM-17 unit1 halftime_adjustment.json",
            7638,
            f"split-half Spearman-Brown {sh}",
        )
        a = u3[arm]
        set_pick(
            row, a["W"], a["L"], a["dLL_vs_served"]["mean"], a["dLL_vs_served"]["prob_positive"]
        )
        rows.append(tag(row, f"sim17_halftime|{side}"))
    c = jload("sim17/unit2_carriers.json")
    for side, grp in (("off_res", ("hc", "oc", "qb")), ("def_res", ("hc", "dc"))):
        who = "offense" if side == "off_res" else "defense"
        for g in grp:
            rows.append(
                tag(
                    rel_row(
                        f"halftime adjustment, {who}",
                        f"{g} spell (>=16 games), own sample",
                        "nfl",
                        "2009-2025",
                        c["reliability_own"][side][g],
                        "SIM-17 unit2 unit2_carriers.json",
                        None,
                        "none stated",
                    ),
                    f"sim17_unit2|{side}|{g}",
                )
            )
    u2 = jload("sim18/unit2.json")
    mh = u2["multiplier"]["all|loso_other_seasons"]
    half = mh["mean_half_reliability"]
    oc_shrink = (
        f"SIM-18 unit2 mean half reliability {half:.3f} at K {mh['median_K_drives']:.0f} drives"
    )
    for k, rec in jload("sim18/unit1.json")["reliability"].items():
        grp, met = k.split("|")
        is_sec = met == "res_sec"
        row = rel_row(
            "lead tempo (sec/play)" if is_sec else "lead run share",
            f"{grp} spell (>=16 games), leader plays",
            "nfl",
            "2009-2025 2nd half",
            rec,
            "SIM-18 unit1 unit1.json",
            None,
            oc_shrink if grp == "oc" else "none stated",
        )
        if grp == "oc":
            arm = "served+sec_adj" if is_sec else "served+run_adj"
            a = u2["pick"]["arms"][arm]
            v = u2["pick"]["vs_served"][arm]
            set_pick(row, a["W"], a["L"], v["dLL"], v["p_positive"])
        rows.append(tag(row, f"sim18_unit1|{k}"))
    for k, rec in jload("sim19/trailing_risk.json")["reliability"].items():
        met, grp, kind = k.split("|")
        label = "style-adjusted shift" if kind == "shift" else "incl. team style"
        rows.append(
            tag(
                rel_row(
                    f"trailing {met}",
                    f"{grp} spell (>=16 games), {label}",
                    "nfl",
                    "2009-2025",
                    rec,
                    "SIM-19 unit1 trailing_risk.json",
                    None,
                    "none stated",
                ),
                f"sim19|{k}",
            )
        )
    x = jload("xlg15/report.json")
    for tr, eras in x["reliability"].items():
        for era, rec in eras.items():
            pr = x["pricing"][tr][era]
            rr = rel_row(
                tr,
                "team-season, alternating games",
                "cfb",
                era,
                rec,
                "XLG-15 report.json",
                pr["n_games"],
                "none stated",
            )
            rr["metric_kind"] = "split-half only"
            rr["pick_term"] = "priced vs opener ATS, not a pick term"
            rr["decisive"] = (
                f"slope {pr['slope_ats_points_per_sd']:.2f} ats pts/sd "
                f"[{pr['ci'][0]:.2f}, {pr['ci'][1]:.2f}]"
            )
            rr["ll_p_pos"] = pr["p_pos"]
            rows.append(tag(rr, f"xlg15|{tr}|{era}"))
    rows.append(
        tag(
            mod40("CPOE gap", "NGS seasons, 10", 320, 0.418, 0.59, 0.0002, "802-701", 4e-05, 0.499),
            "mod40|cpoe",
        )
    )
    rows.append(
        tag(
            mod40(
                "RYOE per attempt gap",
                "2018-2025",
                256,
                0.175,
                0.30,
                0.003,
                "796-707",
                -2e-05,
                0.496,
            ),
            "mod40|ryoe",
        )
    )
    for nm, src in (
        ("offensive line continuity (MOD-42)", "MOD-42 lane: unmeasured"),
        ("repeat-read ensemble spread (LEAD-100)", "LEAD-100 lane: unmeasured"),
    ):
        r = blank()
        r.update(
            trait=nm, key="n/a", league="nfl", status="not measured", source=src, pick_term="no"
        )
        rows.append(r)
    return rows


PER14 = {
    "halftime_off": ("halftime adjustment, offense", "SIM-17"),
    "lead_sec": ("lead tempo (sec/play)", "SIM-18"),
    "lead_run": ("lead run share", "SIM-18"),
    "trail_pass_incl": ("trailing pass rate, incl. team style", "SIM-19"),
    "trail_pass_shift": ("trailing pass rate, style-adjusted", "SIM-19"),
}


def build_per14():
    rep = jload("per14/mover_portability.json")["traits"]
    rng = np.random.default_rng(SEED)
    rows = []
    for name, (nm, origin) in PER14.items():
        ev = pd.read_csv(ART / "per14" / f"events_{name}.csv")
        t = rep[name]
        for which, ycol, rrec in (
            ("person (stint1 vs stint2)", "x2", t["person"]),
            ("team (stint1 vs successor)", "succ", t["team"]),
        ):
            r, lo, hi, p, n = pair_estimate(ev, "x1", ycol, rng)
            clo, chi = rrec["ci95"]
            row = blank()
            row.update(
                trait=nm,
                key=f"oc mover {which}",
                league="nfl",
                era="2009-2025",
                metric_kind="mover portability r",
                n_groups=rrec.get("n", t["n_movers"]),
                tt_r=rrec["r"],
                tt_lo=clo,
                tt_hi=chi,
                tt_pairs=rrec.get("n", t["n_movers"]),
                tt_perm_p=rrec["perm_p"],
                shrink_used="method-of-moments, median long-spell reliability "
                f"{t['reliability_median_long_spell']:.3f}",
                pick_term="no",
                shared_r=r,
                shared_lo=lo,
                shared_hi=hi,
                shared_p=p,
                shared_n=n,
                status="recomputed",
                source=f"PER-14 events_{name}.csv ({origin} definition)",
            )
            if not (clo <= r <= chi):
                row["flag"] = "shared r outside reported interval"
            elif abs(r - rrec["r"]) > TOLERANCE:
                row["flag"] = "inside interval, differs by more than tolerance"
            rows.append(row)
    return rows


def unit_means(df):
    code = pd.factorize(df["unit"])[0]
    cell = code * 2 + df["parity"].to_numpy()
    vw = (df["resid"] * df["weight"]).to_numpy()
    w = df["weight"].to_numpy()
    return code, cell, vw, w


def half_values(cell, vw, w, n_units):
    sv = np.bincount(cell, vw, minlength=2 * n_units).reshape(-1, 2)
    sw = np.bincount(cell, w, minlength=2 * n_units).reshape(-1, 2)
    with np.errstate(all="ignore"):
        return sv / sw, (sw > 0).all(axis=1)


def odd_even(df, rng):
    code, cell, vw, w = unit_means(df)
    n_units = code.max() + 1
    m, ok = half_values(cell, vw, w, n_units)
    a, b = m[ok, 0], m[ok, 1]
    r = corr(a, b)
    n = len(a)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        boots.append(corr(a[i], b[i]))
    lo, hi = np.nanquantile(boots, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    null = []
    for _ in range(N_PERM):
        s = rng.choice([-1.0, 1.0], len(vw))
        mm, _ = half_values(cell, vw * s, w, n_units)
        null.append(corr(mm[ok, 0], mm[ok, 1]))
    p = float((np.sum(np.array(null) >= r) + 1) / (N_PERM + 1))
    return r, float(lo), float(hi), p, n


def season_pairs(df, rng):
    g = (
        df.assign(vw=df["resid"] * df["weight"])
        .groupby(["entity", "season"])
        .agg(vw=("vw", "sum"), w=("weight", "sum"), ng=("game_id", "size"))
        .reset_index()
    )
    g = g[(g.ng >= MIN_SEASON_GAMES) & (g.w > 0)]
    g["mu"] = g.vw / g.w
    nxt = g[["entity", "season", "mu"]].copy()
    nxt["season"] = nxt["season"] - 1
    pr = g.merge(nxt, on=["entity", "season"], suffixes=("_t", "_t1"))
    r = corr(pr.mu_t.to_numpy(), pr.mu_t1.to_numpy())
    groups = [x[["mu_t", "mu_t1"]].to_numpy() for _, x in pr.groupby("entity")]
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        m = np.vstack([groups[i] for i in pick])
        boots.append(corr(m[:, 0], m[:, 1]))
    lo, hi = np.nanquantile(boots, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    return r, float(lo), float(hi), len(pr)


def disagreement(label, value, lo, hi, reported):
    if lo != "" and hi != "" and not (lo <= value <= hi):
        return f"{label} outside reported interval"
    if abs(value - reported) > TOLERANCE:
        return f"{label} inside interval, differs by more than tolerance"
    return ""


def attach_units(rows):
    rng = np.random.default_rng(SEED)
    cache = {}
    for row in rows:
        ref = row.get("unit_ref", "")
        if not ref:
            continue
        study, trait = ref.split("|", 1)
        if study not in cache:
            path = UNITS / f"{study}.parquet"
            cache[study] = pd.read_parquet(path) if path.exists() else None
        data = cache[study]
        if data is None:
            continue
        df = data[data["trait"] == trait]
        if df.empty:
            continue
        r, lo, hi, p, n = odd_even(df, rng)
        both = data[data["trait"] == f"{trait}|season_all"]
        tr, tlo, thi, tn = season_pairs(both if len(both) else df, rng)
        row.update(
            unit_groups=n,
            u_oe_r=r,
            u_oe_lo=lo,
            u_oe_hi=hi,
            u_sb=2 * r / (1 + r),
            u_oe_p=p,
            u_tt_r=tr,
            u_tt_lo=tlo,
            u_tt_hi=thi,
            u_tt_pairs=tn,
            status="recomputed from unit data",
        )
        notes = []
        if row["oe_r"] != "":
            notes.append(disagreement("odd/even r", r, row["oe_lo"], row["oe_hi"], row["oe_r"]))
        if row["tt_r"] != "":
            notes.append(
                disagreement("season t to t+1 r", tr, row["tt_lo"], row["tt_hi"], row["tt_r"])
            )
        row["flag"] = "; ".join(n for n in notes if n)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(attach_units(build_reported()) + build_per14(), columns=COLS)
    df.to_csv(OUT / "trait_reliability.csv", index=False)
    print(len(df), df["status"].value_counts().to_dict())
    pd.set_option("display.width", 250)
    cols = [
        "trait",
        "key",
        "tt_r",
        "tt_lo",
        "tt_hi",
        "shared_r",
        "shared_lo",
        "shared_hi",
        "shared_p",
        "shared_n",
        "tt_pairs",
        "flag",
    ]
    print(df[df["status"] == "recomputed"][cols].round(3).to_string())


if __name__ == "__main__":
    main()
