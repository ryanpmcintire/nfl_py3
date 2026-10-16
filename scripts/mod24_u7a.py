import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_ats.constants import DEFAULT_OFFSEASON_RETENTION, STATE_METRICS, TEAM_ABBREVIATION_ALIASES
from nfl_ats.features import build_team_game_metrics

OUT = ROOT / "artifacts" / "mod24_u7a"
FEATURES = ROOT / "data" / "processed" / "game_features_weak_stack.parquet"
SNAPSHOT = ROOT / "data" / "raw" / "20260929T191306Z"
PLAYERS = ROOT / "data" / "players" / "raw" / "20260910T205112Z"
SPAN = 8
MIN_PERIODS = 3
ALPHA = 2.0 / (SPAN + 1.0)
WINDOW = 48
OL_POS = {"T", "G", "C", "OL", "OT", "OG", "C/G", "G/T", "G/C", "T/G", "G/OT"}
OL_MISSING_PCT = 0.25
OL_NEED = 2
OUT_STATUS = {"Out", "Doubtful"}
NM = len(STATE_METRICS)
OWN_IDX = np.array([i for i, m in enumerate(STATE_METRICS) if m.startswith("off_") or m in ("point_diff", "ats_residual")])
DEF_IDX = np.array([i for i, m in enumerate(STATE_METRICS) if m.startswith("def_")])
ARMS = {
    "a1_exclude": dict(w=0.0, ol=False),
    "a2_w025": dict(w=0.25, ol=False),
    "a2_w050": dict(w=0.5, ol=False),
    "a2_w075": dict(w=0.75, ol=False),
    "a3_exclude_ol": dict(w=0.0, ol=True),
}


def load_qb():
    best = {}
    for p in sorted(glob.glob(str(ROOT / "data/pbp/raw/*/season=*/plays.parquet"))):
        season = int(p.replace("\\", "/").split("season=")[1].split("/")[0])
        best[season] = p
    parts = []
    for season, p in sorted(best.items()):
        d = pd.read_parquet(p, columns=["game_id", "season_type", "posteam", "qb_dropback", "passer_player_id"])
        d = d.loc[d["season_type"].eq("REG") & d["passer_player_id"].notna() & d["posteam"].notna() & d["qb_dropback"].eq(1)]
        parts.append(d.groupby(["game_id", "posteam", "passer_player_id"]).size().rename("n").reset_index())
    q = pd.concat(parts, ignore_index=True)
    q["posteam"] = q["posteam"].replace(TEAM_ABBREVIATION_ALIASES)
    q = q.sort_values("n", ascending=False).drop_duplicates(["game_id", "posteam"])
    return q.rename(columns={"posteam": "team", "passer_player_id": "qb"})[["game_id", "team", "qb"]]


def load_snaps():
    s = pd.read_parquet(PLAYERS / "snap_counts.parquet", columns=["game_id", "season", "team", "position", "pfr_player_id", "offense_snaps", "offense_pct", "game_type"])
    s = s.loc[s["game_type"].eq("REG") & s["position"].isin(OL_POS)].copy()
    s["team"] = s["team"].replace(TEAM_ABBREVIATION_ALIASES)
    return s


def league_means(tg):
    return {m: tg.groupby("season")[m].mean().to_dict() for m in STATE_METRICS}


def team_tables(tg, qb, snaps, games):
    tg = tg.merge(qb, on=["game_id", "team"], how="left")
    gm = games.set_index("game_id")
    home = tg["game_id"].map(gm["home_team"])
    away = tg["game_id"].map(gm["away_team"])
    tg["opp"] = np.where(tg["team"].eq(home), away, home)
    opp = tg[["game_id", "team", "qb"]].rename(columns={"team": "opp", "qb": "opp_qb"})
    tg = tg.merge(opp, on=["game_id", "opp"], how="left")
    tg["gameday"] = pd.to_datetime(tg["gameday"])
    tg = tg.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    snapmap = {}
    top5 = {}
    for (gid, team), g in snaps.groupby(["game_id", "team"]):
        snapmap[(gid, team)] = dict(zip(g["pfr_player_id"], g["offense_pct"].fillna(0.0), strict=True))
        gg = g.dropna(subset=["pfr_player_id"]).sort_values("offense_snaps", ascending=False).head(5)
        top5[(gid, team)] = tuple(gg["pfr_player_id"])
    return tg, snapmap, top5


class Team:
    def __init__(self, g, snapmap, top5, lm):
        self.g = g.reset_index(drop=True)
        self.n = len(g)
        self.V = self.g[list(STATE_METRICS)].to_numpy(dtype=float)
        self.season = self.g["season"].to_numpy(dtype=int)
        self.date = self.g["gameday"].to_numpy(dtype="datetime64[ns]")
        self.qb = self.g["qb"].to_numpy(dtype=object)
        self.oqb = self.g["opp_qb"].to_numpy(dtype=object)
        self.gid = self.g["game_id"].to_numpy(dtype=object)
        self.opp = self.g["opp"].to_numpy(dtype=object)
        self.gidx = {x: i for i, x in enumerate(self.gid)}
        self.snapmap = snapmap
        self.team = self.g["team"].iloc[0]
        self.lm = lm
        self.top5 = top5
        self.prev_top = [None] + [top5.get((self.gid[i - 1], self.team)) for i in range(1, self.n)]
        self.base_cur = np.full((self.n, NM), np.nan)
        self.base_obs = np.zeros((self.n, NM), dtype=int)
        self.run_base()

    def lmv(self, season):
        return np.array([self.lm[m].get(season, 0.0) for m in STATE_METRICS])

    def step(self, cur, obs, i, prev_season, w):
        s = self.season[i]
        if prev_season is not None and s != prev_season:
            gap = max(1, s - prev_season)
            lmv = self.lmv(prev_season)
            cur = np.where(np.isfinite(cur), lmv + DEFAULT_OFFSEASON_RETENTION**gap * (cur - lmv), cur)
        v = self.V[i]
        ok = np.isfinite(v) & (w > 0)
        a = ALPHA * w
        cur = np.where(ok, np.where(np.isfinite(cur), a * v + (1.0 - a) * cur, v), cur)
        return cur, obs + ok.astype(int)

    def run_base(self):
        cur = np.full(NM, np.nan)
        obs = np.zeros(NM, dtype=int)
        ones = np.ones(NM)
        prev = None
        for i in range(self.n):
            cur, obs = self.step(cur, obs, i, prev, ones)
            prev = self.season[i]
            self.base_cur[i] = cur
            self.base_obs[i] = obs

    def finish(self, cur, obs, p, S):
        out = np.where(obs >= MIN_PERIODS, cur, np.nan)
        gap = S - self.season[p - 1]
        if gap > 0:
            lmv = self.lmv(self.season[p - 1])
            out = np.where(np.isfinite(out), lmv + DEFAULT_OFFSEASON_RETENTION ** max(1, gap) * (out - lmv), out)
        return out

    def base_state(self, p, S):
        return self.finish(self.base_cur[p - 1].copy(), self.base_obs[p - 1].copy(), p, S)

    def run(self, p, S, w_own, w_def, lo):
        wm = np.ones((p - lo, NM))
        wm[:, OWN_IDX] = w_own[:, None]
        wm[:, DEF_IDX] = w_def[:, None]
        changed = np.where((wm != 1.0).any(axis=1))[0]
        if len(changed) == 0:
            return self.base_state(p, S), False
        f = lo + int(changed[0])
        if f == 0:
            cur = np.full(NM, np.nan)
            obs = np.zeros(NM, dtype=int)
            prev = None
        else:
            cur = self.base_cur[f - 1].copy()
            obs = self.base_obs[f - 1].copy()
            prev = self.season[f - 1]
        for i in range(f, p):
            cur, obs = self.step(cur, obs, i, prev, wm[i - lo])
            prev = self.season[i]
        res = self.finish(cur, obs, p, S)
        base = self.base_state(p, S)
        return np.where(np.isnan(res) & np.isfinite(base), base, res), True


def is_id(x):
    return isinstance(x, str)


def expected_qb(T, p, inj, kick, season, week):
    e = T.qb[p - 1]
    if not is_id(e):
        return None, False
    t = inj.get((season, week, T.team, e))
    if t is not None and t <= kick - pd.Timedelta(hours=24):
        for j in range(p - 2, max(-1, p - 40), -1):
            if is_id(T.qb[j]) and T.qb[j] != e:
                return T.qb[j], True
    return e, False


def target_weights(T, teams, p, D, E, w, use_ol):
    lo = max(0, p - WINDOW)
    idx = range(lo, p)
    w_own = np.ones(p - lo)
    w_def = np.ones(p - lo)
    if E is not None:
        w_own = np.array([w if is_id(T.qb[i]) and T.qb[i] != E else 1.0 for i in idx])
    for k, i in enumerate(idx):
        o = T.oqb[i]
        orow = teams.get(T.opp[i])
        if not is_id(o) or orow is None:
            continue
        j = orow.gidx.get(T.gid[i])
        if j is None or j == 0:
            continue
        pq = orow.qb[j - 1]
        if not is_id(pq) or o == pq:
            continue
        if j + 1 < orow.n and orow.date[j + 1] < D and is_id(orow.qb[j + 1]) and orow.qb[j + 1] == o:
            continue
        w_def[k] = w
    if use_ol:
        e5 = T.top5.get((T.gid[p - 1], T.team), ())
        if len(e5) == 5:
            for k, i in enumerate(idx):
                sm = T.snapmap.get((T.gid[i], T.team))
                if sm is not None and sum(sm.get(x, 0.0) < OL_MISSING_PCT for x in e5) >= OL_NEED:
                    w_own[k] = w
        for k, i in enumerate(idx):
            orow = teams.get(T.opp[i])
            if orow is None:
                continue
            j = orow.gidx.get(T.gid[i])
            e5o = orow.prev_top[j] if j else None
            sm = orow.snapmap.get((T.gid[i], orow.team))
            if e5o is None or len(e5o) < 5 or sm is None:
                continue
            if sum(sm.get(x, 0.0) < OL_MISSING_PCT for x in e5o) >= OL_NEED:
                w_def[k] = w
    return lo, w_own, w_def


def build(cfg, teams, targets, inj):
    w = cfg["w"]
    stats = dict(targets=0, perturbed=0, override=0)
    cols = {"home": [], "away": []}
    flag8 = {"home": [], "away": []}
    for r in targets.itertuples():
        D = np.datetime64(pd.Timestamp(r.gameday), "ns")
        for side in ("home", "away"):
            T = teams.get(getattr(r, f"{side}_team"))
            vals = np.full(NM, np.nan)
            f8 = False
            if T is not None:
                p = int(np.searchsorted(T.date, D, side="left"))
                if p > 0:
                    E, ov = expected_qb(T, p, inj, r.kickoff, r.season, r.week)
                    stats["override"] += int(ov)
                    lo, w_own, w_def = target_weights(T, teams, p, D, E, w, cfg["ol"])
                    f8 = bool((w_own[-8:] != 1.0).any() or (w_def[-8:] != 1.0).any())
                    vals, ch = T.run(p, int(r.season), w_own, w_def, lo)
                    stats["perturbed"] += int(ch)
                stats["targets"] += 1
            cols[side].append(vals)
            flag8[side].append(f8)
    out = pd.DataFrame({"game_id": targets["game_id"].to_numpy()})
    for side in ("home", "away"):
        arr = np.vstack(cols[side])
        for k, m in enumerate(STATE_METRICS):
            out[f"{side}_{m}"] = arr[:, k]
    for m in STATE_METRICS:
        out[f"diff_{m}"] = out[f"home_{m}"] - out[f"away_{m}"]
    out["flag8"] = np.array(flag8["home"]) | np.array(flag8["away"])
    return out, stats


def prep():
    features = pd.read_parquet(FEATURES)
    schedules = pd.read_parquet(SNAPSHOT / "schedules.parquet")
    team_stats = pd.read_parquet(SNAPSHOT / "team_stats.parquet")
    tg = build_team_game_metrics(schedules, team_stats)
    games = features[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")
    qb = load_qb()
    snaps = load_snaps()
    tg, snapmap, top5 = team_tables(tg, qb, snaps, games)
    lm = league_means(tg)
    teams = {t: Team(g, snapmap, top5, lm) for t, g in tg.groupby("team")}
    inj = pd.read_parquet(PLAYERS / "injuries.parquet", columns=["season", "week", "team", "gsis_id", "report_status", "effective_observed_at"])
    inj = inj.loc[inj["report_status"].isin(OUT_STATUS)].copy()
    inj["effective_observed_at"] = pd.to_datetime(inj["effective_observed_at"], utc=True, errors="coerce")
    inj["team"] = inj["team"].replace(TEAM_ABBREVIATION_ALIASES)
    inj = inj.dropna(subset=["effective_observed_at"])
    g = inj.groupby(["season", "week", "team", "gsis_id"])["effective_observed_at"].min()
    injd = {(int(a), int(b), c, d): t for (a, b, c, d), t in g.items()}
    reg = features.loc[features["game_type"].eq("REG")].copy()
    reg["kickoff"] = pd.to_datetime(reg["kickoff"], utc=True)
    reg["gameday"] = pd.to_datetime(reg["gameday"])
    reg["season"] = reg["season"].astype(int)
    reg["week"] = reg["week"].astype(int)
    return features, reg, teams, injd, tg


def stage_build():
    OUT.mkdir(parents=True, exist_ok=True)
    features, reg, teams, injd, tg = prep()
    have = int(sum(sum(is_id(x) for x in T.qb) for T in teams.values()))
    print("team-games with starter qb", have, "of", len(tg), flush=True)
    cols = [f"{s}_{m}" for s in ("home", "away", "diff") for m in STATE_METRICS]
    sample = reg.loc[reg["season"].ge(2015)].sample(120, random_state=1)
    chk, _ = build({"w": 1.0, "ol": False}, teams, sample, injd)
    base = features.set_index("game_id").loc[sample["game_id"], cols]
    c = chk.set_index("game_id")[cols]
    print("reproduction max abs diff", float(np.nanmax((c - base).abs().to_numpy())), "nan mismatch", int((c.isna() != base.isna()).to_numpy().sum()), flush=True)
    summary = {}
    for name, cfg in ARMS.items():
        out, stats = build(cfg, teams, reg, injd)
        out.to_parquet(OUT / f"features_{name}.parquet")
        b = features.set_index("game_id").loc[out["game_id"], cols].reset_index(drop=True)
        sd = b.std().replace(0, np.nan)
        z = ((out[cols].reset_index(drop=True) - b).abs() / sd).fillna(0).to_numpy()
        material = (z > 0.1).any(axis=1)
        for lo_, hi_ in ((2011, 2019), (2020, 2025)):
            m_ = reg["season"].between(lo_, hi_).to_numpy()
            stats[f"share_recent8_flag_{lo_}_{hi_}"] = float(out["flag8"].to_numpy()[m_].mean())
            stats[f"share_material_{lo_}_{hi_}"] = float(material[m_].mean())
        summary[name] = stats
        print(name, stats, flush=True)
        (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2))


def harness():
    u1 = ROOT / "scripts" / "mod24_u1.py"
    src = u1.read_text().rsplit(chr(10) + "main()", 1)[0]
    ns = {"__file__": str(u1), "__name__": "u1_harness"}
    exec(compile(src, str(u1), "exec"), ns)
    return ns


W_GRID = (0.25, 0.5, 0.75)
W_NAME = {0.25: "a2_w025", 0.5: "a2_w050", 0.75: "a2_w075"}


def arm_features(features, name):
    arm = pd.read_parquet(OUT / f"features_{name}.parquet").drop(columns=["flag8"])
    cols = [c for c in arm.columns if c != "game_id"]
    feat = features.copy()
    idx = feat.reset_index().set_index("game_id")["index"]
    pos = idx.loc[arm["game_id"]].to_numpy()
    for c in cols:
        feat.iloc[pos, feat.columns.get_loc(c)] = arm[c].to_numpy()
    return feat


def stage_select(ns, features):
    FULL = ns["FULL"]
    sel = {}
    for s in ns["PROXY_SEASONS"] + ns["TRUE_SEASONS"]:
        inner = list(range(max(ns["FIRST_INNER"], s - ns["INNER_SEASONS"]), s))
        scores = {}
        for w in W_GRID:
            f = ns["regular_season_rows"](arm_features(features, W_NAME[w]))
            f = f.loc[f["ats_margin"].notna() & f["result"].notna()]
            scores[w] = ns["inner_mse"](f, FULL, ns["BASE_ALPHA"], inner)
        sel[str(s)] = {"w": min(W_GRID, key=lambda w: (scores[w], abs(w - 0.5))), "inner_mse": {str(k): v for k, v in scores.items()}}
        print(s, sel[str(s)], flush=True)
        (OUT / "selection_w.json").write_text(json.dumps(sel, indent=2))


def true_scored(ns, feat):
    cfg = {
        "feature_profile": ns["PROFILE"],
        "regressor": "ridge",
        "target": "market_residual",
        "probability_method": ns["PROB_METHOD"],
        "calibration_method": "none",
        "ridge_alpha": ns["BASE_ALPHA"],
        "mod24_variant": json.dumps(["u7a", id(feat)]),
    }
    with ns["patched"](ns["FULL"]):
        scored = ns["clv"].opener_pick_evaluation(ns["u3"].MARKET_ROOT, feat, active_model_config=cfg, min_train_games=ns["DEFAULT_MIN_TRAIN_GAMES"])
    return scored


def run_variant(ns, features, pop, name):
    feat = arm_features(features, name)
    spec = lambda s: (ns["FULL"], ns["BASE_ALPHA"])
    proxy = ns["proxy_eval"](feat, pop, spec)
    scored = true_scored(ns, feat)
    scored.to_parquet(OUT / f"{name}_true_scored.parquet")
    sc = scored.loc[scored["correct_at_open_probability_rule"].notna()].copy()
    sc["p"] = sc["home_cover_probability_at_open"]
    true = ns["finish"](sc[["game_id", "season", "week", "p", "margin_vs_open"]], sc["p"])
    return proxy, true, scored


def stage_run(ns, features, which):
    pop = ns["proxy_population"](features)
    for name in which:
        proxy, true, scored = run_variant(ns, features, pop, name)
        proxy.to_parquet(OUT / f"{name}_proxy.parquet")
        true.to_parquet(OUT / f"{name}_true.parquet")
        print(name, "proxy", int(proxy["correct"].sum()), len(proxy), "true", int(true["correct"].sum()), len(true), flush=True)


def temp_loso(df):
    from scipy.optimize import minimize
    from scipy.special import expit, logit

    z = logit(df["p"].clip(1e-6, 1 - 1e-6)).to_numpy()
    y = (df["pick_home"] == df["correct"].astype(bool)).astype(float).to_numpy()
    out = np.zeros(len(df))
    seasons = df["season"].to_numpy()
    for s in np.unique(seasons):
        tr = seasons != s
        f = lambda a: float(np.mean(np.logaddexp(0, a[0] * z[tr]) - y[tr] * a[0] * z[tr]))
        a = minimize(f, [1.0], method="BFGS").x[0]
        out[seasons == s] = expit(a * z[seasons == s])
    pc = np.clip(out, 1e-6, 1 - 1e-6)
    return pd.DataFrame({"season": seasons, "ll": -(y * np.log(pc) + (1 - y) * np.log(1 - pc)), "brier": (pc - y) ** 2}, index=df.index)


def season_boot(d, col, seed=20261001, draws=10000):
    g = d.groupby("season")[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (draws, len(g)))
    b = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return {"mean": float(d[col].mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "pp": float((b > 0).mean()), "seasons_pos": int(((g["sum"] / g["count"]) > 0).sum()), "seasons": len(g)}


def stage_grade():
    ns = harness()
    sel = json.loads((OUT / "selection_w.json").read_text())
    base = pd.read_parquet(ROOT / "artifacts" / "mod24_u1" / "base_per_game.parquet")
    parts = {}
    for name in ARMS:
        parts[name] = {k: pd.read_parquet(OUT / f"{name}_{k}.parquet") for k in ("proxy", "true")}
    arms = {"a1_exclude": ("a1_exclude",), "a3_exclude_ol": ("a3_exclude_ol",)}
    frames = {}
    for a in arms:
        frames[a] = pd.concat([parts[a]["proxy"], parts[a]["true"]])
    pieces = []
    for s in range(2011, 2026):
        n = W_NAME[sel[str(s)]["w"]]
        k = "proxy" if s <= 2019 else "true"
        f = parts[n][k]
        pieces.append(f.loc[f["season"].eq(s)])
    frames["a2_down_weight"] = pd.concat(pieces)
    scored = {}
    for a in ("a1_exclude", "a3_exclude_ol"):
        scored[a] = pd.read_parquet(OUT / f"{a}_true_scored.parquet")
    sp = []
    for s in range(2020, 2026):
        sc = pd.read_parquet(OUT / f"{W_NAME[sel[str(s)]['w']]}_true_scored.parquet")
        sp.append(sc.loc[sc["season"].eq(s)])
    scored["a2_down_weight"] = pd.concat(sp)
    report = {"selection_w": {k: v["w"] for k, v in sel.items()}}
    for a, f in frames.items():
        res = ns["compare"](f, base)
        j = f.join(base, rsuffix="_b", how="inner")
        j["d"] = 100.0 * (j["correct"] - j["correct_b"])
        for era, m in (("proxy_2011_2019", j["season"].le(2019)), ("true_2020_2025", j["season"].ge(2020)), ("pooled", j["season"].ge(0))):
            sb = j.loc[m].groupby("season")["d"].mean()
            res[era]["seasons_pos"] = int((sb > 0).sum())
            res[era]["seasons"] = int(len(sb))
            res[era]["acc_season_pp"] = res[era]["acc_season"]["pp"]
            fw = j.loc[m & (j["pick_home"] != j["pick_home_b"])]
            res[era]["flips_won_lost"] = f"{int(fw['correct'].sum())}-{int(len(fw) - fw['correct'].sum())}"
        px = f.loc[f["season"].le(2019)]
        pb = base.loc[base["season"].le(2019)]
        ra, rb = temp_loso(px), temp_loso(pb.loc[px.index])
        jj = ra.join(rb, rsuffix="_b")
        jj["d_ll"] = jj["ll_b"] - jj["ll"]
        jj["d_br"] = jj["brier_b"] - jj["brier"]
        res["proxy_recal"] = {"log_loss_gain": season_boot(jj, "d_ll"), "brier_gain": season_boot(jj, "d_br")}
        report[a] = res
        scored[a].set_index("game_id").to_parquet(OUT / f"{a}_per_game.parquet") if "game_id" in scored[a].columns else scored[a].to_parquet(OUT / f"{a}_per_game.parquet")
        print(a, res["pooled"]["record"], round(res["pooled"]["d_acc"], 3), res["pooled"]["acc_season"], flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


def stage_u2b():
    u2 = ROOT / "scripts" / "mod24_u2b.py"
    src = u2.read_text().rsplit(chr(10) + "main()", 1)[0]
    ns = {"__file__": str(u2), "__name__": "u2b_harness"}
    exec(compile(src, str(u2), "exec"), ns)
    ns["ARMS"] = {a: f"mod24_u7a/{a}" for a in ("a1_exclude", "a2_down_weight", "a3_exclude_ol")}
    ns["OUT"] = OUT / "u2b"
    ns["OUT"].mkdir(exist_ok=True)
    ns["main"]()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "build":
        stage_build()
    elif sys.argv[1] == "select":
        stage_select(harness(), pd.read_parquet(FEATURES))
    elif sys.argv[1] == "grade":
        stage_grade()
    elif sys.argv[1] == "u2b":
        stage_u2b()
    elif sys.argv[1] == "run":
        stage_run(harness(), pd.read_parquet(FEATURES), sys.argv[2].split(","))
