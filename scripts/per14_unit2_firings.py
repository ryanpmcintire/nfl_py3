import glob
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

with open("scripts/per14_mover_portability.py", encoding="utf-8") as fh:
    HEAD = fh.read().split("def corr(")[0]
ns = {}
exec(compile(HEAD, "per14_head", "exec"), ns)
traits = ns["traits"]
raw = ns["raw"]
MIN_STINT_GAMES = ns["MIN_STINT_GAMES"]
N_BOOT = ns["N_BOOT"]
N_PERM = ns["N_PERM"]
K = MIN_STINT_GAMES // 2
MIN_SIDE = K - 1
MIN_OBS = 2
FIRST_SEASON, LAST_SEASON = 2011, 2025
MARKET_WINDOW = 3
N_MARKET_PERM = 300
rng = np.random.default_rng(1402)
OUT = "artifacts/per14"

sch = pd.read_parquet(sorted(glob.glob("data/raw/*/schedules.parquet"))[-1])
sch = sch[(sch.game_type == "REG") & sch.season.between(FIRST_SEASON, LAST_SEASON)]
side = []
for loc in ("home", "away"):
    t = sch[["game_id", "season", "week", "gameday", f"{loc}_team", f"{loc}_coach"]].copy()
    t.columns = ["game_id", "season", "week", "gameday", "team", "coach"]
    side.append(t)
tg = pd.concat(side, ignore_index=True).sort_values(["season", "team", "week"])
tg["idx"] = tg.groupby(["season", "team"]).cumcount()
tg["gameday"] = pd.to_datetime(tg.gameday, utc=True)
NG = int(tg.idx.max()) + 1
ngames = tg.groupby(["season", "team"]).size()

ev = []
for (s, tm), g in tg.groupby(["season", "team"]):
    c = g.coach.tolist()
    for i in range(1, len(c)):
        if c[i] != c[i - 1]:
            ev.append(
                {
                    "season": s,
                    "team": tm,
                    "idx": i,
                    "source": "schedule_head_coach_column",
                    "role": "HC",
                }
            )
ch = pd.read_parquet(sorted(glob.glob("data/raw/coordinators/*/coordinator_history.parquet"))[-1])
ch["eff"] = pd.to_datetime(ch.effective_observed_at, utc=True)
oc = ch[(ch.role == "OC") & ch.season.between(FIRST_SEASON, LAST_SEASON)].sort_values("eff")
for (s, tm), g in oc.groupby(["season", "team"]):
    if g.person.nunique() < 2:
        continue
    second = g[g.person != g.person.iloc[0]].iloc[0]
    games = tg[(tg.season == s) & (tg.team == tm) & (tg.gameday > second.eff)]
    if games.empty:
        continue
    ev.append(
        {
            "season": s,
            "team": tm,
            "idx": int(games.idx.iloc[0]),
            "source": "coordinator_history_oc",
            "role": "OC",
        }
    )
ev = pd.DataFrame(ev)
ev["n_games"] = [int(ngames[(r.season, r.team)]) for r in ev.itertuples()]
n_raw = len(ev)
ev = ev[(ev.idx >= MIN_SIDE) & (ev.n_games - ev.idx >= MARKET_WINDOW)]
ev = (
    ev.sort_values("source")
    .groupby(["season", "team"], as_index=False)
    .agg(
        idx=("idx", "min"),
        source=("source", "+".join),
        role=("role", "+".join),
        n_games=("n_games", "first"),
    )
)
all_change = set()
for (s, tm), g in tg.groupby(["season", "team"]):
    if g.coach.nunique() > 1:
        all_change.add((s, tm))
for (s, tm), g in oc.groupby(["season", "team"]):
    if g.person.nunique() > 1:
        all_change.add((s, tm))

epa = raw[raw.play_type.isin(["pass", "run"])].dropna(subset=["epa", "posteam"])
epa = epa.groupby(["game_id", "posteam"]).epa.agg(["mean", "size"]).reset_index()
traits["off_epa"] = pd.DataFrame(
    {"game_id": epa.game_id, "team": epa.posteam, "v": epa["mean"], "w": epa["size"]}
)

keys = list(ngames.index)
kix = {k: i for i, k in enumerate(keys)}
mat = {}
for name, df in traits.items():
    m = tg[["game_id", "season", "team", "idx"]].merge(df, on=["game_id", "team"], how="left")
    m = m[m.v.notna()]
    V = np.full((len(keys), NG), np.nan)
    W = np.zeros((len(keys), NG))
    rows = [kix[k] for k in zip(m.season, m.team, strict=True)]
    V[rows, m.idx.values] = m.v.values
    W[rows, m.idx.values] = m.w.values
    mat[name] = (V, W)


def delta(V, W, i, c):
    out = []
    for lo, hi in ((max(c - K, 0), c), (c, c + K)):
        v = V[i, lo:hi]
        w = W[i, lo:hi]
        ok = ~np.isnan(v)
        if ok.sum() < MIN_OBS:
            return np.nan
        out.append(np.average(v[ok], weights=w[ok]))
    return out[1] - out[0]


ctrl_idx = {
    s: [kix[k] for k in keys if k[0] == s and k not in all_change]
    for s in range(FIRST_SEASON, LAST_SEASON + 1)
}
cache = {}


def ctrl_delta(name, s, c):
    key = (name, s, c)
    if key not in cache:
        V, W = mat[name]
        d = [delta(V, W, i, c) for i in ctrl_idx[s]]
        d = [x for x in d if x == x]
        cache[key] = float(np.mean(d)) if d else np.nan
    return cache[key]


def did(name, s, tm, c):
    V, W = mat[name]
    return delta(V, W, kix[(s, tm)], c) - ctrl_delta(name, s, c)


res = {}
for name in mat:
    obs = np.array([did(name, r.season, r.team, r.idx) for r in ev.itertuples()])
    ok = obs[~np.isnan(obs)]
    boots = np.array([rng.choice(ok, len(ok)).mean() for _ in range(N_BOOT)])
    stat = float(ok.mean())
    nulls = []
    for _ in range(N_PERM):
        vals = np.array(
            [
                did(
                    name,
                    r.season,
                    r.team,
                    int(rng.integers(MIN_SIDE, r.n_games - MARKET_WINDOW + 1)),
                )
                for r in ev.itertuples()
            ]
        )
        nulls.append(vals[~np.isnan(vals)].mean())
    nulls = np.array(nulls)
    res[name] = {
        "n_events": len(ok),
        "did_mean": stat,
        "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
        "probability_positive": float((boots > 0).mean()),
        "perm_p_two_sided": float((np.sum(np.abs(nulls) >= abs(stat)) + 1) / (N_PERM + 1)),
        "perm_null_sd": float(nulls.std()),
        "se": float(ok.std(ddof=1) / np.sqrt(len(ok))),
    }
    print(name, res[name], flush=True)

sys.path.insert(0, "scripts")
import per07_first_game_fitted_term as p7  # noqa: E402

p7.BOOTSTRAP_SAMPLES = N_BOOT
d = p7.prepare()
parts = d.game_id.str.split("_", expand=True)
d["away_team"], d["home_team"] = parts[2], parts[3]
for c in ("hs3", "as3", "hs1", "as1"):
    d[c] = 0.0
flagged = []
for r in ev.itertuples():
    games = tg[(tg.season == r.season) & (tg.team == r.team)].sort_values("idx")
    for j, g in enumerate(games.iloc[r.idx : r.idx + MARKET_WINDOW].itertuples()):
        mask = d.game_id == g.game_id
        if not mask.any():
            continue
        col = "h" if d.loc[mask, "home_team"].iloc[0] == r.team else "a"
        d.loc[mask, f"{col}s3"] = 1.0
        if j == 0:
            d.loc[mask, f"{col}s1"] = 1.0
        flagged.append((g.game_id, r.team))
market = {
    "events_in_market_years": len({(g.split("_")[0], t) for g, t in flagged}),
    "flagged_team_games": len(flagged),
}
for label, hs, as_ in (("first3", "hs3", "as3"), ("first1", "hs1", "as1")):
    f = d.copy()
    f["home_signal"], f["away_signal"] = d[hs], d[as_]
    f["s"] = f.home_signal - f.away_signal
    r = p7.analyze(f, 2)
    sc, _ = p7.loso(f)
    obs_gain = p7.metrics(sc, "p_base")["log_loss"] - p7.metrics(sc, "p_term")["log_loss"]
    nulls = []
    for _ in range(N_MARKET_PERM):
        g2 = f.copy()
        g2["s"] = g2.groupby("block").s.transform(lambda x: rng.permutation(x.values))
        s2, _ = p7.loso(g2)
        nulls.append(p7.metrics(s2, "p_base")["log_loss"] - p7.metrics(s2, "p_term")["log_loss"])
    r["perm_log_loss_gain_p_upper"] = float(
        (np.sum(np.array(nulls) >= obs_gain) + 1) / (N_MARKET_PERM + 1)
    )
    r["flag_signed_counts"] = {"home": int((f.s > 0).sum()), "away": int((f.s < 0).sum())}
    market[label] = r
    print(label, json.dumps(r, default=str), flush=True)

out = {
    "window_K": K,
    "min_side": MIN_SIDE,
    "raw_events_before_window_filter": n_raw,
    "events": ev.to_dict("records"),
    "did": res,
    "market": market,
    "control_team_seasons": {s: len(v) for s, v in ctrl_idx.items()},
}
with open(f"{OUT}/firings_event_study.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, default=str)
ev.to_csv(f"{OUT}/firings_events.csv", index=False)
