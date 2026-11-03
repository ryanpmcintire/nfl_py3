import glob
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from nfl_ats.home_side_location import TRAILING_SEASONS
from nfl_ats.mass_preserving_lattice import band_read, prior_pool
from nfl_ats.pick_probability import FLAG_SUM_COLUMN, MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
from nfl_ats.pick_probability_fit import FIT_RIDGE, _fit_logit
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "artifacts" / "mod27_replay_inputs"
OUT = REPO / "artifacts" / "mod27_fenced_repairs_replay"
PICK = REPO / "artifacts" / "pick_probability" / "20261101T171318Z"
EPS = 1e-9
SAMPLES = 4000
SEED = 20261102
FOUR = ("model_logit", FLAG_SUM_COLUMN, MOVE_COLUMN, MOVE_AVAILABLE_COLUMN)
MIN_LEADERS = 3
OFFSETS = (-1.0, -0.5, 0.5, 1.0)


def gap(cond, lines, results, line, target, point):
    return cond(band_read(lines, results, line, point)) - target


class Fit:
    def __init__(self, features, train):
        self.features = features
        self.means = {n: float(train[n].mean()) for n in features}
        self.stds = {n: float(train[n].std(ddof=0)) or 1.0 for n in features}
        self.beta = _fit_logit(
            self.design(train), train.home_covered.astype(float).to_numpy(), FIT_RIDGE
        )

    def design(self, frame):
        columns = [np.ones(len(frame))]
        for n in self.features:
            columns.append(((frame[n].astype(float) - self.means[n]) / self.stds[n]).to_numpy())
        return np.column_stack(columns)

    def predict(self, frame):
        z = np.clip(self.design(frame) @ self.beta, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def natural(self):
        out = {n: float(self.beta[i + 1] / self.stds[n]) for i, n in enumerate(self.features)}
        out["intercept"] = float(self.beta[0]) - sum(
            float(self.beta[i + 1]) * self.means[n] / self.stds[n]
            for i, n in enumerate(self.features)
        )
        return out


def loso(frame, features):
    p = pd.Series(np.nan, index=frame.index)
    for season in sorted(frame.season.unique()):
        fit = Fit(features, frame[frame.season != season])
        te = frame[frame.season == season]
        p.loc[te.index] = fit.predict(te)
    return p


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def record(p, y):
    c = (np.asarray(p) >= 0.5) == (np.asarray(y) == 1)
    return f"{int(c.sum())}-{int((~c).sum())}"


def paired(frame, pa, pb, block):
    y = frame.home_covered.to_numpy(float)
    a = np.asarray(pa, float)
    b = np.asarray(pb, float)
    d_ll = log_loss(a, y) - log_loss(b, y)
    d_br = (a - y) ** 2 - (b - y) ** 2
    d_ac = ((b >= 0.5) == (y == 1)).astype(float) - ((a >= 0.5) == (y == 1)).astype(float)
    keys = frame[block].to_numpy()
    uniq = np.unique(keys)
    idx = [np.where(keys == k)[0] for k in uniq]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(SAMPLES):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx[k] for k in pick])
        draws.append((d_ll[ii].mean(), d_br[ii].mean(), d_ac[ii].mean() * 100))
    draws = np.array(draws)
    out = {}
    for j, name in enumerate(("log_loss_gain", "brier_gain", "accuracy_points_gain")):
        point = (d_ll.mean(), d_br.mean(), d_ac.mean() * 100)[j]
        out[name] = {
            "point": float(point),
            "lo": float(np.percentile(draws[:, j], 2.5)),
            "hi": float(np.percentile(draws[:, j], 97.5)),
            "probability_positive": float((draws[:, j] > 0).mean()),
        }
    return out


def week_table(frame, p_served, p_variant):
    y = frame.home_covered.to_numpy(float)
    ps = np.asarray(p_served)
    pv = np.asarray(p_variant)
    rows = []
    for wk in sorted(frame.week.unique()):
        m = (frame.week == wk).to_numpy()
        sw = m & ((ps >= 0.5) != (pv >= 0.5))
        rows.append(
            {
                "week": int(wk),
                "games": int(m.sum()),
                "sides_changed": int(sw.sum()),
                "old_side_record": record(ps[sw], y[sw]) if sw.any() else "0-0",
                "new_side_record": record(pv[sw], y[sw]) if sw.any() else "0-0",
            }
        )
    return rows


def changed_games(frame, p_served, p_variant):
    y = frame.home_covered.to_numpy(float)
    out = []
    for i in np.where((np.asarray(p_served) >= 0.5) != (np.asarray(p_variant) >= 0.5))[0]:
        out.append(
            {
                "game": str(frame.game_id.iloc[i]),
                "week": int(frame.week.iloc[i]),
                "served_p_home": round(float(p_served[i]), 4),
                "variant_p_home": round(float(p_variant[i]), 4),
                "home_covered": float(y[i]),
                "served_correct": bool((p_served[i] >= 0.5) == (y[i] == 1)),
                "variant_correct": bool((p_variant[i] >= 0.5) == (y[i] == 1)),
            }
        )
    return out


def structs(books):
    out = {}
    for g, t in books.groupby("game_id"):
        lead = t[t.bookmaker_key.isin(LEADER_BOOKS)]
        if len(lead) == 0:
            continue
        out[g] = {
            "lb": lead.base.to_numpy(),
            "lf": lead.final.to_numpy(),
            "all_moves": t.move.to_numpy(),
        }
    return out


def move_of(s, variant):
    served = float(np.median(s["lf"] - s["lb"]))
    thin = len(s["lb"]) < MIN_LEADERS
    if variant == "served" or not thin:
        return served, 1.0
    if variant == "thin_drop":
        return 0.0, 0.0
    return float(np.median(s["all_moves"])), 1.0


def move_repair():
    H = pd.read_parquet(PICK / "per_game.parquet").reset_index(drop=True)
    if FLAG_SUM_COLUMN != "composition_flag_sum" and "composition_flag_sum" in H:
        H = H.rename(columns={"composition_flag_sum": FLAG_SUM_COLUMN})
    G = pd.read_parquet(INPUTS / "g26.parquet").reset_index(drop=True)
    G["model_logit"] = np.log(
        np.clip(G.discrete, EPS, 1 - EPS) / (1 - np.clip(G.discrete, EPS, 1 - EPS))
    )
    hist = structs(pd.read_pickle(INPUTS / "hist_books.pkl"))
    live = structs(pd.read_pickle(INPUTS / "live_books.pkl"))
    inc = H.set_index("game_id")[MOVE_COLUMN]
    repro = {
        g: abs(move_of(s, "served")[0] - inc[g]) < 1e-9 for g, s in hist.items() if g in inc.index
    }

    def frame_for(variant, base, structures, gate):
        f = base.copy()
        for i, g in enumerate(f.game_id):
            if g in structures and gate(g):
                m, a = move_of(structures[g], variant)
                if variant != "served" and len(structures[g]["lb"]) < MIN_LEADERS:
                    f.loc[f.index[i], MOVE_COLUMN] = m
                    f.loc[f.index[i], MOVE_AVAILABLE_COLUMN] = a
        return f

    y = H.home_covered.to_numpy(float)
    ps = loso(H, FOUR)
    served_full = Fit(FOUR, H)
    results = {
        "served_loso_record": record(ps, y),
        "stored_oos_record": record(H.out_of_season_home_probability, y),
        "harness_side_agreement_with_feed_2026": float(
            (
                (served_full.predict(frame_for("served", G, live, lambda g: True)) >= 0.5)
                == (G.final_probability >= 0.5)
            ).mean()
        ),
        "thin_games_2020_2025": int(
            sum(1 for g, s in hist.items() if repro.get(g, False) and len(s["lb"]) < MIN_LEADERS)
        ),
        "thin_games_2026": int(sum(1 for s in live.values() if len(s["lb"]) < MIN_LEADERS)),
    }
    for variant in ("thin_drop", "thin_all_books"):
        Hv = frame_for(variant, H, hist, lambda g: repro.get(g, False))
        pv = loso(Hv, FOUR)
        H2 = H.assign(season_week=H.season * 100 + H.week)
        cmp_all = paired(H2, ps.to_numpy(), pv.to_numpy(), "season_week")
        changed = (
            (Hv[MOVE_COLUMN] - H[MOVE_COLUMN]).abs().gt(1e-9)
            | (Hv[MOVE_AVAILABLE_COLUMN] - H[MOVE_AVAILABLE_COLUMN]).abs().gt(1e-9)
        ).to_numpy()
        dis = ((ps >= 0.5) != (pv >= 0.5)).to_numpy()
        fv = Fit(FOUR, Hv)
        ins_s = log_loss(served_full.predict(H), y).mean()
        ins_v = log_loss(fv.predict(Hv), y).mean()
        oos_s = log_loss(ps, y).mean()
        oos_v = log_loss(pv, y).mean()
        fold_coefs = {}
        for season in sorted(H.season.unique()):
            fold_coefs[int(season)] = {
                k: round(v, 3)
                for k, v in Fit(FOUR, Hv[Hv.season != season]).natural().items()
                if k.startswith("market_move")
            }
        Gs = frame_for("served", G, live, lambda g: True)
        Gv = frame_for(variant, G, live, lambda g: True)
        pS = served_full.predict(Gs)
        pV = fv.predict(Gv)
        results[variant] = {
            "rows_changed_2020_2025": int(changed.sum()),
            "loso_record": record(pv, y),
            "loso_vs_served_paired_gain_of_variant": cmp_all,
            "disagreement_2020_2025": {
                "n": int(dis.sum()),
                "served": record(ps[dis], y[dis]) if dis.any() else "0-0",
                "variant": record(pv[dis], y[dis]) if dis.any() else "0-0",
            },
            "in_sample_ll": {"served": float(ins_s), "variant": float(ins_v)},
            "oos_ll": {"served": float(oos_s), "variant": float(oos_v)},
            "gap_ll": {"served": float(oos_s - ins_s), "variant": float(oos_v - ins_v)},
            "move_coefficient_per_fold": fold_coefs,
            "week_table_2026": week_table(G, pS, pV),
            "changed_games_2026": changed_games(G, pS, pV),
            "record_2026": {
                "served": record(pS, G.home_covered.to_numpy(float)),
                "variant": record(pV, G.home_covered.to_numpy(float)),
            },
        }
    return results


def pool_offset():
    d = pd.read_parquet(PICK / "per_game.parquet").reset_index(drop=True)
    d["margin"] = d.margin_vs_open + d.tue_open_home_spread
    feat = pd.read_parquet(REPO / "data" / "processed" / "game_features.parquet")
    pool = prior_pool(feat, d.set_index("game_id")["tue_open_home_spread"])

    def cond(rd):
        t = rd.cover + rd.loss
        return rd.cover / t if t > 0 else 0.5

    def lg(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    rows = []
    for season, g in d.groupby("season"):
        pp = pool[(pool.season < season) & (pool.season >= season - TRAILING_SEASONS)]
        lines = pp.line.to_numpy(float)
        results = pp.result.to_numpy(float)
        for i, r in g.iterrows():
            line = float(r.tue_open_home_spread)
            target = float(r.model_probability)
            try:
                point = optimize.brentq(
                    partial(gap, cond, lines, results, line, target),
                    line - 25,
                    line + 25,
                    xtol=1e-6,
                )
            except ValueError:
                continue
            base = lg(cond(band_read(lines, results, line, point)))
            for off in OFFSETS:
                pl = line + off
                rows.append(
                    (i, off, pl, lg(cond(band_read(lines, results, pl, point))) - base, point)
                )
    o = pd.DataFrame(rows, columns=["i", "offset", "pool_line", "offset_value", "point"])
    st = d.loc[o.i].reset_index(drop=True).join(o.drop(columns="i"))
    push = st.margin == st.pool_line
    st = st.loc[~push].reset_index(drop=True)
    st["home_covered"] = (st.margin > st.pool_line).astype(float)
    st["season_week"] = st.season * 100 + st.week
    five = (*FOUR, "offset_value")
    out = {"stacked_rows": len(st), "games": int(st.game_id.nunique())}
    for name, sub in (
        ("all_offsets", st),
        ("abs_0.5", st[st.offset.abs() == 0.5]),
        ("abs_1.0", st[st.offset.abs() == 1.0]),
    ):
        sub = sub.reset_index(drop=True)
        p4 = loso(sub, FOUR)
        p5 = loso(sub, five)
        coefs = {
            int(s): round(Fit(five, sub[sub.season != s]).natural()["offset_value"], 3)
            for s in sorted(sub.season.unique())
        }
        out[name] = {
            "rows": len(sub),
            "offset_coefficient_per_fold": coefs,
            "paired_gain_of_offset_term": paired(sub, p4.to_numpy(), p5.to_numpy(), "season_week"),
            "accuracy_four": float(((p4 >= 0.5) == (sub.home_covered == 1)).mean()),
            "accuracy_five": float(((p5 >= 0.5) == (sub.home_covered == 1)).mean()),
        }
    fit5 = Fit(five, st)
    nat = fit5.natural()
    out["frozen_offset_coefficient"] = nat["offset_value"]
    served = json.loads((PICK / "coefficients.json").read_text())["coefficients"]
    po = pd.read_csv(INPUTS / "pool_offset2.csv")
    pk = pd.read_parquet(REPO / "artifacts" / "clv_ledger" / "published_picks.parquet")
    pk = pk[pk.season == 2026]
    dl = pk.groupby("game_id").pick_deadline_utc.first()
    feeds = {}
    for f in sorted(glob.glob(str(REPO / "artifacts" / "waterfall_feed" / "2026*" / "feed.json"))):
        j = json.loads(Path(f).read_text())
        t = pd.Timestamp(j["created_at_utc"])
        for g in j["games"]:
            feeds.setdefault(g["game_id"], []).append((t, g))
    pp = pool[(pool.season >= 2021) & (pool.season <= 2025)]
    lines = pp.line.to_numpy(float)
    results = pp.result.to_numpy(float)
    live = []
    skipped = 0
    for _, r in po.iterrows():
        if r.game_id not in feeds or r.game_id not in dl.index or pd.isna(r.tue_mkt):
            continue
        ok = [x for x in feeds[r.game_id] if x[0] <= pd.Timestamp(dl[r.game_id])]
        if not ok:
            continue
        t, g = max(ok, key=lambda x: x[0])
        if "discrete_base_home_cover_probability" not in g:
            skipped += 1
            continue
        base = g["discrete_base_home_cover_probability"]
        point = optimize.brentq(
            partial(gap, cond, lines, results, r.pool, base),
            r.pool - 25,
            r.pool + 25,
            xtol=1e-6,
        )
        b_mkt = cond(band_read(lines, results, r.tue_mkt, point))
        off_val = lg(cond(band_read(lines, results, r.pool, point))) - lg(b_mkt)
        spp = g["served_pick_probability"]
        sph = spp if g["picked_side"] == "HOME" else 1 - spp
        zs = lg(sph)
        za = zs + served["model_logit"] * (lg(b_mkt) - lg(base)) + nat["offset_value"] * off_val
        live.append(
            {
                "game_id": r.game_id,
                "week": int(r.week),
                "pool": r.pool,
                "mkt": r.tue_mkt,
                "off": r.off,
                "off_val": off_val,
                "p_served": sph,
                "p_variant": 1 / (1 + np.exp(-za)),
                "home_covered": r.hc,
                "card_home": r.card_home,
            }
        )
    L = pd.DataFrame(live)
    L = L[L.home_covered != 0].reset_index(drop=True)
    out["live_2026"] = {
        "games": len(L),
        "skipped_no_discrete_base": skipped,
        "week_table": week_table(L, L.p_served.to_numpy(), L.p_variant.to_numpy()),
        "changed_games": changed_games(L, L.p_served.to_numpy(), L.p_variant.to_numpy()),
        "mean_abs_probability_shift": float((L.p_variant - L.p_served).abs().mean()),
        "offset_games": int((L.off.abs() >= 0.5).sum()),
        "served_record": record(L.p_served, np.where(L.home_covered > 0, 1.0, 0.0)),
        "variant_record": record(L.p_variant, np.where(L.home_covered > 0, 1.0, 0.0)),
    }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"move_repair": move_repair(), "pool_offset": pool_offset()}
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
