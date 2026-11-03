import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
for sub in (SCRIPTS, REPO / "src"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

import live_kickoff  # noqa: E402
import mod27_fenced_repairs_replay as base  # noqa: E402
import pool_relative_move_forward_log as plog  # noqa: E402

from nfl_ats.clv import LIVE_CAPTURE_KIND, load_decision_quotes  # noqa: E402
from nfl_ats.mass_preserving_lattice import band_read, prior_pool  # noqa: E402

REPLAY = REPO / "artifacts" / "mod27_fenced_repairs_replay" / "results.json"
OUT = REPO / "artifacts" / "mod27_pool_offset_forward"
FROZEN = OUT / "frozen_coefficients.json"
OBSERVED = OUT / "observed_weeks_1_8.json"
POOL_SEASONS = (2021, 2025)
FLOOR = 1e-6
SAMPLES = base.SAMPLES
SEED = base.SEED


def logit(p):
    p = min(max(float(p), FLOOR), 1 - FLOOR)
    return float(np.log(p / (1 - p)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cond(rd):
    t = rd.cover + rd.loss
    return rd.cover / t if t > 0 else 0.5


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen():
    if FROZEN.is_file():
        return json.loads(FROZEN.read_text())
    replay = json.loads(REPLAY.read_text())
    served = json.loads((base.PICK / "coefficients.json").read_text())["coefficients"]
    body = {
        "offset_value_coefficient": replay["pool_offset"]["frozen_offset_coefficient"],
        "model_logit_coefficient": served["model_logit"],
        "fit_seasons": "2020-2025 stacked simulated offsets",
        "pool_prior_seasons": list(POOL_SEASONS),
        "source_results": str(REPLAY.relative_to(REPO)),
        "source_results_sha256": sha_file(REPLAY),
        "pick_probability_artifact": base.PICK.name,
    }
    body["coefficient_hash"] = sha_obj(body)
    OUT.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(body, indent=1))
    return body


def history():
    d = pd.read_parquet(base.PICK / "per_game.parquet").reset_index(drop=True)
    feat = pd.read_parquet(REPO / "data" / "processed" / "game_features.parquet")
    pool = prior_pool(feat, d.set_index("game_id")["tue_open_home_spread"])
    pool = pool[(pool.season >= POOL_SEASONS[0]) & (pool.season <= POOL_SEASONS[1])]
    return pool.line.to_numpy(float), pool.result.to_numpy(float)


def offset_value(lines, results, pool_line, mkt, discrete_base):
    point = optimize.brentq(
        partial(base.gap, cond, lines, results, pool_line, discrete_base),
        pool_line - 25,
        pool_line + 25,
        xtol=1e-6,
    )
    b_mkt = cond(band_read(lines, results, mkt, point))
    v = logit(cond(band_read(lines, results, pool_line, point))) - logit(b_mkt)
    return v, b_mkt


def variant_probability(fz, served_home, discrete_base, b_mkt, off_val):
    z = (
        logit(served_home)
        + fz["model_logit_coefficient"] * (logit(b_mkt) - logit(discrete_base))
        + fz["offset_value_coefficient"] * off_val
    )
    return float(sigmoid(z))


def feed_row(feeds, gid, limit):
    best = None
    for created, name, payload in feeds:
        if created > limit:
            continue
        for g in payload["games"]:
            if g["game_id"] == gid and (best is None or created > best[0]):
                best = (created, name, g)
    return best


def served_home_probability(g):
    p = float(g["served_pick_probability"])
    return p if g["picked_side"] == "HOME" else 1.0 - p


def scores(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return (p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log1p(-p))


def gain_block(d_br, d_ll, keys):
    uniq = np.unique(keys)
    idx = [np.where(keys == k)[0] for k in uniq]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(SAMPLES):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx[k] for k in pick])
        draws.append((d_br[ii].mean(), d_ll[ii].mean()))
    draws = np.array(draws)
    out = {}
    for j, name in enumerate(("brier_gain", "log_loss_gain")):
        out[name] = {
            "point": float((d_br, d_ll)[j].mean()),
            "lo": float(np.percentile(draws[:, j], 2.5)),
            "hi": float(np.percentile(draws[:, j], 97.5)),
            "probability_positive": float((draws[:, j] > 0).mean()),
        }
    return out


def grade_summary(frame, y, label):
    ps = frame.p_served.to_numpy()
    pv = frame.p_variant.to_numpy()
    b_s, l_s = scores(ps, y)
    b_v, l_v = scores(pv, y)
    d_br, d_ll = b_s - b_v, l_s - l_v
    weeks = []
    for wk in sorted(frame.week.unique()):
        m = (frame.week == wk).to_numpy()
        keys = np.arange(m.sum())
        weeks.append(
            {
                "week": int(wk),
                "games": int(m.sum()),
                "mean_abs_shift": float(np.abs(pv[m] - ps[m]).mean()),
                "max_abs_shift": float(np.abs(pv[m] - ps[m]).max()),
                "sides_changed": int(((ps[m] >= 0.5) != (pv[m] >= 0.5)).sum()),
                "served_brier": float(b_s[m].mean()),
                "variant_brier": float(b_v[m].mean()),
                "served_log_loss": float(l_s[m].mean()),
                "variant_log_loss": float(l_v[m].mean()),
                **gain_block(d_br[m], d_ll[m], keys),
            }
        )
    return {
        "grade": label,
        "games": len(frame),
        "served_brier": float(b_s.mean()),
        "variant_brier": float(b_v.mean()),
        "served_log_loss": float(l_s.mean()),
        "variant_log_loss": float(l_v.mean()),
        "pooled_week_block_gain": gain_block(d_br, d_ll, frame.week.to_numpy()),
        "per_week": weeks,
    }


def observed():
    fz = frozen()
    lines, results = history()
    po = pd.read_csv(base.INPUTS / "pool_offset2.csv")
    pk = pd.read_parquet(REPO / "artifacts" / "clv_ledger" / "published_picks.parquet")
    pk = pk[pk.season == 2026]
    dl = pk.groupby("game_id").pick_deadline_utc.first()
    feeds = []
    for d in sorted((REPO / "artifacts" / "waterfall_feed").glob("2026*")):
        f = d / "feed.json"
        if f.is_file():
            j = json.loads(f.read_text())
            feeds.append((pd.Timestamp(j["created_at_utc"]).tz_convert("UTC"), d.name, j))
    rows = []
    no_base = []
    for r in po.itertuples():
        if r.game_id not in dl.index or pd.isna(r.tue_mkt):
            continue
        hit = feed_row(feeds, r.game_id, pd.Timestamp(dl[r.game_id]).tz_convert("UTC"))
        if hit is None or "discrete_base_home_cover_probability" not in hit[2]:
            no_base.append(r.game_id)
            continue
        g = hit[2]
        db = g["discrete_base_home_cover_probability"]
        v, b_mkt = offset_value(lines, results, r.pool, r.tue_mkt, db)
        sh = served_home_probability(g)
        rows.append(
            {
                "game_id": r.game_id,
                "week": int(r.week),
                "pool": r.pool,
                "mkt": r.tue_mkt,
                "offset": r.off,
                "offset_value": v,
                "p_served": sh,
                "p_variant": variant_probability(fz, sh, db, b_mkt, v),
                "hc_pool": r.hc,
                "result": r.result,
            }
        )
    L = pd.DataFrame(rows)
    pool_graded = L[L.hc_pool != 0].reset_index(drop=True)
    opener_graded = L[L.result != L.mkt].reset_index(drop=True)
    out = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "coefficient_hash": fz["coefficient_hash"],
        "frozen_coefficients": fz,
        "games_with_feed_base": len(L),
        "games_without_discrete_base": sorted(no_base),
        "note": "n per week is 13-16; per-week intervals are wide; pooled interval blocks by week",
        "graded_at_pool_line": grade_summary(
            pool_graded, (pool_graded.hc_pool > 0).to_numpy(float), "pool_line"
        ),
        "graded_at_tuesday_market_line": grade_summary(
            opener_graded,
            (opener_graded.result > opener_graded.mkt).to_numpy(float),
            "tuesday_market",
        ),
        "mean_abs_shift_all": float((L.p_variant - L.p_served).abs().mean()),
        "sides_changed_all": int(((L.p_served >= 0.5) != (L.p_variant >= 0.5)).sum()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    OBSERVED.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def forward(week):
    OUT.mkdir(parents=True, exist_ok=True)
    prior = sorted(OUT.glob(f"2026_week{week:02d}_*.json"))
    if prior:
        print(f"already frozen: {prior[-1].name}")
        return
    now = pd.Timestamp(datetime.now(UTC))
    fz = frozen()
    pool, pool_file = plog._pool_lines(week)
    cards = plog._cards(week)
    feeds = plog._feeds(week)
    if not feeds:
        raise SystemExit(f"no waterfall feed for week {week}")
    quotes = load_decision_quotes(REPO / "data" / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    ids = [g for g in cards.index if g in pool]
    live = live_kickoff.live_kickoffs().set_index("game_id")["live_kickoff"]
    tuesday = plog._tuesday_market(quotes, set(ids), {g: cards.at[g, "kickoff"] for g in ids})
    lines, results = history()
    rows = []
    for gid in ids:
        limit = min(now, cards.at[gid, "pick_deadline_utc"], live.get(gid, now))
        hit = feed_row(feeds, gid, limit)
        mkt = tuesday.get(gid, np.nan)
        if (
            hit is None
            or not np.isfinite(mkt)
            or "discrete_base_home_cover_probability" not in hit[2]
        ):
            continue
        created, name, g = hit
        db = g["discrete_base_home_cover_probability"]
        v, b_mkt = offset_value(lines, results, pool[gid], float(mkt), db)
        sh = served_home_probability(g)
        pv = variant_probability(fz, sh, db, b_mkt, v)
        rows.append(
            {
                "game_id": gid,
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "kickoff_utc": str(cards.at[gid, "kickoff"]),
                "pick_deadline_utc": str(cards.at[gid, "pick_deadline_utc"]),
                "pool_home_spread": pool[gid],
                "tuesday_market_line": float(mkt),
                "offset": float(mkt) - pool[gid],
                "offset_value": v,
                "discrete_base_home_probability": db,
                "served_home_probability": sh,
                "variant_home_probability": pv,
                "served_side": "HOME" if sh >= 0.5 else "AWAY",
                "variant_side": "HOME" if pv >= 0.5 else "AWAY",
                "feed": name,
                "feed_created_at_utc": created.isoformat(),
            }
        )
    if not rows:
        raise SystemExit("no scorable games yet")
    body = {
        "term": "pool_offset_two_stage",
        "season": 2026,
        "week": week,
        "created_at_utc": now.isoformat(),
        "runner": "scripts/mod27_pool_offset_observed.py",
        "coefficient_hash": fz["coefficient_hash"],
        "frozen_coefficients_file": str(FROZEN.relative_to(REPO)),
        "frozen_coefficients": fz,
        "pool_file": pool_file,
        "pool_file_sha256": sha_file(REPO / "data" / "splash" / pool_file),
        "games_scored_pregame": len(rows),
        "sides_changed": sum(r["served_side"] != r["variant_side"] for r in rows),
        "games": rows,
        "scoring_rule": "score only after the week settles, graded at the opener, beside the served card; never served",  # noqa: E501
    }
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    f = OUT / f"2026_week{week:02d}_{stamp}.json"
    f.write_text(json.dumps(body, indent=1, default=str))
    print(f"wrote {f.name} games={len(rows)} sides_changed={body['sides_changed']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("observed", "forward"), required=True)
    ap.add_argument("--week", type=int, default=9)
    a = ap.parse_args()
    if a.mode == "observed":
        observed()
    else:
        forward(a.week)


if __name__ == "__main__":
    main()
