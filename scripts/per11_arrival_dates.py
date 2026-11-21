import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eng54_served_regrade as E  # noqa: E402

from nfl_ats import transaction_flag_features as T  # noqa: E402

OUT = ROOT / "artifacts" / "per11_arrival_rampup"
LANE = "docs/lanes/per-11-per11-acquired-player-ramp.md"
FAMILY = "per11_arrival_rampup"
PREDICTION_TZ = "America/New_York"
PREDICTION_HOUR = 12
LIVE_SEASON = 2026
LIVE_WEEK = 12
LIVE_MONTHS = (10, 11)
WINDOWS = {"g1": 1, "g12": 2}
GROUPS = {
    "all": None,
    "qb": {"QB"},
    "ol": {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"},
    "rush": {"DE", "DT", "NT", "DL", "OLB", "EDGE"},
}
TRADE_TO_RE = re.compile(r"^(?P<giver>.+?)-(?:to-)?trad(?:e|es|ed)-(?P<rest>.+)-to-(?P<recv>[a-z0-9]+)$")
FINAL_RE = re.compile(
    r"^(?P<recv>.+?)-(?:finaliz(?:e|es|ing)|complet(?:e|es|ed|ing)|agree(?:s|d)?-to|land|lands)-"
    r"(?:trade-for|deal-for)-(?P<rest>.+)$"
)
CLAIM_RE = re.compile(r"^(?P<recv>.+?)-claim(?:s|ed)?-(?P<rest>.+)$")
CLAIM_FIX_RE = re.compile(r"^(?P<recv>.+?)-not-(?P<wrong>.+?)-claim(?:s|ed)?-(?P<rest>.+)$")
PAIR_RE = re.compile(r"^(?P<pair>.+?)-(?:complete|completed|completes)-(?P<rest>.+)-trade$")
SUFFIX_RE = re.compile(r"-(?:jr|sr|ii|iii|iv|v)$")
NOT_DONE_RE = re.compile(
    r"not-|wont-|will-not|unlikely|willing|expected|could-|would-|interest|candidate|rumor|"
    r"explor|talks|discuss|monitor|consider|open-to|shopping|planning|eye|pursu|target|radar|"
    r"calling|seeking|request|asking|price|nixed|failed|poll|mailbag|notes|latest|fallout|details"
)
NIX_RE = re.compile(
    r"nixed|failed|voided|rescind|fell-through|falls-through|called-off|not-completed|vetoed|"
    r"backs-out|backed-out|reversed"
)


def prediction_ts(gameday):
    gd = pd.to_datetime(gameday)
    tue = gd - pd.to_timedelta((gd.dt.weekday - 1) % 7, unit="D")
    return (tue + pd.Timedelta(hours=PREDICTION_HOUR)).dt.tz_localize(PREDICTION_TZ).dt.tz_convert("UTC")


def url_dates():
    pub = []
    for f in glob.glob(str(ROOT / "data/raw/pfr_transactions/*/sample_articles/*.json")):
        d = json.load(open(f, encoding="utf-8"))
        p = d.get("json_ld_date_published") or []
        if p:
            pub.append({"url": d["url"], "published": pd.to_datetime(p[0], utc=True)})
    lm = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/pfr_transactions/*/index.parquet"))):
        d = pd.read_parquet(f, columns=["url", "lastmod_contaminated"])
        d["lastmod"] = pd.to_datetime(d["lastmod_contaminated"], utc=True)
        lm.append(d[["url", "lastmod"]])
    lm = pd.concat(lm).groupby("url")["lastmod"].min().reset_index()
    pb = pd.DataFrame(pub, columns=["url", "published"]).groupby("url")["published"].min().reset_index()
    out = lm.merge(pb, on="url", how="outer")
    out["tx_ts"] = out["published"].fillna(out["lastmod"])
    out["date_source"] = np.where(out["published"].notna(), "published", "min_lastmod")
    return out


def one_team(text):
    t = T.match_transaction_teams(text)
    return next(iter(t)) if len(t) == 1 else None


def parse(slug, category):
    if T.SPECULATIVE_ACQUISITION_RE.search(slug) or T.DRAFT_PICK_RE.search(slug):
        return None, "speculative_or_pick"
    if category == "trade":
        if T.ACQUISITION_RE.search(slug):
            if NOT_DONE_RE.search(slug.split("acquir")[0]):
                return None, "not_completed_wording"
            got = T._parse_acquisition(slug)
            return (got, "acquire") if got else (None, "acquire_team_unparsed")
        if NOT_DONE_RE.search(slug):
            return None, "not_completed_wording"
        m = TRADE_TO_RE.match(slug)
        if m and one_team(m["giver"]) and one_team(m["recv"]):
            rest = re.sub(r"-back$", "", m["rest"])
            return (one_team(m["recv"]), rest), "trade_to"
        m = FINAL_RE.match(slug)
        if m and one_team(m["recv"]):
            return (one_team(m["recv"]), m["rest"]), "finalize"
        m = PAIR_RE.match(slug)
        if m and len(T.match_transaction_teams(m["pair"])) == 2:
            return ("|".join(sorted(T.match_transaction_teams(m["pair"]))), m["rest"]), "pair_complete"
        return None, "no_completed_pattern"
    if category == "waiver_claim":
        m = CLAIM_FIX_RE.match(slug)
        if m and one_team(m["recv"]):
            return (one_team(m["recv"]), re.split(r"-off-waivers|-from-", m["rest"])[0]), "claim_correction"
        m = CLAIM_RE.match(slug)
        if m and one_team(m["recv"]) and not NOT_DONE_RE.search(slug):
            rest = re.split(r"-off-waivers|-from-", m["rest"])[0]
            return (one_team(m["recv"]), rest), "claim"
        return None, "claim_unparsed"
    return None, "other_category"


def candidates(tx, dates, name_slugs):
    t = tx.loc[tx["url_month"].isin(T.DEADLINE_WINDOW_MONTHS) & tx["category"].isin(["trade", "waiver_claim"])]
    t = t.merge(dates[["url", "tx_ts", "date_source"]], on="url", how="left")
    rows = []
    for _, r in t.iterrows():
        got, how = parse(str(r["slug"]), r["category"])
        team, player = None, None
        if got:
            team, part = got
            player = T.find_player_in_segment(part, name_slugs)
            if player is None:
                how = how + "_player_unmatched"
        rows.append(
            {
                "season": int(r["url_year"]),
                "month": int(r["url_month"]),
                "slug": r["slug"],
                "category": r["category"],
                "rule": how,
                "team": team,
                "player": player,
                "tx_ts": r["tx_ts"],
                "date_source": r["date_source"] if pd.notna(r["tx_ts"]) else "undated",
            }
        )
    return pd.DataFrame(rows)


def nix_times(tx, dates, cand):
    t = tx.loc[tx["url_month"].isin(T.DEADLINE_WINDOW_MONTHS)].merge(dates[["url", "tx_ts"]], on="url")
    t = t.loc[t["slug"].astype(str).str.contains(NIX_RE)]
    out = {}
    for (season, player), _ in cand.dropna(subset=["player"]).groupby(["season", "player"]):
        ns = T._normalize_name_to_slug(player)
        hit = t.loc[t["url_year"].eq(season) & t["slug"].astype(str).str.contains(f"(?:^|-){ns}(?:-|$)")]
        if len(hit):
            out[(season, player)] = hit["tx_ts"].min()
    return out


def events(cand, nix, sn, reg):
    ok = cand.dropna(subset=["player", "team", "tx_ts"])
    ok = ok.sort_values("tx_ts").drop_duplicates(["season", "player", "team"])
    pos = sn.dropna(subset=["position"]).groupby("player")["position"].agg(lambda s: s.mode().iloc[0])
    out, drops = [], []
    for _, r in ok.iterrows():
        season, team, player, ts = r["season"], r["team"], r["player"], r["tx_ts"]
        if "|" in team:
            seen = sn.loc[sn["player"].eq(player) & sn["season"].eq(season) & sn["week"].lt(
                int(reg.loc[reg["season"].eq(season) & reg["pred_ts"].gt(ts), "week"].min()))]
            if seen.empty:
                drops.append((r["slug"], "pair_direction_unknown"))
                continue
            last = str(seen.sort_values("week")["team"].iloc[-1])
            other = [t for t in team.split("|") if t != last]
            if len(other) != 1:
                drops.append((r["slug"], "pair_direction_unknown"))
                continue
            team = other[0]
        tg = reg.loc[reg["season"].eq(season) & (reg["home_team"].eq(team) | reg["away_team"].eq(team))]
        tg = tg.sort_values("week")
        after = tg.loc[tg["pred_ts"] > ts]
        if after.empty:
            drops.append((r["slug"], "no_game_after_date"))
            continue
        w0 = int(after["week"].min())
        sr = sn.loc[sn["player"].eq(player) & sn["season"].eq(season) & sn["week"].lt(w0)]
        prior = sr.loc[sr["team"].ne(team)]
        if prior.empty:
            drops.append((r["slug"], "no_prior_team_snaps"))
            continue
        lw = int(prior["week"].max())
        giving = str(prior.loc[prior["week"].eq(lw), "team"].iloc[0])
        share = float(prior.loc[prior["team"].eq(giving), "snap_share"].mean())
        if share < T.HIGH_SNAP_SHARE_THRESHOLD:
            drops.append((r["slug"], "prior_share_below_threshold"))
            continue
        nx = nix.get((season, player))
        games = after.loc[after["week"].gt(lw)]
        if nx is not None:
            games = games.loc[games["pred_ts"] <= nx]
        for k, (_, g) in enumerate(games.head(max(WINDOWS.values())).iterrows(), start=1):
            out.append(
                {
                    "season": season,
                    "week": int(g["week"]),
                    "game_id": g["game_id"],
                    "team": team,
                    "home": g["home_team"] == team,
                    "k": k,
                    "player": player,
                    "position": pos.get(player, ""),
                    "from_team": giving,
                    "prior_share": share,
                    "slug": r["slug"],
                    "rule": r["rule"],
                    "tx_ts": ts,
                    "date_source": r["date_source"],
                    "pred_ts": g["pred_ts"],
                    "nixed_after": nx is not None,
                }
            )
    ev = pd.DataFrame(out)
    if len(ev) and not (ev["tx_ts"] < ev["pred_ts"]).all():
        raise AssertionError("an arrival flag uses a transaction not strictly before the prediction timestamp")
    return ev, pd.DataFrame(drops, columns=["slug", "reason"])


def term(ev, games, group, window):
    e = ev.loc[ev["k"].le(WINDOWS[window])]
    if GROUPS[group] is not None:
        e = e.loc[e["position"].isin(GROUPS[group])]
    flag = e[["game_id", "team"]].drop_duplicates()
    h = games.merge(flag, left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left")
    a = games.merge(flag, left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left")
    return h["team"].notna().to_numpy(float) - a["team"].notna().to_numpy(float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sch = T.default_schedule()
    reg = sch.loc[sch["game_type"].eq("REG")].copy()
    reg["season"] = reg["season"].astype(int)
    reg["week"] = reg["week"].astype(int)
    reg["pred_ts"] = prediction_ts(reg["gameday"])
    sn = T.default_snap_counts(season=LIVE_SEASON, before_week=LIVE_WEEK)
    names = T.distinct_player_slugs(sn)
    bare = names.assign(name_slug=names["name_slug"].str.replace(SUFFIX_RE, "", regex=True))
    bare = bare.loc[bare["name_slug"].ne(names["name_slug"])]
    names = pd.concat([names, bare.loc[~bare["name_slug"].isin(names["name_slug"])]])
    names = names.assign(_n=names["name_slug"].str.len()).sort_values("_n", ascending=False)
    names = names.drop(columns="_n").drop_duplicates("name_slug").reset_index(drop=True)
    tx = T.default_transactions_index()
    dates = url_dates()
    cand = candidates(tx, dates, names)
    nix = nix_times(tx, dates, cand)
    ev, drops = events(cand, nix, sn, reg)
    cand.to_csv(OUT / "dated_candidates.csv", index=False)
    ev.to_csv(OUT / "dated_events.csv", index=False)
    drops.to_csv(OUT / "dated_drops.csv", index=False)
    live = cand.loc[cand["season"].eq(LIVE_SEASON) & cand["month"].isin(LIVE_MONTHS)]
    live_trade = live.loc[live["category"].eq("trade")]
    cover = {
        "live_trade_slugs": len(live_trade),
        "live_waiver_slugs": int(live["category"].eq("waiver_claim").sum()),
        "live_rule_counts": live["rule"].value_counts().to_dict(),
        "live_undated": live.loc[live["date_source"].eq("undated"), "slug"].tolist(),
        "live_date_source": live["date_source"].value_counts().to_dict(),
        "live_completed": live.dropna(subset=["player"])[
            ["slug", "team", "player", "tx_ts", "rule"]
        ].astype(str).to_dict("records"),
        "live_nixed": {f"{k[1]}": str(v) for k, v in nix.items() if k[0] == LIVE_SEASON},
        "history_events_by_season": ev.loc[ev["k"].eq(1)].groupby("season").size().to_dict(),
        "history_date_source": ev.loc[ev["k"].eq(1), "date_source"].value_counts().to_dict(),
        "drop_reasons": drops["reason"].value_counts().to_dict(),
    }
    res = {"coverage": cover}
    if "--parse-only" in sys.argv:
        print(json.dumps(cover, indent=1, default=str))
        return
    pick = sorted((ROOT / "artifacts/pick_probability").glob("*/per_game.parquet"))[-1]
    g = pd.read_parquet(
        pick,
        columns=["game_id", "season", "week", "home_team", "away_team", "margin_vs_open", "home_covered",
                 E.SERVED_COLUMN],
    )
    g = g.loc[g["margin_vs_open"].ne(0) & g["season"].between(E.FIRST_SEASON, E.LAST_SEASON)]
    g = g.assign(game_id=g["game_id"].astype(str)).reset_index(drop=True)
    res["pick_table"] = str(pick.relative_to(ROOT))
    looks = len(GROUPS) * len(WINDOWS)
    cells, idx = {}, 0
    for grp in GROUPS:
        for win in WINDOWS:
            idx += 1
            col = f"x_{grp}_{win}"
            g[col] = term(ev, g, grp, win)
            nz = g.loc[g[col].ne(0)]
            served_right = (nz[E.SERVED_COLUMN].ge(0.5) == nz["home_covered"].eq(1)).sum()
            arr_cover = ((nz[col] > 0) == nz["home_covered"].eq(1)).sum()
            st = E.setup(g, [col], False, E.identity, 500 + idx)
            r = E.grade_arm(st, 500 + idx)
            r["flagged_games"] = len(nz)
            r["served_record_on_flagged"] = [int(served_right), int(len(nz) - served_right)]
            r["arrival_team_cover_record"] = [int(arr_cover), int(len(nz) - arr_cover)]
            r["flagged_by_season"] = nz.groupby("season").size().to_dict()
            cells[col] = r
            a = r["acc"]
            print(
                col, len(nz), r["served_record_on_flagged"], r["arrival_team_cover_record"],
                f"acc {a['effect']:+.3f} [{a['low']:+.3f},{a['high']:+.3f}] P+ {a['pp']:.3f} "
                f"null {a['null_percentile']:.1f} chg {r['arm_right']}-{r['ref_right']} "
                f"coef {r['full_term_coefficients']} det {E.det_text(r['detectable']['acc'])}",
                flush=True,
            )
    res["cells"] = cells
    res["looks"] = looks
    wk = reg.loc[reg["season"].eq(LIVE_SEASON) & reg["week"].eq(LIVE_WEEK)].copy()
    wk = wk.assign(game_id=wk["game_id"].astype(str)).reset_index(drop=True)
    live_rows = []
    lev = ev.loc[ev["season"].eq(LIVE_SEASON) & ev["week"].eq(LIVE_WEEK)]
    for _, e in lev.iterrows():
        grp = [k for k, v in GROUPS.items() if v is not None and e["position"] in v]
        col = f"x_{grp[0] if grp else 'all'}_g12"
        coef = cells[col]["full_term_coefficients"][0]
        home_sign = 1 if e["home"] else -1
        live_rows.append(
            {
                "game_id": e["game_id"],
                "team": e["team"],
                "player": e["player"],
                "position": e["position"],
                "group": grp[0] if grp else "other",
                "game_since_arrival": int(e["k"]),
                "dated": str(e["tx_ts"]),
                "term_cell": col,
                "full_fit_coef": coef,
                "home_logit_contribution_sign": int(np.sign(coef * home_sign)),
                "favours": "arrival team" if coef > 0 else "opponent",
            }
        )
    res["live_week"] = {"season": LIVE_SEASON, "week": LIVE_WEEK, "rows": live_rows,
                        "games_on_slate": len(wk)}
    (OUT / "dated_results.json").write_text(json.dumps(res, indent=1, default=str))
    batch = []
    for col, r in cells.items():
        a, ll = r["acc"], r["ll"]
        batch.append(
            {
                "name": f"per11_dated_{col[2:]}_served",
                "description": (
                    f"Date-honest arrival term {col[2:]} (trade or waiver arrival of a prior-team starter, "
                    "dated by article publication or earliest sitemap lastmod, first games whose Tuesday-noon "
                    "prediction time follows the date), signed home minus away, added as a free slope to the "
                    "served out-of-season logit, leave-one-season-out 2020-2025 opener games."
                ),
                "effect": round(a["effect"], 6),
                "effect_units": "accuracy_points",
                "interval_low": round(a["low"], 6),
                "interval_high": round(a["high"], 6),
                "probability_positive": round(a["pp"], 4),
                "sample_games": int(r["n"]),
                "sample_blocks": E.LAST_SEASON - E.FIRST_SEASON + 1,
                "season_start": E.FIRST_SEASON,
                "season_end": E.LAST_SEASON,
                "league": "nfl",
                "classification": "unresolved_below_power",
                "classification_evidence": (
                    f"Matched null ({E.NULL_DRAWS} within-season shuffles) percentile {a['null_percentile']:.1f}; "
                    f"planted detectable size at 80 percent power: {E.det_text(r['detectable']['acc'])}. "
                    "One-term add-ons cannot see effects under about 1.5 accuracy points. No refuted mechanism "
                    "and no positive control at this size."
                ),
                "notes": (
                    f"scripts/per11_arrival_dates.py, artifacts/per11_arrival_rampup/dated_results.json. "
                    f"Flagged games {r['flagged_games']}; served record on flagged "
                    f"{r['served_record_on_flagged'][0]}-{r['served_record_on_flagged'][1]}; arrival team cover "
                    f"{r['arrival_team_cover_record'][0]}-{r['arrival_team_cover_record'][1]}; changed sides "
                    f"{r['arm_right']}-{r['ref_right']}. Log loss {ll['effect']:+.6f} P+ {ll['pp']:.3f} "
                    f"(null pct {ll['null_percentile']:.1f}); out minus in {r['ll_gap_out_minus_in']:+.5f}; "
                    f"fold coefficients {r['fold_term_coefficients']}. Looks {looks} (4 groups x 2 windows)."
                ),
                "family": FAMILY,
                "category": "onfield",
                "source": LANE,
                "batch": "per11_arrival_rampup_u2",
            }
        )
    (OUT / "dated_batch.json").write_text(json.dumps({"cells": batch}, indent=1))
    print(json.dumps(res["live_week"], indent=1, default=str))


if __name__ == "__main__":
    main()
