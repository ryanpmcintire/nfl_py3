import argparse
import glob
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("scripts", "src"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

import live_kickoff  # noqa: E402

OUT = ROOT / "artifacts" / "forward_terms"
SEASON = 2026
ET = "America/New_York"
PFR_URL = "https://github.com/nflverse/nflverse-data/releases/download/pfr_advstats/advstats_week_def_{s}.csv"
PFR_ROOT = ROOT / "data" / "raw" / "pfr_advstats"
MOS_ROOT = ROOT / "data" / "raw" / "forecast_archive" / "mos_bulletins" / "MEX"
STATION_MAP = ROOT / "registry" / "reference" / "stadium_station_map.csv"
MARKET_ROOT = ROOT / "data" / "market" / "raw"
SHARP_CUTOFF_ET = (12, 30)
TERMS = (
    *("sky03", "sky03_separated", "sky04", "env02", "sky01", "sky01snap", "surface"),
    "mod35_cold_exposure",
    "mod41_residual_load",
    "mod41_residual_load_served",
    "mod41_avgwin_served",
)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def now_utc():
    return datetime.now(UTC)


def stamp(t):
    return t.strftime("%Y%m%dT%H%M%SZ")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    return np.log(p / (1.0 - p))


def resolve_week(arg):
    if arg != "auto":
        return int(arg)
    weeks = [
        int(Path(p).name.split("-")[2])
        for p in glob.glob(str(ROOT / "artifacts" / "margin_predictions" / f"{SEASON}-week-*"))
    ]
    if not weeks:
        raise SystemExit("no served card directory for the season")
    return max(weeks)


def dry_card(week):
    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    sc, _ = load_snapshot(latest_snapshot(ROOT / "data" / "raw"))
    g = sc[(sc.season == SEASON) & (sc.week == week) & (sc.game_type == "REG")].copy()
    local = pd.to_datetime(g.gameday + " " + g.gametime.fillna("13:00"))
    g["kickoff_utc"] = local.dt.tz_localize(ET).dt.tz_convert("UTC")
    g["base_home_probability"] = 0.5
    cols = ["game_id", "home_team", "away_team", "kickoff_utc", "base_home_probability", "gameday"]
    return g[cols].reset_index(drop=True), {"card_directory": None}


def base_card(week):
    dirs = sorted(
        glob.glob(str(ROOT / "artifacts" / "margin_predictions" / f"{SEASON}-week-{week:02d}-*"))
    )
    if not dirs:
        raise SystemExit(
            f"MISSING INPUT: served card for week {week} (artifacts/margin_predictions/{SEASON}-week-{week:02d}-*) not produced yet; run after weekly_lock"  # noqa: E501
        )
    d = Path(dirs[-1])
    pc = pd.read_csv(
        d / "pool_card.csv",
        usecols=["game_id", "home_team", "away_team", "pool_side", "pick_probability"],
    )
    pr = pd.read_csv(
        d / "predictions.csv", usecols=["game_id", "kickoff", "gameday"]
    ).drop_duplicates("game_id")
    g = pc.merge(pr, on="game_id", how="left")
    g["base_home_probability"] = np.where(
        g.pool_side.str.upper() == "HOME", g.pick_probability, 1.0 - g.pick_probability
    )
    g["kickoff_utc"] = pd.to_datetime(g.kickoff, utc=True)
    meta = {
        "card_directory": str(d.relative_to(ROOT)),
        "pool_card_sha256": sha_file(d / "pool_card.csv"),
        "predictions_sha256": sha_file(d / "predictions.csv"),
        "card_stamp": d.name.rsplit("-", 1)[-1],
        "probability_column": "pool_card.pick_probability converted to home side",
    }
    return g.reset_index(drop=True), meta


def fetch_pfr_2026():
    t = now_utc()
    d = PFR_ROOT / stamp(t)
    d.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(PFR_URL.format(s=SEASON), timeout=120).read()
    (d / f"advstats_week_def_{SEASON}.csv").write_bytes(raw)
    man = {
        "source": PFR_URL.format(s=SEASON),
        "fetched_utc": stamp(t),
        "files": [
            {
                "file": f"advstats_week_def_{SEASON}.csv",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
    }
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    return d


def sky03_frozen():
    path = OUT / "sky03" / "frozen_coefficients.json"
    if path.exists():
        return json.loads(path.read_text())
    import sky03_def_traits as d
    import sky03_matchups as m

    t, _ = d.load()
    rr = d.build(t)
    c = m.fit_c(rr, "zBc", "rB", "n_ns")
    term_path = d.OUTF / "term.json"
    term = json.loads(term_path.read_text())
    assert term["candidate"] == "B_oN_x_completions_per_target_allowed"
    body = {
        "term": "sky03_oN_x_completions_per_target_allowed",
        "c_epa_per_unit_product": c,
        "b_logit_per_point": term["pooled_coefficient"],
        "c_fit_rows": len(rr),
        "c_fit_seasons": [int(rr.season.min()), int(rr.season.max())],
        "b_fit_seasons": list(map(int, m.EVAL_SEASONS)),
        "b_source_file": str(term_path.relative_to(ROOT)),
        "b_source_sha256": sha_file(term_path),
        "no_2026_outcomes": True,
    }
    body["coefficient_hash"] = sha_obj(
        {k: body[k] for k in ("c_epa_per_unit_product", "b_logit_per_point")}
    )
    body["created_at_utc"] = now_utc().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1))
    return body


def pbp_2026_games(through):
    import sky03_matchups as m

    f = sorted(glob.glob(m.PBP_GLOB.format(s=SEASON)))[-1]
    cols = [
        "game_id",
        "season_type",
        "week",
        "posteam",
        "defteam",
        "qb_dropback",
        "qb_kneel",
        "qb_spike",
        "sack",
        "qb_hit",
        "epa",
    ]
    x = pd.read_parquet(f, columns=cols)
    x = x[
        (x.season_type == "REG")
        & (x.qb_dropback == 1)
        & (x.qb_kneel != 1)
        & (x.qb_spike != 1)
        & x.epa.notna()
        & x.posteam.notna()
    ].copy()
    pbp_max = int(x.week.max())
    x = x[x.week <= through].copy()
    x["season"] = SEASON
    x["press"] = ((x.sack == 1) | (x.qb_hit == 1)).astype(float)
    x["ns"] = (x.sack != 1).astype(float)
    x["epa_ns_v"] = x.epa * x.ns
    g = x.groupby(["game_id", "season", "week", "posteam", "defteam"], as_index=False).agg(
        n_db=("epa", "size"),
        n_ns=("ns", "sum"),
        press_s=("press", "sum"),
        epa_s=("epa", "sum"),
        epa_ns_s=("epa_ns_v", "sum"),
    )
    g = g.rename(columns={"posteam": "off", "defteam": "deff"})
    g["press"] = g.press_s / g.n_db
    g["epa"] = g.epa_s / g.n_db
    g["epa_ns"] = g.epa_ns_s / g.n_ns.clip(lower=1)
    g["key"] = g.season * 100 + g.week
    return g, pbp_max, f


def sky03_week(week, games, fetch):
    import sky03_def_traits as d
    import sky03_matchups as m

    fr = sky03_frozen()
    pdir = (
        fetch_pfr_2026()
        if fetch
        else Path(
            sorted(glob.glob(str(PFR_ROOT / "*" / f"advstats_week_def_{SEASON}.csv")))[-1]
        ).parent
    )
    pf = pdir / f"advstats_week_def_{SEASON}.csv"
    pdf = pd.read_csv(pf)
    pdf = pdf[pdf.game_type == "REG"]
    pfr_max = int(pdf.week.max())
    through = min(pfr_max, week - 1)
    g26, pbp_max, pbp_file = pbp_2026_games(through)
    through = min(through, pbp_max)
    g26 = g26[g26.week <= through]
    pdf = pdf[pdf.week <= through]
    pg = (
        pdf.groupby(["game_id", "team"], as_index=False)
        .agg(
            tg2=("def_targets", "sum"),
            cmp=("def_completions_allowed", "sum"),
            yds=("def_yards_allowed", "sum"),
            hurried=("def_times_hurried", "sum"),
            pres=("def_pressures", "sum"),
        )
        .rename(columns={"team": "deff"})
    )
    t26 = g26.merge(pg, on=["game_id", "deff"], how="left")
    t26 = t26.dropna(subset=["tg2"]).copy()
    t26["hur"] = (t26.press_s + t26.hurried) / t26.n_db
    t26["prd"] = t26.pres / t26.n_db
    t26["ypt"] = (t26.yds / t26.tg2.clip(lower=1)).where(t26.tg2 > 0, 0.0)
    t26["cpt"] = (t26["cmp"] / t26.tg2.clip(lower=1)).where(t26.tg2 > 0, 0.0)
    th, _ = d.load()
    th = pd.concat([th, t26], ignore_index=True)
    base_traits = dict(m.TRAITS)
    m.TRAITS.update(d.NEW)
    try:
        k = m.variance_components(th, SEASON)
        tr, dbm, share = m.as_of_traits(th, SEASON, week, k)
        rows = []
        for r in games.itertuples():
            for off, deff, sign in (
                (r.home_team, r.away_team, 1.0),
                (r.away_team, r.home_team, -1.0),
            ):
                rows.append({"game_id": r.game_id, "off": off, "deff": deff, "sign": sign})
        cur = pd.DataFrame(rows)
        for name, (ser, mu) in tr.items():
            cur[name] = cur[m.grp_of(name)].map(ser).fillna(mu) - mu
        cur["db_exp"] = cur.off.map(dbm).fillna(dbm.mean())
    finally:
        m.TRAITS.clear()
        m.TRAITS.update(base_traits)
    cur["z"] = cur.oN * cur.dC
    cur["pts"] = cur.db_exp * fr["c_epa_per_unit_product"] * cur.z * share
    x = (cur.sign * cur.pts).groupby(cur.game_id).sum()
    timestamps = {
        "pfr_def_file": str(pf.relative_to(ROOT)),
        "pfr_def_file_sha256": sha_file(pf),
        "pfr_def_max_week_on_disk": pfr_max,
        "pbp_file": str(Path(pbp_file).relative_to(ROOT)),
        "pbp_file_sha256": sha_file(pbp_file),
        "pbp_max_week_on_disk": pbp_max,
        "trait_data_through_week": through,
        "weeks_lagged_vs_week_minus_1": int(week - 1 - through),
        "prior_season_traits": SEASON - 1,
        "pfr_fetch_utc": pdir.name,
    }
    return x, fr, timestamps


def sky03_separated_frozen():
    path = OUT / "sky03_separated" / "frozen_coefficients.json"
    body = json.loads(path.read_text())
    assert body["no_2026_outcomes"] is True
    keys = ("c_A_epa_per_unit_product", "c_B_epa_per_unit_product", "b_logit_per_point")
    assert body["coefficient_hash"] == sha_obj({k: body[k] for k in keys})
    return body


def sky03_separated_week(week, games):
    import sky03_matchup as sm
    import sky03_matchups as m

    fr = sky03_separated_frozen()
    g26, pbp_max, pbp_file = pbp_2026_games(week - 1)
    through = min(week - 1, pbp_max)
    g26 = g26[g26.week <= through]
    assert g26.week.max() < week
    wp = pd.read_parquet(
        pbp_file,
        columns=[
            *("game_id", "season_type", "posteam", "qb_dropback", "qb_kneel", "qb_spike"),
            *("epa", "wp", "week"),
        ],
    )
    wp = wp[
        (wp.season_type == "REG")
        & (wp.qb_dropback == 1)
        & (wp.qb_kneel != 1)
        & (wp.qb_spike != 1)
        & wp.epa.notna()
        & wp.posteam.notna()
        & wp.wp.notna()
        & (wp.week <= through)
    ]
    sc = wp.groupby(["game_id", "posteam"], as_index=False).wp.mean()
    sc = sc.rename(columns={"posteam": "off", "wp": "sc"})
    g26 = g26.merge(sc, on=["game_id", "off"], how="left")
    g26["sc"] = g26.sc.fillna(0.5)
    hist = m.load_games().merge(sm.load_script(), on=["game_id", "off"], how="left")
    hist["sc"] = hist.sc.fillna(0.5)
    hist = hist[hist.season < SEASON]
    rows = []
    for r in games.itertuples():
        for off, deff, sign in (
            (r.home_team, r.away_team, 1.0),
            (r.away_team, r.home_team, -1.0),
        ):
            rows.append(
                {
                    "game_id": r.game_id,
                    "season": SEASON,
                    "week": week,
                    "key": SEASON * 100 + week,
                    "off": off,
                    "deff": deff,
                    "sign": sign,
                }
            )
    cur = pd.DataFrame(rows)
    t = pd.concat([hist, g26, cur.drop(columns=["sign", "game_id"])], ignore_index=True)
    t.loc[t.game_id.isna(), "game_id"] = "pending"
    t["sc"] = t.sc.fillna(0.5)
    rr = sm.build_rows(t, (SEASON,))
    rr = rr[(rr.week == week) & (rr.game_id == "pending")]
    rr = rr.drop(columns=["game_id"]).merge(
        cur[["game_id", "off", "deff", "sign"]], on=["off", "deff"], how="inner"
    )
    rr["pts"] = rr.db_exp * (
        fr["c_A_epa_per_unit_product"] * rr.zA
        + fr["c_B_epa_per_unit_product"] * rr.zB * rr.ns_share
    )
    x = (rr.sign * rr.pts).groupby(rr.game_id).sum()
    timestamps = {
        "pbp_file": str(Path(pbp_file).relative_to(ROOT)),
        "pbp_file_sha256": sha_file(pbp_file),
        "pbp_max_week_on_disk": pbp_max,
        "trait_data_through_week": through,
        "weeks_lagged_vs_week_minus_1": int(week - 1 - through),
        "prior_season_traits": SEASON - 1,
        "script_column": "mean win probability on the team's dropbacks, prior games only",
    }
    return x, fr, timestamps


def sky04_frozen():
    path = OUT / "sky04" / "frozen_coefficients.json"
    if path.exists():
        return json.loads(path.read_text())
    import sky04_lead_lag as L
    import sky04_sharpness as S

    s = L.book_series()
    s = s[s.nflverse_game_id.str[:4].isin(["2023", "2024", "2025"])]
    part = s.sort_values(["nflverse_game_id", "bookmaker_key", "observed_at_utc"]).copy()
    part["chg"] = (
        part.groupby(["nflverse_game_id", "bookmaker_key"]).home_spread_line.diff().fillna(0) != 0
    )
    pc = part.groupby("bookmaker_key").chg.sum().sort_values(ascending=False)
    books = [b for b in pc.index if pc[b] >= L.MIN_PARTICIPATION]
    L.BOOKS_REF = books
    mats = L.game_mats(s, books)
    g = pd.read_parquet(L.PERGAME)
    g = g[(g.game_type == "REG") & g.season.isin(S.SEASONS)]
    sub = g[g.margin_vs_open != 0].copy().reset_index(drop=True)
    X = S.noon_lines(mats, sub.game_id.values, sub.gameday.values)
    has = (~np.isnan(X)).sum(axis=1) >= 3
    err2 = (sub.result.values.astype(float)[:, None] - X) ** 2
    w = np.zeros(len(books))
    for j in range(len(books)):
        mk = has & ~np.isnan(X[:, j])
        w[j] = 1 / err2[mk, j].mean() if mk.sum() > 50 else 0
    w = w / w.sum()
    sp = L.OUT / "sharpness.json"
    sj = json.loads(sp.read_text())
    assert sj["books"] == books
    body = {
        "term": "sky04_sharpness_weighted_mean_minus_median",
        "books": books,
        "weights": {b: float(w[j]) for j, b in enumerate(books)},
        "b_logit_per_point": sj["pooled_coefficient"],
        "weight_fit_seasons": list(S.SEASONS),
        "weight_fit_games": int(has.sum()),
        "b_source_file": str(sp.relative_to(ROOT)),
        "b_source_sha256": sha_file(sp),
        "no_2026_outcomes": True,
    }
    body["coefficient_hash"] = sha_obj({k: body[k] for k in ("weights", "b_logit_per_point")})
    body["created_at_utc"] = now_utc().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1))
    return body


def sky04_week(week, games, now):
    fr = sky04_frozen()
    books = fr["books"]
    w = np.array([fr["weights"][b] for b in books])
    cols = ["observed_at_utc", "nflverse_game_id", "bookmaker_key", "market", "home_spread_line"]
    frames = []
    used = []
    for d in sorted(glob.glob(str(MARKET_ROOT / f"{SEASON}*"))):
        f = Path(d) / "quotes.parquet"
        if Path(d).name.endswith("futures") or not f.exists():
            continue
        q = pd.read_parquet(f, columns=cols)
        q = q[
            (q.market == "spreads")
            & q.nflverse_game_id.fillna("").str.startswith(f"{SEASON}_{week:02d}_")
        ]
        if len(q):
            frames.append(q)
            used.append(Path(d).name)
    x = {}
    stamps = {}
    if frames:
        q = pd.concat(frames, ignore_index=True)
        q["observed_at_utc"] = pd.to_datetime(q.observed_at_utc, utc=True)
        q = q.groupby(
            ["nflverse_game_id", "bookmaker_key", "observed_at_utc"], as_index=False
        ).home_spread_line.median()
        for r in games.itertuples():
            gd = pd.Timestamp(r.gameday)
            tue = (
                (gd - pd.Timedelta(days=(gd.weekday() - 1) % 7))
                .replace(hour=SHARP_CUTOFF_ET[0], minute=SHARP_CUTOFF_ET[1])
                .tz_localize(ET)
                .tz_convert("UTC")
            )
            cutoff = min(tue, pd.Timestamp(now))
            week_open = (
                (gd - pd.Timedelta(days=(gd.weekday() - 1) % 7)).tz_localize(ET).tz_convert("UTC")
            )
            qg = q[
                (q.nflverse_game_id == r.game_id)
                & (q.observed_at_utc <= cutoff)
                & (q.observed_at_utc >= week_open)
            ]
            if qg.empty:
                continue
            last = qg.observed_at_utc.max()
            row = qg[qg.observed_at_utc == last].set_index("bookmaker_key").home_spread_line
            v = np.array([row.get(b, np.nan) for b in books], dtype=float)
            stamps[r.game_id] = {
                "observation_utc": last.isoformat(),
                "cutoff_utc": cutoff.isoformat(),
                "books_present": [b for b, a in zip(books, v) if not np.isnan(a)],  # noqa: B905
            }
            if (~np.isnan(v)).sum() < 3:
                continue
            ww = np.where(np.isnan(v), 0.0, w)
            x[r.game_id] = float((np.nan_to_num(v) * ww).sum() / ww.sum() - np.nanmedian(v))
    ts = {
        "market_directories_scanned": {
            "count": len(used),
            "first": used[0] if used else None,
            "last": used[-1] if used else None,
        },
        "per_game": stamps,
        "cutoff_rule": "last observation within the game week (on or after Tuesday 00:00 ET) and at or before min(run time, Tuesday 12:30 ET); fewer than 3 frozen-book lines gives no term value",  # noqa: E501
    }
    return pd.Series(x, dtype=float), fr, ts


def env02_frozen():
    path = OUT / "env02" / "frozen_coefficients.json"
    if path.exists():
        return json.loads(path.read_text())
    import env02_roof_fitted_term as e
    import roof_state_screen as rs
    from line_move_yardstick_paired_eval import _design, _standardise
    from sklearn.linear_model import LogisticRegression

    from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit

    table = rs.build_prediction_table(
        forecast_archive=e.AWARE / "forecasts.parquet", forecast_manifest=e.AWARE / "manifest.json"
    )
    table = table.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(rs._venue_features(table), table["is_open"].astype(int).to_numpy())
    flags = e.roof_open_probability(e.AWARE / "forecasts.parquet", e.AWARE / "manifest.json")
    pop = pd.read_parquet(e.EXTENDED_POPULATION_PATH)
    pop["game_id"] = pop["game_id"].astype(str)
    flags["game_id"] = flags["game_id"].astype(str)
    pop = pop.merge(flags, on="game_id", how="left")
    pop[e.NEW_TERM] = pop[e.NEW_TERM].fillna(0.0)
    feats = (*FIT_FEATURES, e.NEW_TERM)
    means, stds = _standardise(pop, feats)
    beta = _fit_logit(
        _design(pop, feats, means, stds), pop["home_covered"].astype(float).to_numpy(), FIT_RIDGE
    )
    roof_beta = float(beta[len(feats)] / stds[e.NEW_TERM])
    body = {
        "term": "env02_roof_open_probability",
        "open_model_coef": model.coef_[0].tolist(),
        "open_model_intercept": float(model.intercept_[0]),
        "open_model_C": 1.0,
        "open_model_train_games": len(table),
        "open_model_seasons": [int(table.season.min()), int(table.season.max())],
        "roof_beta_natural": roof_beta,
        "roof_term_train_mean": float(pop[e.NEW_TERM].mean()),
        "five_term_fit_games": len(pop),
        "forecast_archive": str(e.AWARE.relative_to(ROOT)),
        "forecast_archive_manifest_sha256": sha_file(e.AWARE / "manifest.json"),
        "no_2026_outcomes": True,
    }
    body["coefficient_hash"] = sha_obj(
        {
            k: body[k]
            for k in (
                "open_model_coef",
                "open_model_intercept",
                "roof_beta_natural",
                "roof_term_train_mean",
            )
        }
    )
    body["created_at_utc"] = now_utc().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1))
    return body


def env02_week(week, games, now):
    import roof_state_screen as rs
    from ingest_forecast_archive import nearest_row

    fr = env02_frozen()
    smap = pd.read_csv(STATION_MAP)
    smap = smap[smap.mappable.astype(str).str.lower() == "true"]
    st = {str(r.teams): r.icao_station for r in smap.itertuples()}
    x = {}
    ts = {"per_game": {}, "missing": []}
    coef = np.array(fr["open_model_coef"])
    for r in games.itertuples():
        if r.home_team not in rs.RETRACT_TEAMS:
            x[r.game_id] = 0.0
            continue
        station = st.get(r.home_team)
        files = sorted(glob.glob(str(MOS_ROOT / str(station) / "*.json"))) if station else []
        files = [
            f
            for f in files
            if pd.Timestamp(Path(f).stem.replace("T", " ").replace("Z", "") + "+00:00") <= now
        ]
        if not files:
            ts["missing"].append({"game_id": r.game_id, "station": station})
            continue
        b = json.loads(Path(files[-1]).read_text())
        rows = [dict(z, ftime_utc=z["ftime"].replace(" ", "T")) for z in b["data"]]
        row = nearest_row(rows, r.kickoff_utc)
        prow = nearest_row(
            [z for z in rows if z.get("p06") is not None]
            or [z for z in rows if z.get("p12") is not None],
            r.kickoff_utc,
        )
        precip = prow.get("p06") if prow.get("p06") is not None else prow.get("p12")
        if row is None or row.get("tmp") is None or precip is None:
            ts["missing"].append(
                {
                    "game_id": r.game_id,
                    "station": station,
                    "reason": "no temp or precip in bulletin",
                }
            )
            continue
        temp = float(row["tmp"])
        feats = [temp, float(precip)]
        feats += [temp if r.home_team == v else 0.0 for v in rs.RETRACT_TEAMS]
        feats += [1.0 if r.home_team == v else 0.0 for v in rs.RETRACT_TEAMS]
        z = float(np.dot(coef, np.array(feats)) + fr["open_model_intercept"])
        x[r.game_id] = float(sigmoid(z))
        ts["per_game"][r.game_id] = {
            "bulletin_file": str(Path(files[-1]).relative_to(ROOT)),
            "bulletin_sha256": sha_file(files[-1]),
            "bulletin_cached_at_utc": b.get("cached_at_utc"),
            "forecast_valid_utc": row["ftime_utc"],
            "forecast_temp_f": temp,
            "forecast_precip_prob_pct": float(precip),
        }
    return pd.Series(x, dtype=float), fr, ts


def sky01_frozen(term):
    path = OUT / term / "frozen_coefficients.json"
    if path.exists():
        return json.loads(path.read_text())
    if term == "sky01":
        src = ROOT / "artifacts" / "sky01" / "fitted_term.json"
        b = json.loads(src.read_text())["pooled_coefficient"]
        name = "sky01_beat_out_count_differential"
        seasons = [2024, 2025]
    else:
        src = ROOT / "artifacts" / "sky01" / "drift_weighted.json"
        b = json.loads(src.read_text())["weighted_term"]["pooled_coefficient"]
        name = "sky01_snap_share_weighted_beat_out_differential"
        seasons = [2024, 2025]
    body = {
        "term": name,
        "b_logit_per_unit": b,
        "b_logit_per_point": b,
        "b_fit_seasons": seasons,
        "b_source_file": str(src.relative_to(ROOT)),
        "b_source_sha256": sha_file(src),
        "no_2026_outcomes": True,
    }
    body["coefficient_hash"] = sha_obj({"b_logit_per_point": b})
    body["created_at_utc"] = now_utc().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1))
    return body


def newest(pattern):
    hits = sorted(glob.glob(str(ROOT / pattern)))
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[-1])


def sky01_week(week, games, now, term):
    import sky01_drift_weighted as dw
    import sky01_news_availability as base

    fr = sky01_frozen(term)
    sched_file = newest("data/raw/*/schedules.parquet")
    base.SCHED = sched_file
    s = base.schedule()
    tg = base.team_games(s)
    p = base.beat_rows(tg)
    p = p[p.created_at <= pd.Timestamp(now)]
    g = games[["game_id", "home_team", "away_team"]].copy()
    g["home_team"] = g.home_team.map(base.alias)
    g["away_team"] = g.away_team.map(base.alias)
    snap_info = {}
    if term == "sky01":
        v = base.feature(p, g)
    else:
        snap_dir = newest("data/players/raw/*/snap_counts.parquet").parent
        dw.SNAPS = snap_dir / "snap_counts.parquet"
        dw.ROSTERS = snap_dir / "weekly_rosters.parquet"
        ps, sm = dw.prior_share()
        v, src, cov = dw.weighted_feature(p, g, ps, sm)
        snap_info = {
            "snap_counts_file": str(dw.SNAPS.relative_to(ROOT)),
            "weekly_rosters_file": str(dw.ROSTERS.relative_to(ROOT)),
            "weight_source_counts": {k: int(n) for k, n in src.items()},
            "weight_positive_share": cov,
            "weight_rule": "in-season mean snap share through the prior week, else prior-season mean share, else 0",  # noqa: E501
        }
    x = pd.Series(np.asarray(v, dtype=float), index=games.game_id.values)
    posts = ROOT / "data" / "processed" / "beat_reporter_posts.parquet"
    pre = p[p.opener_ts.isin(g_openers(s, games)) & (p.created_at < p.opener_ts)]
    ts = {
        "beat_posts_file": str(posts.relative_to(ROOT)),
        "beat_posts_file_sha256": sha_file(posts),
        "beat_posts_max_created_at_utc": str(p.created_at.max()),
        "schedule_file": str(sched_file.relative_to(ROOT)),
        "post_rule": "out-class beat posts with created_at before Tuesday 12:00 ET opener of the game week and at or after the team's previous kickoff; created_at also at or before run time",  # noqa: E501
        "posts_counted_for_week": len(pre),
        **snap_info,
    }
    return x, fr, ts


def g_openers(s, games):
    return s[s.game_id.isin(games.game_id)].opener_ts.unique()


def surface_frozen():
    path = OUT / "surface" / "frozen_coefficients.json"
    if path.exists():
        return json.loads(path.read_text())
    import surface_switch_fitted_term as sw
    from line_move_yardstick_paired_eval import _design, _standardise

    from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit
    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    snap = latest_snapshot(ROOT / "data" / "raw")
    schedules, _ = load_snapshot(snap)
    flags = sw.point_in_time_surface_switch_flags(schedules)
    pop = pd.read_parquet(sw.EXTENDED_POPULATION_PATH)
    pop["game_id"] = pop["game_id"].astype(str)
    pop = pop.merge(flags, on="game_id", how="left")
    pop[sw.NEW_TERM] = pop[sw.NEW_TERM].fillna(0.0)
    assert int(pop.season.max()) < SEASON
    feats = (*FIT_FEATURES, sw.NEW_TERM)
    means, stds = _standardise(pop, feats)
    beta = _fit_logit(
        _design(pop, feats, means, stds), pop["home_covered"].astype(float).to_numpy(), FIT_RIDGE
    )
    b = float(beta[len(feats)] / stds[sw.NEW_TERM])
    body = {
        "term": "surface_switch_flag_pit",
        "b_logit_per_point": b,
        "five_term_fit_games": len(pop),
        "fit_seasons": [int(pop.season.min()), int(pop.season.max())],
        "flag_games_in_fit": int(pop[sw.NEW_TERM].sum()),
        "fit_population": str(sw.EXTENDED_POPULATION_PATH.relative_to(ROOT)),
        "fit_population_sha256": sha_file(sw.EXTENDED_POPULATION_PATH),
        "no_2026_outcomes": True,
    }
    body["coefficient_hash"] = sha_obj({"b_logit_per_point": b})
    body["created_at_utc"] = now_utc().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1))
    return body


def surface_week(week, games, now):
    import surface_switch_fitted_term as sw

    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    fr = surface_frozen()
    snap = latest_snapshot(ROOT / "data" / "raw")
    schedules, _ = load_snapshot(snap)
    flags = sw.point_in_time_surface_switch_flags(schedules)
    x = flags.set_index("game_id")[sw.NEW_TERM].reindex(games.game_id.values)
    ts = {
        "schedule_snapshot": snap.snapshot_id,
        "schedule_file_sha256": sha_file(snap.schedules_path),
        "rule": "flag is 1 when the game's venue surface is turf and the away team's modal home surface from games strictly before the game date is grass; venue surface comes from the published schedule, history from completed earlier games",  # noqa: E501
        "flag_games": [gid for gid, v in x.items() if v == 1.0],
        "games_without_flag_row": [gid for gid, v in x.items() if pd.isna(v)],
    }
    return x, fr, ts


def mod35_frozen():
    path = OUT / "mod35_cold_exposure" / "frozen_coefficients.json"
    body = json.loads(path.read_text())
    assert body["no_2026_outcomes"] is True
    keys = ("raw_coefficients_per_unit", "b_logit_per_point", "intercept")
    assert body["coefficient_hash"] == sha_obj({k: body[k] for k in keys})
    return body


def mod35_week(week, games, now):
    import mod35_apparent_temp as m35
    from ingest_forecast_archive import KNOTS_TO_MPH, nearest_row

    from nfl_ats.snapshots import latest_snapshot, load_snapshot

    fr = mod35_frozen()
    assert sha_file(m35.FC) == fr["recorded_weather_sha256"]
    fc = pd.read_parquet(m35.FC)
    rec = fc.loc[
        (fc.fetch_status == "ok")
        & fc.roof.isin(m35.OUTDOOR)
        & fc.actual_temp_f.notna()
        & fc.actual_wind_mph.notna()
    ].copy()
    rec["home"] = rec.game_id.str.split("_").str[3]
    rec["app"] = m35.apparent(rec.actual_temp_f.to_numpy(), rec.actual_wind_mph.to_numpy())
    assert int(rec.season.max()) < SEASON
    k = fr["normal_half_width_weeks_for_2026"]
    snap = latest_snapshot(ROOT / "data" / "raw")
    schedules, _ = load_snapshot(snap)
    roof = schedules[(schedules.season == SEASON) & (schedules.week == week)].set_index("game_id")[
        "roof"
    ]
    smap = pd.read_csv(STATION_MAP)
    smap = smap[smap.mappable.astype(str).str.lower() == "true"]
    st = {str(r.teams): r.icao_station for r in smap.itertuples()}
    x = {}
    ts = {
        "schedule_snapshot": snap.snapshot_id,
        "forecast_source": "data/raw/forecast_archive/mos_bulletins/MEX, latest bulletin cached at or before the run time; history was built from GFS MOS",  # noqa: E501
        "per_game": {},
        "not_outdoor": [],
        "missing": [],
    }
    for r in games.itertuples():
        if roof.get(r.game_id) not in m35.OUTDOOR:
            ts["not_outdoor"].append({"game_id": r.game_id, "roof": roof.get(r.game_id)})
            continue
        station = st.get(r.home_team)
        files = sorted(glob.glob(str(MOS_ROOT / str(station) / "*.json"))) if station else []
        usable = []
        for f in files:
            b = json.loads(Path(f).read_text())
            cached = pd.Timestamp(b["cached_at_utc"])
            if cached <= now and cached <= r.kickoff_utc:
                usable.append((f, b))
        if not usable:
            ts["missing"].append({"game_id": r.game_id, "station": station})
            continue
        f, b = usable[-1]
        rows = [dict(z, ftime_utc=z["ftime"].replace(" ", "T")) for z in b["data"]]
        row = nearest_row(rows, r.kickoff_utc)
        if row is None or row.get("tmp") is None or row.get("wsp") is None:
            ts["missing"].append({"game_id": r.game_id, "reason": "no temp or wind in bulletin"})
            continue
        wind = float(row["wsp"]) * KNOTS_TO_MPH
        fc_at = float(m35.apparent([float(row["tmp"])], [wind])[0])
        nrm_a = m35.normals(rec, [(r.away_team, SEASON, week)], k)[0]
        nrm_h = m35.normals(rec, [(r.home_team, SEASON, week)], k)[0]
        if np.isnan(nrm_a) or np.isnan(nrm_h):
            ts["missing"].append({"game_id": r.game_id, "reason": "no outdoor-home normal"})
            continue
        gap = float(nrm_a - fc_at)
        x[r.game_id] = max(-gap, 0.0)
        ts["per_game"][r.game_id] = {
            "bulletin_file": str(Path(f).relative_to(ROOT)),
            "bulletin_sha256": sha_file(f),
            "bulletin_cached_at_utc": b["cached_at_utc"],
            "forecast_valid_utc": row["ftime_utc"],
            "forecast_temp_f": float(row["tmp"]),
            "forecast_wind_mph": wind,
            "forecast_apparent_f": fc_at,
            "visitor_normal_f": float(nrm_a),
            "home_normal_f": float(nrm_h),
            "gap": gap,
            "cold_flag": float(gap >= fr["cold_flag_gap_threshold_f"]),
        }
    return pd.Series(x, dtype=float), fr, ts


def mod41_frozen(term="mod41_residual_load"):
    path = OUT / term / "frozen_coefficients.json"
    body = json.loads(path.read_text())
    assert body["no_2026_outcomes"] is True
    keys = ("b_logit_per_unit", "intercept", "decomposition_beta_const_own_opp")
    assert body["coefficient_hash"] == sha_obj({k: body[k] for k in keys})
    body["b_logit_per_point"] = body["b_logit_per_unit"]
    return body


def mod41_week(week, games, now, term="mod41_residual_load"):
    import mod41_freeze_residual_load as f41

    fr = mod41_frozen(term)
    pbp = f41.latest_pbp(ROOT / "data" / "pbp" / "raw")
    t = f41.season_team_games(pbp, SEASON)
    assert int(t.week.max()) < week
    t = t[t.week < week]
    st = f41.team_state(t)
    b0, b_own, b_opp = fr["decomposition_beta_const_own_opp"]
    x = {}
    ts = {
        "pbp_snapshot": pbp.name,
        "trait_data_through_week": int(t.week.max()),
        "per_game": {},
        "missing": [],
    }
    for r in games.itertuples():
        if r.home_team not in st.index or r.away_team not in st.index:
            ts["missing"].append({"game_id": r.game_id, "reason": "team without prior games"})
            continue
        h, a = st.loc[r.home_team], st.loc[r.away_team]
        if not (bool(h.ok) and bool(a.ok)):
            ts["missing"].append({"game_id": r.game_id, "reason": "fewer than 4 prior games"})
            continue
        s_total = float(h.load_total - a.load_total)
        s_own = float(h.own_td - a.own_td)
        s_opp = float(h.opp3 - a.opp3)
        x[r.game_id] = s_total - (b0 + b_own * s_own + b_opp * s_opp)
        ts["per_game"][r.game_id] = {
            "s_load_total": s_total,
            "s_own_tempo": s_own,
            "s_opp_tempo": s_opp,
            "residual": x[r.game_id],
        }
    return pd.Series(x, dtype=float), fr, ts


def avgwin_frozen(term="mod41_avgwin_served"):
    path = OUT / term / "frozen_coefficients.json"
    body = json.loads(path.read_text())
    assert body["no_2026_outcomes"] is True
    keys = ("b_logit_per_unit", "intercept", "decomposition_by_window", "sd_by_window")
    assert body["coefficient_hash"] == sha_obj({k: body[k] for k in keys})
    body["b_logit_per_point"] = body["b_logit_per_unit"]
    return body


def avgwin_week(week, games, now, term="mod41_avgwin_served"):
    import mod41_freeze_residual_load as f41

    fr = avgwin_frozen(term)
    pbp = f41.latest_pbp(ROOT / "data" / "pbp" / "raw")
    t = f41.season_team_games(pbp, SEASON)
    assert int(t.week.max()) < week
    t = t[t.week < week]
    g = t.groupby("team")
    n = g.cumcount() + 1
    state = pd.DataFrame({"team": t["team"], "week": t["week"], "n_played": n})
    state["own_td"] = g["run"].transform(lambda s: s.expanding().mean())
    norm = g["total"].transform(lambda s: s.expanding().mean())
    for w in fr["windows"]:
        state[f"load{w}"] = g["total"].transform(lambda s, w=w: s.rolling(w).mean()) - norm
        state[f"opp{w}"] = g["defended"].transform(lambda s, w=w: s.rolling(w).mean())
    state["ok"] = state["n_played"] >= fr["min_prior_games"]
    st = state.sort_values("week").groupby("team").tail(1).set_index("team")
    x = {}
    ts = {
        "pbp_snapshot": pbp.name,
        "trait_data_through_week": int(t.week.max()),
        "per_game": {},
        "missing": [],
    }
    for r in games.itertuples():
        if r.home_team not in st.index or r.away_team not in st.index:
            ts["missing"].append({"game_id": r.game_id, "reason": "team without prior games"})
            continue
        h, a = st.loc[r.home_team], st.loc[r.away_team]
        if not (bool(h.ok) and bool(a.ok)):
            ts["missing"].append(
                {"game_id": r.game_id, "reason": f"fewer than {fr['min_prior_games']} prior games"}
            )
            continue
        s_own = float(h.own_td - a.own_td)
        z = {}
        for w in fr["windows"]:
            b0, b_own, b_opp = fr["decomposition_by_window"][str(w)]
            s_total = float(h[f"load{w}"] - a[f"load{w}"])
            s_opp = float(h[f"opp{w}"] - a[f"opp{w}"])
            z[w] = (s_total - (b0 + b_own * s_own + b_opp * s_opp)) / fr["sd_by_window"][str(w)]
        x[r.game_id] = float(np.mean(list(z.values())))
        ts["per_game"][r.game_id] = {
            "standardized_residual_by_window": {str(w): v for w, v in z.items()},
            "average": x[r.game_id],
        }
    return pd.Series(x, dtype=float), fr, ts


def served_card_probability(week):
    pub = pd.read_parquet(ROOT / "artifacts" / "clv_ledger" / "published_picks.parquet")
    pub = pub[(pub.season == SEASON) & (pub.week == week)]
    pub = pub[pd.to_datetime(pub.published_at_utc, utc=True) <= pd.Timestamp(now_utc())]
    last = pub.sort_values("published_at_utc").groupby("game_id").tail(1)
    home = np.where(
        last.pick_team == last.home_team, last.displayed_score, 1.0 - last.displayed_score
    )
    return dict(zip(last.game_id, home, strict=True))


SERVED_BASE_TERMS = ("mod41_residual_load_served", "mod41_avgwin_served")


def combine(term, games, x, fr, week):
    out = []
    served = served_card_probability(week) if term in SERVED_BASE_TERMS else {}
    if term == "env02":
        centre = fr["roof_term_train_mean"]
        beta = fr["roof_beta_natural"]
    else:
        centre = 0.0
        beta = fr["b_logit_per_point"]
    for r in games.itertuples():
        if term in SERVED_BASE_TERMS:
            if r.game_id not in served:
                raise SystemExit(f"MISSING INPUT: no published card probability for {r.game_id}")
            r = r._replace(base_home_probability=float(served[r.game_id]))
        v = x.get(r.game_id)
        if v is None or np.isnan(v):
            out.append(
                {
                    "game_id": r.game_id,
                    "away_team": r.away_team,
                    "home_team": r.home_team,
                    "kickoff_utc": r.kickoff_utc.isoformat(),
                    "base_home_probability": float(r.base_home_probability),
                    "term_value": None,
                    "delta_logit": None,
                    "combined_home_probability": None,
                }
            )
            continue
        dl = beta * (v - centre) if term == "env02" else beta * v
        if term in SERVED_BASE_TERMS:
            base_logit = logit(r.base_home_probability)
            dl = (
                fr["intercept"]
                + fr["served_logit_coefficient"] * base_logit
                + beta * v
                - base_logit
            )
        if term == "env02" and r.home_team not in ("ARI", "ATL", "DAL", "HOU", "IND"):
            dl = 0.0
        out.append(
            {
                "game_id": r.game_id,
                "away_team": r.away_team,
                "home_team": r.home_team,
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "base_home_probability": float(r.base_home_probability),
                "term_value": float(v),
                "delta_logit": float(dl),
                "combined_home_probability": float(sigmoid(logit(r.base_home_probability) + dl)),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", default="auto")
    ap.add_argument("--term", default="all", choices=("all",) + TERMS)  # noqa: RUF005
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    week = resolve_week(a.week)
    now = now_utc()
    if a.dry and not glob.glob(
        str(ROOT / "artifacts" / "margin_predictions" / f"{SEASON}-week-{week:02d}-*")
    ):
        games, basemeta = dry_card(week)
    else:
        games, basemeta = base_card(week)
    if not a.dry:
        games["kickoff_utc"] = live_kickoff.earliest_by_game_id(games, "kickoff_utc")
        games = games[games.kickoff_utc > pd.Timestamp(now)].reset_index(drop=True)
    if games.empty:
        raise SystemExit(f"week {week}: every game has kicked off; nothing pregame to freeze")
    status = {}
    for term in TERMS if a.term == "all" else (a.term,):
        d = OUT / term
        if not a.dry:
            d.mkdir(parents=True, exist_ok=True)
        prior = [] if a.dry else sorted(glob.glob(str(d / f"{SEASON}_week{week:02d}_*.json")))
        if prior:
            status[term] = f"already frozen: {Path(prior[-1]).name}"
            continue
        try:
            if term == "sky03":
                x, fr, ts = sky03_week(week, games, not a.no_fetch)
            elif term == "sky03_separated":
                x, fr, ts = sky03_separated_week(week, games)
            elif term == "sky04":
                x, fr, ts = sky04_week(week, games, now)
            elif term == "env02":
                x, fr, ts = env02_week(week, games, now)
            elif term == "surface":
                x, fr, ts = surface_week(week, games, now)
            elif term == "mod35_cold_exposure":
                x, fr, ts = mod35_week(week, games, now)
            elif term == "mod41_avgwin_served":
                x, fr, ts = avgwin_week(week, games, now, term)
            elif term in ("mod41_residual_load", "mod41_residual_load_served"):
                x, fr, ts = mod41_week(week, games, now, term)
            else:
                x, fr, ts = sky01_week(week, games, now, term)
        except SystemExit:
            raise
        except Exception as ex:
            status[term] = f"FAILED {type(ex).__name__}: {ex}"
            continue
        rows = combine(term, games, x, fr, week)
        if a.dry:
            vals = np.array([r["term_value"] for r in rows if r["term_value"] is not None], float)
            status[term] = {
                "n_games": len(rows),
                "n_with_term": len(vals),
                "mean": float(vals.mean()) if len(vals) else None,
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                "rows_sha256": sha_obj(rows),
                "trait_data_through_week": ts.get("trait_data_through_week"),
            }
            continue
        nz = sum(1 for r in rows if r["term_value"] is not None)
        body = {
            "term": term,
            "season": SEASON,
            "week": week,
            "created_at_utc": now.isoformat(),
            "runner": "scripts/forward_terms_week.py",
            "coefficient_hash": fr["coefficient_hash"],
            "frozen_coefficients_file": f"artifacts/forward_terms/{term}/frozen_coefficients.json",
            "frozen_coefficients": {k: v for k, v in fr.items() if k not in ("created_at_utc",)},
            "frozen_created_at_utc": fr["created_at_utc"],
            "feature_timestamps": ts,
            "base": basemeta,
            "games_scored_pregame": len(rows),
            "games_with_term_value": nz,
            "games": rows,
            "scoring_rule": "score only after the week settles, against the opener, alongside the served card; never served",  # noqa: E501
        }
        f = d / f"{SEASON}_week{week:02d}_{stamp(now)}.json"
        f.write_text(json.dumps(body, indent=1, default=str))
        status[term] = f"wrote {f.name} games={len(rows)} with_term={nz}"
    print(json.dumps({"week": week, "status": status}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
