from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g55_u2_expected_snaps as u2
import g55_u3_servable_all as u3
import g55_u5a_usage_v2 as u5a
from mod57_p2c_uncorrected_error_model import (
    BASE_ALPHA,
    BASE_CONFIG,
    FULL_COLUMNS,
    MARKET_ROOT,
    SERVED_FEATURES,
    SET_NAME,
)

from nfl_ats import clv
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.features import _kickoff_utc
from nfl_ats.pbp import load_pbp_snapshot
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root
from nfl_ats.players import (
    attach_snap_player_ids,
    canonicalize_rosters,
    canonicalize_snaps,
    load_player_snapshot,
    load_player_value_snapshot,
    player_snapshot_from_root,
    player_value_snapshot_from_root,
)
from nfl_ats.published_picks import game_deadlines
from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.splash_lines import load_splash_capture, splash_decision_lines

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts/g55_s_all_v2_forward"
DRY_OUT = OUT / "dry_run"
FROZEN = OUT / "frozen.pkl"
FROZEN_JSON = OUT / "frozen.json"
SYNTH_V2 = ROOT / "artifacts/g55_u5b/servable_features.parquet"
U5B_PER_GAME = ROOT / "artifacts/g55_u5b/per_game.parquet"
FORWARD_SNAPS = DATA / "players/forward_snaps"
INACTIVES = DATA / "players/inactives"
PLAYERS_RAW = DATA / "players/raw"
PUBLISHED = ROOT / "artifacts/clv_ledger/published_picks.parquet"
SEASON = 2026
FIRST_WEEK = 12
LAST_WEEK = 18
POSTING_MINUTES = 90
STAMP = "%Y%m%dT%H%M%SZ"
METHOD = BASE_CONFIG["probability_method"]
ARMS = {"base": (), "arm": tuple(u3.COLUMNS)}
OPEN_COLUMNS = ("tue_open_home_spread", "open_home_spread", "spread_at_open", "opener_spread")
ROSTER_ON_53 = ("ACT", "INA")
SCORE_SLOT = ("tue", "09:25")
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp_time(text):
    return pd.Timestamp(datetime.strptime(text, STAMP).replace(tzinfo=UTC))


@contextmanager
def feature_set(extra):
    saved = FEATURE_SETS[SET_NAME]
    FEATURE_SETS[SET_NAME] = FULL_COLUMNS + tuple(extra)
    try:
        yield
    finally:
        FEATURE_SETS[SET_NAME] = saved


def open_column(frame):
    for column in OPEN_COLUMNS:
        if column in frame.columns:
            return column
    raise SystemExit(f"no opener column in evaluation output: {sorted(frame.columns)}")


def freeze_margin(features, stamp):
    frame = clv.regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    cutoff = frame.loc[frame["season"].eq(SEASON), "gameday"].min()
    train = frame.loc[frame["result"].notna() & frame["gameday"].lt(cutoff)]
    history = features.loc[features["season"].lt(SEASON)].copy()
    reference = pd.read_parquet(U5B_PER_GAME).set_index("game_id")
    frozen = {"models": {}, "offsets": {}, "pool": {}, "check": {}}
    for name, extra in ARMS.items():
        streams, pools = [], []
        original_rows, original_pool = clv.prior_rows_before, clv.prior_pool

        def capture_rows(stream, *args, _orig=original_rows, _keep=streams, **kwargs):
            _keep.append(stream)
            return _orig(stream, *args, **kwargs)

        def capture_pool(*args, _orig=original_pool, _keep=pools, **kwargs):
            result = _orig(*args, **kwargs)
            _keep.append(result)
            return result

        with feature_set(extra):
            model = clv.fit_margin_model(
                train,
                target="market_residual",
                model_name=BASE_CONFIG["regressor"],
                feature_profile=BASE_CONFIG["feature_profile"],
                ridge_alpha=BASE_ALPHA,
            )
            clv.prior_rows_before, clv.prior_pool = capture_rows, capture_pool
            try:
                scored = clv.opener_pick_evaluation(
                    MARKET_ROOT,
                    history,
                    active_model_config={
                        **BASE_CONFIG,
                        "ridge_alpha": BASE_ALPHA,
                        "g55_u7_freeze": f"{name}-{stamp}",
                    },
                    min_train_games=DEFAULT_MIN_TRAIN_GAMES,
                )
            finally:
                clv.prior_rows_before, clv.prior_pool = original_rows, original_pool
        if not streams or len(pools) != 1:
            raise SystemExit("evaluation did not run the walk-forward (cache hit?)")
        column = open_column(scored)
        last = scored.loc[scored["season"].eq(scored["season"].max())]
        last = last.loc[last["week"].eq(last["week"].max())]
        results = features.drop_duplicates("game_id").set_index("game_id")["result"]
        tail = pd.DataFrame(
            {
                "game_id": last["game_id"].astype(str).to_numpy(),
                "season": last["season"].astype(int).to_numpy(),
                "week": last["week"].astype(int).to_numpy(),
                "spread_line": last[column].to_numpy(float),
                "point_incumbent": last[column].to_numpy(float)
                + last["residual_at_open"].to_numpy(float),
                "result": pd.to_numeric(last["game_id"].map(results), errors="coerce").to_numpy(),
            }
        )
        stream = pd.concat([streams[-1], tail], ignore_index=True)
        frozen["models"][name] = model
        frozen["offsets"][name] = clv.fit_home_side_offsets(
            clv.prior_rows_before(stream, SEASON, 1)
        )
        frozen["pool"][name] = pools[0]
        mine = scored.loc[scored["season"].between(2020, 2025)].set_index("game_id")
        ref = reference[f"p_{name}"].reindex(mine.index)
        frozen["check"][name] = {
            "games": int(ref.notna().sum()),
            "max_abs_diff_vs_u5b": float(
                (mine["home_cover_probability_at_open"] - ref).abs().max()
            ),
            "stream_rows": len(stream),
            "stream_last": f"{int(tail['season'].max())}-{int(tail['week'].max())}",
        }
    frozen["train_cutoff"] = str(cutoff.date())
    frozen["train_rows"] = len(train)
    frozen["train_last_gameday"] = str(train["gameday"].max().date())
    return frozen


def freeze_usage():
    manifest = json.loads(u5a.V1_MANIFEST.read_text(encoding="utf-8"))
    spans = {int(f["span_games"]) for f in manifest["folds"].values()}
    if len(spans) != 1:
        raise SystemExit("v1 span differs across folds")
    span = spans.pop()
    table = pd.read_parquet(u5a.ACTIVES)
    table = table.loc[table["season"].isin(u5a.SEASONS)].copy()
    table["team"] = table["team"].astype(str)
    table["game_id"] = table["game_id"].astype(str)
    known = table.groupby(["game_id", "team"])["known_at_deadline"].all().reset_index()
    known = known.loc[known["known_at_deadline"], ["game_id", "team"]]
    listed = table.merge(known, on=["game_id", "team"], how="inner")
    listed = listed.loc[listed["active"].fillna(False).astype(bool)]
    actives = listed.assign(player_key=listed["gsis_id"].astype(str), active_source="actives_table")
    actives = actives[[*u2.KEYS, "active_source"]].drop_duplicates(u2.KEYS)
    rosters, snaps = u2.load_inputs(u5a.PLAYER_SNAPSHOT)
    cand, hist = u2.build_candidates(rosters, snaps, actives)
    baseline, alloc = u2.evaluate(cand, hist, span)
    realized = cand["realized"].fillna(0.0).to_numpy(float)
    pop = cand["active"].to_numpy(bool) & cand["row"].notna().to_numpy()
    frac = u2.fit_fraction((realized - np.nan_to_num(baseline))[pop], alloc[pop])
    cand = cand.merge(known, on=["game_id", "team"], how="inner").reset_index(drop=True)
    shares, covered, _ = u5a.neutral_tables()
    hist["neutral"] = u5a.attach_neutral(hist, shares, covered)
    alpha = 2.0 / (span + 1.0)
    state = (
        hist.groupby("player_key", sort=False)["neutral"]
        .ewm(alpha=alpha, adjust=False, ignore_na=True)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(hist.index)
    )
    nb_state = state.groupby(hist["player_key"]).last().dropna()
    has = cand["row"].notna().to_numpy()
    rows = cand["row"].fillna(0).astype(int).to_numpy()
    cand["has"] = has
    cand["nb"] = np.where(has, state.to_numpy(float)[rows], np.nan)
    cand["t1"] = u5a.attach_neutral(cand, shares, covered)
    v1 = pd.read_parquet(u5a.V1, columns=[*u2.KEYS, "baseline_share", "redistributed"])
    cand = cand.merge(v1, on=u2.KEYS, how="left")
    inj, _ = u5a.injury_inputs(table)
    cand = cand.merge(inj, on=["season", "week", "team", "player_key"], how="left")
    cand[["designation", "practice"]] = cand[["designation", "practice"]].fillna("none")
    x = u5a.design(cand)
    y = cand["t1"].to_numpy(float)
    active = cand["active"].to_numpy(bool)
    role = cand["role"].to_numpy(object)
    ok = active & np.isfinite(y)
    usage = {}
    for name in sorted(set(role[active])):
        train = ok & (role == name)
        if not train.any():
            continue
        keep = np.ptp(x[train], axis=0) > 0
        model = QuantileRegressor(quantile=u5a.MEDIAN, alpha=0.0, solver="highs")
        model.fit(x[train][:, keep], y[train])
        coef = np.zeros(x.shape[1])
        coef[keep] = model.coef_
        usage[str(name)] = {
            "intercept": float(model.intercept_),
            "coef": dict(zip(u5a.TERMS, coef.tolist(), strict=True)),
            "clip_max": float(y[train].max()),
            "n": int(train.sum()),
        }
    covered_2026 = int(
        covered["game_id"].astype(str).str.startswith(f"{SEASON}_").sum() if len(covered) else 0
    )
    return {
        "span": span,
        "fraction": float(frac),
        "usage": usage,
        "nb_state": nb_state.to_dict(),
        "participation_team_games_2026": covered_2026,
    }


def cmd_freeze(a):
    start = time.time()
    stamp = datetime.now(UTC).strftime(STAMP)
    u3.SYNTH_TABLE = SYNTH_V2
    features = u3.load_features()
    frozen = freeze_margin(features, stamp)
    print(f"margin frozen at {time.time() - start:.0f}s", json.dumps(frozen["check"]), flush=True)
    frozen.update(freeze_usage())
    frozen["created_at_utc"] = stamp
    OUT.mkdir(parents=True, exist_ok=True)
    with FROZEN.open("wb") as handle:
        pickle.dump(frozen, handle)
    doc = {
        "created_at_utc": stamp,
        "frozen_sha256": sha_file(FROZEN),
        "declaration": "S_ALL_v2 frozen: margin ridge = the walk-forward fit the U3/U5b harness "
        "makes for 2026 week 1 (all completed games before the first 2026 gameday); home-side "
        "offsets and discrete push pool from the same walk-forward through 2025; U2 span from the "
        "v1 manifest and redistribution fraction refit on 2020-25; U5a one-score median "
        "regression per role refit on all 2020-25 (the forward scheme for held season 2026). "
        "Nothing fitted on 2026.",
        "train_cutoff": frozen["train_cutoff"],
        "train_rows": frozen["train_rows"],
        "train_last_gameday": frozen["train_last_gameday"],
        "check_vs_u5b": frozen["check"],
        "u2_span": frozen["span"],
        "u2_fraction": frozen["fraction"],
        "usage": frozen["usage"],
        "nb_players": len(frozen["nb_state"]),
        "participation_team_games_2026": frozen["participation_team_games_2026"],
        "seconds": round(time.time() - start),
    }
    FROZEN_JSON.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in doc.items() if k != "usage"}, indent=1))


def schedules():
    sch, _ = load_snapshot(latest_snapshot(DATA / "raw"))
    sch["game_id"] = sch["game_id"].astype(str)
    return sch


def week_games(sch, week):
    g = sch[(sch["season"] == SEASON) & (sch["game_type"] == "REG") & (sch["week"] == week)]
    g = g.copy().reset_index(drop=True)
    g["kickoff"] = _kickoff_utc(g)
    deadlines = game_deadlines(g[["game_id"]].assign(kickoff=g["kickoff"]))
    g["pick_deadline_utc"] = pd.to_datetime(g["game_id"].map(deadlines), utc=True)
    g["cutoff_utc"] = g[["kickoff", "pick_deadline_utc"]].min(axis=1)
    return g


def resolve_week(arg, sch, now):
    if arg != "auto":
        return int(arg)
    g = sch[(sch["season"] == SEASON) & (sch["game_type"] == "REG")]
    ahead = g.loc[_kickoff_utc(g) > now, "week"]
    if ahead.empty:
        raise SystemExit("no unplayed regular-season week")
    return int(ahead.min())


def player_snapshot(now):
    dirs = [p.parent for p in sorted(PLAYERS_RAW.glob("2*/manifest.json"))]
    dirs = [d for d in dirs if stamp_time(d.name) <= now]
    if not dirs:
        raise SystemExit("no player snapshot at or before the run time")
    return dirs[-1]


def forward_snaps(week):
    files = sorted(FORWARD_SNAPS.glob(f"snap_counts_{SEASON}_before_week_*.parquet"))
    usable = [f for f in files if int(f.stem.rsplit("_", 1)[1]) <= week]
    if not usable:
        raise SystemExit(f"no forward snap file for {SEASON} before week {week}")
    snaps = pd.read_parquet(usable[-1])
    current = snaps["season"].astype(int).eq(SEASON)
    if (snaps.loc[current, "week"].astype(int) >= week).any():
        raise SystemExit("snap rows from the target week or later")
    return snaps, usable[-1]


def norm_name(text):
    text = re.sub(r"[^a-z ]", "", str(text).lower().replace("-", " "))
    return " ".join(SUFFIX.sub("", text).split())


def live_lists(games, rosters_raw, week, now):
    prior = rosters_raw.loc[
        rosters_raw["season"].astype(int).eq(SEASON) & rosters_raw["week"].astype(int).lt(week)
    ].copy()
    prior["team"] = prior["team"].astype(str)
    latest = prior.groupby("team")["week"].transform("max")
    prior = prior.loc[prior["week"].eq(latest)].drop_duplicates(["team", "gsis_id"])
    prior = prior.dropna(subset=["gsis_id"])
    prior["key"] = prior["full_name"].map(norm_name)
    frames = [pd.read_parquet(f) for f in sorted(INACTIVES.glob("2*/inactives.parquet"))]
    frames = [f for f in frames if len(f)]
    cap = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    lists, table_rows, info = {}, [], {"unmatched_names": 0, "captures_used": 0}
    if cap.empty:
        return lists, pd.DataFrame(), info, prior
    cap = cap.loc[cap["season"].astype(int).eq(SEASON) & cap["week"].astype(int).eq(week)].copy()
    cap["captured_at_utc"] = pd.to_datetime(cap["captured_at_utc"], utc=True)
    window = games.set_index("game_id")
    cap = cap.loc[cap["game_id"].astype(str).isin(window.index)].copy()
    kick = cap["game_id"].map(window["kickoff"])
    cut = cap["game_id"].map(window["cutoff_utc"])
    posted = kick - pd.Timedelta(minutes=POSTING_MINUTES)
    stamp = cap["captured_at_utc"]
    cap = cap.loc[(stamp >= posted) & (stamp < cut) & (stamp <= now)]
    for (game_id, team), grp in cap.groupby(["game_id", "team"]):
        grp = grp.loc[grp["captured_at_utc"].eq(grp["captured_at_utc"].max())]
        grp = grp.loc[grp["status"].astype(str).str.upper().str.startswith(u3.INACTIVE_STATUS)]
        roster = prior.loc[prior["team"].eq(str(team))]
        if roster.empty or grp.empty:
            continue
        info["captures_used"] += 1
        if "gsis_id" in grp.columns and grp["gsis_id"].notna().all():
            out_ids = set(grp["gsis_id"].astype(str))
        else:
            by_name = dict(zip(roster["key"], roster["gsis_id"].astype(str), strict=False))
            found = grp["player_name"].map(norm_name).map(by_name)
            info["unmatched_names"] += int(found.isna().sum())
            out_ids = set(found.dropna())
        listed = {}
        for pid, pos, status in zip(
            roster["gsis_id"].astype(str), roster["position"], roster["status"], strict=True
        ):
            active = status in ROSTER_ON_53 and pid not in out_ids
            listed[pid] = (None if pd.isna(pos) else str(pos), bool(active))
            table_rows.append(
                {
                    "game_id": str(game_id),
                    "team": str(team),
                    "gsis_id": pid,
                    "position": pos,
                    "active": bool(active),
                    "status": u3.INACTIVE_STATUS if pid in out_ids else str(status),
                    "known_at_deadline": True,
                }
            )
        lists[(str(game_id), str(team))] = listed
    return lists, pd.DataFrame(table_rows), info, prior


def injury_rows(injuries, games, week, now):
    inj = injuries.loc[
        injuries["season"].astype(int).eq(SEASON)
        & injuries["week"].astype(int).eq(week)
        & injuries["game_type"].eq("REG")
    ].copy()
    team_deadline = pd.concat(
        [
            games[["home_team", "cutoff_utc"]].rename(columns={"home_team": "team"}),
            games[["away_team", "cutoff_utc"]].rename(columns={"away_team": "team"}),
        ]
    )
    inj["team"] = inj["team"].astype(str)
    inj = inj.merge(team_deadline, on="team", how="inner")
    proxy = inj["observed_at_is_proxy"].astype(bool)
    seen = pd.to_datetime(
        inj["date_modified"].where(~proxy, inj["effective_observed_at"]), utc=True
    )
    keep = seen.notna() & (seen <= inj["cutoff_utc"]) & (seen <= now)
    inj = inj.loc[keep].assign(seen=seen[keep]).sort_values("seen")
    inj = inj.drop_duplicates(["team", "gsis_id"], keep="last")
    inj["designation"] = inj["report_status"].map(u5a.DESIGNATIONS).fillna("none")
    inj["practice"] = inj["practice_status"].map(u5a.PRACTICE).fillna("none")
    inj = inj.assign(season=SEASON, week=week, player_key=inj["gsis_id"].astype(str))
    return inj[["season", "week", "team", "player_key", "designation", "practice"]]


def expected_shares(frozen, lists, rosters_raw, snaps_raw, injuries, games, week, now):
    rosters = canonicalize_rosters(rosters_raw)
    snaps = attach_snap_player_ids(canonicalize_snaps(snaps_raw), rosters)
    snaps["player_key"] = snaps["gsis_id"].astype(object).where(snaps["gsis_id"].notna(), None)
    snaps["player_key"] = snaps["player_key"].fillna("pfr:" + snaps["pfr_player_id"].astype(str))
    snaps["player_key"] = snaps["player_key"].astype(str)
    snaps["share"] = snaps["offense_pct"].fillna(0.0) + snaps["defense_pct"].fillna(0.0)
    played = snaps[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0.0).sum(axis=1)
    snaps["played"] = played.gt(0.0)
    snaps = snaps.drop_duplicates(u2.KEYS).reset_index(drop=True)
    rosters = rosters.assign(player_key=rosters["gsis_id"].astype(str))
    targets = pd.DataFrame(list(lists), columns=["game_id", "team"])
    if targets.empty:
        return {}, 0
    holders = targets.assign(
        player_key="team:" + targets["team"], season=SEASON, week=week, share=0.0, played=False
    )
    snaps = pd.concat([snaps, holders], ignore_index=True)
    rows = [
        {"season": SEASON, "week": week, "team": t, "player_key": p, "position": pos}
        for (g, t), listed in lists.items()
        for p, (pos, _active) in listed.items()
    ]
    rosters = pd.concat([rosters, pd.DataFrame(rows)], ignore_index=True)
    actives = pd.DataFrame(
        [
            {"game_id": g, "team": t, "player_key": p, "active_source": "live_capture"}
            for (g, t), listed in lists.items()
            for p, (_pos, active) in listed.items()
            if active
        ]
    )
    cand, hist = u2.build_candidates(rosters, snaps, actives)
    baseline, alloc = u2.evaluate(cand, hist, frozen["span"])
    cand["baseline_share"] = baseline
    cand["redistributed"] = frozen["fraction"] * alloc
    cand = cand.merge(targets, on=["game_id", "team"], how="inner").reset_index(drop=True)
    cand["has"] = cand["row"].notna()
    nb = cand["player_key"].map(frozen["nb_state"]).astype(float)
    cand["nb"] = np.where(cand["has"], nb, np.nan)
    cand = cand.merge(
        injury_rows(injuries, games, week, now),
        on=["season", "week", "team", "player_key"],
        how="left",
    )
    cand[["designation", "practice"]] = cand[["designation", "practice"]].fillna("none")
    x = u5a.design(cand)
    pred = np.zeros(len(cand))
    active = cand["active"].to_numpy(bool)
    fallback = 0
    for i, role in enumerate(cand["role"].to_numpy(object)):
        if not active[i]:
            continue
        fit = frozen["usage"].get(str(role))
        if fit is None:
            pred[i] = np.nan_to_num(cand.at[i, "baseline_share"]) + cand.at[i, "redistributed"]
            fallback += 1
            continue
        coef = np.array([fit["coef"][t] for t in u5a.TERMS])
        pred[i] = float(np.clip(fit["intercept"] + x[i] @ coef, 0.0, fit["clip_max"]))
    keys = zip(cand["game_id"], cand["team"], cand["player_key"], strict=True)
    return {(str(g), str(t), str(p)): float(s) for (g, t, p), s in zip(keys, pred, strict=True)}, (
        fallback
    )


def servable_terms(frozen, games, week, now):
    snap_dir = player_snapshot(now)
    injuries, rosters_raw, _snaps_old = load_player_snapshot(player_snapshot_from_root(snap_dir))
    rosters_raw = rosters_raw.loc[
        ~(rosters_raw["season"].astype(int).eq(SEASON) & rosters_raw["week"].astype(int).ge(week))
    ]
    snaps_raw, snap_path = forward_snaps(week)
    lists, table, info, _prior = live_lists(games, rosters_raw, week, now)
    info.update(
        {
            "player_snapshot": snap_dir.name,
            "forward_snaps": snap_path.name,
            "known_team_games": len(lists),
            "participation_team_games_2026": frozen["participation_team_games_2026"],
        }
    )
    terms = pd.DataFrame({"game_id": games["game_id"].astype(str)})
    for column in u3.COLUMNS:
        terms[column] = 0.0
    if not lists:
        return terms, info
    expected, fallback = expected_shares(
        frozen, lists, rosters_raw, snaps_raw, injuries, games, week, now
    )
    info["usage_role_fallbacks"] = fallback
    base = pd.read_parquet(u3.BASE_FEATURES)
    base = base.loc[
        base["season"].lt(SEASON) | (base["season"].eq(SEASON) & base["week"].le(week))
    ].reset_index(drop=True)
    snaps_ids = attach_snap_player_ids(
        canonicalize_snaps(snaps_raw), canonicalize_rosters(rosters_raw)
    )
    inj, keys, stats = u3.role_rows(base, lists, expected, snaps_ids)
    info["role_rows"] = stats
    known_games = {g for g, _t in lists}
    if len(keys) and not keys["game_id"].isin(known_games).all():
        raise SystemExit("synthetic rows on a team-game without a valid capture")
    if len(inj):
        pbp = load_pbp_snapshot(pbp_snapshot_from_root(u3.PBP_SNAPSHOT))
        values = load_player_value_snapshot(player_value_snapshot_from_root(u3.VALUE_SNAPSHOT))
        for tag, column in (("o1", "sev1"), ("o2", "sev2")):
            if not (inj[column] > 0).any():
                continue
            enriched = u3.run_enrich(base, inj, rosters_raw, snaps_raw, pbp, values, column)
            part = enriched.set_index("game_id")
            value = part["diff_injury_skill_epa_value_lost"].fillna(0.0) + part[
                "diff_injury_defense_disruption_value_lost"
            ].fillna(0.0)
            terms[f"diff_servable_{tag}"] = terms["game_id"].map(value).fillna(0.0).to_numpy()
    served = pd.read_parquet(SERVED_FEATURES)
    served = served.loc[served["game_id"].astype(str).isin(set(terms["game_id"]))]
    o3, _counts = u3.qb_term(served.reset_index(drop=True), table)
    terms["diff_servable_o3"] = (
        terms["game_id"].map(o3.set_index("game_id")["diff_servable_o3"]).fillna(0.0)
    )
    unknown = ~terms["game_id"].isin(known_games)
    if terms.loc[unknown, list(u3.COLUMNS)].abs().to_numpy().sum() != 0.0:
        raise SystemExit("nonzero servable term on a game without a valid capture")
    return terms, info


def frozen_probability(frozen, name, rows, week, cutoff):
    model = frozen["models"][name]
    with feature_set(ARMS[name]):
        raw = model.predict(rows, probability_method=METHOD)
        offsets = frozen["offsets"][name].offset_for(rows["spread_line"])
        offsets = offsets.fillna(0.0).to_numpy(dtype=float)
        served = (
            model.predict(rows, probability_method=METHOD, center_offset=offsets)
            if np.any(offsets != 0.0)
            else raw
        )
        reader = clv.DiscretePushReader.for_week(
            frozen["pool"][name],
            season=SEASON,
            week=week,
            cutoff=cutoff,
            exclude_game_ids=set(rows["game_id"].astype(str)),
        )
        out = clv.serve_discrete_three_way(
            served, rows, reader, residuals=model.residuals, probability_method=METHOD
        )
    return out["home_cover_probability"].to_numpy(float)


def card_now(week, now):
    if not PUBLISHED.is_file():
        return {}
    pub = pd.read_parquet(PUBLISHED)
    pub = pub[(pub["season"] == SEASON) & (pub["week"] == week)]
    pub = pub[pd.to_datetime(pub["published_at_utc"], utc=True) <= now]
    last = pub.sort_values("published_at_utc").groupby("game_id").tail(1)
    home = np.where(
        last["pick_team"] == last["home_team"], last["displayed_score"], 1 - last["displayed_score"]
    )
    return dict(zip(last["game_id"].astype(str), home.astype(float), strict=True))


def cmd_pregame(a):
    now = pd.Timestamp(datetime.now(UTC))
    if not FROZEN.is_file():
        raise SystemExit("frozen parameters missing; run freeze first")
    frozen_sha = sha_file(FROZEN)
    with FROZEN.open("rb") as handle:
        frozen = pickle.load(handle)
    sch = schedules()
    week = resolve_week(a.week, sch, now)
    if not a.dry and not FIRST_WEEK <= week <= LAST_WEEK:
        raise SystemExit(f"week {week} outside the declared forward window")
    games = week_games(sch, week)
    if not a.dry:
        games = games.loc[games["cutoff_utc"] > now].reset_index(drop=True)
    if games.empty:
        print("no game before its cutoff; nothing written")
        return
    terms, info = servable_terms(frozen, games, week, now)
    capture = load_splash_capture(DATA, SEASON, week)
    lines = splash_decision_lines(capture) if capture is not None else {}
    card = card_now(week, now)
    served = pd.read_parquet(SERVED_FEATURES)
    served = served.loc[served["game_id"].astype(str).isin(set(games["game_id"]))].copy()
    served["game_id"] = served["game_id"].astype(str)
    rows = served.merge(terms, on="game_id", how="left")
    rows["spread_line"] = rows["game_id"].map(lines)
    priced = rows.loc[rows["spread_line"].notna()].reset_index(drop=True)
    cutoff = pd.Timestamp(pd.to_datetime(games["gameday"]).min())
    probs = {}
    if len(priced):
        for name in ARMS:
            probs[name] = dict(
                zip(
                    priced["game_id"],
                    frozen_probability(frozen, name, priced, week, cutoff),
                    strict=True,
                )
            )
    out_games = []
    for game in games.itertuples(index=False):
        gid = str(game.game_id)
        lacks = []
        if gid not in lines:
            lacks.append("no_pool_line")
        if gid not in card:
            lacks.append("no_published_card")
        term = terms.set_index("game_id").loc[gid]
        if not (term.abs() > 0).any():
            lacks.append("base_inputs_only")
        out_games.append(
            {
                "game_id": gid,
                "away_team": game.away_team,
                "home_team": game.home_team,
                "kickoff_utc": pd.Timestamp(game.kickoff).isoformat(),
                "pick_deadline_utc": pd.Timestamp(game.pick_deadline_utc).isoformat(),
                "opener_home_line": lines.get(gid),
                "terms": {c: float(term[c]) for c in u3.COLUMNS},
                "frozen_base_home_probability": probs.get("base", {}).get(gid),
                "arm_home_probability": probs.get("arm", {}).get(gid),
                "card_home_probability": card.get(gid),
                "missing": lacks,
            }
        )
    doc = {
        "season": SEASON,
        "week": week,
        "created_at_utc": now.strftime(STAMP),
        "frozen_sha256": frozen_sha,
        "inputs": info,
        "games": out_games,
    }
    target = DRY_OUT if a.dry else OUT
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{SEASON}_week{week:02d}_{now.strftime(STAMP)}.json"
    path.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in doc.items() if k != "games"}, indent=1, default=str))
    for g in out_games:
        print(
            f"{g['game_id']:<18} line {g['opener_home_line']} base "
            f"{g['frozen_base_home_probability']} arm {g['arm_home_probability']} card "
            f"{g['card_home_probability']} missing {g['missing']}"
        )
    print(path)


def record(p, y):
    hit = (p >= 0.5) == (y == 1)
    return f"{int(hit.sum())}-{int((~hit).sum())}"


def cmd_score(a):
    sha = sha_file(FROZEN) if FROZEN.is_file() else None
    rows = []
    for f in sorted(OUT.glob(f"{SEASON}_week*_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["frozen_sha256"] != sha or not FIRST_WEEK <= d["week"] <= LAST_WEEK:
            continue
        made = stamp_time(d["created_at_utc"])
        mtime = pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC")
        for g in d["games"]:
            cut = min(pd.Timestamp(g["kickoff_utc"]), pd.Timestamp(g["pick_deadline_utc"]))
            if made < cut and mtime < cut and g["arm_home_probability"] is not None:
                rows.append({**g, "week": d["week"], "created": d["created_at_utc"]})
    if not rows:
        print("no admissible forward rows yet; nothing graded")
        return
    t = pd.DataFrame(rows).sort_values("created").drop_duplicates("game_id", keep="last")
    sch = schedules().set_index("game_id")
    t["result"] = pd.to_numeric(t["game_id"].map(sch["result"]), errors="coerce")
    t = t.dropna(subset=["result", "opener_home_line"])
    t = t.loc[t["result"] != t["opener_home_line"]]
    if t.empty:
        print("no settled forward game at the opener yet; nothing graded")
        return
    y = (t["result"] > t["opener_home_line"]).astype(float).to_numpy()
    arm = t["arm_home_probability"].to_numpy(float)
    base = t["frozen_base_home_probability"].to_numpy(float)
    card = pd.to_numeric(t["card_home_probability"], errors="coerce").to_numpy(float)
    changed = (arm >= 0.5) != (base >= 0.5)
    has_card = np.isfinite(card)
    loss = {
        k: float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))
        for k, p in (("arm", np.clip(arm, 1e-9, 1 - 1e-9)), ("base", np.clip(base, 1e-9, 1 - 1e-9)))
    }
    report = {
        "games": len(t),
        "weeks": sorted(int(w) for w in t["week"].unique()),
        "arm_record": record(arm, y),
        "frozen_base_record": record(base, y),
        "card_record_same_games": record(card[has_card], y[has_card]),
        "arm_record_card_games": record(arm[has_card], y[has_card]),
        "changed_v_frozen_base": int(changed.sum()),
        "changed_record_arm": record(arm[changed], y[changed]) if changed.any() else "0-0",
        "log_loss": loss,
        "frozen_sha256": sha,
    }
    (OUT / "score.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


def cmd_slots(a):
    import capture_scheduler as cs

    jobs = [
        j
        for v in vars(cs).values()
        if isinstance(v, (list, tuple)) and v and type(v[0]).__name__ == "Job"
        for j in v
    ]
    order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    slots = sorted(
        {(j.day, j.at, j.name) for j in jobs if j.name.startswith("inactives_")},
        key=lambda s: (order.index(s[0]), s[1]),
    )
    argv = [str(cs.UV), "run", "--no-sync", "python", str(ROOT / "scripts" / Path(__file__).name)]
    print(
        json.dumps(
            {
                "pregame_argv": [*argv, "pregame"],
                "pregame_slots_after_capture_job": [list(s) for s in slots],
                "score_argv": [*argv, "score"],
                "score_slot": list(SCORE_SLOT),
            },
            indent=1,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="G55-U7 frozen S_ALL_v2 forward log for 2026 weeks 12-18: model alone at "
        "the Tuesday opener beside the frozen base and the served card; freeze, pregame, score "
        "and slots. Research only; reads no 2026 outcome before score."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze").set_defaults(fn=cmd_freeze)
    pre = sub.add_parser("pregame")
    pre.add_argument("--week", default="auto")
    pre.add_argument("--dry", action="store_true", help="any week; writes under dry_run/")
    pre.set_defaults(fn=cmd_pregame)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    sub.add_parser("slots").set_defaults(fn=cmd_slots)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
