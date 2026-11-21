import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead106_forward_score as F
import lead106_injury_language as L6
import lead108_forward as LF
import lead108_full_mixture as M
import lead108_wording_mixture as U1
import lead110_absorption as A
import mod25_wf_forward as mwf

ROOT = L6.ROOT
SEASON = 2026
FIRST_WEEK = 12
LAST_WEEK = 18
FIRST_FORWARD_LADDER_WEEK = 10
OUT = ROOT / "artifacts" / "lead108_mixture_forward"
SPLASH = ROOT / "data" / "splash"
STAMP = LF.STAMP
CODE = ("scripts/lead108_full_mixture.py", "scripts/lead108_mixture_forward.py")
COLS = ["week", "designation", "practice_status", "played", "snap_share"]
ET = "America/New_York"
EARLIEST_SCORING_ET = "2027-01-12"
DECLARED_MIXTURE_SHA256 = "0214bdaa7ca7d4693091239247605840bf8170a5e806002c2e88e3d16ed87671"
FAMILY = "lead108_wording_mixture"
LANE = "docs/lanes/lead108-forward-2026-11-16.md"
UNIT2_WEEKS = (2, 10)
LOOKS = 8


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def mtime(path):
    return pd.Timestamp(Path(path).stat().st_mtime, unit="s", tz="UTC")


def resolve(arg, now):
    if arg != "auto":
        return int(arg)
    g = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["season", "week", "kickoff", "game_type"],
    )
    g = g[(g["season"] == SEASON) & (g["game_type"] == "REG")]
    k = pd.to_datetime(g["kickoff"], utc=True, format="ISO8601")
    ahead = g.loc[k > pd.Timestamp(now), "week"]
    return int(ahead.min()) if len(ahead) else None


def admissible_reads(week, sched, now):
    fz = json.loads((L6.ART / "frozen.json").read_text())
    cut = {}
    for r in sched.itertuples():
        cut[r.home_team] = r.cutoff_utc
        cut[r.away_team] = r.cutoff_utc
    rows, files = [], []
    for f in sorted(L6.FORWARD.glob(f"{SEASON}_week{week:02d}_*.json")):
        j = json.loads(f.read_text())
        rec = LF.ts(j["recorded_utc"])
        mt = mtime(f)
        enrolled = j["prompt_sha256"] == fz["prompt_sha256"] and j["model"] == fz["model"]
        before_run = rec <= pd.Timestamp(now) and mt <= pd.Timestamp(now)
        inj = pd.read_parquet(LF.INJ / j["capture_utc"] / "injuries.parquet")
        inj = inj[(inj["season"] == SEASON) & (inj["week"] == week)]
        inj = inj.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
        kept = 0
        for o in j["rows"] if enrolled and before_run else []:
            c = cut.get(o["team"])
            if c is None or o["gsis_id"] not in inj.index:
                continue
            if not (rec < c and rec < LF.ts(o["kickoff_utc"]) and mt < c):
                continue
            ps = inj.at[o["gsis_id"], "practice_status"]
            rows.append(
                {
                    "gsis_id": o["gsis_id"],
                    "team": o["team"],
                    "week": week,
                    "position": inj.at[o["gsis_id"], "position"],
                    "designation": o["designation"],
                    "practice_status": ps if pd.notna(ps) else "none",
                    "rp_raw": float(o["p_plays"]),
                    "recorded_utc": rec,
                }
            )
            kept += 1
        files.append(
            {
                "file": f.name,
                "recorded_utc": j["recorded_utc"],
                "modified_utc": mt.strftime(STAMP),
                "enrolled": enrolled,
                "before_run": bool(before_run),
                "rows": len(j["rows"]),
                "kept": kept,
            }
        )
    x = pd.DataFrame(rows)
    if x.empty:
        return x, files
    x = x.sort_values("recorded_utc").drop_duplicates("gsis_id", keep="last")
    return x.reset_index(drop=True), files


def ladder_table(week, now):
    ver = M.verified_population()
    w9 = pd.read_parquet(L6.ART / "forward_score" / f"{SEASON}_week09_scored.parquet")
    w9 = w9[w9["verified"].astype(bool)].copy()
    w9["practice_status"] = w9["practice_status"].fillna("none")
    parts, used, missing = [ver[COLS], w9[COLS]], [], []
    for w in range(FIRST_FORWARD_LADDER_WEEK, week):
        f = F.SNAPS / f"snap_counts_{SEASON}_before_week_{w + 1:02d}.parquet"
        if not f.exists() or mtime(f) > pd.Timestamp(now):
            missing.append(w)
            continue
        fw, _, _ = F.load_forward(w)
        if fw.empty:
            missing.append(w)
            continue
        snaps, _ = F.snap_frame(w)
        fo = F.outcomes(fw, snaps, w, w)
        fv = fo[fo["verified"].astype(bool)].copy()
        fv["practice_status"] = fv["practice_status"].fillna("none")
        parts.append(fv[COLS])
        used.append(w)
    prior = pd.concat(parts, ignore_index=True)
    prior = prior[prior["week"] < week]
    tab = {
        week: (
            prior.groupby(U1.KEYS).agg(**M.AGG),
            prior.groupby("designation").agg(**M.AGG),
        )
    }
    meta = {
        "prior_rows": len(prior),
        "prior_weeks": sorted(int(w) for w in prior["week"].unique()),
        "forward_weeks_used": used,
        "forward_weeks_missing_snaps": missing,
    }
    return tab, meta


def card(week, now):
    pub = pd.read_parquet(ROOT / "artifacts" / "clv_ledger" / "published_picks.parquet")
    pub = pub[(pub["season"] == SEASON) & (pub["week"] == week)]
    pub = pub[pd.to_datetime(pub["published_at_utc"], utc=True) <= pd.Timestamp(now)]
    last = pub.sort_values("published_at_utc").groupby("game_id").tail(1).set_index("game_id")
    last["p_home"] = np.where(
        last["pick_team"] == last["home_team"], last["displayed_score"], 1 - last["displayed_score"]
    )
    return last


def splash(week, now):
    out = {}
    for f in sorted(SPLASH.glob(f"{SEASON}_week{week:02d}_*.json")):
        if mtime(f) > pd.Timestamp(now):
            continue
        for g in json.loads(f.read_text()).get("games", []):
            if g.get("home_spread") is not None:
                out[str(g["game_id"])] = (float(g["home_spread"]), f.name)
    return out


def history_pool():
    gf = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["season", "spread_line", "result"],
    )
    gf = gf[(gf["season"] < SEASON) & gf["spread_line"].notna() & gf["result"].notna()]
    return gf["spread_line"].to_numpy(float), gf["result"].to_numpy(float)


def cmd_log(a):
    real_now = datetime.now(UTC)
    if a.as_of and not a.smoke:
        raise SystemExit("--as-of is allowed only with --smoke")
    now = pd.Timestamp(a.as_of).tz_convert("UTC").to_pydatetime() if a.as_of else real_now
    week = resolve(a.week, now)
    if week is None or (not a.smoke and not FIRST_WEEK <= week <= LAST_WEEK):
        print(f"SKIP: week {week} outside the declared forward window {FIRST_WEEK}-{LAST_WEEK}")
        return
    sched = LF.schedule(week)
    x, files = admissible_reads(week, sched, now)
    tab, lmeta = ladder_table(week, now)
    clip = 1.0 / (len(x) + 2)
    if not x.empty:
        x["lp"] = [L6.ladder_pred(r, tab, U1.KEYS)[0] for _, r in x.iterrows()]
        x["rp"] = x["rp_raw"].clip(clip, 1 - clip)
        x["grp"] = x["position"].map(A.POS_TO_GROUP)
    else:
        x = pd.DataFrame(columns=["gsis_id", "team", "week", "grp", "lp", "rp"])
    base = card(week, now)
    pool = splash(week, now)
    opens = mwf.tuesday_open(sched)
    rows, skipped = [], {}
    for r in sched.itertuples():
        if r.cutoff_utc <= pd.Timestamp(now):
            skipped[r.game_id] = "past_cutoff"
            continue
        if r.game_id not in base.index:
            skipped[r.game_id] = "no_published_card"
            continue
        b = base.loc[r.game_id]
        op = opens.get(r.game_id, (np.nan, None))
        if r.game_id in pool:
            line, src = pool[r.game_id][0], "pool_splash"
        elif pd.notna(b["market_spread"]):
            line, src = float(b["market_spread"]), "card_market_spread"
        else:
            line, src = op[0], "tuesday_opener"
        rows.append(
            {
                "game_id": r.game_id,
                "away_team": r.away_team,
                "home_team": r.home_team,
                "week": week,
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "pick_deadline_utc": r.pick_deadline_utc.isoformat(),
                "cutoff_utc": r.cutoff_utc.isoformat(),
                "line_source": src,
                "open_line": line,
                "pool_line": pool.get(r.game_id, (None, None))[0],
                "pool_capture": pool.get(r.game_id, (None, None))[1],
                "card_line": float(b["market_spread"]),
                "tuesday_opener": op[0],
                "tuesday_opener_observed_utc": op[1],
                "p_served": float(b["p_home"]),
                "base_published_at_utc": str(b["published_at_utc"]),
                "served_pick_team": b["pick_team"],
                "players_home": int((x["team"] == r.home_team).sum()),
                "players_away": int((x["team"] == r.away_team).sum()),
            }
        )
    g = pd.DataFrame(rows)
    games = []
    if not g.empty:
        hl, hm = history_pool()
        val = json.loads((A.OUT / "report.json").read_text())["weights_spread_pts_per_out"]
        mx = M.mixtures(g, x, val, hl, hm)
        g = pd.concat([g, mx], axis=1)
        off = U1.logit(g["p_served"].to_numpy())
        g["delta_logit_ladder"] = U1.logit(g["p_mix_lad"].to_numpy()) - off
        g["delta_logit_read"] = U1.logit(g["p_mix_read"].to_numpy()) - off
        served_home = g["p_served"] >= 0.5
        g["ladder_side_changed_vs_served"] = (g["p_mix_lad"] >= 0.5) != served_home
        g["read_side_changed_vs_served"] = (g["p_mix_read"] >= 0.5) != served_home
        g["read_vs_ladder_side_differs"] = (g["p_mix_read"] >= 0.5) != (g["p_mix_lad"] >= 0.5)
        g = g.rename(
            columns={
                "open_line": "lattice_line",
                "p_served": "base_home_probability",
                "p_mix_lad": "ladder_arm_home_probability",
                "p_mix_read": "read_arm_home_probability",
                "e_lad": "ladder_expected_shift_pts",
                "e_read": "read_expected_shift_pts",
                "n_players": "valued_players",
                "zero_err": "zero_shift_error",
            }
        )
        games = json.loads(g.to_json(orient="records"))
    doc = {
        "lead": "LEAD-108",
        "arm": "unfitted read-weighted PER-10 lattice scenario mixture vs ladder-weighted",
        "season": SEASON,
        "week": week,
        "smoke": bool(a.smoke),
        "scoreable": bool(not a.smoke and FIRST_WEEK <= week <= LAST_WEEK),
        "created_at_utc": real_now.strftime(STAMP),
        "as_of_utc": pd.Timestamp(now).strftime(STAMP),
        "code_sha256": {c: sha(c) for c in CODE},
        "value_source": str(A.OUT / "report.json"),
        "read_files": files,
        "rows_read": len(x),
        "read_clip": clip,
        "ladder": lmeta,
        "skipped": skipped,
        "games": games,
    }
    summary = {k: v for k, v in doc.items() if k != "games"}
    print(json.dumps(summary, indent=1, default=str))
    for gm in games:
        print(
            f"{gm['game_id']:<16} {gm['line_source']:<18} line {gm['lattice_line']:+5.1f} "
            f"open {gm['tuesday_opener']} card {gm['card_line']} "
            f"n {gm['valued_players']:>2} base {gm['base_home_probability']:.4f} "
            f"lad {gm['ladder_arm_home_probability']:.4f} "
            f"read {gm['read_arm_home_probability']:.4f} "
            f"err {gm['zero_shift_error']:.1e} pick {gm['served_pick_team']} "
            f"chg {gm['read_side_changed_vs_served']}"
        )
    if not games:
        print("no game with a published card before its cutoff; nothing written")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{SEASON}_week{week:02d}_{real_now.strftime(STAMP)}.json"
    out = OUT / (f"smoke_{name}" if a.smoke else name)
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(out, len(games), "games")


def utc(v):
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def score_paths(a):
    window = list(range(FIRST_WEEK, LAST_WEEK + 1))
    if not a.dry_run:
        if a.log_dir or a.out_dir or a.weeks:
            raise SystemExit("--log-dir, --out-dir and --weeks are allowed only with --dry-run")
        return OUT, OUT, window
    if not (a.log_dir and a.out_dir):
        raise SystemExit("--dry-run needs --log-dir and --out-dir")
    out = Path(a.out_dir).resolve()
    art = (ROOT / "artifacts").resolve()
    if out == art or art in out.parents:
        raise SystemExit("--dry-run output must stay outside artifacts/")
    weeks = [int(w) for w in a.weeks.split(",")] if a.weeks else window
    return Path(a.log_dir).resolve(), out, weeks


def week_outcomes(weeks, dry):
    cols = ["season", "week", "game_id", "game_type"] + ([] if dry else ["result"])
    gf = pd.read_parquet(ROOT / "data" / "processed" / "game_features.parquet", columns=cols)
    gf = gf[(gf["season"] == SEASON) & (gf["game_type"] == "REG") & gf["week"].isin(weeks)]
    gf = gf.set_index("game_id")
    if dry:
        return gf, []
    done = gf.groupby("week")["result"].apply(lambda s: bool(s.notna().all()))
    return gf, [w for w in weeks if not done.get(w, False)]


def played_card(week):
    pub = pd.read_parquet(ROOT / "artifacts" / "clv_ledger" / "published_picks.parquet")
    pub = pub[(pub["season"] == SEASON) & (pub["week"] == week)]
    pub = pub[pub["published_at_utc"] <= pub["pick_deadline_utc"]]
    last = pub.sort_values("published_at_utc").drop_duplicates("game_id", keep="last")
    last = last.set_index("game_id")
    last["p_home"] = np.where(
        last["pick_team"] == last["home_team"], last["displayed_score"], 1 - last["displayed_score"]
    )
    return last


def logged_rows(log_dir, weeks, dry):
    keep, files = [], []
    for f in sorted(log_dir.glob(f"{SEASON}_week*_*.json")):
        d = json.loads(f.read_text())
        mix = d.get("code_sha256", {}).get(CODE[0])
        why = None
        if d.get("smoke", True) or f.name.startswith("smoke_"):
            why = "smoke"
        elif d.get("test_input") and not dry:
            why = "test_input"
        elif not dry and not d.get("scoreable"):
            why = "not_scoreable"
        elif d.get("week") not in weeks:
            why = "outside_window"
        elif d.get("as_of_utc") != d.get("created_at_utc"):
            why = "as_of_not_creation_time"
        mt = mtime(f)
        files.append(
            {
                "file": f.name,
                "week": d.get("week"),
                "created_at_utc": d.get("created_at_utc"),
                "modified_utc": mt.strftime(STAMP),
                "mixture_sha256": mix,
                "excluded": why,
            }
        )
        if why:
            continue
        if mix != DECLARED_MIXTURE_SHA256:
            raise SystemExit(f"{f.name}: logged by a changed mixture construction ({mix})")
        created = LF.ts(d["created_at_utc"])
        for gm in d["games"]:
            keep.append(
                {
                    "game_id": gm["game_id"],
                    "created": created,
                    "modified": mt,
                    "file": f.name,
                    "zl": float(gm["delta_logit_ladder"]),
                    "zr": float(gm["delta_logit_read"]),
                    "base": float(gm["base_home_probability"]),
                    "base_published": str(gm["base_published_at_utc"]),
                    "logged_line": gm["lattice_line"],
                    "logged_line_source": gm["line_source"],
                    "valued_players": int(gm["valued_players"]),
                }
            )
    cols = [
        "game_id",
        "created",
        "modified",
        "file",
        "zl",
        "zr",
        "base",
        "base_published",
        "logged_line",
        "logged_line_source",
        "valued_players",
    ]
    return pd.DataFrame(keep, columns=cols), files


def gains(pa, pb, y):
    return np.array([(U1.ll(pb, y) - U1.ll(pa, y)).mean(), (M.acc(pa, y) - M.acc(pb, y)).mean()])


def fcell(pa, pb, y, wk, rng):
    out = {"games": len(y), "record_a_vs_b": U1.record(pa, pb, y)}
    for mname, d in (
        ("log_loss", U1.ll(pb, y) - U1.ll(pa, y)),
        ("accuracy_points", M.acc(pa, y) - M.acc(pb, y)),
    ):
        bs = U1.boot(d, wk, rng)
        out[mname] = {
            "gain": float(d.mean()),
            "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "p_positive": float((bs > 0).mean() + 0.5 * (bs == 0).mean()),
            "p_positive_strict": float((bs > 0).mean()),
            "se": float(bs.std()),
            "detectable_80pct": float(M.POWER_Z * bs.std()),
        }
    return out


def arm(pp, z):
    return np.where(z == 0, pp, U1.sig(U1.logit(pp) + z))


def matched_null(zl, zr, pp, y, wk, rng):
    pl, pr = arm(pp, zl), arm(pp, zr)
    obs = {
        "read_vs_ladder": gains(pr, pl, y),
        "read_vs_played_card": gains(pr, pp, y),
        "ladder_vs_played_card": gains(pl, pp, y),
    }
    draws = {k: [] for k in obs}
    for _ in range(M.NULL_REPS):
        zp = zl + U1.perm_x(zr - zl, wk, rng)
        draws["read_vs_ladder"].append(gains(arm(pp, zp), pl, y))
        draws["read_vs_played_card"].append(gains(arm(pp, U1.perm_x(zr, wk, rng)), pp, y))
        draws["ladder_vs_played_card"].append(gains(arm(pp, U1.perm_x(zl, wk, rng)), pp, y))
    return {
        k: dict(
            zip(
                ("log_loss", "accuracy_points"),
                (np.array(v) < obs[k]).mean(0).tolist(),
                strict=True,
            )
        )
        for k, v in draws.items()
    }


def batch_cell(name, desc, look, metric, n_blocks, null_pct, dry):
    m = look[metric]
    lo, hi = m["ci"]
    rec = look["record_a_vs_b"]
    span = (
        "interval spans both signs"
        if lo < 0 < hi
        else ("whole interval below zero" if hi < 0 else "whole interval at or above zero")
    )
    entry = {
        "name": name,
        "description": desc,
        "source": LANE,
        "effect": round(m["gain"], 6),
        "effect_units": "log_loss_improvement" if metric == "log_loss" else "accuracy_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": SEASON,
        "season_end": SEASON,
        "family": FAMILY,
        "category": "health",
        "classification_evidence": (
            f"One of {LOOKS} declared looks (declaration 2, 2026-11-21); {look['games']} "
            f"pool-line-graded games, pushes dropped; decisive {rec['a_wins']}-{rec['b_wins']} "
            f"of {rec['decisive']}; week-stratified game bootstrap {U1.REPS} over {n_blocks} "
            f"weeks; detectable at 80% power {m['detectable_80pct']:.4f}; within-week "
            f"permutation matched-null percentile {null_pct:.3f} ({M.NULL_REPS}); {span}; no "
            "refuted mechanism, no split-half, no positive control bound."
        ),
        "notes": (
            "Scored by scripts/lead108_mixture_forward.py score from the declaration 2 logs: "
            "latest non-smoke row created and file-modified before min(kickoff, pick deadline); "
            "arm = sigmoid(logit(played card) + logged arm shift); y = result above the pool "
            "line (Splash, else card, else Tuesday opener). Nothing fitted. Serves nothing."
            + (" DRY RUN on synthetic outcomes: never record." if dry else "")
        ),
        "interval_low": round(lo, 6),
        "interval_high": round(hi, 6),
        "probability_positive": round(m["p_positive"], 4),
        "sample_games": int(look["games"]),
        "sample_blocks": int(n_blocks),
        "plain_summary": (
            "Checks whether weighting each injured player's chance of playing by the injury "
            "report's own wording, instead of the tag alone, makes the card's probability better."
        ),
    }
    if hi < 0:
        entry["classification"] = "refuted_mechanism"
        entry["closing_ground"] = "wrong_sign_resolved"
    return entry


def cmd_score(a):
    real_now = datetime.now(UTC)
    dry = a.dry_run
    log_dir, out_dir, weeks = score_paths(a)
    earliest = pd.Timestamp(EARLIEST_SCORING_ET).date()
    if not dry and pd.Timestamp(real_now).tz_convert(ET).date() < earliest:
        raise SystemExit(f"declared earliest scoring is {EARLIEST_SCORING_ET} ET")
    outcome, open_weeks = week_outcomes(weeks, dry)
    if open_weeks:
        raise SystemExit(f"weeks {open_weeks} are not final; nothing graded")
    logs, files = logged_rows(log_dir, weeks, dry)
    synth = np.random.default_rng(M.SEED)
    recs, skipped = [], {}
    for w in weeks:
        sched = LF.schedule(w)
        pc = played_card(w)
        opens = mwf.tuesday_open(sched)
        pools = {}
        for r in sched.itertuples():
            gid = r.game_id
            if gid not in pc.index:
                skipped[gid] = "no_published_card"
                continue
            c = pc.loc[gid]
            if utc(c["published_at_utc"]) > r.cutoff_utc:
                skipped[gid] = "card_after_cutoff"
                continue
            if r.cutoff_utc not in pools:
                pools[r.cutoff_utc] = splash(w, r.cutoff_utc)
            pool = pools[r.cutoff_utc]
            if gid in pool:
                line, src = pool[gid][0], "pool_splash"
            elif pd.notna(c["market_spread"]):
                line, src = float(c["market_spread"]), "card_market_spread"
            else:
                line, src = opens.get(gid, (np.nan, None))[0], "tuesday_opener"
            cand = logs[
                (logs["game_id"] == gid)
                & (logs["created"] < r.cutoff_utc)
                & (logs["modified"] < r.cutoff_utc)
            ]
            row = cand.sort_values(["created", "file"]).iloc[-1] if len(cand) else None
            pp = float(c["p_home"])
            rec = {
                "game_id": gid,
                "week": w,
                "cutoff_utc": r.cutoff_utc.isoformat(),
                "played_home_probability": pp,
                "played_published_at_utc": str(c["published_at_utc"]),
                "played_pick_team": c["pick_team"],
                "pool_line": line,
                "line_source": src,
                "logged_row": row is not None,
                "log_file": None if row is None else row["file"],
                "delta_logit_ladder": 0.0 if row is None else row["zl"],
                "delta_logit_read": 0.0 if row is None else row["zr"],
                "valued_players": 0 if row is None else int(row["valued_players"]),
                "logged_base_home_probability": None if row is None else row["base"],
                "base_is_played_publication": (
                    None if row is None else row["base_published"] == str(c["published_at_utc"])
                ),
                "logged_line": None if row is None else row["logged_line"],
                "logged_line_source": None if row is None else row["logged_line_source"],
            }
            if dry:
                rec.update(result=None, push=False, y=float(synth.binomial(1, pp)))
            else:
                res = outcome["result"].get(gid, np.nan)
                if pd.isna(res) or pd.isna(line):
                    raise SystemExit(f"{gid}: no final result or no pool line")
                rec.update(result=float(res), push=bool(res == line), y=float(res > line))
            recs.append(rec)
    if not recs:
        raise SystemExit("no game with a published card before its cutoff; nothing graded")
    g = pd.DataFrame(recs)
    gg = g[~g["push"]].reset_index(drop=True)
    y = gg["y"].to_numpy(float)
    wk = gg["week"].to_numpy(int)
    pp = gg["played_home_probability"].to_numpy(float)
    zl = gg["delta_logit_ladder"].to_numpy(float)
    zr = gg["delta_logit_read"].to_numpy(float)
    pl, pr = arm(pp, zl), arm(pp, zr)
    rng = np.random.default_rng(M.SEED)
    decisive = {
        "read_vs_ladder": U1.record(pr, pl, y),
        "read_vs_played_card": U1.record(pr, pp, y),
        "ladder_vs_played_card": U1.record(pl, pp, y),
    }
    looks = {
        "read_vs_ladder": fcell(pr, pl, y, wk, rng),
        "read_vs_played_card": fcell(pr, pp, y, wk, rng),
        "ladder_vs_played_card": fcell(pl, pp, y, wk, rng),
    }
    nulls = matched_null(zl, zr, pp, y, wk, rng)
    u2 = pd.read_parquet(M.OUT / "games.parquet")
    u2 = u2[u2["week"].between(*UNIT2_WEEKS)]
    u2ps = u2["p_served"].to_numpy(float)
    u2off = U1.logit(u2ps)
    u2zl = U1.logit(u2["p_mix_lad"].to_numpy(float)) - u2off
    u2zr = U1.logit(u2["p_mix_read"].to_numpy(float)) - u2off
    u2y = u2["y"].to_numpy(float)
    u2wk = u2["week"].to_numpy(int)
    q_zl = np.concatenate([u2zl, zl])
    q_zr = np.concatenate([u2zr, zr])
    q_y = np.concatenate([u2y, y])
    q_wk = np.concatenate([u2wk, wk])
    q_pp = np.concatenate([u2ps, pp])
    pooled = fcell(arm(q_pp, q_zr), arm(q_pp, q_zl), q_y, q_wk, rng)
    pooled_null = matched_null(q_zl, q_zr, q_pp, q_y, q_wk, rng)["read_vs_ladder"]
    unit2 = fcell(arm(u2ps, u2zr), arm(u2ps, u2zl), u2y, u2wk, rng)
    for k, v in looks.items():
        for metric in ("log_loss", "accuracy_points"):
            v[metric]["matched_null_percentile"] = nulls[k][metric]
    for metric in ("log_loss", "accuracy_points"):
        pooled[metric]["matched_null_percentile"] = pooled_null[metric]
    n = len(gg)
    if len(np.unique(wk)) > 1:
        p_int, _ = U1.lowo(np.ones((n, 1)), y, np.zeros(n), wk)
        control = fcell(p_int, pp, y, wk, rng)
    else:
        control = {"skipped": "leave-one-week-out intercept needs at least two weeks"}
    logged = g[g["logged_row"]]
    stamp = real_now.strftime(STAMP)
    body = {
        "lead": "LEAD-108",
        "declaration": f"declaration 2 (2026-11-21T15:51Z), {LANE}",
        "family": FAMILY,
        "dry_run": dry,
        "real_outcomes_graded": not dry,
        "outcomes": (
            "synthetic Bernoulli(played card home probability); no real outcome read"
            if dry
            else "data/processed/game_features.parquet result, every scored week final"
        ),
        "scored_at_utc": stamp,
        "weeks": weeks,
        "log_dir": str(log_dir),
        "code_sha256": {c: sha(c) for c in CODE},
        "declared_mixture_sha256": DECLARED_MIXTURE_SHA256,
        "looks": {
            "count": LOOKS,
            "family": FAMILY,
            "detail": (
                "3 contrasts (read vs ladder, read vs played card, ladder vs played card) x 2 "
                "metrics (log loss, accuracy points) on the forward weeks + pooled read vs "
                "ladder with unit 2 wk2_10 x 2 metrics"
            ),
        },
        "decisive_records": decisive,
        "forward": looks,
        "pooled_read_vs_ladder_with_unit2": pooled,
        "unit2_wk2_10_read_vs_ladder_beside_not_a_look": unit2,
        "intercept_only_control_vs_played_card": control,
        "mean_brier": {
            k: float(U1.br(v, y).mean()) for k, v in (("played", pp), ("ladder", pl), ("read", pr))
        },
        "mean_log_loss": {
            k: float(U1.ll(v, y).mean()) for k, v in (("played", pp), ("ladder", pl), ("read", pr))
        },
        "reliability_5_bins": (
            {k: U1.reliability(v, y, 5) for k, v in (("played", pp), ("ladder", pl), ("read", pr))}
            if n >= 5
            else None
        ),
        "population": {
            "games": len(g),
            "pool_line_pushes_dropped": int(g["push"].sum()),
            "graded": n,
            "with_logged_row": len(logged),
            "without_logged_row_zero_difference": int((~g["logged_row"]).sum()),
            "with_valued_players": int((g["valued_players"] > 0).sum()),
            "base_is_played_publication_share": (
                float(logged["base_is_played_publication"].astype(bool).mean())
                if len(logged)
                else None
            ),
            "logged_line_equals_graded_line": int(
                (logged["logged_line"].astype(float) == logged["pool_line"].astype(float)).sum()
            ),
            "line_source": {k: int(v) for k, v in g["line_source"].value_counts().items()},
            "skipped": skipped,
        },
        "log_files": files,
        "games": json.loads(g.to_json(orient="records")),
    }
    desc = {
        "read_vs_ladder": "read-weighted vs ladder-weighted mixture on the played card",
        "read_vs_played_card": "read-weighted mixture vs the played card",
        "ladder_vs_played_card": "ladder-weighted mixture vs the played card",
    }
    tag = {"read_vs_ladder": "read_vs_ladder", "read_vs_played_card": "read_vs_played"}
    tag["ladder_vs_played_card"] = "ladder_vs_played"
    unit = {"log_loss": "ll", "accuracy_points": "acc"}
    span = f"2026 weeks {weeks[0]}-{weeks[-1]}"
    cells = [
        batch_cell(
            f"lead108_fullmix_{tag[k]}_{unit[m]}_fwd_wk{weeks[0]}_{weeks[-1]}",
            f"LEAD-108 declared forward look: {desc[k]}, {m.replace('_', ' ')} at the pool line, "
            f"{span}, unfitted",
            looks[k],
            m,
            len(np.unique(wk)),
            looks[k][m]["matched_null_percentile"],
            dry,
        )
        for k in looks
        for m in ("log_loss", "accuracy_points")
    ]
    cells += [
        batch_cell(
            f"lead108_fullmix_read_vs_ladder_{unit[m]}_pooled_wk2_{weeks[-1]}",
            "LEAD-108 pooled look: read-weighted vs ladder-weighted mixture, unit 2 weeks 2-10 "
            f"at the opener plus forward {span} at the pool line, {m.replace('_', ' ')}",
            pooled,
            m,
            len(np.unique(q_wk)),
            pooled[m]["matched_null_percentile"],
            dry,
        )
        for m in ("log_loss", "accuracy_points")
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "dry_run_" if dry else ""
    res = out_dir / f"{prefix}score_{stamp}.json"
    res.write_text(json.dumps(body, indent=1, default=str))
    batch = out_dir / f"{prefix}weak_signals_batch.json"
    batch.write_text(json.dumps({"cells": cells}, indent=1))
    print(json.dumps({k: v for k, v in body.items() if k not in ("games", "log_files")}, indent=1))
    print(res)
    print(batch, len(cells), "cells, not recorded")


def main():
    p = argparse.ArgumentParser(
        description=(
            "LEAD-108 declared forward log: unfitted read-weighted and ladder-weighted injury "
            "scenario mixtures on the discrete margin lattice, per game, before kickoff, weeks "
            "12-18 of 2026. Research only; never changes a served side."
        )
    )
    sp = p.add_subparsers(dest="cmd", required=True)
    lg = sp.add_parser("log", help="write per-game arm probabilities for one week")
    lg.add_argument("--week", default="auto", help="week number or auto (first unplayed week)")
    lg.add_argument(
        "--smoke", action="store_true", help="any week; file marked smoke and never scored"
    )
    lg.add_argument("--as-of", default=None, help="smoke only: UTC time standing in for now")
    lg.set_defaults(fn=cmd_log)
    sc = sp.add_parser(
        "score", help="grade the declared 8 looks once after Week 18 settles (2027-01-12)"
    )
    sc.add_argument(
        "--dry-run",
        action="store_true",
        help="synthetic outcomes, no finality or date check, output outside artifacts/",
    )
    sc.add_argument("--log-dir", default=None, help="dry-run only: folder of log files")
    sc.add_argument("--out-dir", default=None, help="dry-run only: output folder")
    sc.add_argument("--weeks", default=None, help="dry-run only: comma-separated weeks")
    sc.set_defaults(fn=cmd_score)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
