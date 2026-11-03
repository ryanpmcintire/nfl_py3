import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "xlg12_cfb_forward"
LINES = ROOT / "data" / "raw" / "cfbd_lines"
VARIANTS = ("base", "cv", "slope")
LEDGER = ART / "forward_ledger.jsonl"
RETRO = ART / "retro2026_predictions.csv"
RETRO_OPEN = ART / "retro2026_openinput_predictions.csv"


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def use_version(version):
    global ART_V, LEDGER, RETRO, RETRO_OPEN, TAG
    ART_V = ART / version if version != "v1" else ART
    LEDGER = ART_V / "forward_ledger.jsonl"
    RETRO = ART_V / "retro2026_predictions.csv"
    RETRO_OPEN = ART_V / "retro2026_openinput_predictions.csv"
    TAG = "" if version == "v1" else f"_{version}"


ART_V = ART
TAG = ""


def load_frozen():
    return {v: json.loads((ART_V / f"frozen_{v}.json").read_text()) for v in VARIANTS}


def load_lines(week):
    rows = []
    for manifest in sorted(LINES.glob("*/manifest.json")):
        doc = json.loads(manifest.read_text())
        p = doc["request_parameters"]
        if p["year"] != 2026 or p["week"] != week or p["seasonType"] != "regular":
            continue
        for game in json.loads((manifest.parent / doc["file"]).read_text()):
            for ln in game["lines"]:
                if ln.get("spread") is None:
                    continue
                spread = float(ln["spread"])
                fmt = str(ln.get("formattedSpread", ""))
                tokens = fmt.rsplit(" ", 1)[0]
                if tokens == game["homeTeam"] and spread > 0:
                    raise SystemExit(f"sign convention violated {game['id']}")
                if tokens == game["awayTeam"] and spread < 0:
                    raise SystemExit(f"sign convention violated {game['id']}")
                opn = ln.get("spreadOpen")
                rows.append(
                    {
                        "game_id": int(game["id"]),
                        "provider": ln["provider"],
                        "spread_line": -spread,
                        "spread_open": None if opn is None else -float(opn),
                        "total_line": ln.get("overUnder"),
                        "total_open": ln.get("overUnderOpen"),
                        "observed_at_utc": doc["observed_at_utc"],
                    }
                )
    return pd.DataFrame(rows)


def predict(frame, frozen):
    out = []
    for v, f in frozen.items():
        cols = f["features"]
        x = frame[cols].to_numpy(float)
        med = np.array(f["impute_median"])
        x = np.where(np.isnan(x), med, x)
        a = (x - np.array(f["scaler_mean"])) / np.array(f["scaler_scale"])
        m = f["prediction_scale"] * (a @ np.array(f["coef"]) + f["intercept"])
        sd = f["residual_sd"]
        part = frame[["game_id", "provider"]].copy()
        part["variant"] = v
        part["pred_ats_margin"] = m
        part["p_home_cover_line"] = [phi(z / sd) for z in m]
        proj = frame["spread_line"].to_numpy(float) + m
        part["proj_home_margin"] = proj
        open_ = frame["spread_open"].to_numpy(float)
        part["p_home_cover_open"] = [
            np.nan if np.isnan(o) else phi((pm - o) / sd) for pm, o in zip(proj, open_, strict=True)
        ]
        out.append(part)
    return pd.concat(out, ignore_index=True)


def side(p, home, away):
    if p > 0.5:
        return home
    if p < 0.5:
        return away
    return "none"


def build(week, feats, frozen, open_input=False):
    lines = load_lines(week)
    g = feats.loc[feats["week"].eq(week)].merge(lines, on="game_id", how="inner")
    g["total_line"] = pd.to_numeric(g["total_line"], errors="coerce")
    g["total_open"] = pd.to_numeric(g["total_open"], errors="coerce")
    if open_input:
        g["spread_close"] = g["spread_line"]
        g = g.dropna(subset=["spread_open", "total_open"]).copy()
        g["spread_line"] = g["spread_open"]
        g["total_line"] = g["total_open"]
    pr = predict(g, frozen)
    return g.merge(pr, on=["game_id", "provider"], how="inner")


def forward(week):
    feats = pd.read_parquet(ART / "features_2026.parquet")
    frozen = load_frozen()
    now = datetime.now(UTC)
    d = build(week, feats, frozen)
    d["kickoff"] = pd.to_datetime(d["kickoff"], utc=True)
    d["observed"] = pd.to_datetime(d["observed_at_utc"], utc=True)
    unkicked = d["kickoff"].gt(pd.Timestamp(now)) & d["observed"].lt(d["kickoff"])
    skipped = int((~unkicked).sum())
    d = d.loc[unkicked]
    seen = set()
    if LEDGER.exists():
        for ln in LEDGER.read_text().splitlines():
            r = json.loads(ln)
            seen.add((r["game_id"], r["provider"], r["variant"]))
    written = 0
    with LEDGER.open("a") as fh:
        for r in d.itertuples():
            key = (int(r.game_id), r.provider, r.variant)
            if key in seen:
                continue
            rec = {
                "league": "cfb",
                "season": 2026,
                "week": week,
                "game_id": int(r.game_id),
                "home_team": r.home_team,
                "away_team": r.away_team,
                "kickoff_utc": r.kickoff.isoformat(),
                "provider": r.provider,
                "variant": r.variant,
                "line_home_spread": r.spread_line,
                "opener_home_spread": None if pd.isna(r.spread_open) else r.spread_open,
                "total_line": None if pd.isna(r.total_line) else r.total_line,
                "pred_ats_margin": r.pred_ats_margin,
                "proj_home_margin": r.proj_home_margin,
                "p_home_cover_line": r.p_home_cover_line,
                "side_line": side(r.p_home_cover_line, r.home_team, r.away_team),
                "line_basis": "observed_pregame_line",
                "line_observed_at_utc": r.observed_at_utc,
                "written_at_utc": datetime.now(UTC).isoformat(),
                "frozen_created_at": frozen[r.variant]["created_at"],
            }
            fh.write(json.dumps(rec) + "\n")
            written += 1
    print("week", week, "written", written, "skipped_kicked_or_late", skipped)


def retro(weeks, open_input=False):
    feats = pd.read_parquet(ART / "features_2026.parquet")
    frozen = load_frozen()
    parts = [build(w, feats, frozen, open_input) for w in weeks]
    d = pd.concat(parts, ignore_index=True)
    d = d.loc[d["completed"].astype(bool)].copy()
    d["result"] = d["home_points"] - d["away_points"]
    d["cell"] = d["variant"] + (f"{TAG}_retro2026_openinput" if open_input else f"{TAG}_retro2026")
    d.to_csv(RETRO_OPEN if open_input else RETRO, index=False)
    rows = []
    for (prov, cell), s in d.groupby(["provider", "cell"]):
        for name, line, p in (
            ("open", "spread_open", "p_home_cover_open"),
            ("line", "spread_line", "p_home_cover_line"),
        ):
            t = s.dropna(subset=[line, p])
            ats = t["result"] - t[line]
            dec = ats != 0
            hit = ((t[p] > 0.5) == (ats > 0))[dec]
            n = int(dec.sum())
            k = int(hit.sum())
            ll = -np.mean(
                np.where(
                    ats[dec] > 0,
                    np.log(t.loc[dec, p].clip(1e-9, 1 - 1e-9)),
                    np.log(1 - t.loc[dec, p].clip(1e-9, 1 - 1e-9)),
                )
            )
            rows.append(
                {
                    "provider": prov,
                    "cell": cell,
                    "graded_at": name,
                    "n": n,
                    "hits": k,
                    "acc": k / n if n else np.nan,
                    "logloss": ll,
                    "logloss_coinflip": math.log(2),
                }
            )
    print(pd.DataFrame(rows).to_string())


def main():
    ap = argparse.ArgumentParser(description="XLG-12 frozen CFB forward predictor and retro scorer")
    ap.add_argument("--forward-week", type=int)
    ap.add_argument("--retro-weeks", type=int, nargs="+")
    ap.add_argument("--open-input", action="store_true")
    ap.add_argument("--version", choices=("v1", "v2"), default="v1")
    a = ap.parse_args()
    use_version(a.version)
    if a.forward_week is not None:
        forward(a.forward_week)
    if a.retro_weeks:
        retro(a.retro_weeks, a.open_input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
