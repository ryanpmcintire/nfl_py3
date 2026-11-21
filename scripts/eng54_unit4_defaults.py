import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "artifacts/eng54/label_audit_unit4.json"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def college(ev):
    seasons = {}
    for season in ev.CFB_TRAIN_SEASONS:
        lines = pd.read_parquet(ev.CFB_LINES_ROOT / f"season={season}" / "lines.parquet")
        sched = pd.read_parquet(ev.CFB_SCHEDULES_ROOT / f"season={season}" / "schedules.parquet")
        spread = lines.loc[lines.market_type.eq("spread") & lines.book.eq(ev.CFB_BOOK) & lines.opening_lines.notna()]
        base = sched.loc[sched.completed.fillna(False) & ~sched.neutral_site.fillna(False)]
        matched = spread.merge(base[["game_id", "home_team", "away_team"]], on="game_id", how="inner")
        with_home = matched.loc[matched.abbr.eq(matched.home_team), "game_id"].unique()
        with_away = matched.loc[matched.abbr.eq(matched.away_team), "game_id"].unique()
        games_with_lines = matched["game_id"].unique()
        seasons[int(season)] = {
            "games_with_lines": int(len(games_with_lines)),
            "home_row_by_name": int(len(with_home)),
            "dropped_no_home_name_match": int(len(set(games_with_lines) - set(with_home))),
            "dropped_but_away_row_matched": int(len(set(with_away) - set(with_home))),
            "completed_points_missing": int(base[["home_points", "away_points"]].isna().any(axis=1).sum()),
        }
    pop = ev.load_cfb_population()
    totals = {k: int(sum(v[k] for v in seasons.values())) for k in next(iter(seasons.values()))}
    return {
        "by_season": seasons,
        "totals": totals,
        "population_rows": int(len(pop)),
        "prior_move_filled_zero_rows": int((pop["home_prior_move"].isna() | pop["away_prior_move"].isna()).sum()),
        "rest_filled_median_rows": int((pop["home_rest"].isna() | pop["away_rest"].isna()).sum()),
        "label_nan_margin_rows": int(pop["home_margin"].isna().sum()),
        "label_mismatch_rows": int((pop["home_covered"] != (pop["home_margin"] + pop["home_open"] > 0).astype(float)).sum()),
    }


def nfl(ev, u3):
    pop, _ = ev.build_fit_population(ROOT / "artifacts", ROOT / "data")
    cols = [c for c in pop.columns if any(k in c for k in ("result", "margin", "covered", "open"))]
    res = {"fit_population_rows": int(len(pop)), "columns": cols, "home_covered_nan": int(pop["home_covered"].isna().sum())}
    if "result" in pop.columns:
        line = pop["tue_open_home_spread"]
        mv = pop["result"] - line
        ok = mv.notna() & pop["home_covered"].notna()
        res["label_vs_tue_open_mismatch"] = int((pop.loc[ok, "home_covered"] != (mv[ok] > 0).astype(float)).sum())
        res["push_rows_in_population"] = int(mv.eq(0).sum())
    ext = pd.read_parquet(u3.EXTENDED_POPULATION_PATH)
    ext["game_id"] = ext["game_id"].astype(str)
    true_ids = set(pop["game_id"].astype(str))
    missing = ~ext["game_id"].isin(true_ids)
    ext_seasons = ext.loc[missing, "season"].value_counts().sort_index()
    res["extended_rows"] = int(len(ext))
    res["extended_columns"] = [c for c in ext.columns if any(k in c for k in ("result", "margin", "covered", "open", "spread"))]
    res["extended_open_move_filled_zero_unit3"] = int(missing.sum())
    res["extended_open_move_filled_zero_by_season"] = {int(k): int(v) for k, v in ext_seasons.items()}
    sbr = pd.read_parquet(u3.SBR_SCORED_ARTIFACT)
    sbr["game_id"] = sbr["game_id"].astype(str)
    proxy = sbr.drop_duplicates("game_id").set_index("game_id")["proxy_open_home_spread"]
    tue = pop.assign(game_id=pop["game_id"].astype(str)).drop_duplicates("game_id").set_index("game_id")["tue_open_home_spread"]
    home_open = ext["game_id"].map(tue).fillna(ext["game_id"].map(proxy))
    res["extended_home_open_missing"] = int(home_open.isna().sum())
    if "result" in ext.columns and "home_covered" in ext.columns:
        mv = ext["result"] - home_open
        ok = mv.notna() & ext["home_covered"].notna()
        mism = ok & (ext["home_covered"] != (mv > 0).astype(float))
        res["extended_label_vs_feature_open_mismatch"] = int(mism.sum())
        res["extended_label_vs_feature_open_mismatch_on_push"] = int((mism & mv.eq(0)).sum())
        res["extended_label_checked"] = int(ok.sum())
        res["extended_label_mismatch_by_season"] = {int(k): int(v) for k, v in ext.loc[mism, "season"].value_counts().sort_index().items()}
    return res


def forward_logs():
    res = {}
    specs = {
        "mod24_forward": ("served_side", "served_pool_side", "served_line"),
        "armb_forward": ("served_model_side", "served_pool_side", "served_line"),
        "draftcap_forward": ("served_model_side", "served_pool_side", "served_line"),
    }
    for name, (side, pool_side, served_line) in specs.items():
        rows = []
        for path in sorted((ROOT / "artifacts" / name / "runs").glob("*.json")):
            data = json.loads(path.read_text())
            for r in data.get("rows", []):
                rows.append({**r, "run": path.name})
        frame = pd.DataFrame(rows)
        if frame.empty:
            res[name] = {"rows": 0}
            continue
        latest = frame.sort_values("recorded_at_utc").drop_duplicates("game_id", keep="last")
        both = latest[side].notna() & latest[pool_side].notna()
        res[name] = {
            "runs": int(frame["run"].nunique()),
            "rows": int(len(frame)),
            "games_latest": int(len(latest)),
            "games_without_pool_side": int(latest[pool_side].isna().sum()),
            "model_side_ne_pool_side": int((both & latest[side].ne(latest[pool_side])).sum()),
            "games_compared": int(both.sum()),
            "arm_line_ne_served_line": int((latest["line"].notna() & latest[served_line].notna() & latest["line"].ne(latest[served_line])).sum()),
            "served_line_missing": int(latest[served_line].isna().sum()),
        }
    rep = ROOT / "artifacts/forward_2026_ftn_terms/report.json"
    if rep.is_file():
        data = json.loads(rep.read_text())
        res["forward_2026_ftn_terms"] = {"keys": list(data)[:20]}
    return res


def main() -> int:
    out = {}
    ev = load("opener_error_transfer_eval")
    u3 = load("opener_error_transfer_unit3")
    for name, fn in (("college", lambda: college(ev)), ("nfl", lambda: nfl(ev, u3)), ("forward_logs", forward_logs)):
        try:
            out[name] = fn()
        except Exception:
            out[name] = {"error": traceback.format_exc()[-1500:]}
        print(name, json.dumps(out[name], default=str)[:3000], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
