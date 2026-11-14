import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.published_picks import game_deadlines

ROOT = Path(__file__).resolve().parents[1]
SEASON = 2026
RUN_FAMILIES = {
    "mod24": ("artifacts/mod24_forward/runs/*.json", "arm_home_cover_probability"),
    "mod25_combined": ("artifacts/mod25_forward/runs/*.json", "combined_home_cover_probability"),
    "mod25_wfj": ("artifacts/mod25_wfj_forward/runs/*.json", "student_home_cover_probability"),
    "mod54": ("artifacts/mod54_forward/runs/*.json", "arm_home_cover_probability"),
    "draftcap": ("artifacts/draftcap_forward/runs/*.json", "arm_home_cover_probability"),
}
TERM_DIRS = [
    "sky03",
    "sky03_separated",
    "sky04",
    "env02",
    "sky01",
    "sky01snap",
    "surface",
    "mod35_cold_exposure",
    "mod41_residual_load",
    "mod41_residual_load_served",
    "mod41_avgwin_served",
    "mod39_levdiff_served",
]
CFB_EXCLUDED = ["xlg12_cfb_v1", "xlg12_cfb_v2"]
LEAD99_DECLARED = "lead99"
LEAD99_FIRST_VIEW = "lead99_first_view"
PICK_DEADLINE_CANDIDATES = [LEAD99_DECLARED, LEAD99_FIRST_VIEW]
LATEST_BEFORE_DEADLINE = [LEAD99_DECLARED]
RECORD_RULES = {
    LEAD99_DECLARED: "latest record written before the game's pick deadline, any view",
    LEAD99_FIRST_VIEW: "earliest record written before the game's pick deadline, any view",
}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records():
    rows = []
    for name, (pattern, column) in RUN_FAMILIES.items():
        for path in sorted(glob.glob(str(ROOT / pattern))):
            doc = _load(path)
            if doc.get("season", SEASON) != SEASON:
                continue
            for row in doc["rows"]:
                rows.append(
                    {
                        "candidate": name,
                        "game_id": row["game_id"],
                        "p": row.get(column),
                        "written_at": row["recorded_at_utc"],
                        "row_deadline": row.get("deadline"),
                        "served_p": row.get("served_home_cover_probability"),
                        "frozen_at": None,
                    }
                )
    for term in TERM_DIRS:
        for path in sorted(
            glob.glob(str(ROOT / "artifacts/forward_terms" / term / "2026_week*.json"))
        ):
            doc = _load(path)
            for game in doc["games"]:
                rows.append(
                    {
                        "candidate": f"term_{term}",
                        "game_id": game["game_id"],
                        "p": game.get("combined_home_probability"),
                        "written_at": doc["created_at_utc"],
                        "row_deadline": game["kickoff_utc"],
                        "served_p": game.get("base_home_probability"),
                        "frozen_at": doc.get("frozen_created_at_utc"),
                    }
                )
    for path in sorted(glob.glob(str(ROOT / "artifacts/lead99_llm_read/runs/*.json"))):
        doc = _load(path)
        for row in doc.get("parsed_rows", []):
            for name in PICK_DEADLINE_CANDIDATES:
                rows.append(
                    {
                        "candidate": name,
                        "game_id": row["game_id"],
                        "p": row.get("home_cover_probability"),
                        "written_at": doc["recorded_at_utc"],
                        "row_deadline": None,
                        "served_p": None,
                        "frozen_at": None,
                    }
                )
    pool = pd.read_parquet(ROOT / "artifacts/pool_relative_forward_log/log.parquet")
    for row in pool.loc[~pool["backfill"].astype(bool)].itertuples():
        rows.append(
            {
                "candidate": "mod27_pool_move",
                "game_id": row.game_id,
                "p": row.challenger_home_probability,
                "written_at": row.run_at_utc.isoformat(),
                "row_deadline": row.pick_deadline_utc.isoformat(),
                "served_p": row.served_home_probability,
                "frozen_at": None,
            }
        )
    return pd.DataFrame(rows)


def _played_card():
    picks = pd.read_parquet(ROOT / "artifacts/clv_ledger/published_picks.parquet")
    picks = picks.loc[
        picks["season"].eq(SEASON) & picks["published_at_utc"].le(picks["pick_deadline_utc"])
    ]
    picks = picks.sort_values("published_at_utc").drop_duplicates("game_id", keep="last").copy()
    pick_home = picks["pick_team"].eq(picks["home_team"])
    picks["served_p"] = np.where(pick_home, picks["displayed_score"], 1 - picks["displayed_score"])
    picks["served_side"] = np.where(pick_home, "HOME", "AWAY")
    picks["card_line"] = picks["market_spread"]
    return picks[["game_id", "served_p", "served_side", "card_line"]]


def _pol_records(ids):
    if not ids:
        return pd.DataFrame()
    frame = pd.read_parquet(ROOT / "artifacts/prospective/challenger_decisions.parquet")
    frame = frame.loc[frame["challenger_id"].isin(ids) & frame["season"].eq(SEASON)]
    return pd.DataFrame(
        {
            "candidate": "pol_" + frame["challenger_id"],
            "game_id": frame["game_id"],
            "p": np.nan,
            "side": frame["pick_side"],
            "written_at": frame["recorded_at_utc"].map(
                lambda value: pd.Timestamp(value).isoformat()
            ),
            "row_deadline": frame["kickoff"].map(lambda value: pd.Timestamp(value).isoformat()),
            "served_p": np.nan,
            "frozen_at": None,
        }
    )


def _utc(series):
    return pd.to_datetime(series, utc=True, format="ISO8601")


def _pick_deadlines(games):
    deadlines = {}
    for _, week in games.groupby("week"):
        deadlines.update(game_deadlines(week[["game_id"]].assign(kickoff=week["kickoff_utc"])))
    return deadlines


def _build(weeks, pol_ids, require):
    games = pd.read_parquet(ROOT / "data/processed/game_features.parquet")
    games = games.loc[games["season"].eq(SEASON) & games["week"].isin(weeks)].copy()
    games["kickoff_utc"] = _utc(games["kickoff"])
    records = pd.concat([_records(), _pol_records(pol_ids)], ignore_index=True)
    if "side" not in records:
        records["side"] = None
    records["written_at"] = _utc(records["written_at"])
    records["row_deadline"] = _utc(records["row_deadline"])
    records["frozen_at"] = _utc(records["frozen_at"])
    records = records.loc[records["game_id"].isin(games["game_id"])]
    deadline = records.groupby("game_id")["row_deadline"].min()
    games["deadline_utc"] = games["game_id"].map(deadline)
    games["deadline_utc"] = games[["deadline_utc", "kickoff_utc"]].min(axis=1)
    games["pick_deadline_utc"] = games["game_id"].map(_pick_deadlines(games))
    records = records.merge(games[["game_id", "deadline_utc", "pick_deadline_utc"]], on="game_id")
    records["cutoff_utc"] = records["deadline_utc"].where(
        ~records["candidate"].isin(PICK_DEADLINE_CANDIDATES),
        records[["deadline_utc", "pick_deadline_utc"]].min(axis=1),
    )
    records["verified"] = records["written_at"].lt(records["cutoff_utc"]) & (
        records["frozen_at"].isna() | records["written_at"].ge(records["frozen_at"])
    )
    ordered = records.loc[records["verified"]].sort_values("written_at")
    latest = ordered["candidate"].isin(LATEST_BEFORE_DEADLINE)
    kept = pd.concat(
        [
            ordered.loc[~latest].drop_duplicates(["candidate", "game_id"], keep="first"),
            ordered.loc[latest].drop_duplicates(["candidate", "game_id"], keep="last"),
        ]
    )
    kept["p_side"] = np.where(
        kept["p"].notna(), np.where(kept["p"].ge(0.5), "HOME", "AWAY"), kept["side"]
    )
    base = (
        records.loc[records["served_p"].notna()]
        .sort_values("written_at")
        .drop_duplicates("game_id")
        .set_index("game_id")
    )
    games["base_at_record_p"] = games["game_id"].map(base["served_p"])
    games = games.merge(_played_card(), on="game_id", how="left")
    wide = games[
        [
            "game_id",
            "week",
            "deadline_utc",
            "served_side",
            "served_p",
            "base_at_record_p",
            "card_line",
            "result",
            "home_cover",
        ]
    ].copy()
    for name, part in kept.groupby("candidate"):
        part = part.set_index("game_id")
        wide[f"{name}__side"] = wide["game_id"].map(part["p_side"])
        wide[f"{name}__p"] = wide["game_id"].map(part["p"])
        wide[f"{name}__written_at"] = wide["game_id"].map(part["written_at"])
        wide[f"{name}__base_p"] = wide["game_id"].map(part["served_p"])
    present = sorted(set(kept["candidate"]))
    candidates = sorted(set(present).intersection(require)) if require else present
    wide["matched"] = (
        wide[[f"{c}__side" for c in candidates]].notna().all(axis=1) if candidates else False
    )
    wide["status"] = np.where(
        wide["result"].isna(), "pending", np.where(wide["home_cover"].isna(), "push", "graded")
    )
    return wide, candidates, records


def _score(wide, candidates):
    done = wide.loc[wide["matched"] & wide["status"].eq("graded")]
    out = []
    for name in ["served", *candidates]:
        side = done["served_side"] if name == "served" else done[f"{name}__side"]
        p = done["served_p"] if name == "served" else done[f"{name}__p"]
        home_won = done["home_cover"].eq(1.0)
        correct = side.eq("HOME") == home_won
        decisive = done["served_side"].ne(side) & side.notna() & done["served_side"].notna()
        valid = p.notna()
        clipped = p.clip(1e-6, 1 - 1e-6)
        loss = -pd.Series(
            np.where(home_won, np.log(clipped), np.log(1 - clipped)), index=done.index
        )
        brier = (p - home_won.astype(float)) ** 2
        out.append(
            {
                "candidate": name,
                "games": len(done),
                "accuracy": float(correct.mean()) if len(done) else np.nan,
                "decisive_games": int(decisive.sum()),
                "decisive_wins": int((correct & decisive).sum()),
                "log_loss": float(loss[valid].mean()) if valid.any() else np.nan,
                "brier": float(brier[valid].mean()) if valid.any() else np.nan,
            }
        )
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(
        description="Matched-game ledger across frozen NFL forward candidates"
    )
    parser.add_argument(
        "--weeks", type=int, nargs="+", default=list(range(4, 10)), help="2026 weeks to include"
    )
    parser.add_argument(
        "--require",
        nargs="*",
        default=None,
        help="candidates needing a verified pregame record (default: all with any record)",
    )
    parser.add_argument(
        "--pol-ids",
        nargs="*",
        default=[],
        help="POL-10 challenger ids to include as side-only candidates",
    )
    parser.add_argument("--out", default="artifacts/forward_ledger", help="output directory")
    args = parser.parse_args()
    wide, candidates, records = _build(
        args.weeks, args.pol_ids, set(args.require) if args.require else None
    )
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out / "ledger.parquet", index=False)
    all_candidates = sorted(set(records["candidate"]))
    drop = {name: int(wide[f"{name}__side"].isna().sum()) for name in candidates}
    no_log = [t for t in TERM_DIRS if f"term_{t}" not in all_candidates]
    manifest = {
        "weeks": args.weeks,
        "games_in_universe": len(wide),
        "required_candidates": candidates,
        "matched_games": int(wide["matched"].sum()),
        "matched_graded": int((wide["matched"] & wide["status"].eq("graded")).sum()),
        "matched_pending": int((wide["matched"] & wide["status"].eq("pending")).sum()),
        "games_missing_by_candidate": drop,
        "candidates_with_no_verified_record": no_log,
        "unverified_records_by_candidate": {
            k: int(v)
            for k, v in records.loc[~records["verified"]].groupby("candidate").size().items()
        },
        "excluded_cfb_ledgers": CFB_EXCLUDED,
        "record_rule_default": "earliest verified record",
        "record_rule_by_candidate": RECORD_RULES,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    per_week = (
        wide.groupby("week")[["matched"]].sum().join(wide.groupby("week").size().rename("games"))
    )
    print(per_week.to_string())
    print(_score(wide, candidates).to_string(index=False))
    show = ["game_id", "served_side", "served_p", "status"] + [
        c for c in wide.columns if c.endswith(("__side", "__p"))
    ]
    print(wide.loc[wide["matched"], show].to_string(index=False))


if __name__ == "__main__":
    main()
