import argparse
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from playcaller_change_screen import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED  # noqa: E402

PAIRED = REPO / "artifacts" / "playcaller_change_leads" / "production_paired_lead29_full.parquet"
OUT = REPO / "artifacts" / "per07" / "first_game_fitted_term.json"
OUT_STAFF = REPO / "artifacts" / "per07" / "first_game_fitted_term_staff_state.json"
GAME_FEATURES = REPO / "data" / "processed" / "game_features.parquet"
EASTERN = ZoneInfo("America/New_York")
LINE_FREEZE_HOUR_ET = 12


def prepare() -> pd.DataFrame:
    d = pd.read_parquet(PAIRED)
    d = d.loc[d["margin_vs_open"].ne(0.0)].reset_index(drop=True)
    p = d["home_cover_probability_at_open"].clip(1e-6, 1 - 1e-6)
    d["logit_p"] = np.log(p / (1 - p))
    d["s"] = d["home_signal"] - d["away_signal"]
    d["y"] = d["margin_vs_open"].gt(0.0).astype(int)
    d["block"] = d["season"] * 100 + d["week"]
    return d


def prediction_timestamps(kickoff_utc: pd.Series) -> pd.Series:
    local = pd.to_datetime(kickoff_utc, utc=True).dt.tz_convert(EASTERN)
    days_back = (local.dt.weekday - 1) % 7
    tuesday = (local.dt.normalize() - pd.to_timedelta(days_back, unit="D")).dt.tz_localize(None)
    stamped = (tuesday + pd.Timedelta(hours=LINE_FREEZE_HOUR_ET)).dt.tz_localize(
        EASTERN, nonexistent="shift_forward", ambiguous="NaT"
    )
    return stamped.dt.tz_convert("UTC")


def staff_events(staff_state: Path) -> pd.DataFrame:
    s = pd.read_parquet(staff_state)
    s = s.loc[s["playcaller_status"].ne("unknown") & s["playcaller_usable_from"].notna()]
    s = s.sort_values("first_observed_at")
    first = s.groupby(["season", "team", "playcaller_name"], as_index=False).first()
    return first[["season", "team", "playcaller_name", "playcaller_side", "playcaller_usable_from"]]


def apply_staff_events(d: pd.DataFrame, events: pd.DataFrame, side: str | None) -> pd.DataFrame:
    meta = pd.read_parquet(GAME_FEATURES, columns=["game_id", "kickoff"])
    out = d.merge(meta, on="game_id", how="left", validate="one_to_one")
    out["predicted_at"] = prediction_timestamps(out["kickoff"])
    parts = out["game_id"].str.split("_", expand=True)
    out["away_team"] = parts[2]
    out["home_team"] = parts[3]
    out["home_signal"] = 0.0
    out["away_signal"] = 0.0
    used = []
    ev = events if side is None else events.loc[events["playcaller_side"].eq(side)]
    for e in ev.itertuples():
        usable_from = pd.Timestamp(e.playcaller_usable_from)
        mask = out["season"].eq(e.season) & (
            out["home_team"].eq(e.team) | out["away_team"].eq(e.team)
        )
        games = out.loc[mask & out["predicted_at"].gt(usable_from)].sort_values("predicted_at")
        if games.empty:
            continue
        g = games.iloc[0]
        assert g["predicted_at"] > usable_from
        col = "home_signal" if g["home_team"] == e.team else "away_signal"
        out.loc[g.name, col] = 1.0
        used.append(
            {
                "game_id": g["game_id"],
                "team": e.team,
                "side": e.playcaller_side,
                "usable_from": str(usable_from),
                "predicted_at": str(g["predicted_at"]),
            }
        )
    out["s"] = out["home_signal"] - out["away_signal"]
    out.attrs["events_used"] = used
    return out


def fit(train: pd.DataFrame, cols: list[str]) -> LogisticRegression:
    return LogisticRegression(penalty=None, max_iter=1000).fit(train[cols], train["y"])


def loso(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["p_base"] = np.nan
    out["p_term"] = np.nan
    coefs = {}
    for season in sorted(d["season"].unique()):
        train = d.loc[d["season"].ne(season)]
        test = d["season"].eq(season)
        base = fit(train, ["logit_p"])
        term = fit(train, ["logit_p", "s"])
        out.loc[test, "p_base"] = base.predict_proba(d.loc[test, ["logit_p"]])[:, 1]
        out.loc[test, "p_term"] = term.predict_proba(d.loc[test, ["logit_p", "s"]])[:, 1]
        coefs[int(season)] = {
            "s_coef": float(term.coef_[0][1]),
            "logit_p_coef": float(term.coef_[0][0]),
        }
    return out, coefs


def metrics(frame: pd.DataFrame, p_col: str) -> dict:
    p = frame[p_col].clip(1e-9, 1 - 1e-9)
    y = frame["y"]
    return {
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
    }


def block_bootstrap(frame: pd.DataFrame, stat, rng) -> np.ndarray:
    groups = [g for _, g in frame.groupby("block")]
    draws = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(groups), len(groups))
        draws[i] = stat(pd.concat([groups[j] for j in idx]))
    return draws


def summarize(draws: np.ndarray, point: float) -> dict:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "point": point,
        "ci95": [float(lo), float(hi)],
        "probability_positive": float((draws > 0).mean()),
    }


def analyze(d: pd.DataFrame, looks: int) -> dict:
    scored, coefs = loso(d)
    flagged = scored.loc[scored["s"].ne(0)]
    decisive = scored.loc[((scored["p_term"] >= 0.5) != (scored["p_base"] >= 0.5))]
    decisive_term_correct = int(((decisive["p_term"] >= 0.5).astype(int) == decisive["y"]).sum())
    decisive_base_correct = int(((decisive["p_base"] >= 0.5).astype(int) == decisive["y"]).sum())

    full_base = fit(d, ["logit_p"])
    full_term = fit(d, ["logit_p", "s"])
    ins = d.copy()
    ins["p_base"] = full_base.predict_proba(d[["logit_p"]])[:, 1]
    ins["p_term"] = full_term.predict_proba(d[["logit_p", "s"]])[:, 1]

    def acc_gain(f):
        return (
            ((f["p_term"] >= 0.5).astype(int) == f["y"]).mean()
            - ((f["p_base"] >= 0.5).astype(int) == f["y"]).mean()
        ) * 100.0

    def ll_gain(f):
        return metrics(f, "p_base")["log_loss"] - metrics(f, "p_term")["log_loss"]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    oos_acc = summarize(block_bootstrap(scored, acc_gain, rng), float(acc_gain(scored)))
    oos_ll = summarize(block_bootstrap(scored, ll_gain, rng), float(ll_gain(scored)))
    ins_acc = float(acc_gain(ins))
    ins_ll = float(ll_gain(ins))

    def coef_stat(f):
        return fit(f, ["logit_p", "s"]).coef_[0][1]

    coef_draws = block_bootstrap(d, coef_stat, rng)
    coef = summarize(coef_draws, float(full_term.coef_[0][1]))

    result = {
        "family": "per07_first_game_new_playcaller_fitted_term",
        "grade": "opener",
        "games": len(d),
        "flagged_games": len(flagged),
        "looks": looks,
        "decisive_games_pick_changed_oos": len(decisive),
        "decisive_term_correct": decisive_term_correct,
        "decisive_base_correct": decisive_base_correct,
        "market_only_accuracy": metrics(scored.assign(p_mkt=0.5), "p_mkt")["accuracy"],
        "baseline_oos": metrics(scored, "p_base"),
        "term_oos": metrics(scored, "p_term"),
        "oos_accuracy_points_gain": oos_acc,
        "oos_log_loss_gain": oos_ll,
        "in_sample_accuracy_points_gain": ins_acc,
        "in_sample_log_loss_gain": ins_ll,
        "gap_accuracy_points": ins_acc - oos_acc["point"],
        "gap_log_loss": ins_ll - oos_ll["point"],
        "s_coefficient_full_sample": coef,
        "s_coefficient_per_heldout_season": coefs,
        "flagged_record_baseline_pick": [
            int(((flagged["p_base"] >= 0.5).astype(int) == flagged["y"]).sum()),
            len(flagged),
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staff-state", type=Path, default=None)
    args = parser.parse_args()
    d = prepare()
    if args.staff_state is None:
        result = analyze(d, 3)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    events = staff_events(args.staff_state)
    previous = json.loads(OUT.read_text())
    result = {"events_in_table": len(events), "previous_12_event_run": previous}
    for label, side in (("all", None), ("offence_only", "offence")):
        frame = apply_staff_events(d, events, side)
        r = analyze(frame, 2)
        r["family"] = "per07_first_game_new_playcaller_fitted_term"
        r["events_used"] = frame.attrs["events_used"]
        r["difference_from_12_event_run"] = {
            "oos_accuracy_points_gain": r["oos_accuracy_points_gain"]["point"]
            - previous["oos_accuracy_points_gain"]["point"],
            "oos_log_loss_gain": r["oos_log_loss_gain"]["point"]
            - previous["oos_log_loss_gain"]["point"],
        }
        result[label] = r
    OUT_STAFF.parent.mkdir(parents=True, exist_ok=True)
    OUT_STAFF.write_text(json.dumps(result, indent=2, default=str))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "previous_12_event_run"}, indent=2, default=str
        )
    )


if __name__ == "__main__":
    main()
