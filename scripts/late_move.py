from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_candidate as v4
from mod27_fenced_repairs_replay import FOUR, Fit, log_loss, paired, record

from nfl_ats.sharp_book_movement_features import LEADER_BOOKS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/late_move"
QUOTES = ROOT / "artifacts/sharp_book_weighted_movement/spread_quotes.parquet"
QW_OPENER = ROOT / "artifacts/v4_candidate/opener_QW.parquet"
ET = "America/New_York"
QUOTE_SEASONS = (2023, 2024, 2025)
POOL_FREEZE_DAYS_BEFORE_SUNDAY = 5
POOL_FREEZE_HOUR_ET = 12
TERM = "late_move_pred"
ORACLE = "late_move_oracle"
FAMILY = "late_move_v1"
NULL_DRAWS = 200
NULL_SEED = 20261116


def anchors(table):
    kick, cutoff = v4.kickoff_and_cutoff(table)
    frame = pd.DataFrame(
        {
            "game_id": table["game_id"].to_numpy(),
            "kick": kick.to_numpy(),
            "cutoff": cutoff.to_numpy(),
        }
    )
    frame["kick"] = pd.to_datetime(frame["kick"], utc=True)
    frame["cutoff"] = pd.to_datetime(frame["cutoff"], utc=True)
    frame["after_cap"] = frame["kick"] > frame["cutoff"]
    sunday = frame["cutoff"].dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    tuesday = (
        sunday
        - pd.Timedelta(days=POOL_FREEZE_DAYS_BEFORE_SUNDAY)
        + pd.Timedelta(hours=POOL_FREEZE_HOUR_ET)
    )
    frame["tuesday"] = tuesday.dt.tz_localize(ET).dt.tz_convert("UTC")
    return frame.drop_duplicates("game_id").set_index("game_id")


def load_quotes(ids):
    q = pd.read_parquet(
        QUOTES,
        columns=[
            "nflverse_game_id",
            "bookmaker_key",
            "observed_at_utc",
            "bookmaker_last_update_utc",
            "snapshot_timestamp_utc",
            "home_spread_line",
        ],
    )
    q = q.loc[q["nflverse_game_id"].isin(ids) & q["home_spread_line"].notna()]
    q = q.rename(columns={"nflverse_game_id": "game_id", "bookmaker_key": "book"})
    q["t"] = q[["snapshot_timestamp_utc", "observed_at_utc", "bookmaker_last_update_utc"]].max(
        axis=1
    )
    return q[["game_id", "book", "t", "home_spread_line"]].sort_values(["game_id", "book", "t"])


def naive(s):
    return pd.to_datetime(s, utc=True).dt.tz_localize(None).astype("datetime64[ns]").to_numpy()


def as_of(q, when, books=None):
    sub = q if books is None else q.loc[q["book"].isin(books)]
    sub = sub.loc[naive(sub["t"]) <= naive(when.reindex(sub["game_id"]))]
    last = sub.groupby(["game_id", "book"])["home_spread_line"].last()
    return last.groupby(level=0).median(), last.groupby(level=0).size()


def quote_lines(q, anc):
    ids = anc.index
    tue, _ = as_of(q, anc["tuesday"])
    dead, dead_books = as_of(q, anc["cutoff"])
    lead, lead_books = as_of(q, anc["cutoff"], LEADER_BOOKS)
    close, close_books = as_of(q, anc["kick"] - pd.Timedelta(seconds=1))
    return pd.DataFrame(
        {
            "tue_line": tue.reindex(ids),
            "deadline_line": dead.reindex(ids),
            "deadline_books": dead_books.reindex(ids),
            "leader_deadline_line": lead.reindex(ids),
            "leader_books": lead_books.reindex(ids),
            "close_line": close.reindex(ids),
            "close_books": close_books.reindex(ids),
        }
    )


def ols_no_intercept(x, y):
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def predictor(frame, holdout):
    fit_rows = (
        frame["quote_game"]
        & frame["season"].ne(holdout)
        & frame[["m", "g", "D"]].notna().all(axis=1)
    )
    beta = ols_no_intercept(
        frame.loc[fit_rows, ["m", "g"]].to_numpy(float), frame.loc[fit_rows, "D"].to_numpy(float)
    )
    x = frame[["m", "g"]].fillna(0.0).to_numpy(float)
    value = np.where(frame["quote_game"], x @ beta, 0.0)
    return value, beta, int(fit_rows.sum())


def loso_arm(frame, term_fn):
    p = pd.Series(np.nan, index=frame.index)
    folds = {}
    features = (*FOUR, TERM)
    for season in sorted(frame["season"].unique()):
        work = frame.copy()
        work[TERM], beta, n = term_fn(work, season)
        fit = Fit(features, work[work["season"] != season])
        held = work["season"] == season
        p.loc[held] = fit.predict(work[held])
        folds[str(int(season))] = {
            "ols_beta_m_g": [float(b) for b in beta],
            "ols_games": n,
            "stage2": fit.natural(),
        }
    return p.to_numpy(float), folds


def loso_fixed(frame, column):
    p = pd.Series(np.nan, index=frame.index)
    features = (*FOUR, column)
    coefs = {}
    for season in sorted(frame["season"].unique()):
        fit = Fit(features, frame[frame["season"] != season])
        held = frame["season"] == season
        p.loc[held] = fit.predict(frame[held])
        coefs[str(int(season))] = fit.natural()[column]
    return p.to_numpy(float), coefs


def wl(p, y, mask):
    return record(np.asarray(p)[mask], np.asarray(y)[mask]) if mask.any() else "0-0"


def acc(p, y, mask):
    return (
        float(((np.asarray(p)[mask] >= 0.5) == (np.asarray(y)[mask] == 1)).mean())
        if mask.any()
        else float("nan")
    )


def side_record(home_pick, y, mask):
    c = (home_pick[mask]) == (y[mask] == 1)
    return f"{int(c.sum())}-{int((~c).sum())}"


def main():
    parser = argparse.ArgumentParser(
        description="Post-deadline line move on games kicking off after the pick cap."
    )
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(v4.TABLE, columns=["game_id", "gameday", "gametime", "spread_line"])
    table["game_id"] = table["game_id"].astype(str)
    pop = pd.read_parquet(v4.PICK / "per_game.parquet")
    pop["game_id"] = pop["game_id"].astype(str)
    pop = pop.reset_index(drop=True)
    qw = pd.read_parquet(QW_OPENER)
    anc = anchors(table.loc[table["game_id"].isin(set(qw.index.astype(str)) | set(pop["game_id"]))])
    ids = pop["game_id"].to_numpy()
    y = pop["home_covered"].to_numpy(float)
    mp = qw["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
    p_qw, qw_folds, _, _ = v4.stage_two(pop, mp)
    frame = pop.copy()
    frame["model_logit"] = v4.logit(mp)
    frame["after_cap"] = anc["after_cap"].reindex(ids).fillna(False).to_numpy(bool)
    frame["week_block"] = frame["season"] * 100 + frame["week"]
    cap_quote_ids = frame.loc[frame["after_cap"] & frame["season"].isin(QUOTE_SEASONS), "game_id"]
    quotes = load_quotes(set(cap_quote_ids))
    lines = quote_lines(quotes, anc.loc[cap_quote_ids])
    tue_check = pd.concat(
        [lines["tue_line"], pop.set_index("game_id")["tue_open_home_spread"]], axis=1
    ).dropna()
    corr = float(tue_check.corr().iloc[0, 1])
    sign = float(np.sign(corr))
    lines["m"] = sign * (lines["deadline_line"] - lines["tue_line"])
    lines["g"] = sign * (lines["leader_deadline_line"] - lines["deadline_line"])
    lines["D"] = sign * (lines["close_line"] - lines["deadline_line"])
    for c in ("m", "g", "D"):
        frame[c] = lines[c].reindex(ids).to_numpy(float)
    frame["quote_game"] = frame["game_id"].isin(set(cap_quote_ids)) & frame["D"].notna()
    cap = frame["after_cap"].to_numpy(bool)
    cap_q = (frame["after_cap"] & frame["season"].isin(QUOTE_SEASONS)).to_numpy(bool)
    served_home = p_qw >= 0.5
    d = frame["D"].to_numpy(float)
    has_d = cap_q & ~np.isnan(d)
    moved = has_d & (np.abs(d) > 0)
    against = moved & ((served_home & (d < 0)) | (~served_home & (d > 0)))
    with_move = moved & ~against
    push_ids = set(qw.index[qw["margin_vs_open"].eq(0.0)].astype(str))
    after_cap_ids = set(anc.index[anc["after_cap"]])
    season_of = qw["season"].copy()
    season_of.index = season_of.index.astype(str)
    pushes_cap = {
        "2020-25": len(push_ids & after_cap_ids),
        "2023-25": int(
            sum(1 for g in push_ids & after_cap_ids if int(season_of.get(g, 0)) in QUOTE_SEASONS)
        ),
    }
    close_side = np.sign(
        pop["close_home_spread"].to_numpy(float) - pop["tue_open_home_spread"].to_numpy(float)
    )
    close_pick = np.where(close_side > 0, True, np.where(close_side < 0, False, served_home))
    d_side = np.where(moved & (d > 0), True, np.where(moved & (d < 0), False, served_home))
    frame[ORACLE] = np.where(has_d, d, 0.0)
    p_oracle, oracle_coef = loso_fixed(frame, ORACLE)
    dead = lines["deadline_line"].reindex(ids).to_numpy(float)
    d_close = pop["close_home_spread"].to_numpy(float) - dead
    has_c = cap_q & ~np.isnan(d_close)
    moved_c = has_c & (np.abs(d_close) > 0)
    against_c = moved_c & ((served_home & (d_close < 0)) | (~served_home & (d_close > 0)))
    c_side = np.where(
        moved_c & (d_close > 0), True, np.where(moved_c & (d_close < 0), False, served_home)
    )
    frame["late_move_oracle_close"] = np.where(has_c, d_close, 0.0)
    p_oracle_c, oracle_c_coef = loso_fixed(frame, "late_move_oracle_close")
    arch_close = lines["close_line"].reindex(ids).to_numpy(float)
    close_check = {
        "archive_close_equals_pool_close": int(
            (has_c & (arch_close == pop["close_home_spread"].to_numpy(float))).sum()
        ),
        "games": int(has_c.sum()),
        "close_source_counts": pop.loc[cap_q, "close_source"].value_counts().to_dict(),
        "moved_deadline_to_true_close": int(moved_c.sum()),
        "against_served": int(against_c.sum()),
        "qw_record_when_true_close_move_against": wl(p_qw, y, against_c),
        "qw_record_when_true_close_move_with": wl(p_qw, y, moved_c & ~against_c),
        "upper_bound_true_close_side_else_served_2023_25": side_record(c_side, y, cap_q),
        "upper_bound_true_close_oracle_term_after_cap_2023_25": wl(p_oracle_c, y, cap_q),
        "true_close_oracle_coef_per_fold": oracle_c_coef,
    }
    nfl_close = (
        table.drop_duplicates("game_id")
        .set_index("game_id")["spread_line"]
        .reindex(ids)
        .to_numpy(float)
    )
    d_nfl = nfl_close - dead
    has_n = cap_q & ~np.isnan(d_nfl)
    moved_n = has_n & (np.abs(d_nfl) > 0)
    against_n = moved_n & ((served_home & (d_nfl < 0)) | (~served_home & (d_nfl > 0)))
    n_side = np.where(
        moved_n & (d_nfl > 0), True, np.where(moved_n & (d_nfl < 0), False, served_home)
    )
    frame["late_move_oracle_nfl"] = np.where(has_n, d_nfl, 0.0)
    p_oracle_n, oracle_n_coef = loso_fixed(frame, "late_move_oracle_nfl")
    nfl_check = {
        "games": int(has_n.sum()),
        "moved_deadline_to_kickoff_close": int(moved_n.sum()),
        "mean_abs_move": float(np.nanmean(np.abs(d_nfl[has_n]))),
        "against_served": int(against_n.sum()),
        "qw_record_when_move_against": wl(p_qw, y, against_n),
        "qw_record_when_move_with": wl(p_qw, y, moved_n & ~against_n),
        "qw_record_when_no_move": wl(p_qw, y, has_n & ~moved_n),
        "upper_bound_kickoff_close_side_else_served_2023_25": side_record(n_side, y, cap_q),
        "upper_bound_kickoff_close_oracle_term_after_cap_2023_25": wl(p_oracle_n, y, cap_q),
        "upper_bound_kickoff_close_oracle_term_all_games": wl(p_oracle_n, y, np.ones(len(y), bool)),
        "oracle_coef_per_fold": oracle_n_coef,
    }
    oos = pd.Series(np.nan, index=frame.index)
    for season in QUOTE_SEASONS:
        value, _, _ = predictor(frame, season)
        held = (frame["season"] == season).to_numpy()
        oos.loc[held] = value[held]
    oos = oos.to_numpy(float)
    dir_rows = moved & (np.abs(oos) > 0)
    step1 = {
        "sign_archive_to_pool_convention": sign,
        "tue_line_corr_with_pool_tuesday": corr,
        "after_cap_games_2020_25": int(cap.sum()),
        "after_cap_games_2023_25": int(cap_q.sum()),
        "after_cap_pushes_at_pool_line": pushes_cap,
        "with_post_deadline_quotes": int(has_d.sum()),
        "qw_record_after_cap_2020_25": wl(p_qw, y, cap),
        "qw_record_after_cap_2023_25": wl(p_qw, y, cap_q),
        "qw_record_before_cap_2023_25": wl(
            p_qw, y, ~cap & frame["season"].isin(QUOTE_SEASONS).to_numpy()
        ),
        "post_deadline_moved_games": int(moved.sum()),
        "post_deadline_no_move_games": int((has_d & ~moved).sum()),
        "move_against_served": int(against.sum()),
        "move_with_served": int(with_move.sum()),
        "share_against_of_moved": float(against.sum() / max(moved.sum(), 1)),
        "qw_record_when_move_against": wl(p_qw, y, against),
        "qw_record_when_move_with": wl(p_qw, y, with_move),
        "qw_record_when_no_move": wl(p_qw, y, has_d & ~moved),
        "mean_abs_D": float(np.nanmean(np.abs(d[has_d]))),
        "upper_bound_close_side_at_pool_line_2023_25": side_record(close_pick, y, cap_q),
        "upper_bound_close_side_at_pool_line_2020_25": side_record(close_pick, y, cap),
        "upper_bound_post_deadline_side_else_served_2023_25": side_record(d_side, y, cap_q),
        "upper_bound_oracle_D_term_after_cap_2023_25": wl(p_oracle, y, cap_q),
        "upper_bound_oracle_D_term_all_games": wl(p_oracle, y, np.ones(len(y), bool)),
        "upper_bound_oracle_D_coef_per_fold": oracle_coef,
        "pool_table_close_sunday_snapshot": close_check,
        "nflverse_kickoff_close": nfl_check,
        "label": "upper bounds use post-deadline information and are not results",
    }
    p_arm, folds = loso_arm(frame, predictor)
    work = frame.copy()
    work[TERM], beta_full, _ = predictor(work, -1)
    full = Fit((*FOUR, TERM), work)
    insample = full.predict(work)
    changed = (p_arm >= 0.5) != (p_qw >= 0.5)
    full_mask = np.ones(len(y), bool)
    sub = frame.loc[cap_q].reset_index(drop=True)
    real_gain = (acc(p_arm, y, full_mask) - acc(p_qw, y, full_mask)) * 100
    real_gain_cap = (acc(p_arm, y, cap_q) - acc(p_qw, y, cap_q)) * 100
    rng = np.random.default_rng(NULL_SEED)
    null_all, null_cap = [], []
    rows = np.flatnonzero(frame["quote_game"].to_numpy(bool))
    seasons = frame["season"].to_numpy()
    for _ in range(args.null_draws):
        shuffled = frame.copy()
        for season in QUOTE_SEASONS:
            r = rows[seasons[rows] == season]
            perm = rng.permutation(r)
            shuffled.loc[shuffled.index[r], ["m", "g", "D"]] = frame.loc[
                frame.index[perm], ["m", "g", "D"]
            ].to_numpy()
        p_null, _ = loso_arm(shuffled, predictor)
        null_all.append((acc(p_null, y, full_mask) - acc(p_qw, y, full_mask)) * 100)
        null_cap.append((acc(p_null, y, cap_q) - acc(p_qw, y, cap_q)) * 100)
    null_all = np.array(null_all)
    null_cap = np.array(null_cap)
    step2 = {
        "declared": "docs/lanes/late-move-2026-11-16.md State, 2026-11-16 17:29 -0500",
        "qw_identity_record": record(p_qw, y),
        "arm_record_all": record(p_arm, y),
        "arm_record_after_cap_2023_25": wl(p_arm, y, cap_q),
        "qw_record_after_cap_2023_25": wl(p_qw, y, cap_q),
        "arm_record_after_cap_2020_25": wl(p_arm, y, cap),
        "qw_record_after_cap_2020_25": wl(p_qw, y, cap),
        "changed_picks": int(changed.sum()),
        "changed_after_cap": int((changed & cap).sum()),
        "changed_record_arm": wl(p_arm, y, changed),
        "changed_record_qw": wl(p_qw, y, changed),
        "predictor_oos_direction_hits_on_moved": side_record(
            np.sign(oos) > 0, (np.sign(d) > 0).astype(float), dir_rows
        ),
        "predictor_oos_corr_with_D": float(np.corrcoef(oos[has_d], d[has_d])[0, 1]),
        "ols_beta_full_m_g": [float(b) for b in beta_full],
        "folds": folds,
        "qw_folds": qw_folds,
        "full_stage2": full.natural(),
        "log_loss_arm": float(log_loss(p_arm, y).mean()),
        "log_loss_qw": float(log_loss(p_qw, y).mean()),
        "in_sample_log_loss_arm": float(log_loss(insample, y).mean()),
        "brier_arm": float(((p_arm - y) ** 2).mean()),
        "brier_qw": float(((p_qw - y) ** 2).mean()),
        "vs_qw_all_season_block": paired(frame, p_qw, p_arm, "season"),
        "vs_qw_all_week_block": paired(frame, p_qw, p_arm, "week_block"),
        "vs_qw_after_cap_week_block": paired(sub, p_qw[cap_q], p_arm[cap_q], "week_block"),
        "matched_null": {
            "draws": int(args.null_draws),
            "real_gain_all_points": float(real_gain),
            "real_gain_after_cap_points": float(real_gain_cap),
            "null_mean_all": float(null_all.mean()),
            "null_mean_after_cap": float(null_cap.mean()),
            "share_null_ge_real_all": float((null_all >= real_gain).mean()),
            "share_null_ge_real_after_cap": float((null_cap >= real_gain_cap).mean()),
            "null_p95_all": float(np.percentile(null_all, 95)),
        },
        "after_cap_per_season": {
            str(s): {
                "qw": wl(p_qw, y, cap & (seasons == s)),
                "arm": wl(p_arm, y, cap & (seasons == s)),
            }
            for s in sorted(np.unique(seasons))
        },
        "looks": (
            "1 declared predictor x (accuracy, log loss, Brier) on one population; "
            "upper bounds are not looks at serving"
        ),
    }
    per_game = frame[
        ["game_id", "season", "week", "after_cap", "quote_game", "m", "g", "D", "home_covered"]
    ].assign(p_qw=p_qw, p_arm=p_arm, p_oracle=p_oracle, late_move_oos=oos)
    per_game.to_parquet(OUT / "per_game.parquet", index=False)
    lines.to_parquet(OUT / "lines.parquet")
    result = {"step1_support": step1, "step2_attack": step2}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, default=str)[:7000])


if __name__ == "__main__":
    main()
