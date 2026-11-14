import glob
import json
import re
from math import comb, erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "lead99_decision_read"
SEASON = 2026
RESAMPLES = 20000
SEED = 20261114
FAMILY = "lead99_forward_decision_read"
STAMP = re.compile(r"\d{8}T\d{6}Z")
VIEWS = ["ledger", "tuesday", "thursday", "canonical"]
BASELINES = ["card", "card_at_read", "no_information", "raw_model", "market_move_only"]


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def _ll(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _brier(p, y):
    return (np.asarray(p, dtype=float) - y) ** 2


def _stamp(text):
    return pd.Timestamp(
        f"{text[:4]}-{text[4:6]}-{text[6:8]}T{text[9:11]}:{text[11:13]}:{text[13:15]}Z"
    )


def _boot(values, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    idx = rng.integers(0, n, size=(RESAMPLES, n))
    means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "probability_positive": float((means > 0).mean()),
        "games": int(n),
    }


def _corr_boot(x, y, rng):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    idx = rng.integers(0, n, size=(RESAMPLES, n))
    xs = x[idx] - x[idx].mean(axis=1, keepdims=True)
    ys = y[idx] - y[idx].mean(axis=1, keepdims=True)
    denom = np.sqrt((xs**2).sum(axis=1) * (ys**2).sum(axis=1))
    r = np.full(RESAMPLES, np.nan)
    good = denom > 0
    r[good] = (xs * ys).sum(axis=1)[good] / denom[good]
    return {
        "mean": float(np.corrcoef(x, y)[0, 1]),
        "low": float(np.nanquantile(r, 0.025)),
        "high": float(np.nanquantile(r, 0.975)),
        "probability_positive": float(np.nanmean(r > 0)),
        "games": int(n),
    }


def _fit(x, y, offset):
    beta = np.zeros(x.shape[1])
    hess = np.eye(x.shape[1])
    for _ in range(200):
        p = _sigmoid(offset + x @ beta)
        grad = x.T @ (y - p)
        hess = (x * (p * (1 - p))[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return beta, np.linalg.inv(hess)


def _runs():
    rows = []
    for path in sorted(glob.glob(str(ROOT / "artifacts/lead99_llm_read/runs/*.json"))):
        doc = _load(path)
        games = {g["game_id"]: g for g in doc["packet"]["games"]}
        for row in doc.get("parsed_rows", []):
            game = games.get(row["game_id"], {})
            market = game.get("market") or {}
            pool = game.get("pool_line") or {}
            rows.append(
                {
                    "file": Path(path).name,
                    "week": int(doc["week"]),
                    "view": doc["view"],
                    "recorded_at": pd.Timestamp(doc["recorded_at_utc"]),
                    "game_id": row["game_id"],
                    "p": row.get("home_cover_probability"),
                    "pool_home_spread": pool.get("home_spread"),
                    "pool_captured_at": pool.get("captured_at_et"),
                    "shown_market": market.get("home_spread_latest"),
                }
            )
    frame = pd.DataFrame(rows)
    frame["pool_captured_at"] = pd.to_datetime(frame["pool_captured_at"], utc=True)
    return frame


def _quotes(start, end, game_ids):
    frames = []
    for folder in sorted((ROOT / "data" / "market" / "raw").iterdir()):
        name = folder.name[:16]
        if not STAMP.fullmatch(name) or not (start <= _stamp(name) <= end):
            continue
        path = folder / "quotes.parquet"
        if not path.exists():
            continue
        q = pd.read_parquet(path)
        q = q.loc[
            (q["market"] == "spreads")
            & (q["outcome_side"] == "HOME")
            & q["nflverse_game_id"].isin(game_ids)
        ]
        frames.append(q[["observed_at_utc", "nflverse_game_id", "bookmaker_key", "line"]])
    q = pd.concat(frames, ignore_index=True).drop_duplicates()
    q["observed_at_utc"] = pd.to_datetime(q["observed_at_utc"], utc=True)
    return {gid: part.sort_values("observed_at_utc") for gid, part in q.groupby("nflverse_game_id")}


def _market_at(quotes, game_id, since, at):
    part = quotes.get(game_id)
    if part is None or pd.isna(since) or pd.isna(at):
        return np.nan
    part = part.loc[(part["observed_at_utc"] >= since) & (part["observed_at_utc"] <= at)]
    if part.empty:
        return np.nan
    latest = part.drop_duplicates("bookmaker_key", keep="last")
    return float(-latest["line"].median())


def _raw_model(games):
    folders = {}
    for path in glob.glob(str(ROOT / "artifacts/margin_predictions" / f"{SEASON}-week-*")):
        parts = Path(path).name.split("-")
        folders.setdefault(int(parts[2]), []).append((_stamp(parts[3]), Path(path)))
    cache = {}
    out = {}
    for row in games.itertuples():
        chosen = [item for item in sorted(folders.get(row.week, [])) if item[0] < row.deadline_utc]
        if not chosen:
            continue
        path = chosen[-1][1] / "recommendations.csv"
        if path not in cache:
            cache[path] = pd.read_csv(path).set_index("game_id")
        table = cache[path]
        if row.game_id in table.index:
            out[row.game_id] = (
                float(table.loc[row.game_id, "home_cover_probability"]),
                float(table.loc[row.game_id, "market_spread"]),
            )
    return out


def _card_history():
    picks = pd.read_parquet(ROOT / "artifacts/clv_ledger/published_picks.parquet")
    return picks.loc[picks["season"].eq(SEASON)].sort_values("published_at_utc")


def _assemble():
    core = pd.read_parquet(ROOT / "artifacts/forward_ledger_core/ledger.parquet")
    core = core.loc[core["lead99_first_view__p"].notna() & core["status"].eq("graded")].copy()
    runs = _runs()
    runs = runs.merge(core[["game_id", "deadline_utc"]], on="game_id", how="left")
    before = runs.loc[runs["recorded_at"] < runs["deadline_utc"]].sort_values("recorded_at")
    first = before.drop_duplicates("game_id", keep="first").set_index("game_id")
    last = before.drop_duplicates("game_id", keep="last").set_index("game_id")
    ordered = runs.sort_values("recorded_at")
    tuesday_files = ordered.loc[ordered["view"].eq("tuesday")].drop_duplicates("week")["file"]
    thursday_files = ordered.loc[ordered["view"].eq("deadline")].drop_duplicates("week")["file"]
    tuesday = before.loc[before["file"].isin(tuesday_files)].set_index("game_id")
    thursday = before.loc[before["file"].isin(thursday_files)].set_index("game_id")
    frame = pd.DataFrame(
        {
            "game_id": core["game_id"].to_numpy(),
            "week": core["week"].to_numpy().astype(int),
            "y": core["home_cover"].to_numpy().astype(float),
            "result": core["result"].to_numpy(),
            "card_line": core["card_line"].to_numpy(),
            "card": core["served_p"].to_numpy(),
            "card_home": core["served_side"].eq("HOME").to_numpy(),
            "ledger": core["lead99_first_view__p"].to_numpy(),
        }
    )
    frame["deadline_utc"] = pd.to_datetime(core["deadline_utc"].to_numpy(), utc=True)
    gid = frame["game_id"]
    frame["away"] = gid.str.split("_").str[2]
    frame["home"] = gid.str.split("_").str[3]
    frame["ledger_view"] = gid.map(first["view"])
    frame["ledger_at"] = gid.map(first["recorded_at"])
    frame["ledger_check_p"] = gid.map(first["p"])
    frame["tuesday"] = gid.map(tuesday["p"])
    frame["thursday"] = gid.map(thursday["p"])
    frame["canonical"] = gid.map(last["p"])
    frame["canonical_at"] = gid.map(last["recorded_at"])
    frame["pool_home_spread"] = gid.map(first["pool_home_spread"])
    frame["week_start"] = gid.map(first["pool_captured_at"]).dt.normalize()
    frame["shown_market"] = gid.map(first["shown_market"])
    quotes = _quotes(frame["week_start"].min(), frame["deadline_utc"].max(), set(gid))
    frame["market_at_read"] = [
        _market_at(quotes, r.game_id, r.week_start, r.ledger_at) for r in frame.itertuples()
    ]
    frame["market_at_deadline"] = [
        _market_at(quotes, r.game_id, r.week_start, r.deadline_utc) for r in frame.itertuples()
    ]
    frame["offset_shown"] = frame["shown_market"] - frame["pool_home_spread"]
    frame["offset_at_read"] = frame["market_at_read"] - frame["pool_home_spread"]
    frame["offset_at_deadline"] = frame["market_at_deadline"] - frame["pool_home_spread"]
    frame["move_after_read"] = frame["market_at_deadline"] - frame["market_at_read"]
    raw = _raw_model(frame)
    frame["raw_model"] = gid.map({k: v[0] for k, v in raw.items()})
    frame["raw_model_line"] = gid.map({k: v[1] for k, v in raw.items()})
    served = sorted(glob.glob(str(ROOT / "artifacts/pick_probability/*/coefficients.json")))[-1]
    coef = _load(served)["coefficients"]
    frame["market_move_only"] = _sigmoid(
        coef["intercept"]
        + coef["move_available"]
        + coef["move_toward_home"] * frame["offset_at_deadline"]
    )
    frame["no_information"] = 0.5
    read_at = frame.set_index("game_id")["ledger_at"]
    first_side, final_at, changes, at_read = {}, {}, {}, {}
    for game_id, part in _card_history().groupby("game_id"):
        part = part.loc[part["published_at_utc"] <= part["pick_deadline_utc"]]
        if part.empty or game_id not in read_at.index:
            continue
        home = part["pick_team"].eq(part["home_team"]).to_numpy()
        score = part["displayed_score"].to_numpy()
        home_p = np.where(home, score, 1 - score)
        seen = (part["published_at_utc"] <= read_at[game_id]).to_numpy()
        first_side[game_id] = bool(home[0])
        final_at[game_id] = part["published_at_utc"].iloc[-1]
        changes[game_id] = int((home[1:] != home[:-1]).sum())
        at_read[game_id] = float(home_p[seen][-1]) if seen.any() else float(home_p[0])
    frame["card_first_home"] = gid.map(first_side)
    frame["card_final_at"] = gid.map(final_at)
    frame["card_side_changes"] = gid.map(changes)
    frame["card_at_read"] = gid.map(at_read)
    pool = pd.read_parquet(ROOT / "artifacts/pool_relative_forward_log/log.parquet")
    pool = pool.merge(frame[["game_id", "deadline_utc"]], on="game_id")
    pool = pool.loc[pool["run_at_utc"] <= pool["deadline_utc"]].sort_values("run_at_utc")
    pool = pool.drop_duplicates("game_id", keep="last").set_index("game_id")
    frame["logged_pool_move"] = gid.map(pool["pool_move"])
    return frame, runs, coef, Path(served).parent.name


def _sides(frame, name):
    if name == "no_information":
        return np.full(len(frame), np.nan)
    if name == "card":
        return frame["card_home"].to_numpy().astype(float)
    if name == "market_move_only":
        offset = frame["offset_at_deadline"].to_numpy(dtype=float)
        return np.where(np.isfinite(offset) & (offset != 0), (offset > 0).astype(float), np.nan)
    p = frame[name].to_numpy(dtype=float)
    return np.where(np.isfinite(p), (p >= 0.5).astype(float), np.nan)


def _row(frame, name):
    y = frame["y"].to_numpy()
    p = frame[name].to_numpy(dtype=float)
    ok = np.isfinite(p)
    side = _sides(frame, name)
    has = np.isfinite(side)
    wins = int((side[has] == y[has]).sum())
    return {
        "games": int(ok.sum()),
        "wins": wins,
        "losses": int(has.sum()) - wins,
        "no_side": int(len(frame) - has.sum()),
        "log_loss": float(_ll(p[ok], y[ok]).mean()),
        "brier": float(_brier(p[ok], y[ok]).mean()),
    }


def _view_block(frame, view, rng):
    part = frame.loc[frame[view].notna()].copy()
    y = part["y"].to_numpy()
    read = part[view].to_numpy(dtype=float)
    home = read >= 0.5
    differs = home != part["card_home"].to_numpy()
    right = home == (y == 1)
    out = {
        "games": len(part),
        "decisive_vs_card": [int((differs & right).sum()), int((differs & ~right).sum())],
        "rows": {"read": _row(part, view)},
        "read_better_by": {},
    }
    for name in BASELINES:
        out["rows"][name] = _row(part, name)
        base = part[name].to_numpy(dtype=float)
        out["read_better_by"][name] = {
            "log_loss": _boot(_ll(base, y) - _ll(read, y), rng),
            "brier": _boot(_brier(base, y) - _brier(read, y), rng),
        }
    return out


def _decisive(frame):
    part = frame.loc[(frame["ledger"] >= 0.5) != frame["card_home"]].copy()
    read_home = (part["ledger"] >= 0.5).to_numpy()
    sign = np.where(read_home, 1.0, -1.0)
    part["card_side"] = np.where(part["card_home"], part["home"], part["away"])
    part["read_side"] = np.where(read_home, part["home"], part["away"])
    part["covered"] = np.where(part["y"] == 1, part["home"], part["away"])
    part["read_won"] = read_home == (part["y"].to_numpy() == 1)
    part["toward_read_at_read"] = part["offset_at_read"] * sign
    part["toward_read_at_deadline"] = part["offset_at_deadline"] * sign
    part["toward_read_after_read"] = part["move_after_read"] * sign
    part["card_began_on_read_side"] = part["card_first_home"].to_numpy() == read_home
    columns = [
        "week",
        "game_id",
        "card_side",
        "card",
        "card_at_read",
        "read_side",
        "ledger",
        "covered",
        "read_won",
        "ledger_view",
        "ledger_at",
        "card_final_at",
        "card_side_changes",
        "card_began_on_read_side",
        "card_line",
        "result",
        "toward_read_at_read",
        "toward_read_at_deadline",
        "toward_read_after_read",
    ]
    return part[columns]


def _split_record(decisive, column):
    out = {}
    value = decisive[column].to_numpy(dtype=float)
    won = decisive["read_won"].to_numpy()
    for label, mask in (
        ("toward_read", value > 0),
        ("against_read", value < 0),
        ("no_move", value == 0),
        ("no_quote", ~np.isfinite(value)),
    ):
        out[label] = [int(won[mask].sum()), int((~won[mask]).sum())]
    return out


def _permutation(frame, rng):
    y = frame["y"].to_numpy()
    card_home = frame["card_home"].to_numpy()
    read = frame["ledger"].to_numpy(dtype=float)
    weeks = frame["week"].to_numpy()
    groups = [np.flatnonzero(weeks == w) for w in sorted(set(weeks))]
    card_loss = _ll(frame["card"].to_numpy(), y)

    def stats(p):
        home = p >= 0.5
        decisive = home != card_home
        correct = home == (y == 1)
        return (
            float((correct & decisive).sum() - (~correct & decisive).sum()),
            float((card_loss - _ll(p, y)).mean()),
            float(correct.sum()),
        )

    observed = np.array(stats(read))
    draws = np.empty((RESAMPLES, 3))
    for i in range(RESAMPLES):
        perm = read.copy()
        for group in groups:
            perm[group] = read[rng.permutation(group)]
        draws[i] = stats(perm)
    out = {}
    for j, name in enumerate(["decisive_net_wins", "log_loss_gain_vs_card", "correct"]):
        tail = float(((draws[:, j] >= observed[j] - 1e-12).sum() + 1) / (RESAMPLES + 1))
        out[name] = {
            "observed": float(observed[j]),
            "null_mean": float(draws[:, j].mean()),
            "null_q95": float(np.quantile(draws[:, j], 0.95)),
            "p_value": tail,
        }
    return out


def _stack(frame, rng):
    y = frame["y"].to_numpy()
    card = _logit(frame["card"].to_numpy())
    read = _logit(frame["ledger"].to_numpy())
    weeks = frame["week"].to_numpy()
    card_loss = _ll(frame["card"].to_numpy(), y)
    half_loss = _ll(np.full(len(y), 0.5), y)
    forms = {
        "card_offset_plus_read": (read[:, None], card),
        "card_and_read_both_fitted": (np.column_stack([card, read]), np.zeros(len(y))),
    }
    out = {}
    for name, (x, offset) in forms.items():
        beta, cov = _fit(x, y, offset)
        inside = _ll(_sigmoid(offset + x @ beta), y)
        held = np.full(len(y), np.nan)
        folds = {}
        for week in sorted(set(weeks)):
            test = weeks == week
            fold_beta, _ = _fit(x[~test], y[~test], offset[~test])
            held[test] = _sigmoid(offset[test] + x[test] @ fold_beta)
            folds[int(week)] = [round(float(v), 3) for v in fold_beta]
        outside = _ll(held, y)
        home = held >= 0.5
        flips = home != frame["card_home"].to_numpy()
        correct = home == (y == 1)
        out[name] = {
            "in_sample_coefficients": [round(float(v), 3) for v in beta],
            "in_sample_standard_errors": [round(float(v), 3) for v in np.sqrt(np.diag(cov))],
            "fold_coefficients": folds,
            "in_sample_gain_vs_card": float((card_loss - inside).mean()),
            "held_out_gain_vs_card": _boot(card_loss - outside, rng),
            "gap_in_sample_minus_held_out": float((outside - inside).mean()),
            "held_out_gain_vs_no_information": _boot(half_loss - outside, rng),
            "held_out_record": [int(correct.sum()), int((~correct).sum())],
            "held_out_changed_record": [
                int((correct & flips).sum()),
                int((~correct & flips).sum()),
            ],
            "held_out_gain_by_week": {
                int(w): round(float((card_loss - outside)[weeks == w].mean()), 4)
                for w in sorted(set(weeks))
            },
            "held_out_probability_range": [float(held.min()), float(held.max())],
        }
    part = frame.loc[frame["offset_at_deadline"].notna()]
    x = np.column_stack(
        [_logit(part["ledger"].to_numpy()), part["offset_at_deadline"].to_numpy(dtype=float)]
    )
    beta, cov = _fit(x, part["y"].to_numpy(), _logit(part["card"].to_numpy()))
    se = np.sqrt(np.diag(cov))
    out["in_sample_joint_read_and_deadline_offset"] = {
        "games": len(part),
        "read_logit_coefficient": float(beta[0]),
        "read_logit_interval": [float(beta[0] - 1.96 * se[0]), float(beta[0] + 1.96 * se[0])],
        "read_logit_probability_positive": float(0.5 * (1 + erf(beta[0] / se[0] / sqrt(2)))),
        "offset_coefficient": float(beta[1]),
        "offset_interval": [float(beta[1] - 1.96 * se[1]), float(beta[1] + 1.96 * se[1])],
    }
    return out


def _week10(runs, read_coefficient):
    now = pd.Timestamp.now(tz="UTC")
    picks = _card_history()
    picks = picks.loc[picks["week"].eq(10) & picks["published_at_utc"].le(now)]
    picks = picks.drop_duplicates("game_id", keep="last")
    picks = picks.loc[picks["pick_deadline_utc"] > now].copy()
    week = runs.loc[runs["week"].eq(10)].sort_values("recorded_at")
    tuesday_file = week.loc[week["view"].eq("tuesday"), "file"].iloc[0]
    tuesday = week.loc[week["file"].eq(tuesday_file)].set_index("game_id")["p"]
    latest = week.drop_duplicates("game_id", keep="last").set_index("game_id")
    rows = []
    for row in picks.itertuples():
        card_home = row.pick_team == row.home_team
        card_p = row.displayed_score if card_home else 1 - row.displayed_score
        read_p = float(latest["p"].get(row.game_id, np.nan))
        stacked = float(_sigmoid(_logit(card_p) + read_coefficient * _logit(read_p)))
        rows.append(
            {
                "game_id": row.game_id,
                "deadline_utc": str(row.pick_deadline_utc),
                "pool_home_spread": float(row.market_spread),
                "card_pick": row.pick_team,
                "card_home_p": float(card_p),
                "card_published_at": str(row.published_at_utc),
                "read_pick": row.home_team if read_p >= 0.5 else row.away_team,
                "read_home_p": read_p,
                "read_at": str(latest["recorded_at"].get(row.game_id)),
                "read_view": latest["view"].get(row.game_id),
                "read_home_p_tuesday": float(tuesday.get(row.game_id, np.nan)),
                "sides_differ": bool((read_p >= 0.5) != card_home),
                "stacked_home_p_information_only": stacked,
            }
        )
    return pd.DataFrame(rows).sort_values(["deadline_utc", "game_id"])


def _cell(name, description, summary, stat, units, category="modeling"):
    return {
        "name": name,
        "family": FAMILY,
        "source": "artifacts/lead99_decision_read/summary.json",
        "description": description,
        "plain_summary": summary,
        "effect": round(stat["mean"], 4),
        "interval_low": round(stat["low"], 4),
        "interval_high": round(stat["high"], 4),
        "probability_positive": round(stat["probability_positive"], 4),
        "effect_units": units,
        "classification": "unresolved_below_power",
        "classification_evidence": (
            "Forward log only, three weeks; interval and probability_positive from a game "
            "bootstrap; no wrong-sign interval, no zero reliability and no positive control "
            "bound applies, so no admissible closing ground exists and the cell stays unresolved."
        ),
        "league": "nfl",
        "season_start": SEASON,
        "season_end": SEASON,
        "sample_games": int(stat["games"]),
        "sample_blocks": 3,
        "category": category,
    }


def _fmt(stat, digits=4):
    return (
        f"{stat['mean']:+.{digits}f} [{stat['low']:+.{digits}f}, {stat['high']:+.{digits}f}] "
        f"P+ {stat['probability_positive']:.3f} n {stat['games']}"
    )


def _cells(summary, n):
    ledger = summary["views"]["ledger"]["read_better_by"]
    canonical = summary["views"]["canonical"]["read_better_by"]
    perm = summary["permutation_within_week"]
    stack = summary["stack"]["card_offset_plus_read"]
    wins, losses = summary["decisive"]["record"]
    tail = "Three weeks is too few to tell luck from skill, so it changes no pick."
    return [
        _cell(
            "lead99_dr_read_vs_no_information_log_loss",
            f"LEAD-99 read as scored by the forward ledger (earliest pregame record, raw read "
            f"probability, no fitted coefficient) against a constant 0.5 on {n} graded games, "
            "Weeks 7-9; paired log-loss improvement, game bootstrap",
            "Research only: an AI model's written read of each game was compared with a coin "
            f"flip on {n} finished games. {tail}",
            ledger["no_information"]["log_loss"],
            "log_loss_improvement",
        ),
        _cell(
            "lead99_dr_read_vs_raw_model_log_loss",
            "LEAD-99 read (ledger view) against the raw margin model's home cover probability "
            f"at the pool line from the last forecast before each deadline, {n} graded games, "
            "Weeks 7-9; paired log-loss improvement, game bootstrap",
            "Research only: an AI model's written read of each game was compared with the "
            f"football model's own probability on {n} finished games. {tail}",
            ledger["raw_model"]["log_loss"],
            "log_loss_improvement",
        ),
        _cell(
            "lead99_dr_read_vs_market_move_log_loss",
            "LEAD-99 read (ledger view) against the deadline market offset from the pool line "
            "mapped through the served card's own fitted move coefficients (no refit), Weeks "
            "7-9; paired log-loss improvement, game bootstrap",
            "Research only: an AI model's written read of each game was compared with simply "
            f"following the way the betting line had moved by the pick deadline. {tail}",
            ledger["market_move_only"]["log_loss"],
            "log_loss_improvement",
            "market",
        ),
        _cell(
            "lead99_dr_read_vs_card_at_read_time_log_loss",
            "LEAD-99 read (ledger view) against the card's published probability at the moment "
            "the read was made, Weeks 7-9; paired log-loss improvement, game bootstrap; removes "
            "the card's later line-move changes from the comparison",
            "Research only: an AI model's written read of each game was compared with what the "
            f"card said at the same moment, before the card's later changes. {tail}",
            ledger["card_at_read"]["log_loss"],
            "log_loss_improvement",
        ),
        _cell(
            "lead99_dr_canonical_view_vs_card_log_loss",
            "LEAD-99 declared canonical row (latest read before each game's deadline) against "
            f"the played card on {n} graded games, Weeks 7-9; paired log-loss improvement, game "
            "bootstrap; the forward ledger scores the earliest record instead",
            "Research only: the last AI read made before each game's pick deadline was compared "
            f"with the card's own probabilities on {n} finished games. {tail}",
            canonical["card"]["log_loss"],
            "log_loss_improvement",
        ),
        _cell(
            "lead99_dr_canonical_view_vs_no_information_log_loss",
            "LEAD-99 declared canonical row (latest read before each game's deadline) against a "
            f"constant 0.5 on {n} graded games, Weeks 7-9; paired log-loss improvement, game "
            "bootstrap",
            "Research only: the last AI read made before each game's pick deadline was compared "
            f"with a coin flip on {n} finished games. {tail}",
            canonical["no_information"]["log_loss"],
            "log_loss_improvement",
        ),
        _cell(
            "lead99_dr_stack_held_out_week_vs_card_log_loss",
            "Card logit as offset plus one coefficient on the LEAD-99 read logit, coefficient "
            "fitted on two of Weeks 7-9 and scored on the third; held-out paired log-loss "
            f"improvement against the card, game bootstrap; in-sample gain "
            f"{stack['in_sample_gain_vs_card']:+.4f}, gap "
            f"{stack['gap_in_sample_minus_held_out']:+.4f}",
            "Research only: the card's probability was blended with an AI model's written read, "
            f"with the blend set on two weeks and checked on the third. {tail}",
            stack["held_out_gain_vs_card"],
            "log_loss_improvement",
        ),
        _cell(
            "lead99_dr_read_vs_week_lean_log_loss",
            "LEAD-99 read (ledger view) against a constant equal to its own mean probability in "
            "that week, Weeks 7-9; paired log-loss improvement, game bootstrap; isolates the "
            "game-by-game part of the read from its weekly lean; within-week permutation p for "
            f"the gain against the card {perm['log_loss_gain_vs_card']['p_value']:.3f}",
            "Research only: an AI model's game-by-game read was compared with its own average "
            f"lean for the week, to see whether it sorts games or only leans one way. {tail}",
            summary["lean_controls"]["read_better_than_its_week_mean"],
            "log_loss_improvement",
        ),
        _cell(
            "lead99_dr_read_logit_vs_later_line_move_correlation",
            "Correlation between the LEAD-99 read logit (ledger view) and the market line move "
            "toward the home team from the read time to the pick deadline (median of books), "
            "Weeks 7-9; game bootstrap",
            "Research only: this checks whether an AI model's written read of a game points the "
            f"same way the betting line later moves. {tail}",
            summary["correlations"]["read_logit_vs_move_after_read"],
            "correlation",
            "market",
        ),
        _cell(
            "lead99_dr_read_minus_card_accuracy",
            "LEAD-99 read side (ledger view) minus the played card side, pick accuracy in "
            f"points at the pool line on {n} graded games, Weeks 7-9; game bootstrap; decisive "
            f"games {wins}-{losses}; within-week permutation p "
            f"{perm['decisive_net_wins']['p_value']:.3f}",
            "Research only: an AI model's written read of each game picked more winners than "
            f"the card over {n} finished games, in three weeks when the card itself ran cold. "
            f"{tail}",
            summary["read_minus_card_accuracy_points"],
            "accuracy_points",
        ),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    frame, runs, coef, served_artifact = _assemble()
    y = frame["y"].to_numpy()
    n = len(frame)
    summary = {
        "family": FAMILY,
        "games": n,
        "resamples": RESAMPLES,
        "seed": SEED,
        "served_coefficients_artifact": served_artifact,
        "served_move_coefficients": {
            k: coef[k] for k in ("intercept", "move_available", "move_toward_home")
        },
    }
    summary["checks"] = {
        "ledger_p_equals_earliest_record": int(
            np.isclose(frame["ledger"], frame["ledger_check_p"]).sum()
        ),
        "ledger_view_counts": {
            f"week{w}_{v}": int(c)
            for (w, v), c in frame.groupby(["week", "ledger_view"]).size().items()
        },
        "canonical_row_is_later_than_ledger_row": int(
            (frame["canonical_at"] > frame["ledger_at"]).sum()
        ),
        "canonical_side_differs_from_ledger_side": int(
            ((frame["canonical"] >= 0.5) != (frame["ledger"] >= 0.5)).sum()
        ),
        "pool_line_equals_card_line": int(
            np.isclose(frame["pool_home_spread"], frame["card_line"]).sum()
        ),
        "raw_model_line_equals_card_line": int(
            np.isclose(frame["raw_model_line"], frame["card_line"]).sum()
        ),
        "read_time_quote_games": int(frame["offset_at_read"].notna().sum()),
        "deadline_quote_games": int(frame["offset_at_deadline"].notna().sum()),
        "logged_pool_move_games": int(frame["logged_pool_move"].notna().sum()),
        "logged_pool_move_equals_quote_offset": int(
            np.isclose(frame["logged_pool_move"], frame["offset_at_deadline"]).sum()
        ),
        "read_range": [float(frame["ledger"].min()), float(frame["ledger"].max())],
        "read_home_sides": int((frame["ledger"] >= 0.5).sum()),
        "card_home_sides": int(frame["card_home"].sum()),
        "home_covers": int(y.sum()),
    }
    summary["views"] = {view: _view_block(frame, view, rng) for view in VIEWS}
    by_week = {}
    for week, part in frame.groupby("week"):
        cell = {"games": len(part), "rows": {"read": _row(part, "ledger")}}
        week_y = part["y"].to_numpy()
        for name in BASELINES:
            cell["rows"][name] = _row(part, name)
            cell[f"read_better_than_{name}"] = _boot(
                _ll(part[name].to_numpy(dtype=float), week_y) - _ll(part["ledger"], week_y), rng
            )
        for view in VIEWS[1:]:
            sub = part.loc[part[view].notna()]
            if len(sub):
                cell[f"{view}_view_vs_card"] = _boot(
                    _ll(sub["card"], sub["y"].to_numpy())
                    - _ll(sub[view].to_numpy(dtype=float), sub["y"].to_numpy()),
                    rng,
                )
        by_week[int(week)] = cell
    summary["by_week"] = by_week
    week_mean = frame.groupby("week")["ledger"].transform("mean").to_numpy()
    read_loss = _ll(frame["ledger"], y)
    read_home = (frame["ledger"] >= 0.5).to_numpy()
    summary["lean_controls"] = {
        "read_better_than_its_week_mean": _boot(_ll(week_mean, y) - read_loss, rng),
        "read_better_than_its_overall_mean": _boot(
            _ll(np.full(n, frame["ledger"].mean()), y) - read_loss, rng
        ),
        "by_week_mean_read_home_cover_rate_card_home_read_home": {
            int(w): [
                round(float(part["ledger"].mean()), 3),
                round(float(part["y"].mean()), 3),
                int(part["card_home"].sum()),
                int((part["ledger"] >= 0.5).sum()),
            ]
            for w, part in frame.groupby("week")
        },
    }
    read_correct = (read_home == (y == 1)).astype(float)
    card_correct = (frame["card_home"].to_numpy() == (y == 1)).astype(float)
    summary["read_minus_card_accuracy_points"] = _boot(100 * (read_correct - card_correct), rng)
    decisive = _decisive(frame)
    decisive.to_csv(OUT / "decisive_games.csv", index=False)
    began = decisive["card_began_on_read_side"].to_numpy(dtype=bool)
    won = decisive["read_won"].to_numpy()
    summary["decisive"] = {
        "record": [int(won.sum()), int((~won).sum())],
        "by_line_position_at_deadline": _split_record(decisive, "toward_read_at_deadline"),
        "by_line_position_at_read": _split_record(decisive, "toward_read_at_read"),
        "by_move_after_read": _split_record(decisive, "toward_read_after_read"),
        "card_began_on_read_side_then_left": [int(won[began].sum()), int((~won[began]).sum())],
        "card_never_on_read_side": [int(won[~began].sum()), int((~won[~began]).sum())],
    }
    read_logit = _logit(frame["ledger"])
    card_logit = _logit(frame["card"])
    shown = frame["offset_at_read"].to_numpy(dtype=float)
    later = frame["move_after_read"].to_numpy(dtype=float)
    final = frame["offset_at_deadline"].to_numpy(dtype=float)
    keep = np.isfinite(shown)
    design = np.column_stack([np.ones(keep.sum()), shown[keep]])
    fitted = np.linalg.lstsq(design, read_logit[keep], rcond=None)[0]
    residual = np.full(n, np.nan)
    residual[keep] = read_logit[keep] - design @ fitted
    moved = np.isfinite(later) & (later != 0)
    summary["correlations"] = {
        "read_logit_vs_card_logit": _corr_boot(read_logit, card_logit, rng),
        "read_logit_vs_card_at_read_logit": _corr_boot(
            read_logit, _logit(frame["card_at_read"]), rng
        ),
        "read_logit_vs_raw_model_logit": _corr_boot(read_logit, _logit(frame["raw_model"]), rng),
        "read_logit_vs_offset_at_read": _corr_boot(read_logit, shown, rng),
        "read_logit_vs_offset_at_deadline": _corr_boot(read_logit, final, rng),
        "read_logit_vs_move_after_read": _corr_boot(read_logit, later, rng),
        "read_residual_on_offset_vs_move_after_read": _corr_boot(residual, later, rng),
        "card_logit_vs_offset_at_deadline": _corr_boot(card_logit, final, rng),
    }
    summary["read_logit_per_point_of_offset_at_read"] = float(fitted[1])
    summary["read_side_matches_later_move"] = [
        int((read_home[moved] == (later[moved] > 0)).sum()),
        int(moved.sum()),
    ]
    summary["permutation_within_week"] = _permutation(frame, rng)
    perm = summary["permutation_within_week"]
    wins, losses = summary["decisive"]["record"]
    games = wins + losses
    exact = sum(comb(games, k) for k in range(wins, games + 1)) / 2**games
    per_term = _load(ROOT / "artifacts/forward_monitor/per_term.json")
    scored = [t for t in per_term["terms"] if t["games"] > 0 and t.get("final_e") is not None]
    terms = len(scored)
    summary["best_of_terms"] = {
        "terms_with_a_probability": terms,
        "exact_binomial_p_decisive": exact,
        "exact_binomial_p_best_of_terms": 1 - (1 - exact) ** terms,
        "permutation_p_decisive_best_of_terms": 1
        - (1 - perm["decisive_net_wins"]["p_value"]) ** terms,
        "permutation_p_gain_best_of_terms": 1
        - (1 - perm["log_loss_gain_vs_card"]["p_value"]) ** terms,
        "mean_evidence_value_across_terms": float(np.mean([t["final_e"] for t in scored])),
        "evidence_threshold_split_across_terms": per_term["threshold"] * terms,
    }
    summary["stack"] = _stack(frame, rng)
    read_b = summary["stack"]["card_offset_plus_read"]["in_sample_coefficients"][0]
    week10 = _week10(runs, read_b)
    week10.to_csv(OUT / "week10_information_only.csv", index=False)
    summary["week10"] = {
        "unplayed_games": len(week10),
        "sides_differ": week10.loc[week10["sides_differ"]].to_dict("records"),
        "stack_read_coefficient_fitted_on_weeks_7_to_9": read_b,
    }
    summary["looks"] = (
        len(VIEWS) * len(BASELINES) * 2
        + len(by_week) * len(BASELINES)
        + sum(1 for w in by_week.values() for k in w if k.endswith("_view_vs_card"))
        + len(summary["correlations"])
        + len(perm)
        + 2 * 2
        + 1
        + 2
        + 1
        + 3 * 3
        + 2
    )
    cells = _cells(summary, n)
    (OUT / "weak_signals_batch.json").write_text(
        json.dumps({"cells": cells}, indent=1), encoding="utf-8"
    )
    frame.to_csv(OUT / "per_game.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print("CHECK", json.dumps(summary["checks"]))
    for view in VIEWS:
        block = summary["views"][view]
        print(f"VIEW {view} games {block['games']} decisive vs card {block['decisive_vs_card']}")
        for name, row in block["rows"].items():
            print(
                f"  {name:17s} {row['wins']}-{row['losses']} (no side {row['no_side']}) "
                f"LL {row['log_loss']:.4f} Brier {row['brier']:.4f}"
            )
        for name, pair in block["read_better_by"].items():
            print(f"  read vs {name:17s} LL {_fmt(pair['log_loss'])} | Brier {_fmt(pair['brier'])}")
    for week, cell in by_week.items():
        records = " ".join(f"{k} {v['wins']}-{v['losses']}" for k, v in cell["rows"].items())
        print(f"WEEK {week} games {cell['games']} {records}")
        for key, value in cell.items():
            if isinstance(value, dict) and "probability_positive" in value:
                print(f"  {key:34s} {_fmt(value)}")
    print("LEAN", json.dumps(summary["lean_controls"]))
    print("ACCURACY read minus card", _fmt(summary["read_minus_card_accuracy_points"], 2))
    print("DECISIVE", json.dumps(summary["decisive"]))
    table = decisive.copy()
    for column in ("ledger_at", "card_final_at"):
        table[column] = pd.to_datetime(table[column], utc=True).dt.strftime("%m-%d %H:%M")
    print(table.round(3).to_string(index=False))
    for key, value in summary["correlations"].items():
        print(f"CORR {key:44s} {_fmt(value, 3)}")
    print(
        "READ PER POINT",
        round(summary["read_logit_per_point_of_offset_at_read"], 4),
        "SIDE MATCHES LATER MOVE",
        summary["read_side_matches_later_move"],
    )
    print("PERMUTATION", json.dumps(perm))
    print("BEST_OF", json.dumps(summary["best_of_terms"]))
    print("STACK", json.dumps(summary["stack"]))
    print("WEEK10")
    print(
        week10.drop(columns=["deadline_utc", "card_published_at", "pool_home_spread"])
        .round(3)
        .to_string(index=False)
    )
    print("LOOKS", summary["looks"])


if __name__ == "__main__":
    main()
