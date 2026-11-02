import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats import pick_probability as pp  # noqa: E402
from nfl_ats.clv import LIVE_CAPTURE_KIND, load_decision_quotes  # noqa: E402
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS  # noqa: E402

ARTIFACTS = REPO / "artifacts"
LOG_DIR = ARTIFACTS / "pool_relative_forward_log"
LOG_PATH = LOG_DIR / "log.parquet"
COEFFICIENT_PATH = LOG_DIR / "challenger_coefficients.json"
FEED_ROOT = ARTIFACTS / "waterfall_feed"
PUBLISHED_PATH = ARTIFACTS / "clv_ledger" / "published_picks.parquet"
GRADED_PATH = ARTIFACTS / "clv_ledger" / "decisions.graded.parquet"
SPLASH_ROOT = REPO / "data" / "splash"
PROBABILITY_FLOOR = 1e-6
SERVED_MOVE_KEY = "move_toward_home"
SERVED_AVAILABLE_KEY = "move_available"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _feeds(week):
    out = []
    for directory in sorted(FEED_ROOT.iterdir()):
        path = directory / "feed.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("season") == 2026 and payload.get("week") == week:
            out.append(
                (pd.Timestamp(payload["created_at_utc"]).tz_convert("UTC"), directory.name, payload)
            )
    return sorted(out, key=lambda x: x[0])


def _current_week():
    latest = json.loads((FEED_ROOT / "latest.json").read_text(encoding="utf-8"))["latest"]
    return int(json.loads((FEED_ROOT / latest / "feed.json").read_text(encoding="utf-8"))["week"])


def _pool_lines(week):
    files = sorted(SPLASH_ROOT.glob(f"2026_week{week:02d}_*.json"))
    if not files:
        raise SystemExit(f"no pool line file for week {week}")
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    return {g["game_id"]: float(g["home_spread"]) for g in payload["games"]}, files[-1].name


def _cards(week):
    pub = pd.read_parquet(PUBLISHED_PATH)
    pub = pub.loc[pub["season"].eq(2026) & pub["week"].eq(week)].copy()
    for column in ("published_at_utc", "pick_deadline_utc", "kickoff"):
        pub[column] = pd.to_datetime(pub[column], utc=True)
    deadline = pub.drop_duplicates("game_id").set_index("game_id")[
        ["home_team", "away_team", "kickoff", "pick_deadline_utc"]
    ]
    latest = pub.sort_values("published_at_utc").groupby("game_id").tail(1).set_index("game_id")
    deadline["card_team"] = latest["pick_team"]
    deadline["card_side"] = np.where(
        deadline["card_team"].eq(deadline["home_team"]), "HOME", "AWAY"
    )
    return deadline


def _coefficients():
    served = json.loads(
        (
            ARTIFACTS
            / json.loads(
                (ARTIFACTS / pp.ACTIVE_PICK_PROBABILITY_FILENAME).read_text(encoding="utf-8")
            )["artifact"]
            / "coefficients.json"
        ).read_text(encoding="utf-8")
    )["coefficients"]
    challenger = json.loads(COEFFICIENT_PATH.read_text(encoding="utf-8"))
    return (
        float(served[SERVED_MOVE_KEY]),
        float(served[SERVED_AVAILABLE_KEY]),
        float(challenger["full_challenger"][pp.MOVE_COLUMN]),
    )


def _leader_median(quotes, game_ids, cutoffs):
    q = quotes.loc[
        quotes["market"].eq("spreads")
        & quotes["bookmaker_key"].isin(LEADER_BOOKS)
        & np.isfinite(quotes["home_spread_line"])
    ].rename(columns={"nflverse_game_id": "game_id"})
    q = q.loc[q["game_id"].isin(game_ids)].copy()
    q["cutoff"] = q["game_id"].map(cutoffs)
    q = q.loc[q["observed_at_utc"].le(q["cutoff"])]
    q = (
        q.sort_values(["game_id", "bookmaker_key", "observed_at_utc"])
        .groupby(["game_id", "bookmaker_key"])
        .tail(1)
    )
    return q.groupby("game_id")["home_spread_line"].median()


def build(week, backfill):
    cards = _cards(week)
    pool, pool_file = _pool_lines(week)
    feeds = _feeds(week)
    if not feeds:
        raise SystemExit(f"no waterfall feed for week {week}")
    now = pd.Timestamp(datetime.now(UTC))
    cm, ca, cc = _coefficients()
    quotes = load_decision_quotes(REPO / "data" / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    chosen = {}
    for created, name, payload in feeds:
        for game in payload["games"]:
            gid = game["game_id"]
            if gid not in cards.index:
                continue
            limit = (
                cards.at[gid, "pick_deadline_utc"]
                if backfill
                else min(now, cards.at[gid, "pick_deadline_utc"])
            )
            if created <= limit and (backfill or created <= now):
                chosen[gid] = (created, name, game)
    if not backfill:
        chosen = {g: v for g, v in chosen.items() if now < cards.at[g, "pick_deadline_utc"]}
        latest_created = max(v[0] for v in chosen.values()) if chosen else None
        chosen = {g: v for g, v in chosen.items() if v[0] == latest_created}
    if not chosen:
        print("no games before their pick deadline; nothing written")
        return None
    ids = sorted(chosen)
    cutoffs = {
        g: (
            cards.at[g, "pick_deadline_utc"]
            if backfill
            else min(now, cards.at[g, "pick_deadline_utc"])
        )
        for g in ids
    }
    leader = _leader_median(quotes, set(ids), cutoffs)
    move_s = pd.Series(0.0, index=ids)
    avail_s = pd.Series(0.0, index=ids)
    by_feed = {}
    for g in ids:
        by_feed.setdefault(chosen[g][0], []).append(g)
    for created, group in by_feed.items():
        preds = pd.DataFrame({"game_id": group, "kickoff": [cards.at[g, "kickoff"] for g in group]})
        m = pp.market_move_toward_home(
            preds,
            REPO / "data",
            now=created.to_pydatetime(),
            feature_version=pp.MARKET_MOVE_FEATURE_SUNDAY,
        ).set_index("game_id")
        move_s.loc[group] = m.loc[group, pp.MOVE_COLUMN].to_numpy()
        avail_s.loc[group] = m.loc[group, pp.MOVE_AVAILABLE_COLUMN].to_numpy()
    rows = []
    for g in ids:
        created, name, game = chosen[g]
        p_served = float(game["final_probability"])
        pool_line = pool.get(g)
        lead = leader.get(g, np.nan)
        pool_move = float(lead - pool_line) if pool_line is not None and np.isfinite(lead) else 0.0
        z_chal = float(_logit(p_served)) - cm * move_s[g] - ca * avail_s[g] + cc * pool_move
        p_chal = float(_sigmoid(z_chal))
        rows.append(
            {
                "run_at_utc": created if backfill else now,
                "backfill": bool(backfill),
                "season": 2026,
                "week": week,
                "game_id": g,
                "feed": name,
                "feed_created_at_utc": created,
                "pick_deadline_utc": cards.at[g, "pick_deadline_utc"],
                "card_side": cards.at[g, "card_side"],
                "served_home_probability": p_served,
                "pool_home_spread": pool_line,
                "pool_file": pool_file,
                "leader_median_line": float(lead) if np.isfinite(lead) else np.nan,
                "pool_move": pool_move,
                "served_move": float(move_s[g]),
                "served_move_available": float(avail_s[g]),
                "challenger_home_probability": p_chal,
                "challenger_side": "HOME" if p_chal >= 0.5 else "AWAY",
                "served_move_coefficient": cm,
                "challenger_move_coefficient": cc,
            }
        )
    return pd.DataFrame(rows)


def append(frame):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.is_file():
        old = pd.read_parquet(LOG_PATH)
        frame = pd.concat([old, frame], ignore_index=True)
        frame = frame.drop_duplicates(["game_id", "run_at_utc", "backfill"], keep="last")
    frame.to_parquet(LOG_PATH, index=False)
    return frame


def _record(side, settle):
    d = pd.DataFrame({"side": side, "settle": settle}).dropna()
    d = d.loc[d["settle"].ne(0)]
    won = (d["settle"] > 0) == d["side"].eq("HOME")
    return int(won.sum()), int((~won).sum())


def grade():
    log = pd.read_parquet(LOG_PATH)
    frozen = log.sort_values("run_at_utc").groupby("game_id").tail(1).set_index("game_id")
    graded = pd.read_parquet(GRADED_PATH)
    graded = graded.loc[graded["arm"].eq("played")].drop_duplicates("game_id").set_index("game_id")
    frozen = frozen.join(graded[["result"]], how="inner")
    frozen["settle"] = frozen["result"] - frozen["pool_home_spread"]
    decided = frozen.loc[frozen["settle"].notna() & frozen["settle"].ne(0)]
    cw, cl = _record(decided["card_side"], decided["settle"])
    hw, hl = _record(decided["challenger_side"], decided["settle"])
    differ = decided.loc[decided["card_side"].ne(decided["challenger_side"])]
    dcw, dcl = _record(differ["card_side"], differ["settle"])
    dhw, dhl = _record(differ["challenger_side"], differ["settle"])
    print(f"games logged {len(frozen)}, decided at the pool line {len(decided)}")
    print(f"card {cw}-{cl}, challenger {hw}-{hl}")
    print(f"sides differ {len(differ)}: card {dcw}-{dcl}, challenger {dhw}-{dhl}")
    print(f"backfilled rows in the frozen set {int(decided['backfill'].sum())} of {len(decided)}")
    by_week = decided.groupby("week").apply(
        lambda d: pd.Series(
            {
                "n": len(d),
                "card_wins": _record(d["card_side"], d["settle"])[0],
                "challenger_wins": _record(d["challenger_side"], d["settle"])[0],
                "differ": int(d["card_side"].ne(d["challenger_side"]).sum()),
            }
        ),
        include_groups=False,
    )
    print(by_week.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Forward log of the pool-relative challenger beside the card"
    )
    parser.add_argument("--week", type=int)
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--grade", action="store_true")
    args = parser.parse_args()
    if args.grade:
        grade()
        return
    week = args.week if args.week is not None else _current_week()
    frame = build(week, args.backfill)
    if frame is None:
        return
    total = append(frame)
    print(
        f"wrote {len(frame)} rows for week {week} "
        f"({'backfill' if args.backfill else 'live'}); log rows {len(total)}"
    )
    print(
        frame[
            [
                "game_id",
                "card_side",
                "served_home_probability",
                "pool_home_spread",
                "leader_median_line",
                "pool_move",
                "challenger_home_probability",
                "challenger_side",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
