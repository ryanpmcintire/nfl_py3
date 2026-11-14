import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

import nfl_ats.card_view as card_view
from nfl_ats.clv import LIVE_CAPTURE_KIND, load_decision_quotes
from nfl_ats.pick_probability import MARKET_MOVE_FEATURE_SUNDAY, PROBABILITY_EPSILON
from nfl_ats.pick_refresh import load_pick_revisions, original_card
from nfl_ats.pool_hook import HOOK_COLUMN, served_hook_frame
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS, sharp_book_movement_features

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
DATA = REPO / "data"
OUT = ARTIFACTS / "flip_attribution_audit"
SEASON = 2026
WEEKS = (7, 8, 9, 10)
FULL_LEADER_DEPTH = len(LEADER_BOOKS)
INTERVAL_LEVEL = 0.95
EVEN_SPLIT = 0.5
FAMILY = "flip_attribution_audit_2026_11_14"
GROUPS = {
    "model": ("model",),
    "line_move": ("move", "move_available"),
    "flags": ("flags",),
    "intercept": ("intercept",),
    "residual": ("residual",),
}
MARGIN_COLUMNS = ("result", "home_margin", "final_margin")


def logit(probability: float) -> float:
    if not np.isfinite(probability):
        return float("nan")
    bounded = min(max(float(probability), PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON)
    return float(np.log(bounded / (1.0 - bounded)))


def stamp(name: str) -> pd.Timestamp:
    return pd.to_datetime(name, format="%Y%m%dT%H%M%SZ", utc=True)


def load_coefficient_sets() -> list[tuple[pd.Timestamp, str, dict]]:
    rows = []
    for path in (ARTIFACTS / "pick_probability").iterdir():
        file = path / "coefficients.json"
        if not path.is_dir() or not file.is_file():
            continue
        try:
            at = stamp(path.name)
        except ValueError:
            continue
        rows.append((at, path.name, json.loads(file.read_text(encoding="utf-8"))))
    rows.sort(key=lambda row: row[0])
    return rows


def coefficients_at(sets: list, when: pd.Timestamp) -> tuple[str, dict]:
    chosen = (sets[0][1], sets[0][2])
    for at, name, payload in sets:
        if at <= when:
            chosen = (name, payload)
    return chosen


def load_feeds() -> list[dict]:
    feeds = []
    for path in sorted((ARTIFACTS / "waterfall_feed").iterdir()):
        file = path / "feed.json"
        if not file.is_file():
            continue
        payload = json.loads(file.read_text(encoding="utf-8"))
        if int(payload.get("season", 0)) != SEASON or int(payload.get("week", 0)) not in WEEKS:
            continue
        feeds.append(
            {
                "feed": path.name,
                "created": pd.Timestamp(payload["created_at_utc"]).tz_convert("UTC"),
                "week": int(payload["week"]),
                "source_forecast": str(payload.get("source_forecast", "")),
                "model_id": str(payload.get("active_model_id", "")),
                "feature_sha": str(payload.get("feature_table_sha256", "")),
                "games": {str(game["game_id"]): game for game in payload.get("games", [])},
            }
        )
    return feeds


def load_captures() -> pd.DataFrame:
    rows = []
    pattern = "independent_validation*/captures/*.json"
    for path in sorted((ARTIFACTS / "prospective").glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        capture = payload.get("capture", payload)
        for row in capture.get("rows", []):
            if int(row.get("season", 0)) == SEASON and int(row.get("week", 0)) in WEEKS:
                rows.append({**row, "capture_file": path.name})
    frame = pd.DataFrame(rows)
    frame["recorded"] = pd.to_datetime(frame["recorded_at_utc"], utc=True, format="ISO8601")
    return frame


def margins_from(frame: pd.DataFrame) -> dict[str, float]:
    if "game_id" not in frame.columns:
        return {}
    scoped = frame
    if "season" in frame.columns:
        scoped = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(SEASON)]
    margin = None
    for column in MARGIN_COLUMNS:
        if column in scoped.columns:
            margin = pd.to_numeric(scoped[column], errors="coerce")
            break
    if margin is None and {"home_score", "away_score"} <= set(scoped.columns):
        margin = pd.to_numeric(scoped["home_score"], errors="coerce") - pd.to_numeric(
            scoped["away_score"], errors="coerce"
        )
    if margin is None:
        return {}
    return {
        str(game): float(value)
        for game, value in zip(scoped["game_id"], margin, strict=True)
        if pd.notna(value)
    }


def load_results() -> tuple[dict[str, float], dict[str, str]]:
    schedules, _ = card_view.load_verified_snapshot(card_view.latest_snapshot(DATA / "raw"))
    frames = [("verified schedule snapshot", schedules)]
    graded = ARTIFACTS / "prospective" / "pick_revisions.graded.parquet"
    extra = sorted((DATA / "raw_schedules").rglob("*.parquet"), key=lambda p: p.stat().st_mtime)
    for path in [*extra[-2:], graded]:
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError) as error:
            print(f"unreadable {path.name}: {error}")
            continue
        frames.append((str(path.relative_to(REPO)), frame))
        print("RESULT SOURCE", path.name, list(frame.columns)[:60])
    margins: dict[str, float] = {}
    sources: dict[str, str] = {}
    for source, frame in frames:
        for game, value in margins_from(frame).items():
            if game not in margins:
                margins[game] = value
                sources[game] = source
    return margins, sources


def move_at(
    quotes: pd.DataFrame, card: pd.DataFrame, when: pd.Timestamp, include_sunday: bool
) -> pd.DataFrame:
    kickoff = pd.to_datetime(card["kickoff"], utc=True)
    games = pd.DataFrame(
        {
            "game_id": card["game_id"].astype(str),
            "commence_time_utc": kickoff,
            "week_first_commence_utc": kickoff.min(),
            "cutoff_utc": when,
        }
    )
    scoped = quotes.loc[quotes["nflverse_game_id"].isin(games["game_id"])]
    out = sharp_book_movement_features(scoped, games, include_sunday=include_sunday)
    return out.set_index("game_id")[["leader_median_net_move", "leader_books", "eligible_books"]]


def nearest_capture(
    captures: pd.DataFrame, game_id: str, when: pd.Timestamp
) -> tuple[float, float, float]:
    rows = captures.loc[captures["game_id"].eq(game_id)]
    if rows.empty:
        return 0.0, float("nan"), float("nan")
    gap = (rows["recorded"] - when).abs()
    row = rows.loc[gap.idxmin()]
    hours = float(gap.min().total_seconds() / 3600.0)
    return float(row["flag_sum"]), float(row["market_move"]), hours


def feed_for(
    feeds: list[dict], week: int, when: pd.Timestamp, model_id: str, feature_sha: str
) -> tuple[dict | None, str]:
    pool = [feed for feed in feeds if feed["week"] == week]
    if not pool:
        return None, "none"

    def nearest(candidates: list[dict]) -> dict:
        return min(candidates, key=lambda feed: abs((feed["created"] - when).total_seconds()))

    both = [f for f in pool if f["model_id"] == model_id and f["feature_sha"] == feature_sha]
    if model_id and both:
        return nearest(both), "model_and_features"
    same_model = [f for f in pool if f["model_id"] == model_id]
    if model_id and same_model:
        return nearest(same_model), "model"
    earlier = [f for f in pool if f["created"] <= when]
    return (earlier[-1] if earlier else pool[0]), "nearest_earlier"


def build_record(
    *,
    game_id: str,
    week: int,
    label: str,
    when: pd.Timestamp,
    side: str,
    served: float,
    feed: dict | None,
    feed_match: str,
    captures: pd.DataFrame,
    sets: list,
    quotes: pd.DataFrame,
    card: pd.DataFrame,
    cache: dict,
    ledger_move: float | None = None,
    ledger_books: float | None = None,
) -> dict:
    name, payload = coefficients_at(sets, when)
    coefficient = payload["coefficients"]
    has_hook = bool(payload.get("hook_push_mass")) and "pool_hook" in coefficient
    include_sunday = payload.get("market_move_feature_version") == MARKET_MOVE_FEATURE_SUNDAY
    key = (week, when, include_sunday)
    if key not in cache:
        cache[key] = move_at(quotes, card, when, include_sunday)
    moved = cache[key]
    replay_books = float(moved.loc[game_id, "leader_books"]) if game_id in moved.index else 0.0
    replay_move = float(moved.loc[game_id, "leader_median_net_move"]) if replay_books > 0 else 0.0
    game = feed["games"].get(game_id, {}) if feed is not None else {}
    model_probability = float(game.get("discrete_base_home_cover_probability", np.nan))
    capture_flags, capture_move, capture_hours = nearest_capture(captures, game_id, when)
    served_logit = logit(served)
    options = []
    if ledger_books is not None and not has_hook:
        stored = float(ledger_move) if ledger_move is not None and np.isfinite(ledger_move) else 0.0
        options.append(("ledger", stored if ledger_books > 0 else 0.0, float(ledger_books)))
    options.append(("replay", replay_move, replay_books))
    flag_options = [("capture", capture_flags)]
    if not has_hook and capture_flags != 0.0:
        flag_options.append(("not_counted", 0.0))
    best: dict | None = None
    for move_source, move, books in options:
        for flag_source, flag_sum in flag_options:
            available = 1.0 if books > 0 else 0.0
            terms = {
                "intercept": float(coefficient["intercept"]),
                "model": float(coefficient["model_logit"]) * logit(model_probability),
                "flags": float(coefficient["flag_sum"]) * flag_sum,
                "move": float(coefficient["move_toward_home"]) * move,
                "move_available": float(coefficient["move_available"]) * available,
            }
            terms["residual"] = served_logit - sum(terms.values())
            candidate = {
                "move_source": move_source,
                "flag_source": flag_source,
                "move_points": move,
                "leader_books": books,
                "flag_sum": flag_sum,
                "terms": terms,
            }
            better = best is None or (
                abs(terms["residual"]) < abs(best["terms"]["residual"])
                and not np.isclose(terms["residual"], best["terms"]["residual"])
            )
            if has_hook:
                better = best is None
            if better:
                best = candidate
    assert best is not None
    return {
        "game_id": game_id,
        "week": week,
        "label": label,
        "when_utc": when.isoformat(),
        "side": side,
        "served_home_probability": served,
        "served_logit": served_logit,
        "coefficient_set": name,
        "has_offset_term": has_hook,
        "feed": feed["feed"] if feed is not None else "",
        "feed_match": feed_match,
        "model_home_probability": model_probability,
        "flag_sum": best["flag_sum"],
        "flag_source": best["flag_source"],
        "move_points": best["move_points"],
        "leader_books": best["leader_books"],
        "move_source": best["move_source"],
        "replay_move_points": replay_move,
        "replay_leader_books": replay_books,
        "ledger_move_points": ledger_move,
        "ledger_leader_books": ledger_books,
        "capture_move_points": capture_move,
        "capture_hours_from_record": capture_hours,
        **{f"term_{term}": value for term, value in best["terms"].items()},
    }


def transition(before: dict, after: dict, kind: str) -> dict:
    direction = 1.0 if after["side"] == "HOME" else -1.0
    hook_era = bool(before["has_offset_term"] or after["has_offset_term"])
    contributions = {}
    for group, members in GROUPS.items():
        delta = sum(after[f"term_{m}"] - before[f"term_{m}"] for m in members)
        contributions[group] = direction * delta
    finite = {k: v for k, v in contributions.items() if np.isfinite(v)}
    driver = max(finite, key=lambda k: finite[k]) if finite else "undetermined"
    residual_name = "pool_line_offset" if hook_era else "unexplained_gap"
    shown = residual_name if driver == "residual" else driver
    needed = -direction * before["served_logit"]
    return {
        "game_id": after["game_id"],
        "week": after["week"],
        "kind": kind,
        "to_label": after["label"],
        "from_side": before["side"],
        "to_side": after["side"],
        "from_probability": before["served_home_probability"],
        "to_probability": after["served_home_probability"],
        "logit_needed_to_cross": needed,
        "driver": shown,
        "driver_alone_crosses": bool(finite.get(driver, 0.0) >= needed) if finite else False,
        "c_model": contributions["model"],
        "c_line_move": contributions["line_move"],
        "c_flags": contributions["flags"],
        "c_intercept": contributions["intercept"],
        "c_residual": contributions["residual"],
        "residual_meaning": residual_name,
        "served_move_points": after["move_points"],
        "served_leader_books": after["leader_books"],
        "move_source": after["move_source"],
        "flag_source": after["flag_source"],
        "replay_move_points": after["replay_move_points"],
        "replay_leader_books": after["replay_leader_books"],
        "coefficient_set_before": before["coefficient_set"],
        "coefficient_set_after": after["coefficient_set"],
    }


def exact(wins: int, losses: int) -> dict:
    total = wins + losses
    if total == 0:
        return {"wins": 0, "losses": 0, "games": 0}
    test = binomtest(wins, total, EVEN_SPLIT)
    interval = test.proportion_ci(confidence_level=INTERVAL_LEVEL, method="exact")
    return {
        "wins": wins,
        "losses": losses,
        "games": total,
        "rate": wins / total,
        "interval_low": float(interval.low),
        "interval_high": float(interval.high),
        "exact_binomial_p_two_sided": float(test.pvalue),
        "probability_positive": float(beta.sf(EVEN_SPLIT, wins + 1, losses + 1)),
    }


def record_of(frame: pd.DataFrame, column: str) -> dict:
    settled = frame.loc[frame[column].notna()]
    wins = int(settled[column].astype(bool).sum())
    return exact(wins, int(len(settled) - wins))


def cell(name: str, description: str, plain: str, stats: dict, weeks: int, notes: str) -> dict:
    return {
        "name": name,
        "description": description,
        "plain_summary": plain,
        "effect": round(100.0 * (stats["rate"] - EVEN_SPLIT), 2),
        "interval_low": round(100.0 * (stats["interval_low"] - EVEN_SPLIT), 2),
        "interval_high": round(100.0 * (stats["interval_high"] - EVEN_SPLIT), 2),
        "probability_positive": round(stats["probability_positive"], 3),
        "sample_games": stats["games"],
        "sample_blocks": weeks,
        "season_start": SEASON,
        "season_end": SEASON,
        "effect_units": "accuracy_points",
        "notes": notes,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 200)
    sets = load_coefficient_sets()
    feeds = load_feeds()
    captures = load_captures()
    quotes = load_decision_quotes(DATA / "market" / "raw", capture_kind=LIVE_CAPTURE_KIND)
    ledger = load_pick_revisions(ARTIFACTS)
    ledger = ledger.loc[
        ledger["season"].astype(int).eq(SEASON) & ledger["week"].astype(int).isin(WEEKS)
    ].sort_values("revision_recorded_at_utc")
    results, result_sources = load_results()
    age = json.loads((ARTIFACTS / "eng62" / "baseline_age.json").read_text(encoding="utf-8"))
    baseline_rows = pd.DataFrame(age["live_move_term"]["rows"])
    baseline_books = (
        baseline_rows.loc[baseline_rows["n_pre"].gt(0)].groupby("game_id")["book"].nunique()
    )
    anchor_rows = pd.DataFrame(age["live_offset_term"]["rows"]).set_index("game_id")
    cache: dict = {}
    records: list[dict] = []
    transitions: list[dict] = []
    games: list[dict] = []
    offset_replay: dict[str, float] = {}
    latest_payload = sets[-1][2]
    push_table = {float(a): float(m) for a, m in latest_payload.get("hook_push_mass", [])}
    hook_coefficient = float(latest_payload["coefficients"].get("pool_hook", 0.0))
    for week in WEEKS:
        card = original_card(ARTIFACTS, season=SEASON, week=week)
        lock = pd.Timestamp(card["recorded_at_utc"].iloc[0]).tz_convert("UTC")
        forecast = str(card["forecast_artifact"].iloc[0])
        week_feeds = [feed for feed in feeds if feed["week"] == week]
        tuesday_feeds = [feed for feed in week_feeds if feed["source_forecast"] == forecast]
        tuesday_feed = tuesday_feeds[0] if tuesday_feeds else None
        tuesday_match = "lock_forecast"
        if tuesday_feed is None:
            tuesday_feed, tuesday_match = feed_for(feeds, week, lock, "", "")
        frame = pd.DataFrame(
            {
                "game_id": card["game_id"].astype(str),
                "season": SEASON,
                "week": week,
                "spread_line": pd.to_numeric(card["decision_home_spread"]),
            }
        )
        try:
            hook = served_hook_frame(frame, DATA, push_table)
            for game, value in zip(hook["game_id"], hook[HOOK_COLUMN], strict=True):
                offset_replay[str(game)] = hook_coefficient * float(value)
        except (OSError, ValueError, KeyError, AttributeError) as error:
            print(f"week {week} offset replay unavailable: {error}")
        for row in card.itertuples():
            game_id = str(row.game_id)
            tuesday_side = str(row.pick_side)
            spread = float(row.decision_home_spread)
            kickoff = pd.Timestamp(row.kickoff).tz_convert("UTC")
            feed_game = tuesday_feed["games"].get(game_id, {}) if tuesday_feed else {}
            tuesday_probability = float(feed_game.get("final_probability", np.nan))
            revisions = ledger.loc[ledger["game_id"].astype(str).eq(game_id)]
            shared = {
                "game_id": game_id,
                "week": week,
                "captures": captures,
                "sets": sets,
                "quotes": quotes,
                "card": card,
                "cache": cache,
            }
            chain = [
                build_record(
                    label="tuesday_lock",
                    when=lock,
                    side=tuesday_side,
                    served=tuesday_probability,
                    feed=tuesday_feed,
                    feed_match=tuesday_match,
                    **shared,
                )
            ]
            chain[0]["feed_side"] = str(feed_game.get("picked_side", ""))
            texts = []
            for revision in revisions.itertuples():
                try:
                    when = stamp(str(revision.refresh_run_id))
                except ValueError:
                    when = pd.Timestamp(revision.revision_recorded_at_utc).tz_convert("UTC")
                feed, match = feed_for(
                    feeds, week, when, str(revision.model_id), str(revision.feature_table_sha256)
                )
                reason = str(revision.reason)
                if "(" in reason:
                    reason = reason[reason.rfind("(") + 1 : reason.rfind(")")]
                stored_move = revision.late_week_net_move
                record = build_record(
                    label=reason,
                    when=when,
                    side=str(revision.new_pick_side),
                    served=float(revision.new_home_cover_probability),
                    feed=feed,
                    feed_match=match,
                    ledger_move=None if pd.isna(stored_move) else float(stored_move),
                    ledger_books=float(revision.late_week_eligible_books),
                    **shared,
                )
                record["trigger_type"] = str(revision.trigger_type)
                chain.append(record)
                texts.append(
                    f"{when.strftime('%m-%d %H:%M')} {revision.new_pick_side} "
                    f"{float(revision.new_home_cover_probability):.3f} {reason}"
                )
            records.extend(chain)
            final = chain[-1]
            final_side = final["side"]
            pairs = pairwise(chain)
            changes = sum(1 for a, b in pairs if a["side"] != b["side"])
            if final_side != tuesday_side:
                status = "net_flip"
            elif changes:
                status = "flip_back"
            else:
                status = "held"
            final_probability = final["served_home_probability"]
            if len(chain) == 1:
                before_kick = [feed for feed in week_feeds if feed["created"] <= kickoff]
                last = before_kick[-1] if before_kick else None
                held = last["games"].get(game_id, {}) if last else {}
                final_probability = float(held.get("final_probability", np.nan))
            margin = results.get(game_id)
            covered = None if margin is None else bool(margin > spread)
            tuesday_right = None if covered is None else (tuesday_side == "HOME") == covered
            final_right = None if covered is None else (final_side == "HOME") == covered
            anchored = game_id in anchor_rows.index
            entry = {
                "week": week,
                "game_id": game_id,
                "pool_home_spread": spread,
                "tuesday_side": tuesday_side,
                "tuesday_home_probability": tuesday_probability,
                "tuesday_feed_side": chain[0]["feed_side"],
                "revisions": " | ".join(texts),
                "side_changes": changes,
                "final_side": final_side,
                "final_home_probability": final_probability,
                "status": status,
                "home_margin": margin,
                "result_source": result_sources.get(game_id, ""),
                "home_covered": covered,
                "tuesday_side_right": tuesday_right,
                "final_side_right": final_right,
                "baseline_leader_books_today": int(baseline_books.get(game_id, 0)),
                "anchor_books": int(anchor_rows.loc[game_id, "n_books"]) if anchored else 0,
                "anchor_source": str(anchor_rows.loc[game_id, "source"]) if anchored else "",
                "served_leader_books": final["leader_books"],
                "served_move_points": final["move_points"],
                "move_source": final["move_source"],
                "replay_leader_books": final["replay_leader_books"],
                "replay_move_points": final["replay_move_points"],
                "offset_term_replayed_now": offset_replay.get(game_id, np.nan),
            }
            if status == "net_flip":
                step = transition(chain[0], final, "tuesday_to_final")
                transitions.append(step)
                entry.update({k: step[k] for k in step if k.startswith("c_") or k == "driver"})
                entry["logit_needed_to_cross"] = step["logit_needed_to_cross"]
                entry["driver_alone_crosses"] = step["driver_alone_crosses"]
                entry["flag_source"] = step["flag_source"]
            if status == "flip_back":
                away = next(i for i, record in enumerate(chain) if record["side"] != tuesday_side)
                leave = transition(chain[away - 1], chain[away], "leave_tuesday_side")
                back = next(
                    i for i in range(len(chain) - 1, 0, -1) if chain[i - 1]["side"] != tuesday_side
                )
                came = transition(chain[back - 1], chain[back], "return_to_tuesday_side")
                transitions.extend([leave, came])
                entry["driver"] = leave["driver"]
                entry["return_driver"] = came["driver"]
                entry.update({k: leave[k] for k in leave if k.startswith("c_")})
                entry["logit_needed_to_cross"] = leave["logit_needed_to_cross"]
                entry["served_leader_books"] = chain[away]["leader_books"]
                entry["served_move_points"] = chain[away]["move_points"]
                entry["move_source"] = chain[away]["move_source"]
                entry["replay_leader_books"] = chain[away]["replay_leader_books"]
                entry["replay_move_points"] = chain[away]["replay_move_points"]
            games.append(entry)
    table = pd.DataFrame(games)
    record_table = pd.DataFrame(records)
    step_table = pd.DataFrame(transitions)
    table.to_csv(OUT / "games.csv", index=False)
    record_table.to_csv(OUT / "records.csv", index=False)
    step_table.to_csv(OUT / "transitions.csv", index=False)
    table["g"] = table["game_id"].str[8:]
    flips = table.loc[table["status"].eq("net_flip")].copy()
    backs = table.loc[table["status"].eq("flip_back")].copy()
    backs["left_side_right"] = backs["final_side_right"].map(
        lambda value: None if value is None or pd.isna(value) else not bool(value)
    )
    depth = flips["served_leader_books"]
    thin = flips.loc[depth.gt(0) & depth.lt(FULL_LEADER_DEPTH)]
    full = flips.loc[depth.ge(FULL_LEADER_DEPTH)]
    no_move = flips.loc[depth.eq(0)]
    one_book = flips.loc[depth.eq(1)]
    show = [
        "week",
        "g",
        "tuesday_side",
        "final_side",
        "side_changes",
        "driver",
        "logit_needed_to_cross",
        "c_model",
        "c_line_move",
        "c_residual",
        "c_flags",
        "c_intercept",
        "served_move_points",
        "served_leader_books",
        "move_source",
        "replay_move_points",
        "replay_leader_books",
        "baseline_leader_books_today",
        "anchor_books",
        "offset_term_replayed_now",
    ]
    print("NET FLIPS")
    print(
        flips[[*show, "driver_alone_crosses", "final_side_right"]].round(3).to_string(index=False)
    )
    print("FLIP AND BACK")
    print(backs[[*show, "return_driver", "left_side_right"]].round(3).to_string(index=False))
    revised = record_table.loc[record_table["ledger_leader_books"].notna()].copy()
    differs = revised.loc[
        ~np.isclose(revised["ledger_leader_books"], revised["replay_leader_books"])
        | ~np.isclose(revised["ledger_move_points"].fillna(0.0), revised["replay_move_points"])
    ]
    print("SERVE-TIME LEDGER VALUES THAT TODAY'S QUOTE HISTORY DOES NOT REPRODUCE")
    print(
        differs[
            [
                "game_id",
                "label",
                "when_utc",
                "side",
                "ledger_move_points",
                "ledger_leader_books",
                "replay_move_points",
                "replay_leader_books",
                "move_source",
                "flag_source",
                "term_residual",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    early = record_table.loc[~record_table["has_offset_term"]]
    late = record_table.loc[record_table["has_offset_term"]]
    fidelity = {
        "records": len(record_table),
        "no_offset_era_records": len(early),
        "no_offset_era_abs_residual_max": float(early["term_residual"].abs().max()),
        "no_offset_era_records_residual_above_0.001": int(
            early["term_residual"].abs().gt(1e-3).sum()
        ),
        "offset_era_records": len(late),
        "feed_match_counts": record_table["feed_match"].value_counts().to_dict(),
        "move_source_counts": revised["move_source"].value_counts().to_dict(),
        "flag_source_counts": record_table["flag_source"].value_counts().to_dict(),
        "tuesday_feed_side_disagrees": int(
            table["tuesday_feed_side"].ne(table["tuesday_side"]).sum()
        ),
        "ledger_revision_records": len(revised),
        "ledger_values_not_reproduced_by_week": differs.groupby("week").size().to_dict(),
        "offset_era_ledger_values_not_reproduced": int(differs["has_offset_term"].sum()),
    }
    print("FIDELITY", json.dumps(fidelity, indent=1, default=str))
    summary: dict = {
        "season": SEASON,
        "weeks": list(WEEKS),
        "fidelity": fidelity,
        "flip_counts_by_week_and_driver": (
            flips.groupby(["week", "driver"]).size().unstack(fill_value=0).to_dict("index")
        ),
        "flip_back_counts_by_week": backs.groupby("week").size().to_dict(),
        "all_games": {
            "tuesday_sides": record_of(table, "tuesday_side_right"),
            "final_sides": record_of(table, "final_side_right"),
        },
        "net_flips_final_side": record_of(flips, "final_side_right"),
        "net_flips_final_side_by_driver": {
            str(driver): record_of(group, "final_side_right")
            for driver, group in flips.groupby("driver")
        },
        "net_flips_final_side_by_week": {
            int(week): record_of(group, "final_side_right") for week, group in flips.groupby("week")
        },
        "net_flips_thin_leader_depth_final_side": record_of(thin, "final_side_right"),
        "net_flips_one_leader_book_final_side": record_of(one_book, "final_side_right"),
        "net_flips_full_leader_depth_final_side": record_of(full, "final_side_right"),
        "net_flips_no_move_input_final_side": record_of(no_move, "final_side_right"),
        "thin_games": thin["game_id"].tolist(),
        "flip_back_left_side": record_of(backs, "left_side_right"),
        "week_10_changes": table.loc[
            table["week"].eq(10) & table["status"].ne("held"),
            [
                "game_id",
                "status",
                "tuesday_side",
                "final_side",
                "driver",
                "served_leader_books",
                "home_margin",
                "final_side_right",
            ],
        ].to_dict("records"),
        "looks": {
            "family": FAMILY,
            "count": int(8 + flips["driver"].nunique() + flips["week"].nunique()),
            "members": "all flips, each driving term, each week, thin depth, one book, full "
            "depth, no move input, flip-back side left, all games Tuesday, all games final",
        },
    }
    audit = pd.read_csv(REPO / "tests" / "scratch" / "fixpicks" / "root_m2_games.csv")
    differ = audit.loc[audit["tuesday_card_side"].ne(audit["final_card_side"])]
    context = {}
    for last in (6, 8, 9):
        part = differ.loc[differ["week"].le(last), "card"].astype(bool)
        context[f"weeks_1_to_{last}_final_side"] = exact(int(part.sum()), int((~part).sum()))
    summary["season_audit_table_changed_sides"] = context
    print("SUMMARY", json.dumps(summary, indent=1, default=str))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    cells = []
    settled = flips.loc[flips["final_side_right"].notna()]
    stats = summary["net_flips_final_side"]
    if stats["games"]:
        cells.append(
            cell(
                "flipaudit_final_side_on_changed_games_2026_w7_w10",
                "Games of 2026 Weeks 7-10 whose final card side differs from the Tuesday card "
                "side; effect is the final side's cover rate at the pool Tuesday line minus an "
                "even split, exact binomial interval",
                "When the card ended the week on a different team than it started on, this "
                "counts how often the team it ended on covered the pool number.",
                stats,
                int(settled["week"].nunique()),
                f"final sides {stats['wins']}-{stats['losses']}, Tuesday sides the mirror; exact "
                f"two-sided p {stats['exact_binomial_p_two_sided']:.3f}",
            )
        )
    for driver, stats in summary["net_flips_final_side_by_driver"].items():
        if stats["games"]:
            cells.append(
                cell(
                    f"flipaudit_final_side_driver_{driver}_2026_w7_w10",
                    "Changed-side games of 2026 Weeks 7-10 where the fitted term contributing "
                    f"most to the served logit crossing even was {driver}; effect is the final "
                    "side's cover rate at the pool Tuesday line minus an even split",
                    "Among games where the card changed teams during the week, this counts the "
                    "ones where one input did most of the pushing and how the new team did.",
                    stats,
                    int(settled.loc[settled["driver"].eq(driver), "week"].nunique()),
                    f"final sides {stats['wins']}-{stats['losses']}; exact two-sided p "
                    f"{stats['exact_binomial_p_two_sided']:.3f}; driver is the largest signed "
                    "term change between the Tuesday record and the last ledger revision",
                )
            )
    extra = (
        (
            "thin_leader_depth",
            summary["net_flips_thin_leader_depth_final_side"],
            thin,
            "final_side_right",
            "the card changed side and the served line-move input rested on one or two of the "
            "three leader books at the final revision",
        ),
        (
            "flip_back_left_side",
            summary["flip_back_left_side"],
            backs,
            "left_side_right",
            "the card left its Tuesday side and came back; effect is the cover rate of the side "
            "it visited and left",
        ),
    )
    for label, stats, group, column, text in extra:
        if stats["games"]:
            cells.append(
                cell(
                    f"flipaudit_{label}_2026_w7_w10",
                    f"2026 Weeks 7-10 games where {text}; cover rate at the pool Tuesday line "
                    "minus an even split, exact binomial interval",
                    "A count of how a small group of changed picks did against the pool number.",
                    stats,
                    int(group.loc[group[column].notna(), "week"].nunique()),
                    f"{stats['wins']}-{stats['losses']}; exact two-sided p "
                    f"{stats['exact_binomial_p_two_sided']:.3f}",
                )
            )
    batch = {
        "family": FAMILY,
        "classification": "unresolved_below_power",
        "league": "nfl",
        "category": "market",
        "source": "artifacts/flip_attribution_audit/summary.json",
        "classification_evidence": "No closing ground met: no resolved wrong sign and no positive "
        "control; a diagnosis of which served term moved picks, recorded as measured, not a rule.",
        "cells": cells,
    }
    (OUT / "record_batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print("CELLS", len(cells), [c["name"] for c in cells])


if __name__ == "__main__":
    main()
