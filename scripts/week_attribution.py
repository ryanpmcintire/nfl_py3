import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats import pick_probability as pp  # noqa: E402
from nfl_ats.clv import pick_correct  # noqa: E402
from nfl_ats.four_overlay_composition import protection_flags_for_card  # noqa: E402
from nfl_ats.pick_refresh import latest_revisions_by_game  # noqa: E402
from nfl_ats.pool_hook import HOOK_COLUMN, served_hook_frame  # noqa: E402
from nfl_ats.snapshots import latest_snapshot, load_verified_snapshot  # noqa: E402

ARTIFACTS = REPO / "artifacts"
DATA = REPO / "data"
OUT = ARTIFACTS / "week_attribution"
LAYERS = ("base", "calibrated", "flags", "move", "hook")


def _stamp(name):
    return pd.Timestamp(name.replace("T", " ").rstrip("Z"), tz="UTC")


def _card(season, week):
    pub = pd.read_parquet(ARTIFACTS / "clv_ledger" / "published_picks.parquet")
    pub = pub.loc[pub["season"].eq(season) & pub["week"].eq(week)].copy()
    if pub.empty:
        raise SystemExit(f"no published card for {season} week {week}")
    for column in ("published_at_utc", "pick_deadline_utc", "kickoff"):
        pub[column] = pd.to_datetime(pub[column], utc=True)
    latest = pub.sort_values("published_at_utc").groupby("game_id").tail(1).set_index("game_id")
    latest["card_side"] = np.where(latest["pick_team"].eq(latest["home_team"]), "HOME", "AWAY")
    revisions = latest_revisions_by_game(ARTIFACTS, season=season, week=week)
    latest["ledger_side"] = [
        str(revisions.get(g, {}).get("new_pick_side") or "") for g in latest.index
    ]
    return latest[["home_team", "away_team", "kickoff", "pick_deadline_utc", "card_side", "ledger_side"]]


def _pool(season, week):
    files = sorted((DATA / "splash").glob(f"{season}_week{week:02d}_*.json"))
    if not files:
        raise SystemExit(f"no pool line file for {season} week {week}")
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    return {g["game_id"]: float(g["home_spread"]) for g in payload["games"]}, files[-1].name


def _feeds(season, week):
    out = []
    for directory in sorted((ARTIFACTS / "waterfall_feed").iterdir()):
        path = directory / "feed.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("season") == season and payload.get("week") == week:
            out.append((pd.Timestamp(payload["created_at_utc"]).tz_convert("UTC"), directory.name, payload))
    return sorted(out, key=lambda x: x[0])


def _model(directory):
    payload = json.loads((directory / pp.COEFFICIENTS_FILENAME).read_text(encoding="utf-8"))
    meta_path = directory / pp.METADATA_FILENAME
    if "counted_flag_columns" not in payload and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "counted_flag_columns" in meta:
            payload["counted_flag_columns"] = meta["counted_flag_columns"]
    return pp.PickProbabilityModel.from_dict(payload, artifact=directory.name)


def _model_at(created):
    root = ARTIFACTS / pp.PICK_PROBABILITY_ARTIFACT_ROOT
    dirs = [d for d in sorted(root.iterdir()) if (d / pp.COEFFICIENTS_FILENAME).is_file()]
    eligible = [d for d in dirs if _stamp(d.name) <= created]
    return eligible[-1] if eligible else None


def _active_dir():
    pointer = json.loads((ARTIFACTS / pp.ACTIVE_PICK_PROBABILITY_FILENAME).read_text(encoding="utf-8"))
    return ARTIFACTS / pointer["artifact"]


def _terms(model, frame, schedules, now):
    pred = frame[["game_id", "season", "week", "gameday", "home_team", "away_team"]].copy()
    flags = pp.signed_composition_flags_fail_open(
        pred, schedules, protection_back_side=protection_flags_for_card(pred, DATA)
    )
    counted = [c for c in model.counted_flag_columns if c in flags.columns]
    flag_sum = flags.set_index("game_id")[counted].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    move = pp.market_move_toward_home(
        frame[["game_id", "kickoff"]],
        DATA,
        now=now.to_pydatetime(),
        feature_version=model.market_move_feature_version,
    ).set_index("game_id")
    if model.has_pool_hook:
        hook = served_hook_frame(
            frame[["game_id", "season", "week", "spread_line"]], DATA, dict(model.hook_push_mass)
        ).set_index("game_id")[HOOK_COLUMN]
    else:
        hook = pd.Series(0.0, index=frame["game_id"])
    ids = frame["game_id"]
    out = pd.DataFrame(
        {
            "flag_sum": flag_sum.reindex(ids).fillna(0.0).to_numpy(),
            "move": move[pp.MOVE_COLUMN].reindex(ids).fillna(0.0).to_numpy(),
            "move_available": move[pp.MOVE_AVAILABLE_COLUMN].reindex(ids).fillna(0.0).to_numpy(),
            "hook": hook.reindex(ids).fillna(0.0).to_numpy(),
        },
        index=frame.index,
    )
    for column in counted:
        out[column] = flags.set_index("game_id")[column].reindex(ids).fillna(0.0).to_numpy()
    return out, counted


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), pp.PROBABILITY_EPSILON, 1 - pp.PROBABILITY_EPSILON)
    return np.log(p / (1 - p))


def _layers(model, base, terms):
    terms = terms.set_index(base.index)
    zero = pd.Series(0.0, index=base.index)
    hook_zero = zero if model.has_pool_hook else None
    hook = terms["hook"] if model.has_pool_hook else None
    return {
        "base": base,
        "calibrated": model.home_probability(base, zero, zero, zero, hook_zero),
        "flags": model.home_probability(base, terms["flag_sum"], zero, zero, hook_zero),
        "move": model.home_probability(base, terms["flag_sum"], terms["move"], terms["move_available"], hook_zero),
        "hook": model.home_probability(base, terms["flag_sum"], terms["move"], terms["move_available"], hook),
    }


def _graded(pick_home, margin):
    return pick_correct(pick_home, margin).where(margin.notna())


def _record(correct):
    s = pd.Series(correct, dtype=float).dropna()
    return f"{int(s.eq(1).sum())}-{int(s.eq(0).sum())}"


def _side(p):
    return np.where(np.asarray(p, dtype=float) >= 0.5, "HOME", "AWAY")


def build(season, week):
    card = _card(season, week)
    pool, pool_file = _pool(season, week)
    feeds = _feeds(season, week)
    if not feeds:
        raise SystemExit(f"no waterfall feed for {season} week {week}")
    chosen = {}
    for created, name, payload in feeds:
        for game in payload["games"]:
            g = game["game_id"]
            if g in card.index and created <= card.at[g, "pick_deadline_utc"]:
                chosen[g] = (created, name, game)
    snapshot = latest_snapshot(DATA / "raw")
    schedules, _ = load_verified_snapshot(snapshot)
    sched = schedules.set_index("game_id")
    rows = []
    for g in card.index:
        if g not in chosen:
            continue
        created, name, game = chosen[g]
        rows.append(
            {
                "game_id": g,
                "season": season,
                "week": week,
                "gameday": sched.at[g, "gameday"],
                "home_team": card.at[g, "home_team"],
                "away_team": card.at[g, "away_team"],
                "kickoff": card.at[g, "kickoff"],
                "spread_line": pool.get(g, np.nan),
                "feed": name,
                "feed_created": created,
                "base_home": float(game["discrete_base_home_cover_probability"]),
                "feed_final_home": float(game["final_probability"]),
                "feed_side": game["picked_side"],
                "card_side": card.at[g, "card_side"],
                "ledger_side": card.at[g, "ledger_side"],
                "result": pd.to_numeric(sched.at[g, "result"], errors="coerce"),
            }
        )
    frame = pd.DataFrame(rows)
    parts = []
    for created, group in frame.groupby("feed_created"):
        model_dir = _model_at(created)
        model = _model(model_dir)
        group = group.copy()
        terms, counted = _terms(model, group, schedules, created)
        probs = _layers(model, group["base_home"], terms)
        for key, value in probs.items():
            group[f"p_{key}"] = np.asarray(value, dtype=float)
        for column in terms.columns:
            group[f"term_{column}"] = terms[column].to_numpy()
        group["model_artifact"] = model_dir.name
        group["counted_flags"] = ",".join(counted)
        parts.append(group)
    frame = pd.concat(parts, ignore_index=True)
    current = _model(_active_dir())
    cur_terms, _ = _terms(current, frame, schedules, frame["feed_created"].max())
    frame["p_current"] = np.asarray(
        _layers(current, frame["base_home"], cur_terms)["hook"], dtype=float
    )
    frame["settle_margin"] = frame["result"] - frame["spread_line"]
    for key in (*LAYERS, "current"):
        frame[f"side_{key}"] = _side(frame[f"p_{key}"])
    frame["side_feed"] = frame["feed_side"]
    frame["side_card"] = frame["card_side"]
    for key in (*LAYERS, "current", "feed", "card"):
        frame[f"ok_{key}"] = _graded(frame[f"side_{key}"].eq("HOME"), frame["settle_margin"])
    frame["status"] = np.where(
        frame["settle_margin"].isna(), "open", np.where(frame["settle_margin"].eq(0), "push", "settled")
    )
    frame["replay_error"] = (frame["p_hook"] - frame["feed_final_home"]).abs()
    return frame, pool_file, Path(str(getattr(snapshot, "root", snapshot))).name


def _term_record(frame, mask, direction):
    sub = frame.loc[mask]
    ok = _graded(pd.Series(direction[mask], index=sub.index).gt(0), sub["settle_margin"])
    return int(mask.sum()), _record(ok)


def summarize(frame):
    lines = []
    settled = frame["status"].eq("settled")
    lines.append("| layer | record | sides changed v previous | changed games settled |")
    lines.append("|---|---|---|---|")
    prev = None
    for key in (*LAYERS, "feed", "card", "current"):
        rec = _record(frame[f"ok_{key}"])
        if prev is None or key == "current":
            ref = "base" if key == "current" else None
        else:
            ref = prev
        if ref is None:
            lines.append(f"| {key} | {rec} | - | - |")
        else:
            changed = frame[f"side_{key}"].ne(frame[f"side_{ref}"])
            names = []
            for _, r in frame.loc[changed].iterrows():
                outcome = "open" if r["status"] == "open" else ("push" if r["status"] == "push" else ("W" if r[f"ok_{key}"] == 1 else "L"))
                names.append(f"{r['game_id'][8:]} {r[f'side_{key}']} {outcome}")
            lines.append(f"| {key} (v {ref}) | {rec} | {int(changed.sum())}: {'; '.join(names) or 'none'} | {_record(frame.loc[changed & settled, f'ok_{key}'])} |")
        if key in LAYERS or key == "feed":
            prev = key
    lines.append("")
    lines.append("| term | nonzero games | term-side record |")
    lines.append("|---|---|---|")
    for name, column, prev_layer in (
        ("flags", "term_flag_sum", "calibrated"),
        ("move", "term_move", "flags"),
        ("hook", "term_hook", "move"),
    ):
        values = (frame[f"p_{name}"] - frame[f"p_{prev_layer}"]).to_numpy(dtype=float)
        mask = np.abs(frame[column].to_numpy(dtype=float)) > 0
        n, rec = _term_record(frame, mask, values)
        lines.append(f"| {name} | {n} | {rec} |")
    for column in [c for c in frame.columns if c.startswith("term_flag_") and c != "term_flag_sum"]:
        values = frame[column].to_numpy(dtype=float)
        mask = np.abs(values) > 0
        n, rec = _term_record(frame, mask, values)
        lines.append(f"| {column[5:]} | {n} | {rec} |")
    lines.append("")
    open_games = frame.loc[frame["status"].ne("settled"), ["game_id", "status"]]
    lines.append("open/push: " + (", ".join(f"{a} {b}" for a, b in open_games.itertuples(index=False)) or "none"))
    lines.append(f"card v ledger side mismatches: {int((frame['ledger_side'].ne('') & frame['ledger_side'].ne(frame['card_side'])).sum())}")
    lines.append(f"card v feed side mismatches: {int(frame['card_side'].ne(frame['feed_side']).sum())}")
    lines.append(f"max |replayed served - feed final|: {frame['replay_error'].max():.4f}; models {sorted(frame['model_artifact'].unique())}")
    return "\n".join(lines)


def _card_only(season, week, have):
    card = _card(season, week)
    card = card.loc[~card.index.isin(have)]
    if card.empty:
        return pd.DataFrame()
    pool, _ = _pool(season, week)
    schedules, _ = load_verified_snapshot(latest_snapshot(DATA / "raw"))
    sched = schedules.set_index("game_id")
    frame = pd.DataFrame(
        {
            "game_id": card.index,
            "season": season,
            "week": week,
            "home_team": card["home_team"].to_numpy(),
            "away_team": card["away_team"].to_numpy(),
            "spread_line": [pool.get(g, np.nan) for g in card.index],
            "card_side": card["card_side"].to_numpy(),
            "ledger_side": card["ledger_side"].to_numpy(),
            "result": [pd.to_numeric(sched.at[g, "result"], errors="coerce") for g in card.index],
        }
    )
    frame["settle_margin"] = frame["result"] - frame["spread_line"]
    frame["side_card"] = frame["card_side"]
    frame["ok_card"] = _graded(frame["side_card"].eq("HOME"), frame["settle_margin"])
    frame["status"] = np.where(
        frame["settle_margin"].isna(), "open", np.where(frame["settle_margin"].eq(0), "push", "settled")
    )
    frame["replay_flag"] = "card only"
    return frame


def _p_positive(wins, losses):
    from scipy.stats import beta

    return float(beta.sf(0.5, wins + 1, losses + 1))


def _wl(ok):
    s = pd.Series(ok, dtype=float).dropna()
    return int(s.eq(1).sum()), int(s.eq(0).sum())


def _fmt(w, n_lost):
    return f"{w}-{n_lost} (P+ {_p_positive(w, n_lost):.2f})" if w + n_lost else "0-0"


def season_run(season, weeks):
    parts = []
    for week in weeks:
        try:
            frame, _, _ = build(season, week)
            frame["replay_flag"] = np.where(
                (frame["replay_error"] > 0.005) | frame["card_side"].ne(frame["feed_side"]), "replay mismatch", ""
            )
        except (SystemExit, KeyError):
            frame = pd.DataFrame()
        extra = _card_only(season, week, set(frame["game_id"]) if len(frame) else set())
        parts.append(pd.concat([frame, extra], ignore_index=True))
    return pd.concat(parts, ignore_index=True)


def _term_rows(rep, weeks, specs):
    rows = []
    for name, column, values_of in specs:
        cells = []
        tw = tl = tn = 0
        for w in weeks:
            f = rep.loc[rep["week"].eq(w)]
            if column not in f or f[column].isna().all():
                cells.append("-")
                continue
            values = values_of(f)
            mask = np.abs(f[column].to_numpy(dtype=float)) > 0
            ok = _graded(pd.Series(values[mask] > 0, index=f.index[mask]), f.loc[mask, "settle_margin"])
            wi, lo = _wl(ok)
            tw += wi
            tl += lo
            tn += int(mask.sum())
            cells.append(f"{int(mask.sum())}: {wi}-{lo}")
        rows.append(f"| {name} | " + " | ".join(cells) + f" | {tn} | {_fmt(tw, tl)} |")
    return rows


def season_summary(frame):
    out = []
    weeks = sorted(frame["week"].unique())
    out.append("| week | games | card-only | replay-mismatch | " + " | ".join(LAYERS) + " | card | open/push |")
    out.append("|" + "---|" * (len(LAYERS) + 6))
    for w in weeks:
        f = frame.loc[frame["week"].eq(w)]
        cells = [_record(f[f"ok_{k}"]) if f[f"ok_{k}"].notna().any() else "-" for k in LAYERS]
        out.append(
            f"| {w} | {len(f)} | {int(f['replay_flag'].eq('card only').sum())} | {int(f['replay_flag'].eq('replay mismatch').sum())} | "
            + " | ".join(cells)
            + f" | {_record(f['ok_card'])} | {int(f['status'].ne('settled').sum())} |"
        )
    rep = frame.loc[frame["replay_flag"].ne("card only")]
    cells = [_fmt(*_wl(rep[f"ok_{k}"])) for k in LAYERS]
    out.append(
        f"| total (replayed games {len(rep)}) | | | | " + " | ".join(cells)
        + f" | {_fmt(*_wl(frame['ok_card']))} (all {len(frame)}) | |"
    )
    out.append("")
    out.append("Layer flips (side changed v previous layer; record of the new side):")
    out.append("")
    out.append("| layer | " + " | ".join(f"W{w}" for w in weeks) + " | flips settled | record, P+ |")
    out.append("|" + "---|" * (len(weeks) + 3))
    pairs = [("calibrated", "base"), ("flags", "calibrated"), ("move", "flags"), ("hook", "move"), ("card", "hook")]
    for key, ref in pairs:
        cells = []
        tw = tl = 0
        for w in weeks:
            f = rep.loc[rep["week"].eq(w)]
            if f[f"side_{ref}"].isna().all():
                cells.append("-")
                continue
            ch = f[f"side_{key}"].ne(f[f"side_{ref}"])
            wi, lo = _wl(f.loc[ch, f"ok_{key}"])
            tw += wi
            tl += lo
            cells.append(f"{wi}-{lo}")
        out.append(f"| {key} v {ref} | " + " | ".join(cells) + f" | {tw + tl} | {_fmt(tw, tl)} |")
    out.append("")
    out.append("Terms (nonzero games; record on the side the term pushed relative to the layer before it):")
    out.append("")
    out.append("| term | " + " | ".join(f"W{w}" for w in weeks) + " | nonzero | pushed-side record, P+ |")
    out.append("|" + "---|" * (len(weeks) + 3))
    specs = [
        (name, column, (lambda f, n=name, p=prev: (f[f"p_{n}"] - f[f"p_{p}"]).to_numpy(dtype=float)))
        for name, column, prev in (
            ("flags", "term_flag_sum", "calibrated"),
            ("move", "term_move", "flags"),
            ("hook", "term_hook", "move"),
        )
    ]
    flag_cols = sorted(c for c in frame.columns if c.startswith("term_flag_") and c != "term_flag_sum")
    specs += [(c[5:], c, (lambda f, c=c: f[c].to_numpy(dtype=float))) for c in flag_cols]
    out.extend(_term_rows(rep, weeks, specs))
    out.append("")
    bad = sorted(rep.loc[rep["replay_flag"].eq("replay mismatch"), "week"].unique().tolist())
    co = frame.loc[frame["replay_flag"].eq("card only"), "week"].value_counts().sort_index().to_dict()
    out.append(f"weeks with replay mismatch: {bad}; card-only games by week: {co}")
    mism = int((frame["ledger_side"].ne("") & frame["ledger_side"].ne(frame["card_side"])).sum())
    out.append(f"card v ledger side mismatches: {mism}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="per-week served-side attribution by layer at the pool line")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int)
    parser.add_argument("--weeks", help="inclusive range such as 1-10")
    args = parser.parse_args()
    if args.weeks:
        lo, hi = (int(x) for x in args.weeks.split("-"))
        frame = season_run(args.season, range(lo, hi + 1))
        OUT.mkdir(parents=True, exist_ok=True)
        stem = OUT / f"{args.season}_season"
        frame.to_csv(stem.with_suffix(".csv"), index=False)
        text = season_summary(frame)
        stem.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
        print(text)
        return
    if args.week is None:
        parser.error("--week or --weeks required")
    frame, pool_file, snapshot = build(args.season, args.week)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{args.season}_week{args.week:02d}"
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    text = summarize(frame)
    header = f"season {args.season} week {args.week}; pool {pool_file}; schedules {snapshot}; games {len(frame)}"
    stem.with_suffix(".md").write_text(header + "\n\n" + text + "\n", encoding="utf-8")
    print(header)
    print(text)
    cols = ["game_id", "spread_line", "base_home", "p_calibrated", "term_flag_sum", "term_move", "term_hook", "p_hook", "feed_final_home", "card_side", "result", "status", "ok_card"]
    print(frame[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
