import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.published_picks import game_deadlines

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "lead99_llm_read" / "runs"
REPEATS = ROOT / "artifacts" / "lead99_llm_read" / "repeats"
OUT = ROOT / "artifacts" / "lead104_revision_path"
SEASON = 2026
SEED = 20261103
RESAMPLES = 10000
PERMUTATIONS = 20000
COIN = 0.5
POINTS = 100
FAMILY = "lead104_revision_path"
BETWEEN_READ_SD = None
CONSTRUCTION = {
    "tuesday": "Tuesday view to the last view before each pick deadline",
    "earliest": "earliest view to the last view before each pick deadline",
}
RECORDED = {
    "earliest": [
        "revision_vs_move_correlation",
        "last_vs_first_ll",
        "revision_alone_vs_half_ll",
        "revision_direction_accuracy",
        "changed_first_vs_half_ll",
        "changed_last_vs_half_ll",
        "changed_last_vs_first_ll",
        "changed_revision_direction_accuracy",
        "residual_direction_accuracy",
        "residual_alone_vs_half_ll",
        "first_plus_residual_vs_first_ll",
        "first_plus_tracking_vs_first_ll",
        "move_direction_accuracy",
        "first_plus_move_part_vs_first_ll",
        "first_plus_drift_vs_first_ll",
    ],
    "tuesday": [
        "revision_vs_move_correlation",
        "last_vs_first_ll",
        "residual_direction_accuracy",
    ],
}
TEXT = {
    "first_plus_move_part_vs_first_ll": (
        "first view plus the fitted slope times the line move between the views (no intercept) "
        "against the first view alone, paired log-loss improvement",
        "whether shifting the AI read's first look by the betting line's later move improved on it",
    ),
    "first_plus_drift_vs_first_ll": (
        "first view plus the fitted intercept of revision on move (the read's average drift "
        "toward the home team between views) against the first view alone, paired log-loss "
        "improvement",
        "whether the AI read's general drift toward home teams between its first and last "
        "look improved on its first look",
    ),
    "revision_vs_move_correlation": (
        "correlation of the read's revision (last view minus first view, logit of the home "
        "cover probability) with the market line move toward the home team between the views",
        "how closely the change in an AI model's read of a game during the week tracked the "
        "betting line's move",
    ),
    "first_vs_half_ll": (
        "first view against a constant 0.5, paired log-loss improvement",
        "whether the AI read's first look at a game beat a coin flip",
    ),
    "last_vs_half_ll": (
        "last view against a constant 0.5, paired log-loss improvement",
        "whether the AI read's last look at a game beat a coin flip",
    ),
    "last_vs_first_ll": (
        "last view against the first view, paired log-loss improvement",
        "whether the AI read's last look at a game beat its first look",
    ),
    "first_accuracy": (
        "first view's side against the cover side, accuracy over 50 percent",
        "how often the AI read's first pick covered",
    ),
    "last_accuracy": (
        "last view's side against the cover side, accuracy over 50 percent",
        "how often the AI read's last pick covered",
    ),
    "revision_alone_vs_half_ll": (
        "the revision alone as a probability (logistic of last minus first) against a constant "
        "0.5 on games whose read changed, paired log-loss improvement",
        "whether the direction and size of the AI read's change of mind pointed at the side "
        "that covered",
    ),
    "revision_direction_accuracy": (
        "direction of the revision against the cover side on games whose read changed, "
        "accuracy over 50 percent",
        "how often the AI read moved toward the side that went on to cover",
    ),
    "changed_first_vs_half_ll": (
        "first view against a constant 0.5 on games where the two views pick different sides, "
        "paired log-loss improvement",
        "whether the AI read's first look beat a coin flip in games where it later switched sides",
    ),
    "changed_last_vs_half_ll": (
        "last view against a constant 0.5 on games where the two views pick different sides, "
        "paired log-loss improvement",
        "whether the AI read's last look beat a coin flip in games where it had switched sides",
    ),
    "changed_last_vs_first_ll": (
        "last view against the first view on games where the two views pick different sides, "
        "paired log-loss improvement",
        "whether the AI read's last look beat its first look in games where it switched sides",
    ),
    "changed_revision_direction_accuracy": (
        "last view's side (the direction of the revision) against the cover side on games "
        "where the two views pick different sides, accuracy over 50 percent",
        "how often the AI read's later pick covered when it had switched sides during the week",
    ),
    "changed_revision_alone_vs_half_ll": (
        "the revision alone as a probability against a constant 0.5 on games where the two "
        "views pick different sides, paired log-loss improvement",
        "whether the size of the AI read's change of mind pointed at the side that covered in "
        "games where it switched sides",
    ),
    "residual_direction_accuracy": (
        "direction of the revision net of the line move (residual of revision on move, slope "
        "fitted without outcomes) against the cover side, accuracy over 50 percent",
        "how often the part of the AI read's change that the betting line's move does not "
        "explain pointed at the side that covered",
    ),
    "residual_alone_vs_half_ll": (
        "the revision net of the line move as a probability against a constant 0.5, paired "
        "log-loss improvement",
        "whether the part of the AI read's change that the betting line's move does not "
        "explain pointed at the side that covered, weighted by its size",
    ),
    "first_plus_residual_vs_first_ll": (
        "first view plus the revision net of the line move against the first view alone, "
        "paired log-loss improvement",
        "whether the part of the AI read's change that the betting line's move does not "
        "explain improved on its first look",
    ),
    "first_plus_tracking_vs_first_ll": (
        "first view plus the part of the revision fitted by the line move against the first "
        "view alone, paired log-loss improvement",
        "whether the part of the AI read's change that follows the betting line improved on "
        "its first look",
    ),
    "move_direction_accuracy": (
        "direction of the market line move between the two views against the cover side on "
        "games whose line moved, accuracy over 50 percent",
        "how often the side the betting line moved toward between the two looks covered the "
        "pool's Tuesday number",
    ),
}


def logit(p):
    return math.log(p / (1 - p))


def logits(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def loss(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def view_rows(run):
    market = {g["game_id"]: g for g in run["packet"]["games"]}
    out = {}
    for r in run["parsed_rows"]:
        g = market[r["game_id"]]
        m = g.get("market")
        out[r["game_id"]] = {
            "p": r["home_cover_probability"],
            "spread": m["home_spread_latest"] if m else None,
            "pool": g["pool_line"]["home_spread"],
        }
    return out


def between_read_sd():
    runs = [load(p) for p in sorted(REPEATS.glob("*.json"))]
    runs.append(load(next(RUNS.glob("2026-week-09-tuesday-*.json"))))
    stacks = {}
    for run in runs:
        for r in run.get("parsed_rows") or run.get("rows") or []:
            stacks.setdefault(r["game_id"], []).append(logit(r["home_cover_probability"]))
    sds = [np.std(v, ddof=1) for v in stacks.values() if len(v) > 1]
    return float(np.mean(sds)) if sds else None


def schedule(weeks):
    g = pd.read_parquet(ROOT / "data" / "processed" / "game_features.parquet")
    g = g[g["season"].eq(SEASON) & g["week"].isin(weeks)].copy()
    g["kickoff"] = pd.to_datetime(g["kickoff"], utc=True, format="ISO8601")
    deadlines = {}
    for _, part in g.groupby("week"):
        deadlines.update(game_deadlines(part[["game_id", "kickoff"]]))
    g["deadline"] = g["game_id"].map(deadlines)
    g["margin"] = g["home_score"] - g["away_score"]
    return g.set_index("game_id")


def paired(week, games, first_view):
    records = []
    for path in sorted(RUNS.glob(f"{SEASON}-week-{week:02d}-*.json")):
        run = load(path)
        written = pd.Timestamp(run["recorded_at_utc"])
        for gid, v in view_rows(run).items():
            if gid in games.index and written < games.at[gid, "deadline"]:
                records.append({"game_id": gid, "view": run["view"], "written": written, **v})
    if not records:
        return []
    frame = pd.DataFrame(records).sort_values("written", kind="stable")
    start = frame[frame["view"].eq("tuesday")] if first_view == "tuesday" else frame
    first = start.drop_duplicates("game_id", keep="first").set_index("game_id")
    last = frame.drop_duplicates("game_id", keep="last").set_index("game_id")
    rows = []
    for gid, f in first.iterrows():
        v = last.loc[gid]
        both = pd.notna(f["spread"]) and pd.notna(v["spread"])
        rows.append(
            {
                "week": week,
                "game_id": gid,
                "first_view": f["view"],
                "two_views": bool(v["written"] > f["written"]),
                "p_first": f["p"],
                "p_last": v["p"],
                "revision": logit(v["p"]) - logit(f["p"]),
                "move": v["spread"] - f["spread"] if both else np.nan,
                "pool": f["pool"],
                "pool_last": v["pool"],
            }
        )
    return rows


def grade(df, games):
    df["margin"] = df["game_id"].map(games["margin"])
    df["cover"] = df["margin"] - df["pool"]
    df["y"] = np.where(df["cover"] > 0, 1.0, np.where(df["cover"] < 0, 0.0, np.nan))
    if {"spread_line", "home_cover"}.issubset(games.columns):
        line = df["game_id"].map(games["spread_line"])
        ref = df["game_id"].map(games["home_cover"]).astype(float)
        same = line.eq(df["pool"]) & df["y"].notna() & ref.notna()
        agree = int(ref[same].eq(df.loc[same, "y"]).sum())
        differ = df["y"].notna() & ref.notna() & ref.ne(df["y"])
        print(
            "pool line equals the feature table line on",
            int(line.eq(df["pool"]).sum()),
            "of",
            len(df),
            "games; pool-line grade agrees with the feature table on",
            agree,
            "of",
            int(same.sum()),
            "same-line graded games; grade differs on",
            int(differ.sum()),
            "games overall",
        )
        if agree < int(same.sum()) - agree:
            raise SystemExit("pool-line grade has the opposite sign to game_features home_cover")
    print(
        "pool line differs between first and last view on",
        int(df["pool"].ne(df["pool_last"]).sum()),
        "games; pushes at the pool line",
        int((df["cover"] == 0).sum()),
    )


def permuted(y, week, rng):
    out = np.tile(y, (PERMUTATIONS, 1))
    for w in np.unique(week):
        cols = np.flatnonzero(week == w)
        out[:, cols] = rng.permuted(out[:, cols], axis=1)
    return out


def gain_score(p, q):
    p = np.asarray(p, dtype=float)
    q = np.broadcast_to(np.asarray(q, dtype=float), p.shape)
    return lambda y: loss(q, y) - loss(p, y)


def hit_score(home):
    home = np.asarray(home, dtype=bool)
    return lambda y: ((y == 1) == home).astype(float) - COIN


def cell(score, y, week, rng):
    s = score(y)
    n = len(s)
    obs = float(s.mean())
    boot = s[rng.integers(0, n, (RESAMPLES, n))].mean(axis=1)
    null = score(permuted(y, week, rng)).mean(axis=1)
    level = np.isclose(boot, 0)
    tied = np.isclose(null, obs)
    return {
        "games": n,
        "blocks": len(np.unique(week)),
        "effect": obs,
        "interval": np.percentile(boot, [2.5, 97.5]).tolist(),
        "probability_positive": float(((boot > 0) & ~level).mean() + COIN * level.mean()),
        "null_mean": float(null.mean()),
        "permutation_p_upper": float((((null > obs) | tied).sum() + 1) / (PERMUTATIONS + 1)),
        "permutation_p_lower": float((((null < obs) | tied).sum() + 1) / (PERMUTATIONS + 1)),
    }


def sign_test(wins, n):
    ref = math.comb(n, wins)
    return sum(math.comb(n, k) for k in range(n + 1) if math.comb(n, k) <= ref) * COIN**n


def revision_fit(df, noise, rng):
    quoted = df[df["two_views"] & df["move"].notna()]
    rev = quoted["revision"].to_numpy()
    mv = quoted["move"].to_numpy()
    n = len(quoted)

    def stats(r, m):
        if np.std(m) == 0 or np.std(r) == 0:
            return np.nan, np.nan, np.nan
        c = np.corrcoef(r, m)[0, 1]
        s = np.polyfit(m, r, 1)[0]
        return c, s, c * c

    point = stats(rev, mv)
    boot = np.array(
        [stats(rev[i], mv[i]) for i in (rng.integers(0, n, n) for _ in range(RESAMPLES))]
    )
    out = {"games": n, "blocks": int(quoted["week"].nunique())}
    for k, name in enumerate(["corr", "slope", "R2"]):
        b = boot[:, k]
        b = b[~np.isnan(b)]
        out[name] = {
            "effect": float(point[k]),
            "interval": np.percentile(b, [2.5, 97.5]).tolist(),
            "probability_positive": float((b > 0).mean()),
            "valid": len(b),
        }
        print(
            name,
            round(point[k], 4),
            "CI",
            np.round(out[name]["interval"], 4),
            "P(>0)",
            round(out[name]["probability_positive"], 4),
            "valid",
            len(b),
        )
    coef = np.polyfit(mv, rev, 1)
    resid = rev - np.polyval(coef, mv)
    rs = [np.std(resid[i], ddof=1) for i in (rng.integers(0, n, n) for _ in range(RESAMPLES))]
    out["intercept"] = float(coef[1])
    out["residual_sd"] = float(resid.std(ddof=2))
    out["residual_sd_interval"] = np.percentile(rs, [2.5, 97.5]).tolist()
    out["revision_sd"] = float(rev.std(ddof=1))
    out["games_with_a_move"] = int((mv != 0).sum())
    print(
        "games",
        n,
        "with a line move",
        out["games_with_a_move"],
        "intercept",
        coef[1],
        "residual SD",
        out["residual_sd"],
        "CI",
        np.round(out["residual_sd_interval"], 4),
        "raw revision SD",
        out["revision_sd"],
    )
    if noise:
        print(
            "resid SD / read noise",
            out["residual_sd"] / noise,
            "/ revision noise",
            out["residual_sd"] / (noise * math.sqrt(2)),
        )
    df["move_part"] = np.where(df["two_views"], coef[0] * df["move"].to_numpy(), np.nan)
    df["drift"] = coef[1]
    df["tracking"] = df["move_part"] + df["drift"]
    df["resid"] = df["revision"] - df["tracking"]
    return out


def outcome_cells(df, rng, wanted):
    g = df[df["y"].notna()]
    revised = g[g["two_views"] & g["revision"].ne(0)]
    changed = g[(g["p_first"] >= COIN) != (g["p_last"] >= COIN)]
    net = g[g["resid"].notna()]
    turned = net[net["resid"].ne(0)]
    moved = g[g["two_views"] & g["move"].notna() & g["move"].ne(0)]
    specs = [
        ("first_vs_half_ll", g, g["p_first"], COIN),
        ("last_vs_half_ll", g, g["p_last"], COIN),
        ("last_vs_first_ll", g, g["p_last"], g["p_first"]),
        ("first_accuracy", g, g["p_first"] >= COIN, None),
        ("last_accuracy", g, g["p_last"] >= COIN, None),
        ("revision_alone_vs_half_ll", revised, sigmoid(revised["revision"]), COIN),
        ("revision_direction_accuracy", revised, revised["revision"] > 0, None),
        ("changed_first_vs_half_ll", changed, changed["p_first"], COIN),
        ("changed_last_vs_half_ll", changed, changed["p_last"], COIN),
        ("changed_last_vs_first_ll", changed, changed["p_last"], changed["p_first"]),
        ("changed_revision_direction_accuracy", changed, changed["p_last"] >= COIN, None),
        ("changed_revision_alone_vs_half_ll", changed, sigmoid(changed["revision"]), COIN),
        ("residual_direction_accuracy", turned, turned["resid"] > 0, None),
        ("residual_alone_vs_half_ll", net, sigmoid(net["resid"]), COIN),
        (
            "first_plus_residual_vs_first_ll",
            net,
            sigmoid(logits(net["p_first"]) + net["resid"].to_numpy()),
            net["p_first"],
        ),
        (
            "first_plus_tracking_vs_first_ll",
            net,
            sigmoid(logits(net["p_first"]) + net["tracking"].to_numpy()),
            net["p_first"],
        ),
        ("move_direction_accuracy", moved, moved["move"] > 0, None),
        (
            "first_plus_move_part_vs_first_ll",
            net,
            sigmoid(logits(net["p_first"]) + net["move_part"].to_numpy()),
            net["p_first"],
        ),
        (
            "first_plus_drift_vs_first_ll",
            net,
            sigmoid(logits(net["p_first"]) + net["drift"].to_numpy()),
            net["p_first"],
        ),
    ]
    results = {}
    for name, part, p, q in specs:
        if (wanted is not None and name not in wanted) or part.empty:
            continue
        y = part["y"].to_numpy()
        week = part["week"].to_numpy()
        hit = q is None
        home = np.asarray(p, dtype=bool) if hit else np.asarray(p, dtype=float) >= COIN
        result = cell(hit_score(home) if hit else gain_score(p, q), y, week, rng)
        wins = int(((y == 1) == home).sum())
        result["units"] = "accuracy_points" if hit else "log_loss_improvement"
        result["wins"] = wins
        result["losses"] = len(y) - wins
        results[name] = result
    return results, g, changed


def show(results):
    for name, r in results.items():
        scale = POINTS if r["units"] == "accuracy_points" else 1
        print(
            f"{name:38s} n {r['games']:3d} rec {r['wins']:2d}-{r['losses']:2d} "
            f"eff {r['effect'] * scale:+8.4f} [{r['interval'][0] * scale:+8.4f}, "
            f"{r['interval'][1] * scale:+8.4f}] P+ {r['probability_positive']:.4f} "
            f"null {r['null_mean'] * scale:+8.4f} p_up {r['permutation_p_upper']:.4f} "
            f"p_lo {r['permutation_p_lower']:.4f}"
        )


def splits(g, changed, noise):
    out = {"by_week": {}, "changed_by_move": {}}
    for w, part in g.groupby("week"):
        won = part["y"].eq(1)
        first = int(((part["p_first"] >= COIN) == won).sum())
        last = int(((part["p_last"] >= COIN) == won).sum())
        out["by_week"][int(w)] = {"games": len(part), "first_wins": first, "last_wins": last}
        print(
            "week",
            w,
            "games",
            len(part),
            "first",
            f"{first}-{len(part) - first}",
            "last",
            f"{last}-{len(part) - last}",
        )
    quoted = changed[changed["move"].notna()]
    align = np.sign(quoted["move"] * quoted["revision"])
    for label, value in (("with the move", 1), ("against the move", -1), ("no move", 0)):
        part = quoted[align.eq(value)]
        wins = int(((part["p_last"] >= COIN) == part["y"].eq(1)).sum())
        out["changed_by_move"][label] = {"games": len(part), "last_wins": wins}
        print("side changed,", label, "games", len(part), "last view", f"{wins}-{len(part) - wins}")
    if len(changed):
        wins = int(((changed["p_last"] >= COIN) == changed["y"].eq(1)).sum())
        out["changed_sign_test_two_sided"] = sign_test(wins, len(changed))
        size = changed["revision"].abs()
        print(
            "side changed: last view",
            f"{wins}-{len(changed) - wins}",
            "exact two-sided sign test",
            round(out["changed_sign_test_two_sided"], 4),
            "median |revision|",
            round(float(size.median()), 4),
        )
        if noise:
            unit = noise * math.sqrt(2)
            out["changed_median_revision_over_noise"] = float(size.median() / unit)
            out["changed_within_one_noise_sd"] = int((size < unit).sum())
            print(
                "median |revision| over revision noise",
                round(out["changed_median_revision_over_noise"], 3),
                "games within one revision-noise SD",
                out["changed_within_one_noise_sd"],
            )
        print(
            changed[
                ["game_id", "p_first", "p_last", "revision", "move", "pool", "margin", "y"]
            ].to_string(index=False)
        )
    return out


def registry_cell(construction, name, result, weeks, looks, source):
    description, plain = TEXT[name]
    outcome = "wins" in result
    scale = POINTS if result.get("units") == "accuracy_points" else 1
    tail = "no outcomes used"
    if outcome:
        tail = (
            f"graded at the pool's Tuesday line, record {result['wins']}-{result['losses']}; "
            f"within-week permutation p {result['permutation_p_upper']:.3f} upper, "
            f"{result['permutation_p_lower']:.3f} lower (null mean "
            f"{result['null_mean'] * scale:+.4f}); one of {looks} outcome looks in this run"
        )
    return {
        "name": f"lead104_{construction}_{name}",
        "family": FAMILY,
        "source": source,
        "description": (
            f"LEAD-104, {CONSTRUCTION[construction]}: {description}; Weeks "
            f"{', '.join(str(w) for w in weeks)}, {result['games']} games; game bootstrap; {tail}"
        ),
        "plain_summary": (
            f"Research only: this measures {plain}, on {result['games']} finished games. That "
            "is too few to tell luck from skill, so it changes no pick."
        ),
        "effect": round(result["effect"] * scale, 4),
        "interval_low": round(result["interval"][0] * scale, 4),
        "interval_high": round(result["interval"][1] * scale, 4),
        "probability_positive": round(result["probability_positive"], 4),
        "effect_units": result.get("units", "correlation"),
        "classification": "unresolved_below_power",
        "classification_evidence": (
            f"Forward log only, {result['games']} games in {result['blocks']} weeks; interval "
            "and probability_positive from a game bootstrap; the mechanism is not refuted, no "
            "split-half reliability of zero and no positive control bound exists for it, so no "
            "admissible closing ground applies and the cell stays unresolved."
        ),
        "league": "nfl",
        "season_start": SEASON,
        "season_end": SEASON,
        "sample_games": result["games"],
        "sample_blocks": result["blocks"],
        "category": "modeling",
    }


def main():
    parser = argparse.ArgumentParser(
        description="LEAD-104 read revision against the line move between two views"
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="+",
        default=list(range(1, 19)),
        help="2026 weeks to consider; unsettled weeks and weeks without a first view are skipped",
    )
    parser.add_argument(
        "--first-view",
        nargs="+",
        choices=sorted(CONSTRUCTION),
        default=["tuesday"],
        help="first view of each game: the Tuesday view, or the earliest record before deadline",
    )
    parser.add_argument("--cells", nargs="*", default=None, help="outcome cells (default: all)")
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    games = schedule(args.weeks)
    settled = [
        w
        for w in args.weeks
        if games["week"].eq(w).any() and games.loc[games["week"].eq(w), "margin"].notna().all()
    ]
    print("requested weeks", args.weeks, "settled", settled)
    noise = between_read_sd()
    print(
        "between-read SD (logit)",
        noise,
        "revision noise SD sqrt2*",
        None if noise is None else noise * math.sqrt(2),
    )
    rng = np.random.default_rng(SEED)
    wanted = set(args.cells) if args.cells else None
    source = f"artifacts/lead104_revision_path/summary{args.tag}.json"
    summary = {}
    pending = []
    looks = 0
    OUT.mkdir(parents=True, exist_ok=True)
    for construction in args.first_view:
        rows = []
        weeks = []
        for w in settled:
            r = paired(w, games, construction)
            if r:
                rows += r
                weeks.append(w)
            else:
                print(construction, "week", w, "has no game with a first view")
        if not rows:
            continue
        df = pd.DataFrame(rows)
        print()
        print("construction", construction, "weeks", weeks, "games", len(df))
        print("first views", df.groupby(["week", "first_view"]).size().to_dict())
        print("games with two distinct views", int(df["two_views"].sum()))
        grade(df, games)
        fit = revision_fit(df, noise, rng)
        print(
            "mean |logit| first",
            round(float(np.abs(logits(df["p_first"])).mean()), 4),
            "last",
            round(float(np.abs(logits(df["p_last"])).mean()), 4),
            "revisions toward home",
            int((df["revision"] > 0).sum()),
            "away",
            int((df["revision"] < 0).sum()),
            "line moves toward home",
            int((df["move"] > 0).sum()),
            "away",
            int((df["move"] < 0).sum()),
            "home covers",
            int(df["y"].eq(1).sum()),
            "of",
            int(df["y"].notna().sum()),
        )
        results, g, changed = outcome_cells(df, rng, wanted)
        show(results)
        extra = {} if wanted is not None else splits(g, changed, noise)
        looks += len(results)
        if extra:
            looks += 2 * len(extra["by_week"]) + len(extra["changed_by_move"])
        summary[construction] = {"weeks": weeks, "fit": fit, "cells": results, "splits": extra}
        corr = {**fit["corr"], "games": fit["games"], "blocks": fit["blocks"]}
        pending.append((construction, "revision_vs_move_correlation", corr, weeks))
        for name, result in results.items():
            pending.append((construction, name, result, weeks))
        df.to_csv(OUT / f"per_game_{construction}{args.tag}.csv", index=False)
    print()
    print("outcome looks in this run", looks, "family", FAMILY)
    summary["outcome_looks"] = looks
    cells = [
        registry_cell(construction, name, result, weeks, looks, source)
        for construction, name, result, weeks in pending
        if name in RECORDED[construction]
    ]
    (OUT / f"summary{args.tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / f"weak_signals_batch{args.tag}.json").write_text(
        json.dumps({"cells": cells}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
