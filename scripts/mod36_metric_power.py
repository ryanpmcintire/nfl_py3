import json
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from nfl_ats.mass_preserving_lattice import (
    BAND_HALF_WIDTH,
    BAND_STEP,
    MAX_BAND,
    MIN_BAND_GAMES,
    prior_pool,
    prior_pool_for_week,
)

HARNESS = Path(__file__).with_name("mod36_addone_noise_floor.py")


def load_harness():
    source = HARNESS.read_text(encoding="utf-8").rstrip()
    body, _, call = source.rpartition("\n")
    if call != "main()":
        raise ValueError("The harness entry call is not where it was")
    module = types.ModuleType("mod36_addone_noise_floor")
    module.__file__ = str(HARNESS)
    exec(compile(body, str(HARNESS), "exec"), module.__dict__)
    return module


H = load_harness()
OUT = H.ROOT / "artifacts/mod36/metric_power"
OUT = OUT if H.DRAWS == H.DECLARED_DRAWS else OUT / "smoke"
PRIOR = H.ROOT / "artifacts/mod36/summary.json"
FAMILY = "mod36_metric_power"
UNIT_STREAM = 3
METRICS = [
    "accuracy_points",
    "log_loss_improvement",
    "brier_improvement",
    "ats_points",
    "rps_improvement",
]
SHORT = ["acc", "ll", "brier", "ats", "rps"]
COVER = SHORT[:4]
RECORDED = ("ats", "rps")
DIRECTION = np.array([1.0, -1.0, -1.0, 1.0, -1.0])
LOGIT_SLOPE_AT_HALF = 4.0
LATTICE_STEP = 1.0
LADDER_PARTS = 8
LADDER_FIRST = 2
LADDER_LAST = 12
PRIMARY_LEVEL = H.NOTABLE[0]
WORDS = {
    "ats": "the average winning margin against the opening line of the side picked",
    "rps": "how close the predicted spread of final margins sits to the real final margin",
}
LOG = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    LOG.append(line)
    (OUT / "run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def frame():
    pick, d = H.games()
    extra = pd.read_parquet(
        pick, columns=["game_id", "tue_open_home_spread", "result", "margin_vs_open"]
    )
    d = d.merge(extra, on="game_id", how="left", validate="one_to_one")
    ats = d.result - d.tue_open_home_spread
    if (ats - d.margin_vs_open).abs().max() > H.EPS:
        raise ValueError("The opener margin is not the result minus the opener line")
    if not ((ats > 0) == d.home_covered.eq(1)).all():
        raise ValueError("The home cover flag disagrees with the opener margin")
    return pick, d


def full_lattice(pick, d, pad):
    features = pd.read_parquet(
        H.FEATURES,
        columns=["game_id", "season", "week", "gameday", "spread_line", "result", "game_type"],
    )
    features["gameday"] = pd.to_datetime(features["gameday"])
    opener = pd.read_parquet(pick, columns=["game_id", "tue_open_home_spread"])
    pool = prior_pool(features, opener.set_index("game_id").tue_open_home_spread)
    cutoffs = features.groupby(["season", "week"]).gameday.min()
    low = int(min(pool.result.min(), d.result.min())) - pad
    high = int(max(pool.result.max(), d.result.max())) + pad
    grid = np.arange(low, high + 1, dtype=float)
    counts = np.zeros((len(d), len(grid)))
    bands = np.empty(len(d))
    for (season, week), group in d.groupby(["season", "week"], sort=True):
        prior = prior_pool_for_week(
            pool,
            season=int(season),
            week=int(week),
            cutoff=cutoffs.loc[(season, week)],
            exclude_game_ids=group.game_id,
        )
        lines = prior["line"].to_numpy(dtype=float)
        margins = prior["result"].to_numpy(dtype=float)
        for i, line in zip(group.index, group.tue_open_home_spread, strict=True):
            width = BAND_HALF_WIDTH
            while True:
                selected = np.abs(lines - line) <= width
                if selected.sum() >= MIN_BAND_GAMES or width >= MAX_BAND:
                    break
                width = min(width + BAND_STEP, MAX_BAND)
            np.add.at(counts[i], (margins[selected] - low).astype(int), 1.0)
            bands[i] = width
    return {
        "grid": grid,
        "counts": counts,
        "bands": bands,
        "line": d.tue_open_home_spread.to_numpy(dtype=float),
        "result": d.result.to_numpy(dtype=float),
        "pool_rows": len(pool),
    }


def template_lattice(full, keep):
    line = full["line"][keep]
    counts = full["counts"][keep]
    offsets = full["grid"][None, :] - line[:, None]
    above = (offsets > 0).astype(float)
    below = (offsets < 0).astype(float)
    if ((counts * above).sum(axis=1) == 0).any() or ((counts * below).sum(axis=1) == 0).any():
        raise ValueError("A line band has no prior margin on one side of the line")
    return {
        "grid": full["grid"],
        "counts": counts,
        "offsets": offsets,
        "above": above,
        "below": below,
        "line": line,
        "result": full["result"][keep],
    }


def tilt(lat, p, start):
    goal = logit(np.clip(p, H.EPS, 1 - H.EPS))
    theta = start.copy()
    miss = np.inf
    for _ in range(H.NEWTON_STEPS):
        mass = lat["counts"] * np.exp(theta[:, None] * lat["offsets"])
        up = (mass * lat["above"]).sum(axis=1)
        down = (mass * lat["below"]).sum(axis=1)
        mean_up = (mass * lat["above"] * lat["offsets"]).sum(axis=1) / up
        mean_down = (mass * lat["below"] * lat["offsets"]).sum(axis=1) / down
        gap = np.log(up / down) - goal
        miss = float(np.abs(gap).max())
        step = gap / (mean_up - mean_down)
        theta = theta - step
        if np.abs(step).max() < H.NEWTON_TOL:
            break
    return theta, miss


def cdf_of(lat, theta):
    mass = lat["counts"] * np.exp(theta[:, None] * lat["offsets"])
    mass /= mass.sum(axis=1, keepdims=True)
    return np.cumsum(mass, axis=1)


def rps(lat, cdf, actual):
    reached = (actual[:, None] <= lat["grid"][None, :]).astype(float)
    return np.square(cdf - reached).sum(axis=1)


def cover_scores(p, y, played, ats):
    pc = np.clip(p, H.EPS, 1 - H.EPS)
    home = p >= 0.5
    return np.column_stack(
        [
            played * ((home == (y == 1)) * 100.0),
            played * -(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
            played * (p - y) ** 2,
            np.where(home, ats, -ats),
        ]
    )


def loso(xd, y, z, folds, played):
    p = np.empty(len(y))
    for te in folds:
        tr = ~te & played
        b = H.fit_offset(xd[tr], y[tr], z[tr])
        p[te] = expit(z[te] + xd[te] @ b)
    return p


def run_cell(st, tpl, lat, fixed, delta, rng):
    rows, kept = [], []
    served = st["refs"]["served"]["p"]
    for i in range(H.DRAWS):
        x = H.permuted(st, rng)
        actual = lat["result"]
        if delta > 0:
            want = delta * x
            whole = np.floor(want)
            actual = actual + whole + (rng.random(st["n"]) < want - whole)
        ats = actual - lat["line"]
        played = ats != 0
        y = (ats > 0).astype(float)
        xd = H.design(x, tpl["intercept"])
        p = loso(xd, y, st["z"], st["folds"], played)
        coef = float(H.fit_offset(xd[played], y[played], st["z"][played])[-1])
        if fixed is None:
            p_ref = loso(np.ones((st["n"], 1)), y, st["z"], st["folds"], played)
            theta_ref, _ = tilt(lat, p_ref, np.zeros(st["n"]))
            cdf_ref = cdf_of(lat, theta_ref)
        else:
            p_ref, theta_ref, cdf_ref = fixed
        theta, miss = tilt(lat, p, theta_ref)
        arm = np.column_stack(
            [cover_scores(p, y, played, ats), rps(lat, cdf_of(lat, theta), actual)]
        )
        ref = np.column_stack([cover_scores(p_ref, y, played, ats), rps(lat, cdf_ref, actual)])
        v = (arm - ref) * DIRECTION
        bs = st["w"] @ v / st["n"]
        pp = (bs > 0).mean(axis=0) + 0.5 * (bs == 0).mean(axis=0)
        changed = ((p >= 0.5) != (p_ref >= 0.5)) & played
        right = (p >= 0.5) == (y == 1)
        row = {
            "template": tpl["name"],
            "delta": delta,
            "draw": i,
            "coef": coef,
            "pushes": int((~played).sum()),
            "changed": int(changed.sum()),
            "arm_right": int((changed & right).sum()),
            "ref_right": int((changed & ~right).sum()),
            "tilt_miss": miss,
        }
        for k, name in enumerate(SHORT):
            row[name] = float(v[:, k].mean())
            row[f"pp_{name}"] = float(pp[k])
        rows.append(row)
        kept.append((x, y, played, ats))
    true_coef = float(np.mean([r["coef"] for r in rows]))
    for row, (x, y, played, ats) in zip(rows, kept, strict=True):
        best = cover_scores(expit(st["z"] + true_coef * x), y, played, ats)
        gain = ((best - cover_scores(served, y, played, ats)) * DIRECTION[:4]).mean(axis=0)
        for k, name in enumerate(COVER):
            row[f"oracle_{name}"] = float(gain[k])
    return rows


def cell_summary(g, logit_size):
    out = {
        "delta": float(g.delta.iloc[0]),
        "designed_logit_per_sd": logit_size,
        "coef_mean": float(g.coef.mean()),
        "coef_sd": float(g.coef.std(ddof=1)),
        "pushes_mean": float(g.pushes.mean()),
        "changed_median": float(g.changed.median()),
        "arm_right_median": float(g.arm_right.median()),
        "ref_right_median": float(g.ref_right.median()),
        "losing_record_share": float((g.arm_right < g.ref_right).mean()),
        "tilt_miss_max": float(g.tilt_miss.max()),
    }
    for name in SHORT:
        out[name] = H.band(g[name])
        out[name]["share_positive"] = float((g[name] > 0).mean() + 0.5 * (g[name] == 0).mean())
        pp = g[f"pp_{name}"]
        out[f"pp_{name}"] = {}
        for level in H.NOTABLE:
            out[f"pp_{name}"][f"at_or_above_{level:.2f}"] = float((pp >= level).mean())
            out[f"pp_{name}"][f"at_or_below_{1 - level:.2f}"] = float((pp <= 1 - level).mean())
    for name in COVER:
        out[f"oracle_{name}"] = float(g[f"oracle_{name}"].mean())
    return out


def smallest(cells, name, level):
    key = f"at_or_above_{level:.2f}"
    planted = [c for c in cells if c["delta"] > 0]
    hit = [c for c in planted if c[f"pp_{name}"][key] >= H.POWER_TARGET]
    if not hit:
        return {
            "reached": False,
            "largest_delta": planted[-1]["delta"],
            "share_at_largest": planted[-1][f"pp_{name}"][key],
        }
    c = hit[0]
    return {
        "reached": True,
        "delta_points_per_sd": c["delta"],
        "share": c[f"pp_{name}"][key],
        "fitted_logit_per_sd": c["coef_mean"],
        "true_accuracy_points": c["oracle_acc"],
        "true_log_loss_improvement": c["oracle_ll"],
        "true_ats_points": c["oracle_ats"],
        "holds_at_every_larger_size": all(
            x[f"pp_{name}"][key] >= H.POWER_TARGET for x in planted if x["delta"] >= c["delta"]
        ),
    }


def null_cells(tpl, st, null):
    form = "an intercept and one coefficient" if tpl["intercept"] else "one coefficient"
    versus = "the served probability" if tpl["matched"] == "served" else "an intercept-only fit"
    cells = []
    for name, units in zip(SHORT, METRICS, strict=True):
        if name not in RECORDED:
            continue
        s = null[name]
        cells.append(
            {
                "name": f"mod36_metric_null_{tpl['name']}_{units}",
                "description": (
                    f"Matched-marginal null: the real {tpl['column']} column shuffled among "
                    f"games within each season, {H.DRAWS} draws (seed {H.SEED}), entered as "
                    f"{form} on the served out-of-season logit, leave-one-season-out "
                    f"{H.FIRST_SEASON}-{H.LAST_SEASON}, versus {versus}; graded on {units} "
                    "(picked-side margin against the opener, or whole-margin ranked "
                    "probability score of the discrete line-conditional lattice tilted to "
                    "the cover probability); effect is the mean of draws, interval the 5th "
                    "to 95th percentile, probability_positive the share of draws above zero."
                ),
                "source": H.LANE,
                "effect": round(s["mean"], 6),
                "effect_units": units,
                "classification": "unresolved_below_power",
                "league": "nfl",
                "season_start": H.FIRST_SEASON,
                "season_end": H.LAST_SEASON,
                "family": FAMILY,
                "category": "control",
                "classification_evidence": (
                    "Negative-control null distribution of a term with no game link; it is "
                    "an instrument reading, closes nothing and is no closing ground."
                ),
                "notes": (
                    f"Family {FAMILY}: {len(H.TEMPLATES)} templates x {len(RECORDED)} margin "
                    "metrics; accuracy, log loss and Brier nulls are the identical draws "
                    "already recorded under mod36_addone_noise_floor; planted-effect cells "
                    "and detection shares stay in the lane."
                ),
                "interval_low": round(s["q05"], 6),
                "interval_high": round(s["q95"], 6),
                "probability_positive": round(s["share_positive"], 4),
                "sample_games": int(st["n"]),
                "sample_blocks": int(H.LAST_SEASON - H.FIRST_SEASON + 1),
                "plain_summary": (
                    f"A check that scrambles {tpl['what']}, to show how much a meaningless "
                    f"extra ingredient moves {WORDS[name]}."
                ),
            }
        )
    return cells


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pick, d = frame()
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    reported = prior["reported_sizes_logit_per_sd"]
    say("base", pick.relative_to(H.ROOT), "games", len(d), "draws", H.DRAWS)
    states, keeps, designs = [], [], []
    for t, tpl in enumerate(H.TEMPLATES):
        st = H.setup(d, tpl, t)
        keep = d[tpl["column"]].notna().to_numpy()
        ats = (d.result - d.tue_open_home_spread).to_numpy(dtype=float)[keep]
        density = float((np.abs(ats) <= LATTICE_STEP).mean() / (2 * LATTICE_STEP))
        points_per_logit = 1 / (LOGIT_SLOPE_AT_HALF * density)
        found = prior["templates"][tpl["name"]]["detectable"][tpl["matched"]]
        anchor = found[f"detect_acc_{PRIMARY_LEVEL:.2f}"]
        anchor = anchor["beta_per_sd"] if anchor["reached"] else anchor["largest_beta_tried"]
        ladder = [anchor * j / LADDER_PARTS for j in range(LADDER_FIRST, LADDER_LAST + 1)]
        sizes = sorted({*reported.values(), *ladder})
        states.append(st)
        keeps.append(keep)
        designs.append(
            {
                "cover_density_per_point": density,
                "points_per_logit": points_per_logit,
                "unit2_accuracy_detectable_logit": anchor,
                "logit_sizes": sizes,
                "deltas": [s * points_per_logit for s in sizes],
            }
        )
    pad = 1 + int(
        np.ceil(
            max(
                max(g["deltas"]) * np.abs(st["xs"]).max()
                for g, st in zip(designs, states, strict=True)
            )
        )
    )
    full = full_lattice(pick, d, pad)
    say(
        "lattice pool rows",
        full["pool_rows"],
        "grid",
        int(full["grid"][0]),
        int(full["grid"][-1]),
        "band median/max",
        float(np.median(full["bands"])),
        float(full["bands"].max()),
        f"{time.time() - started:.0f}s",
    )
    summary = {
        "family": FAMILY,
        "base_artifact": str(pick.relative_to(H.ROOT)),
        "base_column": H.SERVED_COLUMN,
        "draws": H.DRAWS,
        "boot": H.BOOT,
        "seed": H.SEED,
        "unit_stream": UNIT_STREAM,
        "notable_levels": list(H.NOTABLE),
        "power_target": H.POWER_TARGET,
        "metrics": METRICS,
        "templates": {},
    }
    frames, cells, table = [], [], []
    for t, (tpl, st, keep, plan) in enumerate(
        zip(H.TEMPLATES, states, keeps, designs, strict=True)
    ):
        lat = template_lattice(full, keep)
        fixed = None
        if tpl["matched"] == "served":
            served = st["refs"]["served"]["p"]
            theta, _ = tilt(lat, served, np.zeros(st["n"]))
            fixed = (served, theta, cdf_of(lat, theta))
        rows = run_cell(st, tpl, lat, fixed, 0.0, np.random.default_rng([H.SEED, t, 0]))
        for c, delta in enumerate(plan["deltas"]):
            rng = np.random.default_rng([H.SEED, UNIT_STREAM, t, c])
            rows.extend(run_cell(st, tpl, lat, fixed, delta, rng))
        g = pd.DataFrame(rows)
        frames.append(g)
        sizes = dict(zip([0.0, *plan["deltas"]], [0.0, *plan["logit_sizes"]], strict=True))
        summaries = [cell_summary(x, sizes[k]) for k, x in g.groupby("delta", sort=True)]
        null = summaries[0]
        detect = {
            f"{name}_{level:.2f}": smallest(summaries, name, level)
            for name in SHORT
            for level in H.NOTABLE
        }
        summary["templates"][tpl["name"]] = {
            "n": st["n"],
            "matched_reference": tpl["matched"],
            "design": plan,
            "null": null,
            "planted": summaries[1:],
            "detectable": detect,
        }
        cells.extend(null_cells(tpl, st, null))
        say(
            f"TEMPLATE {tpl['name']} n={st['n']} vs {tpl['matched']} density/pt "
            f"{plan['cover_density_per_point']:.4f} points/logit {plan['points_per_logit']:.2f} "
            f"{time.time() - started:.0f}s"
        )
        for name in SHORT:
            b, pp = null[name], null[f"pp_{name}"]
            say(
                f"  NULL {name} mean {b['mean']:+.5f} q05/50/95 {b['q05']:+.5f}/{b['q50']:+.5f}/"
                f"{b['q95']:+.5f} share+ {b['share_positive']:.3f} | P+>=.90 "
                f"{pp['at_or_above_0.90']:.3f} >=.95 {pp['at_or_above_0.95']:.3f} <=.10 "
                f"{pp['at_or_below_0.10']:.3f} <=.05 {pp['at_or_below_0.05']:.3f}"
            )
        say(
            f"  NULL changed median {null['changed_median']:.0f} record "
            f"{null['arm_right_median']:.0f}-{null['ref_right_median']:.0f} losing "
            f"{null['losing_record_share']:.2f} tilt miss {null['tilt_miss_max']:.1e}"
        )
        for c in summaries[1:]:
            shares = " ".join(
                f"{name} {c[f'pp_{name}']['at_or_above_0.90']:.2f}/"
                f"{c[f'pp_{name}']['at_or_above_0.95']:.2f}"
                for name in SHORT
            )
            say(
                f"  PLANT {c['delta']:.3f} pts/sd (designed {c['designed_logit_per_sd']:.3f} "
                f"fitted {c['coef_mean']:.3f} logit) true acc {c['oracle_acc']:+.3f} ll "
                f"{c['oracle_ll']:+.5f} ats {c['oracle_ats']:+.3f} | det {shares} | pushes "
                f"{c['pushes_mean']:.1f} changed {c['changed_median']:.0f} miss "
                f"{c['tilt_miss_max']:.1e}"
            )
        for key, v in detect.items():
            if v["reached"]:
                say(
                    f"  DETECT {key}: {v['delta_points_per_sd']:.3f} pts/sd share "
                    f"{v['share']:.2f} fitted logit {v['fitted_logit_per_sd']:.3f} true acc "
                    f"{v['true_accuracy_points']:+.3f} ll {v['true_log_loss_improvement']:+.5f} "
                    f"ats {v['true_ats_points']:+.3f} monotone {v['holds_at_every_larger_size']}"
                )
            else:
                say(
                    f"  DETECT {key}: not reached; largest {v['largest_delta']:.3f} pts/sd "
                    f"share {v['share_at_largest']:.2f}"
                )
            table.append({"template": tpl["name"], "cell": key, **v})
    pd.concat(frames).to_parquet(OUT / "draws.parquet", index=False)
    pd.DataFrame(table).to_csv(OUT / "detectable.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(
        json.dumps({"cells": cells}, indent=1), encoding="utf-8"
    )
    say("cells", len(cells), f"elapsed {time.time() - started:.0f}s")


main()
