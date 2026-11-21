import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "lead99_llm_read"
OUT = ROOT / "artifacts" / "lead100_repeat_reads"
SEED = 20261103
N_BOOT = 10000
MIN_READS = 3


def logit(p):
    return math.log(p / (1 - p))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_of(d):
    return {r["game_id"]: logit(r["home_cover_probability"]) for r in d["parsed_rows"]}


def read_time(d):
    t = d.get("recorded_at_utc") or d["run_at_utc"]
    for fmt in ("%Y%m%dT%H%M%S%fZ", "%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(t, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def split_half(a, b, rng):
    r = float(np.corrcoef(a, b)[0, 1])
    n = len(a)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if a[i].std() == 0 or b[i].std() == 0:
            continue
        bs.append(np.corrcoef(a[i], b[i])[0, 1])
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {
        "r": r,
        "ci95": [float(lo), float(hi)],
        "probability_positive": float((bs > 0).mean()),
        "spearman_brown": 2 * r / (1 + r),
        "n_games": n,
    }


def block(m, ids):
    sd = m.std(axis=0, ddof=1)
    p = 1 / (1 + np.exp(-m))
    psd = p.std(axis=0, ddof=1)
    rng = np.random.default_rng(SEED)
    out = {
        "reads": int(m.shape[0]),
        "games": len(ids),
        "mean_between_read_sd_logit": float(sd.mean()),
        "mean_between_read_sd_prob": float(psd.mean()),
        "max_between_read_sd_prob": float(psd.max()),
        "between_game_sd_of_mean_logit": float(m.mean(axis=0).std(ddof=1)),
        "split_half_stored_v_repeats": split_half(m[0], m[1:].mean(axis=0), rng),
    }
    if m.shape[0] >= 4:
        rng2 = np.random.default_rng(SEED)
        out["split_half_repeat1_v_other_repeats"] = split_half(m[1], m[2:].mean(axis=0), rng2)
    out["per_game_sd_logit"] = dict(zip(ids, np.round(sd, 4).tolist(), strict=True))
    return out


def main():
    groups = {}
    for p in sorted((ART / "repeats").glob("*-repeat-*.json")):
        d = load(p)
        if d["valid"]:
            groups.setdefault(d["source_run"], []).append(d)
    weeks = {}
    pooled_cols = []
    pooled_ids = []
    for src, reps in sorted(groups.items()):
        base = load(ART / "runs" / src)
        if 1 + len(reps) < MIN_READS:
            continue
        last_read = max(read_time(r) for r in reps)
        live = set()
        for g in base["packet"]["games"]:
            cut = min(
                datetime.fromisoformat(g["kickoff_et"]),
                datetime.fromisoformat(g["deadline_et"]),
            )
            if cut > last_read:
                live.add(g["game_id"])
        reads = [rows_of(base)] + [rows_of(r) for r in reps]
        ids = sorted(live.intersection(*[set(r) for r in reads]))
        m = np.array([[r[g] for g in ids] for r in reads])
        k = min(len(reps), 3)
        mk = np.array([[r[g] for g in ids] for r in reads[: 1 + k]])
        weeks[src] = {
            "excluded_kicked_off": sorted({g["game_id"] for g in base["packet"]["games"]} - live),
            "last_repeat_utc": last_read.isoformat(),
            **block(m, ids),
        }
        pooled_cols.append(mk)
        pooled_ids += [f"{src}:{g}" for g in ids]
    pooled = block(np.concatenate(pooled_cols, axis=1), pooled_ids)
    pooled.pop("per_game_sd_logit")
    views = sorted((ART / "runs").glob("*.json"))
    by_game = {}
    for v in views:
        for g, x in rows_of(load(v)).items():
            by_game.setdefault(g, []).append(x)
    multi = {g: float(np.std(x, ddof=1)) for g, x in by_game.items() if len(x) > 1}
    summary = {
        "min_reads_per_game": MIN_READS,
        "seed": SEED,
        "n_boot": N_BOOT,
        "by_source": weeks,
        "pooled_stored_plus_3_repeats": pooled,
        "across_view_sd_logit": {"games": len(multi), "mean": float(np.mean(list(multi.values())))},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for src, w in weeks.items():
        print(src, "reads", w["reads"], "games", w["games"], "excluded", w["excluded_kicked_off"])
        print(
            "  sd_read",
            round(w["mean_between_read_sd_logit"], 4),
            "sd_game",
            round(w["between_game_sd_of_mean_logit"], 4),
            "maxprob",
            round(w["max_between_read_sd_prob"], 4),
        )
        print("  stored_v_repeats", w["split_half_stored_v_repeats"])
        if "split_half_repeat1_v_other_repeats" in w:
            print("  repeat1_v_others", w["split_half_repeat1_v_other_repeats"])
    print("POOLED", json.dumps(pooled))
    print("across-view", summary["across_view_sd_logit"])
    print(OUT / "summary.json")


main()
