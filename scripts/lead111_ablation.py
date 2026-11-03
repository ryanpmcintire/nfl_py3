import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead99_llm_read as L99
import lead106_injury_language as M

ART = M.ART / "ablation"
CONDITIONS = ("practice", "bodypart", "headlines", "repeat")
OUT_DOUBTFUL_SUBSAMPLE = 40
PREFIXES = (
    "position ",
    "official game designation:",
    "injury:",
    "also:",
    "practice:",
    "headlines:",
)
DROPPED = {
    "practice": ("practice:",),
    "bodypart": ("injury:", "also:"),
    "headlines": ("headlines:",),
    "repeat": (),
}


def split_parts(text):
    parts = []
    for piece in text.split("; "):
        if piece.startswith(PREFIXES) or not parts:
            parts.append(piece)
        else:
            parts[-1] += "; " + piece
    return parts


def ablate(text, cond):
    drop = DROPPED[cond]
    return "; ".join(p for p in split_parts(text) if not p.startswith(drop))


def verified_sample():
    pop = M.id_outcomes(pd.read_parquet(M.ART / "population.parquet"))
    s0 = pd.read_parquet(M.ART / "sample.parquet")
    ver = pop[pop["verified"]].copy()
    ver["practice_status"] = ver["practice_status"].fillna("none")
    s = s0[["rid", "gsis_id", "week", "text", "has_headline"]].merge(
        ver.drop(columns=["text", "has_headline", "rid"], errors="ignore"),
        on=["gsis_id", "week"],
        how="inner",
    )
    return s, ver


def subsample(s):
    rng = np.random.default_rng(M.seed_value())
    carry = s[s["designation"].isin(["Questionable", "none"])]
    rest = s[~s["designation"].isin(["Questionable", "none"])]
    pick = rng.choice(rest.index, size=OUT_DOUBTFUL_SUBSAMPLE, replace=False)
    return pd.concat([carry, rest.loc[np.sort(pick)]]).sort_values("rid").reset_index(drop=True)


def cmd_freeze(_):
    ART.mkdir(parents=True, exist_ok=True)
    s, _ = verified_sample()
    sub = subsample(s)
    for c in CONDITIONS:
        sub[f"text_{c}"] = [ablate(t, c) for t in sub["text"]]
    assert (sub["text"] == sub["text_repeat"]).all()
    sub.to_parquet(ART / "subsample.parquet")
    fz = {
        "prompt_sha256": M.sha(M.PROMPT.read_bytes()),
        "rows": len(sub),
        "by_designation": sub["designation"].value_counts().to_dict(),
        "headline_rows": int(sub["has_headline"].sum()),
        "changed_rows": {c: int((sub[f"text_{c}"] != sub["text"]).sum()) for c in CONDITIONS},
    }
    (ART / "frozen.json").write_text(json.dumps(fz, indent=1))
    print(json.dumps(fz, indent=1))


def cmd_run(a):
    L99.TOOLS = ""
    fz = json.loads((ART / "frozen.json").read_text())
    assert fz["prompt_sha256"] == M.sha(M.PROMPT.read_bytes())
    sub = pd.read_parquet(ART / "subsample.parquet")
    c = a.cond
    if c == "headlines":
        sub = sub[sub["has_headline"]]
    raw = ART / "raw" / c
    raw.mkdir(parents=True, exist_ok=True)
    cwd = ART / "cwd" / c
    cwd.mkdir(parents=True, exist_ok=True)
    prompt = M.PROMPT.read_text()
    for b in range(0, len(sub), M.BATCH_ROWS):
        out = raw / f"batch_{b // M.BATCH_ROWS:03d}.json"
        if out.exists():
            continue
        chunk = sub.iloc[b : b + M.BATCH_ROWS]
        packet = json.dumps(
            [{"id": r.rid, "text": getattr(r, f"text_{c}")} for r in chunk.itertuples()], indent=1
        )
        env, text = L99.call_claude(prompt, packet, cwd)
        out.write_text(json.dumps({"envelope": env, "text": text}))
        print("done", c, out.name, flush=True)


def read_cond(c):
    rows = {}
    for f in sorted((ART / "raw" / c).glob("batch_*.json")):
        t = json.loads(f.read_text())["text"]
        for o in json.loads(t[t.index("[") : t.rindex("]") + 1]):
            rows[o["id"]] = float(o["p_plays"])
    return rows


def cmd_score(_):
    sub = pd.read_parquet(ART / "subsample.parquet")
    _, ver = verified_sample()
    full = M.parse_reads()
    n = len(sub)
    clip = 1.0 / (n + 2)

    def cl_(v):
        return min(max(v, clip), 1 - clip)

    p = {"full": np.array([cl_(full[r][0]) for r in sub["rid"]])}
    for c in CONDITIONS:
        r = read_cond(c)
        p[c] = np.array([cl_(r[x]) if x in r else p["full"][i] for i, x in enumerate(sub["rid"])])
    popn = ver[ver["week"] > M.FIRST_WEEK].groupby("designation").size()
    samn = sub.groupby("designation").size()
    w = sub["designation"].map(lambda d: popn[d] / samn[d]).to_numpy()
    y = sub["played"].to_numpy()
    cl = (sub["team"] + sub["week"].astype(str)).to_numpy()

    def ll(q):
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))

    for nm, keys in (
        ("designation", ["designation"]),
        ("designation_x_practice", ["designation", "practice_status"]),
    ):
        tab = M.ladder_tables(ver, keys)
        p[nm] = np.array([cl_(M.ladder_pred(r, tab, keys)[0]) for _, r in sub.iterrows()])

    def logit(q):
        return np.log(q / (1 - q))

    out = []
    pairs = []
    for c in CONDITIONS:
        pairs.append((f"{c}_vs_full", p["full"], p[c]))
    for c in ("practice", "bodypart", "headlines"):
        pairs.append((f"{c}_vs_repeat", p["repeat"], p[c]))
    for c in ("full", "repeat", "practice", "bodypart", "headlines"):
        for nm in ("designation", "designation_x_practice"):
            pairs.append((f"{c}_gain_vs_{nm}", p[nm], p[c]))
    for name, base, alt in pairs:
        rng = np.random.default_rng(M.seed_value())
        r = M.summarize(name, ll(base) - ll(alt), w, cl, rng)
        r["rows"] = n
        out.append(r)
    noise = (logit(p["repeat"]) - logit(p["full"])) / np.sqrt(2)
    shifts = {
        c: float(np.sqrt(np.mean((logit(p[c]) - logit(p["full"])) ** 2)) / np.sqrt(2))
        for c in ("repeat", "practice", "bodypart", "headlines")
    }
    res = {
        "rows": n,
        "weights_by_designation": {d: float(popn[d] / samn[d]) for d in samn.index},
        "between_read_sd_logit_repeat_vs_full": float(noise.std()),
        "rms_logit_shift_over_sqrt2": shifts,
        "cells": out,
    }
    (ART / "score.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("freeze").set_defaults(fn=cmd_freeze)
    r = sp.add_parser("run")
    r.add_argument("cond", choices=CONDITIONS)
    r.set_defaults(fn=cmd_run)
    sp.add_parser("score").set_defaults(fn=cmd_score)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
