import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead99_llm_read as L

SOURCE = L.ART / "runs" / "2026-week-09-tuesday-20261103T183210Z.json"
OUT = L.ART / "ablations"
BLOCKS = ("market", "injuries", "injury_headlines", "beat_posts", "weather", "rest")
SEED = 20261103
N_BOOT = 10000
READ_NOISE_SD = 0.052


def logit(p):
    return math.log(p / (1 - p))


def rows_of(d):
    return {r["game_id"]: logit(r["home_cover_probability"]) for r in d["parsed_rows"]}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(blocks):
    d = load(SOURCE)
    fz = json.loads(L.FROZEN.read_text())
    if fz["prompt_sha256"] != L.prompt_sha() or fz["model"] != L.MODEL:
        raise SystemExit("prompt or model differs from frozen enrollment")
    if d["prompt_sha256"] != L.prompt_sha():
        raise SystemExit("stored run used a different prompt")
    full_text = json.dumps(d["packet"], ensure_ascii=False, sort_keys=True)
    if L.sha256_bytes(full_text.encode("utf-8")) != d["packet_sha256"]:
        raise SystemExit("stored packet hash mismatch")
    ids = [g["game_id"] for g in d["packet"]["games"]]
    prompt_text = L.PROMPT.read_text(encoding="utf-8")
    cwd = L.ART / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        packet = json.loads(full_text)
        for g in packet["games"]:
            del g[block]
        text = json.dumps(packet, ensure_ascii=False, sort_keys=True)
        env, raw = L.call_claude(prompt_text, text, str(cwd))
        rows = None
        err = None
        try:
            rows = L.validate(L.extract_json(raw), ids)
        except Exception as e:
            err = repr(e)
        t = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = OUT / f"{SOURCE.stem}-drop-{block}-{t}.json"
        path.write_text(
            json.dumps(
                {
                    "source_run": SOURCE.name,
                    "dropped_block": block,
                    "full_packet_sha256": d["packet_sha256"],
                    "ablated_packet_sha256": L.sha256_bytes(text.encode("utf-8")),
                    "prompt_sha256": L.prompt_sha(),
                    "model": L.MODEL,
                    "raw_response": raw,
                    "envelope": env,
                    "valid": rows is not None,
                    "error": err,
                    "parsed_rows": rows,
                    "recorded_at_utc": t,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(path, rows is not None, flush=True)


def analyze():
    reads = [rows_of(load(SOURCE))]
    for p in sorted((L.ART / "repeats").glob(SOURCE.stem + "-repeat-*.json")):
        d = load(p)
        if d["valid"]:
            reads.append(rows_of(d))
    ids = sorted(reads[0])
    full = np.array([[r[g] for g in ids] for r in reads]).mean(axis=0)
    print("full reads", len(reads), "games", len(ids))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ids), (N_BOOT, len(ids)))
    for block in BLOCKS:
        ps = sorted(OUT.glob(f"*-drop-{block}-*.json"))
        ps = [p for p in ps if load(p)["valid"]]
        if not ps:
            print(block, "no valid run")
            continue
        r = rows_of(load(ps[-1]))
        diff = np.array([r[g] for g in ids]) - full
        a = np.abs(diff)
        boots = a[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        signed = diff[idx].mean(axis=1)
        print(
            f"{block}: mean|d| {a.mean():.4f} ratio {a.mean() / READ_NOISE_SD:.2f} "
            f"CI95 [{lo:.4f},{hi:.4f}] ratio [{lo / READ_NOISE_SD:.2f},{hi / READ_NOISE_SD:.2f}] "
            f"signed {diff.mean():+.4f} P(signed>0) {(signed > 0).mean():.2f} max {a.max():.3f}"
        )


def main():
    ap = argparse.ArgumentParser(description="LEAD-103 leave-one-block-out ablation")
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--blocks", nargs="*", default=list(BLOCKS))
    a = ap.parse_args()
    run(a.blocks) if a.cmd == "run" else analyze()


main()
