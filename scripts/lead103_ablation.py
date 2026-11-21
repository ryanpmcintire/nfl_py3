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


def stamp(d):
    s = d.get("recorded_at_utc") or d.get("run_at_utc")
    return datetime.strptime("".join(c for c in s if c.isdigit())[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def run(blocks, source):
    d = load(source)
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
        path = OUT / f"{source.stem}-drop-{block}-{t}.json"
        path.write_text(
            json.dumps(
                {
                    "source_run": source.name,
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


def analyze(source):
    src = load(source)
    docs = [src] + [
        d
        for d in (load(p) for p in sorted((L.ART / "repeats").glob(source.stem + "-repeat-*.json")))
        if d["valid"]
    ]
    abl = {
        b: [d for d in (load(p) for p in sorted(OUT.glob(f"{source.stem}-drop-{b}-*.json"))) if d["valid"]]
        for b in BLOCKS
    }
    cutoff = max(stamp(d) for d in docs + [d for v in abl.values() for d in v])
    kick = {
        g["game_id"]: datetime.fromisoformat(g["kickoff_et"])
        for g in src["packet"]["games"]
        if g.get("kickoff_et")
    }
    all_ids = sorted(rows_of(src))
    ids = [g for g in all_ids if g not in kick or kick[g] > cutoff]
    print(f"latest read {cutoff.isoformat()}; excluded as kicked off before it:", sorted(set(all_ids) - set(ids)))
    F = np.array([[rows_of(d)[g] for g in ids] for d in docs])
    full = F.mean(axis=0)
    sd_full = float(np.sqrt(F.var(axis=0, ddof=1).mean())) if len(F) > 1 else float("nan")
    print("full reads", len(F), "games", len(ids), f"measured within-read SD {sd_full:.4f} noise used {READ_NOISE_SD}")
    rng = np.random.default_rng(SEED)
    n = len(ids)
    idx = rng.integers(0, n, (N_BOOT, n))
    for block in BLOCKS:
        if not abl[block]:
            print(block, "no valid run")
            continue
        A = np.array([[rows_of(d)[g] for g in ids] for d in abl[block]])
        k = len(A)
        diff = A.mean(axis=0) - full
        a = np.abs(diff)
        se = READ_NOISE_SD * math.sqrt(1 / k + 1 / len(F))
        null_mean = se * math.sqrt(2 / math.pi)
        null = np.abs(rng.normal(0, se, (N_BOOT, n))).mean(axis=1)
        boots = a[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        signed = diff[idx].mean(axis=1)
        am = A.mean(axis=0)
        cs = np.array([np.corrcoef(full[i], am[i])[0, 1] for i in idx if np.std(full[i]) > 0 and np.std(am[i]) > 0])
        sd_abl = float(np.sqrt(A.var(axis=0, ddof=1).mean())) if k > 1 else float("nan")
        print(
            f"{block}: reads {k} within-read SD {sd_abl:.4f} mean|d| {a.mean():.4f} CI95 [{lo:.4f},{hi:.4f}] "
            f"null mean|d| {null_mean:.4f} ratio {a.mean() / null_mean:.2f} [{lo / null_mean:.2f},{hi / null_mean:.2f}] "
            f"matched-null percentile {(null < a.mean()).mean():.3f} "
            f"signed {diff.mean():+.4f} CI95 [{np.percentile(signed, 2.5):+.4f},{np.percentile(signed, 97.5):+.4f}] "
            f"P(signed>0) {(signed > 0).mean():.3f} max {a.max():.3f} "
            f"corr full v ablated {np.corrcoef(full, am)[0, 1]:.3f} CI95 [{np.percentile(cs, 2.5):.3f},{np.percentile(cs, 97.5):.3f}] "
            f"P(corr>0) {(cs > 0).mean():.3f}"
        )
        if k > 1 and len(F) > 1:
            sd_m = math.sqrt(
                ((len(F) - 1) * sd_full**2 + (k - 1) * sd_abl**2) / (len(F) + k - 2)
            )
            se_m = sd_m * math.sqrt(1 / k + 1 / len(F))
            null_m = np.abs(rng.normal(0, se_m, (N_BOOT, n))).mean(axis=1)
            nm_m = se_m * math.sqrt(2 / math.pi)
            print(
                f"   measured-noise check: pooled within-read SD {sd_m:.4f} null mean|d| {nm_m:.4f} "
                f"ratio {a.mean() / nm_m:.2f} [{lo / nm_m:.2f},{hi / nm_m:.2f}] "
                f"matched-null percentile {(null_m < a.mean()).mean():.3f}"
            )
        for i in np.argsort(-a)[:3]:
            print(f"   {ids[i]} full {full[i]:+.3f} ablated {am[i]:+.3f} d {diff[i]:+.3f} z {diff[i] / se:+.1f}")


def main():
    ap = argparse.ArgumentParser(description="LEAD-103 leave-one-block-out ablation")
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--blocks", nargs="*", default=list(BLOCKS))
    ap.add_argument("--source", type=Path, default=SOURCE)
    a = ap.parse_args()
    run(a.blocks, a.source) if a.cmd == "run" else analyze(a.source)


main()
