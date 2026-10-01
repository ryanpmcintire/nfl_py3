import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod25_generator as g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--budget-seconds", type=float, default=1800.0)
    ap.add_argument("--target-seasons", type=int, default=2000)
    ap.add_argument("--worlds-per-batch", type=int, default=6)
    ap.add_argument("--seasons", type=int, default=17)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=9000)
    args = ap.parse_args()
    setting = dict(g.SETTING_DEFAULTS)
    setting.update(json.loads(args.setting))
    out = g.SYN_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_seasons = 0
    total_games = 0
    batch = 0
    while total_seasons < args.target_seasons and time.time() - t0 < args.budget_seconds:
        games, plays, latents, elapsed = g.run_generation(
            setting, args.worlds_per_batch, args.seasons, args.workers, args.seed + batch, progress=False
        )
        offset = batch * args.worlds_per_batch
        games["world"] = games["season"] // 1000 + offset
        games["game_id"] = games["game_id"].map(lambda x, o=offset: f"b{o:05d}" + x)
        plays["game_id"] = plays["game_id"].map(lambda x, o=offset: f"b{o:05d}" + x)
        games.to_parquet(out / f"games_{batch:04d}.parquet")
        plays.to_parquet(out / f"plays_{batch:04d}.parquet")
        weekly = {f"w{w}_s{s}": v[0] for (w, s), v in latents.items()}
        qbout = {f"q{w}_s{s}": v[1] for (w, s), v in latents.items()}
        np.savez_compressed(out / f"latents_{batch:04d}.npz", **weekly, **qbout)
        total_seasons += args.worlds_per_batch * args.seasons
        total_games += len(games)
        batch += 1
        print(f"batch {batch} seasons {total_seasons} games {total_games} {time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    summary = {
        "setting": setting, "seasons": total_seasons, "games": total_games, "wall_seconds": wall,
        "games_per_sec": total_games / wall, "batches": batch, "dir": str(out),
    }
    (g.OUT_DIR / f"produce_{args.tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
