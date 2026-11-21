import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/lead75/leak_rows.json"
OUT = ROOT / "artifacts/lead75/interim_wk7_bootstrap.json"
SAMPLES = 4000
SEED = 20261121
LEVEL = 0.95
TERM_INPUT = "pass_diff_stored"


def slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    denom = float((x * x).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def main() -> int:
    source = json.loads(SOURCE.read_text())
    rows = pd.DataFrame(source["interim_rows"])
    x = (rows["revision"] * rows[TERM_INPUT]).to_numpy(float)
    y = rows["ats_margin"].to_numpy(float)
    estimate = slope_through_origin(x, y)
    rng = np.random.default_rng(SEED)
    draws = np.array(
        [
            slope_through_origin(x[idx], y[idx])
            for idx in rng.integers(0, len(x), size=(SAMPLES, len(x)))
        ]
    )
    valid = draws[np.isfinite(draws)]
    tail = (1.0 - LEVEL) / 2.0
    out = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "snapshot": source.get("snapshot"),
        "term_input": TERM_INPUT,
        "n_games": len(x),
        "nonzero_terms": int((np.abs(x) > 0).sum()),
        "estimate": estimate,
        "bootstrap": (
            "game-level resample with replacement, slope of ats_margin on "
            "revision x pass-rate diff through origin, percentile interval"
        ),
        "samples": SAMPLES,
        "seed": SEED,
        "level": LEVEL,
        "degenerate_draws_dropped": int(SAMPLES - len(valid)),
        "interval": [float(np.quantile(valid, tail)), float(np.quantile(valid, 1.0 - tail))],
        "standard_error": float(valid.std(ddof=1)),
        "probability_positive": float((valid > 0).mean()),
        "game_ids": rows["game_id"].tolist(),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "game_ids"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
