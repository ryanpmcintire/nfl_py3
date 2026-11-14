import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "artifacts" / "forward_ledger_core" / "ledger.parquet"
DEFAULT_OUT = ROOT / "artifacts" / "forward_monitor" / "monitor.json"
CANDIDATES = ["mod24", "mod25_combined", "lead99"]
MONITOR_ONLY = ["lead99_first_view"]
FORWARD_FROM_WEEK = {"lead99_first_view": 10}
TRUNCATION = 0.5


def _game_terms(ps, pc, y):
    up = math.log(pc / ps)
    down = math.log((1.0 - pc) / (1.0 - ps))
    diff = up if y == 1.0 else down
    worst = max(-up, -down, 0.0)
    scale = 0.5 * (up * up + down * down)
    return diff, worst, scale


def e_process(frame, alpha):
    log_e = 0.0
    seen = []
    rows = []
    for item in frame.itertuples(index=False):
        diff, worst, scale = _game_terms(item.served_p, item.cand_p, item.home_cover)
        t = len(seen) + 1
        mu_hat = sum(seen) / t
        var_hat = (scale + sum((d - mu_hat) ** 2 for d in seen)) / t
        lam = math.sqrt(2.0 * math.log(1.0 / alpha) / (var_hat * t * math.log(1.0 + t)))
        cap = TRUNCATION / worst if worst > 0 else math.inf
        lam = min(lam, cap)
        log_e += math.log1p(lam * diff)
        seen.append(diff)
        rows.append(
            {
                "game_id": item.game_id,
                "week": int(item.week),
                "diff": diff,
                "lambda": lam,
                "log_e": log_e,
                "e": math.exp(log_e),
            }
        )
    return pd.DataFrame(rows)


def games_needed(proc, alpha):
    d = proc["diff"].to_numpy()
    if len(d) < 2:
        return None
    mu = float(d.mean())
    var = float(d.var(ddof=1))
    if mu <= 0 or var <= 0:
        return None
    rate = mu * mu / (2.0 * var)
    gap = math.log(1.0 / alpha) - float(proc["log_e"].iloc[-1])
    return max(0.0, gap / rate), rate


def run(ledger_path, candidates, alpha):
    ledger = pd.read_parquet(ledger_path)
    graded = ledger[ledger["matched"] & ledger["status"].eq("graded")]
    graded = graded[graded["home_cover"].isin([0.0, 1.0])]
    graded = graded.sort_values(["deadline_utc", "game_id"])
    pending = int((ledger["matched"] & ledger["status"].ne("graded")).sum())
    out = {"alpha": alpha, "threshold": 1.0 / alpha, "pending_matched": pending, "candidates": {}}
    for name in candidates:
        frame = graded.assign(cand_p=graded[f"{name}__p"]).dropna(subset=["cand_p", "served_p"])
        proc = e_process(frame, alpha)
        weekly = proc.groupby("week").tail(1).set_index("week")
        need = games_needed(proc, alpha)
        start = FORWARD_FROM_WEEK.get(name)
        forward = frame[frame["week"].ge(start)] if start is not None else frame.iloc[0:0]
        out["candidates"][name] = {
            "monitor_only": name in MONITOR_ONLY,
            "forward_from_week": start,
            "forward_games": len(forward) if start is not None else len(proc),
            "forward_final_e": float(e_process(forward, alpha)["e"].iloc[-1])
            if len(forward)
            else None,
            "games": len(proc),
            "mean_log_loss_gain": float(proc["diff"].mean()),
            "weekly_e": {int(w): float(r["e"]) for w, r in weekly.iterrows()},
            "final_e": float(proc["e"].iloc[-1]),
            "max_e": float(proc["e"].max()),
            "inferred_additional_games_to_threshold": None if need is None else float(need[0]),
            "inferred_log_growth_per_game": None if need is None else float(need[1]),
        }
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only betting e-process on paired per-game log loss of each matched forward "
            "candidate against the served probability. Predictable plug-in bet (Waudby-Smith "
            "and Ramdas 2024, Eq. 24 form with one-sided log(1/alpha)), truncated at "
            f"{TRUNCATION} of the largest adverse log ratio known before the game. "
            "Reports evidence only; closes nothing."
        )
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--candidates", nargs="+", default=[*CANDIDATES, *MONITOR_ONLY])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.ledger, args.candidates, args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
