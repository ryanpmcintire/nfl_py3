# ruff: noqa: E501
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "mod36_noise"
d = pd.DataFrame([json.loads(x) for x in (OUT / "draws.jsonl").read_text().splitlines()])
Q = (0.05, 0.5, 0.95)
summary = {}
for k, g in d.groupby("k"):
    s = {"n": len(g)}
    for col in ("d_acc", "ll_gain", "flips", "pp"):
        s[col] = {f"q{int(q * 100):02d}": float(g[col].quantile(q)) for q in Q}
    s["share_pp_ge_90"] = float((g["pp"] >= 0.9).mean())
    s["share_pp_ge_95"] = float((g["pp"] >= 0.95).mean())
    s["share_pp_le_10"] = float((g["pp"] <= 0.1).mean())
    s["share_pp_le_05"] = float((g["pp"] <= 0.05).mean())
    summary[int(k)] = s
cands = {
    "mod23_u13_A_pass_rush": (-0.4657, 0.0953, 75, -0.000505),
    "mod23_u13_B_total": (-0.4657, 0.2461, 59, -0.000146),
    "mod23_u14_D_drive": (-0.3327, 0.31125, 161, 0.001853),
    "mod23_u14_P_playquality": (-0.8649, 0.036, 217, 0.0000591),
}
rows = {}
for name, (da, pp, fl, llg) in cands.items():
    for k, g in d.groupby("k"):
        rows[f"{name}|K={k}"] = {
            "p_two_sided_acc": float((g["d_acc"].abs() >= abs(da)).mean()),
            "p_pp_as_extreme": float(((g["pp"] <= pp) | (g["pp"] >= 1 - pp)).mean()) if pp < 0.5 else float((g["pp"] >= pp).mean()),
            "p_flips_ge": float((g["flips"] >= fl).mean()),
            "p_ll_gain_le": float((g["ll_gain"] <= llg).mean()),
        }
(OUT / "summary.json").write_text(json.dumps({"null": summary, "placement": rows}, indent=1))
print(json.dumps({"null": summary, "placement": rows}, indent=1))
