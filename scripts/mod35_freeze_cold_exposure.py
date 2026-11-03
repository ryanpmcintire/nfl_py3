import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod35_apparent_temp as m
from mod35_unit2_controls import COLD_GAP, CONTROLS

TARGET_SEASON = 2026
FIT_LAST = 2025
OUTPATH = (
    m.ROOT / "artifacts" / "forward_terms" / "mod35_cold_exposure" / "frozen_coefficients.json"
)


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def recorded():
    fc = pd.read_parquet(m.FC)
    fc = fc.loc[fc.fetch_status == "ok"].copy()
    fc["fc_at"] = m.apparent(fc.forecast_temp_f.to_numpy(), fc.forecast_wind_mph.to_numpy())
    rec = fc.loc[
        fc.roof.isin(m.OUTDOOR) & fc.actual_temp_f.notna() & fc.actual_wind_mph.notna()
    ].copy()
    rec["home"] = rec.game_id.str.split("_").str[3]
    rec["app"] = m.apparent(rec.actual_temp_f.to_numpy(), rec.actual_wind_mph.to_numpy())
    return fc, rec[["home", "season", "week", "app"]]


def half_width_for(rec, season):
    chosen = m.select_half_widths(rec)
    if season in chosen:
        return chosen[season][0]
    err = {}
    for u in sorted(rec.season.unique()):
        tgt = rec.loc[rec.season == u]
        keys = list(zip(tgt.home, tgt.season, tgt.week, strict=True))
        preds = np.vstack([m.normals(rec, keys, k) for k in m.HALF_WIDTHS])
        ok = ~np.isnan(preds).any(axis=0)
        if ok.any():
            err[u] = (((preds[:, ok] - tgt.app.to_numpy()[ok]) ** 2).sum(axis=1), int(ok.sum()))
    prior = [u for u in err if u < season]
    tot = sum(err[u][0] for u in prior)
    return m.HALF_WIDTHS[int(np.argmin(tot))]


def main():
    fc, rec = recorded()
    chosen = m.select_half_widths(rec)
    base = m.build_base().merge(fc[["game_id", "roof", "fc_at"]], on="game_id", how="left")
    base["nrm_away"] = np.nan
    base["nrm_home"] = np.nan
    for s in range(m.FIRST, m.LAST + 1):
        k = chosen[s][0]
        mk = (base.season == s).to_numpy()
        for col, tcol in (("nrm_away", "away"), ("nrm_home", "home")):
            keys = list(
                zip(base.loc[mk, tcol], base.loc[mk, "season"], base.loc[mk, "week"], strict=True)
            )
            base.loc[mk, col] = m.normals(rec, keys, k)
    base["gap"] = base.nrm_away - base.fc_at
    base["cold_flag"] = (base.gap >= COLD_GAP).astype(float)
    base["wk_x_nrm_home"] = base.week * base.nrm_home
    base["gap_pos"] = base.gap.clip(lower=0)
    base["gap_neg"] = (-base.gap).clip(lower=0)
    ok = base.gap.notna() & base.nrm_home.notna() & base.roof.isin(m.OUTDOOR)
    po = base.loc[ok & (base.season <= FIT_LAST)].reset_index(drop=True)
    assert int(po.season.max()) <= FIT_LAST
    cols = [*CONTROLS, "gap_pos", "gap_neg"]
    X = po[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    F = np.column_stack([po.z0.to_numpy(), (X - mu) / sd])
    mm = m.fit(F, po.y.to_numpy())
    coef = mm.coef_[0]
    b_std = dict(zip(["z0", *cols], coef.tolist(), strict=True))
    raw = {c: float(coef[1 + i] / sd[i]) for i, c in enumerate(cols)}
    body = {
        "term": "mod35_cold_exposure_visitor_colder_climate_gap",
        "term_definition": "gap_neg = max(forecast apparent temperature at site - visitor outdoor-home prior-season recorded mean apparent temperature for the week window, 0), outdoor games only",  # noqa: E501
        "b_logit_per_point": raw["gap_neg"],
        "raw_coefficients_per_unit": raw,
        "standardised_coefficients": b_std,
        "intercept": float(mm.intercept_[0]),
        "standardisation_mean": dict(zip(cols, mu.tolist(), strict=True)),
        "standardisation_sd": dict(zip(cols, sd.tolist(), strict=True)),
        "controls": list(CONTROLS),
        "cold_flag_gap_threshold_f": COLD_GAP,
        "fit_seasons": [int(po.season.min()), int(po.season.max())],
        "fit_games": len(po),
        "fit_base": "proxy-opener main_scored 2011-19 plus opener-evaluation per_game 2020-25, recalibrated z0 as covariate",  # noqa: E501
        "normal_half_width_weeks_for_2026": int(half_width_for(rec, TARGET_SEASON)),
        "normal_half_width_rule": "walk-forward minimum squared error of recorded apparent temperature over seasons before the target",  # noqa: E501
        "recorded_weather_source": str(m.FC.relative_to(m.ROOT)),
        "recorded_weather_sha256": sha_file(m.FC),
        "fit_script": "scripts/mod35_freeze_cold_exposure.py",
        "unit2_result": "artifacts/mod35/unit2.json",
        "no_2026_outcomes": True,
    }
    body["coefficient_hash"] = sha_obj(
        {k: body[k] for k in ("raw_coefficients_per_unit", "b_logit_per_point", "intercept")}
    )
    body["created_at_utc"] = datetime.now(UTC).isoformat()
    if OUTPATH.exists():
        raise SystemExit(f"already frozen: {OUTPATH}")
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPATH.write_text(json.dumps(body, indent=1))
    print(
        json.dumps(
            {
                k: body[k]
                for k in (
                    "b_logit_per_point",
                    "fit_games",
                    "fit_seasons",
                    "normal_half_width_weeks_for_2026",
                    "coefficient_hash",
                    "created_at_utc",
                )
            },
            indent=1,
        )
    )
    print(json.dumps(b_std, indent=1))


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
