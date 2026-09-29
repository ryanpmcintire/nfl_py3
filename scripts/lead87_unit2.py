from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit, logit
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from nfl_ats.home_side_location import fit_home_side_offsets
from nfl_ats.margin import fit_margin_model
from nfl_ats.mass_preserving_lattice import DiscretePushReader, prior_pool, residual_location
from nfl_ats.pick_probability_fit import FIT_RIDGE, _design, _fit_logit, _natural_coefficients, _standardisers

SOURCE = Path("tests/scratch/codex/lead87_unit1")
FEATURES = Path("data/processed/game_features_weak_stack.parquet")
OUTPUT = Path("tests/scratch/codex/lead87_unit2")
REPORT = Path("docs/lead87_unit2.md")
LANE = Path("docs/lanes/lead87.md")
ARMS = ("candidate", "served", "model_only", "market", "elo")
METRICS = ("log_loss", "brier", "accuracy_points", "conditional_rps")
OUTER = (2023, 2024, 2025)
BOOTSTRAPS = 10000
SEED = 20260929


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> pd.DataFrame:
    return pq.read_table(path, use_threads=False).to_pandas(use_threads=False)


def clipped(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    metadata = json.loads(Path("artifacts/opener_evaluation/20260929T192743Z/metadata.json").read_text())
    if digest(FEATURES) != metadata["feature_table_sha256"]:
        raise ValueError("Frozen served feature-table hash changed")
    features = read(FEATURES)
    features["gameday"] = pd.to_datetime(features.gameday)
    reg = read(SOURCE / "regular_base_join.parquet")
    post = read(SOURCE / "postseason_join.parquet")
    if reg.game_id.duplicated().any() or post.game_id.duplicated().any():
        raise ValueError("Repeated source game")
    if not reg.home_covered.eq(reg.actual_margin.gt(reg.tue_open_home_spread)).all():
        raise ValueError("Opener grade changed")
    if not reg.margin_vs_open.ne(0).all():
        raise ValueError("Conditional REG source contains a push")
    for frame, available in ((reg, reg.reconstructed_move_available), (post, post.conditional_fit_eligible)):
        selected = frame.loc[available]
        if not pd.to_datetime(selected.latest_observed_at, utc=True).lt(pd.to_datetime(selected.kickoff, utc=True)).all():
            raise ValueError("Market input reaches kickoff")
    audit = {
        "source_hashes": {str(p): digest(p) for p in (
            FEATURES, SOURCE / "regular_base_join.parquet", SOURCE / "postseason_join.parquet",
            Path("docs/lead87_protocol.md"), LANE,
        )},
        "original_reg": len(reg), "eligible_reg": int(reg.reconstructed_move_available.sum()),
        "paired_post": int(post.leader_books.gt(0).sum()),
        "eligible_post": int(post.conditional_fit_eligible.sum()), "post_pushes": int(post.push.sum()),
    }
    return features, reg, post, audit


def refit_base(features: pd.DataFrame, reg: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    pool = prior_pool(features, reg.set_index("game_id").tue_open_home_spread)
    outputs, audits, train_ids, stream = [], [], [], []
    for season in range(2020, 2026):
        scoring = reg.loc[reg.season.eq(season)].copy()
        games = features.set_index("game_id").loc[scoring.game_id].reset_index()
        if not games.season.eq(season).all():
            raise ValueError("Feature season mismatch")
        cutoff = features.loc[features.season.eq(season), "gameday"].min()
        train = features.loc[
            features.season.lt(season) & features.game_type.eq("REG")
            & features.result.notna() & features.gameday.lt(cutoff)
        ].copy()
        if train.empty or train.season.max() >= season:
            raise ValueError("Margin fit does not exclude the entire season")
        model = fit_margin_model(train, target="market_residual", model_name="ridge", feature_profile="weak_stack", ridge_alpha=10.0)
        games["spread_line"] = scoring.tue_open_home_spread.to_numpy()
        predicted = model.predict(games, probability_method="gaussian_median")
        point = games.spread_line.to_numpy() + predicted.predicted_market_residual.to_numpy()
        previous = pd.concat(stream, ignore_index=True) if stream else pd.DataFrame()
        if len(previous):
            previous = previous.loc[previous.season.between(season - 5, season - 1)]
            fitted_offsets = fit_home_side_offsets(previous)
            offsets = fitted_offsets.offset_for(games.spread_line).fillna(0).to_numpy()
        else:
            offsets = np.zeros(len(games))
        reader = DiscretePushReader.for_week(pool.loc[pool.season.lt(season)], season=season, week=1, cutoff=cutoff)
        location = residual_location(model.residuals, "gaussian_median")
        model_p, market_p, pushes = [], [], []
        for line, center, shift, move, available in zip(
            games.spread_line, point, offsets, scoring.reconstructed_move_toward_home,
            scoring.reconstructed_move_available, strict=True,
        ):
            distribution = reader.read(float(line), float(center + shift + location))
            model_p.append(distribution.conditional_cover_probability)
            pushes.append(distribution.push)
            market_p.append(reader.read(float(line), float(line + move)).conditional_cover_probability if available else np.nan)
        scoring["model_probability"] = clipped(np.asarray(model_p))
        scoring["model_logit"] = logit(scoring.model_probability)
        scoring["market_probability"] = market_p
        scoring["push_probability"] = pushes
        scoring["market_move_toward_home"] = scoring.reconstructed_move_toward_home
        scoring["market_move_available"] = scoring.reconstructed_move_available.astype(float)
        scoring["elo_diff"] = games.elo_diff.to_numpy()
        scoring["base_train_max_season"] = int(train.season.max())
        scoring["base_train_max_gameday"] = model.training_max_gameday
        outputs.append(scoring)
        stream.append(pd.DataFrame({
            "game_id": games.game_id, "season": season, "spread_line": games.spread_line,
            "point_incumbent": point, "result": games.result,
        }))
        training_ids = train[["game_id", "season", "gameday"]].copy()
        training_ids["prediction_season"] = season
        train_ids.append(training_ids)
        audits.append({
            "prediction_season": season, "training_games": len(train),
            "train_max_season": int(train.season.max()), "train_max_gameday": model.training_max_gameday,
            "prediction_games": len(scoring), "discrete_prior_games": reader.prior_rows,
            "discrete_max_gameday": reader.max_gameday, "residual_rows": len(model.residuals),
            "offset_prior_games": len(previous), "offsets": fitted_offsets.offsets if len(previous) else {},
        })
        print(f"base {season}: train={len(train)} max_season={int(train.season.max())} predictions={len(scoring)}", flush=True)
    result = pd.concat(outputs, ignore_index=True)
    pd.concat(train_ids, ignore_index=True).to_parquet(OUTPUT / "margin_training_ids.parquet", index=False)
    result.to_parquet(OUTPUT / "season_excluded_base.parquet", index=False)
    return result.loc[result.reconstructed_move_available].reset_index(drop=True), audits


def fit_joint(train: pd.DataFrame, post: pd.DataFrame, auxiliary: bool) -> tuple[dict, dict]:
    means, stds = _standardisers(train)
    x = _design(train, means, stds)
    y = train.home_covered.to_numpy(dtype=float)
    if auxiliary:
        x = np.column_stack((x, np.zeros(len(train))))
        xp = np.zeros((len(post), x.shape[1]))
        xp[:, 3] = post.market_move_toward_home.to_numpy(dtype=float) / stds["market_move_toward_home"]
        xp[:, -1] = 1
        x = np.vstack((x, xp))
        y = np.r_[y, post.home_covered.to_numpy(dtype=float)]
    beta = _fit_logit(x, y, FIT_RIDGE)
    natural = _natural_coefficients(beta[:5], means, stds)
    p = expit(x @ beta)
    hessian = (x.T * (p * (1 - p))) @ x + FIT_RIDGE * np.eye(x.shape[1])
    move_se = float(np.sqrt(np.linalg.inv(hessian)[3, 3]) / stds["market_move_toward_home"])
    move = natural["market_move_toward_home"]
    detail = {"coefficients": natural, "move_wald_95": [move - 1.959964 * move_se, move + 1.959964 * move_se]}
    if auxiliary:
        detail["post_intercept"] = float(beta[-1])
    return {"beta": beta[:5], "means": means, "stds": stds}, detail


def joint_predict(frame: pd.DataFrame, fitted: dict) -> np.ndarray:
    return clipped(expit(_design(frame, fitted["means"], fitted["stds"]) @ fitted["beta"]))


def platt_fit(raw: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _fit_logit(np.column_stack((np.ones(len(raw)), logit(clipped(raw)))), y, FIT_RIDGE)


def platt_predict(raw: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return clipped(expit(beta[0] + beta[1] * logit(clipped(raw))))


def replay(reg: pd.DataFrame, post: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    oos, ins, folds = [], [], []
    post = post.loc[post.conditional_fit_eligible].copy()
    for season in OUTER:
        train = reg.loc[reg.season.le(season - 3)].copy()
        tune = reg.loc[reg.season.eq(season - 2)]
        calibration = reg.loc[reg.season.eq(season - 1)].copy()
        test = reg.loc[reg.season.eq(season)].copy()
        aux = post.loc[post.season.le(season - 3)]
        if any(f.empty for f in (train, tune, calibration, test, aux)):
            raise ValueError("An immutable fold partition is empty")
        fold = {"outer": season, "fit_through": season - 3, "reg_fit": len(train), "post_fit": len(aux), "tune": len(tune), "calibration": len(calibration), "test": len(test), "arms": {}}
        raw = {}
        for name in ("served", "candidate"):
            fitted, detail = fit_joint(train, aux, name == "candidate")
            raw[name] = [joint_predict(f, fitted) for f in (train, calibration, test)]
            fold["arms"][name] = detail
        for arm, column in (("model_only", "model_probability"), ("market", "market_probability")):
            raw[arm] = [f[column].to_numpy() for f in (train, calibration, test)]
            fold["arms"][arm] = {}
        names = ["elo_diff", "tue_open_home_spread"]
        means = train[names].mean().to_numpy()
        stds = train[names].std(ddof=0).replace(0, 1).to_numpy()
        designs = [np.column_stack((np.ones(len(f)), (f[names].to_numpy() - means) / stds)) for f in (train, calibration, test)]
        elo_beta = _fit_logit(designs[0], train.home_covered.to_numpy(dtype=float), FIT_RIDGE)
        raw["elo"] = [clipped(expit(x @ elo_beta)) for x in designs]
        fold["arms"]["elo"] = {"standardized_coefficients": elo_beta.tolist(), "means": means.tolist(), "stds": stds.tolist()}
        for arm in ARMS:
            beta = platt_fit(raw[arm][1], calibration.home_covered.to_numpy(dtype=float))
            train[arm] = platt_predict(raw[arm][0], beta)
            test[arm] = platt_predict(raw[arm][2], beta)
            fold["arms"][arm]["calibration"] = beta.tolist()
            if arm in ("served", "candidate"):
                fold["arms"][arm]["calibrated_move_coefficient"] = float(fold["arms"][arm]["coefficients"]["market_move_toward_home"] * beta[1])
        for frame in (train, test):
            frame["outer"] = season
            for arm in ARMS:
                frame[f"{arm}_cover_mass"] = (1 - frame.push_probability) * frame[arm]
                frame[f"{arm}_loss_mass"] = (1 - frame.push_probability) * (1 - frame[arm])
        ins.append(train)
        oos.append(test)
        folds.append(fold)
        print(f"joint {season}: REG={len(train)} POST={len(aux)} calibration={len(calibration)} outer={len(test)}", flush=True)
    return pd.concat(oos, ignore_index=True), pd.concat(ins, ignore_index=True), folds


def metric_rows(frame: pd.DataFrame) -> np.ndarray:
    y = frame.home_covered.to_numpy(dtype=float)[:, None]
    p = clipped(frame[list(ARMS)].to_numpy())
    brier = (p - y) ** 2
    return np.stack((-(y * np.log(p) + (1 - y) * np.log(1 - p)), brier, 100 * ((p >= 0.5) == y), brier), axis=-1)


def boot(frame: pd.DataFrame, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flat = values.reshape(len(frame), -1)
    totals = np.zeros((BOOTSTRAPS, flat.shape[1]))
    counts = np.zeros(BOOTSTRAPS)
    for _, group in frame.groupby("season", sort=True):
        indices = group.index.to_numpy()
        _, inverse = np.unique(group.week.to_numpy(), return_inverse=True)
        n = int(inverse.max()) + 1
        sums = np.zeros((n, flat.shape[1]))
        np.add.at(sums, inverse, flat[indices])
        sizes = np.bincount(inverse, minlength=n)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAPS)
        totals += weights @ sums
        counts += weights @ sizes
    return (totals / counts[:, None]).reshape((BOOTSTRAPS, *values.shape[1:]))


def estimate(value: float, draws: np.ndarray) -> dict:
    low, high = np.quantile(draws, (0.025, 0.975))
    return {"estimate": float(value), "low": float(low), "high": float(high), "probability_positive": float(np.mean(draws > 0) + 0.5 * np.mean(draws == 0)), "standard_error": float(np.std(draws, ddof=1))}


def summarize(oos: pd.DataFrame, ins: pd.DataFrame) -> dict:
    panels = {}
    rng = np.random.default_rng(SEED)
    signs = np.asarray([-1, -1, 1, -1])
    for label, select in [("pooled", None), *((str(y), y) for y in OUTER)]:
        out = oos if select is None else oos.loc[oos.outer.eq(select)].reset_index(drop=True)
        inside = ins if select is None else ins.loc[ins.outer.eq(select)].reset_index(drop=True)
        metrics, training = metric_rows(out), metric_rows(inside)
        draws, train_draws = boot(out, metrics, rng), boot(inside, training, rng)
        point, train_point = metrics.mean(axis=0), training.mean(axis=0)
        panel = {"games": len(out), "arms": {}, "contrasts": {}, "decisive": {}}
        for i, arm in enumerate(ARMS):
            panel["arms"][arm] = {}
            for j, metric in enumerate(METRICS):
                panel["arms"][arm][metric] = {
                    "oos": estimate(point[i, j], draws[:, i, j]),
                    "is": estimate(train_point[i, j], train_draws[:, i, j]),
                    "gap_oos_minus_is": estimate(point[i, j] - train_point[i, j], draws[:, i, j] - train_draws[:, i, j]),
                }
        for i, comparator in enumerate(ARMS[1:], 1):
            panel["contrasts"][comparator] = {}
            for j, metric in enumerate(METRICS):
                effect = signs[j] * (point[0, j] - point[i, j])
                effect_is = signs[j] * (train_point[0, j] - train_point[i, j])
                effect_draws = signs[j] * (draws[:, 0, j] - draws[:, i, j])
                is_draws = signs[j] * (train_draws[:, 0, j] - train_draws[:, i, j])
                panel["contrasts"][comparator][metric] = {
                    "oos": estimate(effect, effect_draws), "is": estimate(effect_is, is_draws),
                    "gap_oos_minus_is": estimate(effect - effect_is, effect_draws - is_draws),
                }
            different = out.candidate.ge(0.5).ne(out[comparator].ge(0.5))
            wins = int(out.loc[different, "candidate"].ge(0.5).eq(out.loc[different, "home_covered"].astype(bool)).sum())
            n = int(different.sum())
            exact = binomtest(wins, n) if n else None
            interval = exact.proportion_ci() if exact else None
            panel["decisive"][comparator] = {
                "wins": wins, "losses": n - wins, "games": n,
                "win_rate_95": [float(interval.low), float(interval.high)] if interval else None,
                "exact_two_sided_p": float(exact.pvalue) if exact else None,
            }
        panels[label] = panel
    reliability = []
    for arm in ARMS:
        bins = np.minimum((oos[arm].to_numpy() * 5).astype(int), 4)
        for b in range(5):
            frame = oos.loc[bins == b]
            reliability.append({"arm": arm, "band": f"{b / 5:.1f}-{(b + 1) / 5:.1f}", "games": len(frame), "mean_probability": float(frame[arm].mean()) if len(frame) else None, "observed_home_cover": float(frame.home_covered.mean()) if len(frame) else None})
    return {"panels": panels, "reliability": reliability}


def cell(result: dict) -> str:
    return f"{result['estimate']:.6f} [{result['low']:.6f}, {result['high']:.6f}]"


def table(headers: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |", *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def write_report(summary: dict) -> None:
    pooled = summary["panels"]["pooled"]
    move_ranges = {arm: (min(f["arms"][arm]["coefficients"]["market_move_toward_home"] for f in summary["folds"]), max(f["arms"][arm]["coefficients"]["market_move_toward_home"] for f in summary["folds"])) for arm in ("served", "candidate")}
    rows = [
        "# LEAD-87 unit 2: whole-season-excluded postseason replay", "",
        "**Measured:** `.tools/uv.exe run --no-sync python scripts/lead87_unit2.py` executed the declared replay once.",
        "The pre-score amendment and hash are saved in scratch. No registry or served-card changes.", "",
        "## Decisive record", "",
        "**Measured:** candidate record on held-out REG games where calibrated sides disagree, with exact binomial",
        "95% intervals and two-sided equal-chance null. Historical forced picks are not each game's probability.", "",
    ]
    rows += table(["Comparator", "Candidate W-L", "Win-rate 95% interval", "Exact null p"], [
        [arm, f"{r['wins']}-{r['losses']}", r["win_rate_95"], r["exact_two_sided_p"]]
        for arm, r in pooled["decisive"].items()
    ])
    rows += ["## Protocol and season exclusion", "",
        "**Read:** `docs/lead87_protocol.md` plus the lane amendment declared before scoring.",
        "Original 2020–2025 opener population; historical opener is the frozen pool-line proxy.",
        "One joint likelihood retains all four REG terms and shares only the move coefficient with a POST-only",
        "intercept. No postseason model or situational flags are invented.", "",
        f"**Measured:** {summary['audit']['original_reg']} original nonpush REG rows; {summary['audit']['eligible_reg']} with admissible leader moves;",
        f"{pooled['games']} outer REG games. {summary['audit']['paired_post']} paired POST, {summary['audit']['eligible_post']} nonpush.",
        "Only earlier fit-partition POST labels enter a fold; later POST games are unused.", "",
        "**Measured:** fresh margin ridge fits, residual samples, discrete readers, and home-side offsets exclude each",
        "entire prediction season and later seasons. Four-term likelihoods and REG-only Platt calibration are independently",
        "refitted for each arm. Reserved tuning seasons select nothing: ridge and the single specification were fixed.",
        "Observed earlier-game rolling inputs remain allowed; cached companion predictions are never consumed.", "",
    ]
    rows += table(["Prediction season", "Margin training games", "Last training season/date", "Discrete prior games", "Offset prior games"], [
        [r["prediction_season"], r["training_games"], f"{r['train_max_season']} / {r['train_max_gameday']}", r["discrete_prior_games"], r["offset_prior_games"]]
        for r in summary["base_audit"]
    ])
    rows += table(["Outer", "Fit through", "REG/POST fit", "Tune REG", "Calibrate REG", "Outer REG"], [
        [f["outer"], f["fit_through"], f"{f['reg_fit']}/{f['post_fit']}", f["tune"], f["calibration"], f["test"]] for f in summary["folds"]
    ])
    rows += [
        "**Read/inferred limitation:** feature-table hash matches the frozen served artifact. This repairs fitted-model",
        "season exclusion, not historical feature construction. Weather, totals, injury/roster features and cached flags",
        "are inherited without a new source-clock audit. Margin fitting retains the served historical spread labels;",
        "the discrete prior pool substitutes known openers where available and otherwise retains historical spreads.",
        "Scoring uses the opener and verified pre-kick move. This is a season-excluded recipe replay, not live-fit parity.",
        "The discrete reader keeps its served five-year pool. Home offsets cold-start at zero in 2020, then use only",
        "prior fresh season-excluded opener predictions; no contaminated offset cache is borrowed.", "",
        "## Paired held-out results", "",
        "**Measured:** 95% week-block bootstrap intervals, 10,000 paired draws stratified by season, seed 20260929.",
        "Positive gains favor the candidate: comparator minus candidate for losses, candidate minus comparator for",
        "accuracy points. Tied draws receive half credit in probability_positive. Intervals condition on fitted models.",
        "Conditional binary RPS equals Brier exactly; it is not full-margin RPS.", "",
    ]
    rows += table(["Arm", "Log loss [95%]", "Brier / conditional RPS [95%]", "Accuracy points [95%]"], [
        [arm, *(cell(pooled["arms"][arm][metric]["oos"]) for metric in METRICS[:3])] for arm in ARMS
    ])
    rows += table(["Comparator", "Endpoint", "Candidate gain [95%]", "probability_positive"], [
        [arm, metric, cell(pooled["contrasts"][arm][metric]["oos"]), f"{pooled['contrasts'][arm][metric]['oos']['probability_positive']:.4f}"]
        for arm in ARMS[1:] for metric in METRICS[:3]
    ])
    rows += ["## In-sample, out-of-sample, and gap", "",
        "**Measured:** IS reuses likelihood-fit REG games and is optimistic; IS/OOS apply the same independent",
        "calibration map. Pooled IS includes repeated training games across folds; bootstrap keeps those repeats",
        "together in their original season/week block. Gap is OOS minus IS; positive loss gaps mean worse OOS.",
        "Complete interval-valued fold/pooled arm and contrast IS/OOS/gaps are in scratch summary.json.", "",
    ]
    rows += table(["Arm", "Endpoint", "IS [95%]", "OOS [95%]", "OOS−IS [95%]"], [
        [arm, metric, *(cell(pooled["arms"][arm][metric][phase]) for phase in ("is", "oos", "gap_oos_minus_is"))]
        for arm in ARMS for metric in METRICS[:3]
    ])
    rows += table(["Outer", "Endpoint", "Candidate gain IS", "Candidate gain OOS [95%]", "Gain gap", "probability_positive"], [
        [season, metric, f"{r['is']['estimate']:.6f}", cell(r["oos"]), f"{r['gap_oos_minus_is']['estimate']:.6f}", f"{r['oos']['probability_positive']:.4f}"]
        for season in OUTER for metric in METRICS[:3]
        for r in [summary["panels"][str(season)]["contrasts"]["served"][metric]]
    ])
    rows += ["## Fitted coefficients", "",
        "**Measured:** natural log-odds move coefficients per point. Wald intervals describe likelihood fits, not",
        "paired held-out effects. Constant move-availability remains present with coefficient zero on this subset.", "",
    ]
    rows += table(["Outer", "Arm", "Move [Wald 95%]", "Intercept", "Model logit", "Flags", "POST intercept", "Calibration intercept/slope", "Calibrated move"], [
        [f["outer"], arm, f"{a['coefficients']['market_move_toward_home']:.6f} [{a['move_wald_95'][0]:.6f}, {a['move_wald_95'][1]:.6f}]",
         *(f"{a['coefficients'][c]:.6f}" for c in ("intercept", "model_logit", "composition_flag_sum")),
         f"{a.get('post_intercept', 0):.6f}" if arm == "candidate" else "n/a",
         "/".join(f"{v:.6f}" for v in a["calibration"]), f"{a['calibrated_move_coefficient']:.6f}"]
        for f in summary["folds"] for arm in ("served", "candidate") for a in [f["arms"][arm]]
    ])
    rows += ["## Reliability", "", "**Measured:** five predeclared equal-width home-cover bands; empty cells remain explicit.", ""]
    rows += table(["Arm", "Band", "Games", "Mean predicted", "Observed home cover"], [
        [r["arm"], r["band"], r["games"], *("empty" if r[c] is None else f"{r[c]:.6f}" for c in ("mean_probability", "observed_home_cover"))]
        for r in summary["reliability"]
    ])
    rows += ["## Interpretation and handoff", "",
        "**Inferred:** one retrospective auxiliary arm, not a serving decision. The 497-look family is unchanged",
        "(F=3, B=6, K=4); repeated identical refits add no specification. The conditional RPS panel repeats Brier.",
        "Week intervals do not establish postseason transportability; no positive control was run.",
        "AGENTS.md lines 65–105 require that zero crossing closes nothing and one fitted probability selects the side.",
        "Provisional classification: unresolved_below_power, pending serial orchestrator recording.",
        "No terminal classification or promotion is asserted.", "",
        "**Measured:** candidate-versus-served Brier gain is positive in 2023 and negative in 2024/2025.",
        f"Move slopes remain positive in all fits: served {move_ranges['served'][0]:.6f} to {move_ranges['served'][1]:.6f},",
        f"candidate {move_ranges['candidate'][0]:.6f} to {move_ranges['candidate'][1]:.6f}.",
        "The extra POST labels change the slope upward in 2023 and downward in 2024/2025; no stable improvement",
        "is established. Only 8/17/26 POST games enter the respective chronological fit partitions.", "",
        "**Measured:** predictions, training membership, source hashes, coefficients, metric panels, and the pre-score",
        "amendment are in `tests/scratch/codex/lead87_unit2/`. Record commands are in `docs/lanes/lead87.md`.",
        "Only candidate-versus-served commands are prepared; this script never writes to a registry.", "",
    ]
    REPORT.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    amendment = LANE.read_text(encoding="utf-8-sig")
    if "Amendment declared before scoring" not in amendment or "497 looks" not in amendment:
        raise ValueError("The pre-score amendment is required")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "summary.json").exists():
        raise ValueError("The declared scored replay has already completed")
    (OUTPUT / "prescore_amendment.md").write_text(amendment, encoding="utf-8")
    features, reg, post, audit = load_sources()
    reg, base_audit = refit_base(features, reg)
    oos, ins, folds = replay(reg, post)
    summary = {"audit": audit, "base_audit": base_audit, "folds": folds, "looks": 497, "bootstrap_draws": BOOTSTRAPS, "seed": SEED, **summarize(oos, ins)}
    oos.to_parquet(OUTPUT / "outer_predictions.parquet", index=False)
    ins.to_parquet(OUTPUT / "fit_predictions.parquet", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary)
    print(json.dumps({"outer_games": len(oos), "decisive": summary["panels"]["pooled"]["decisive"]["served"], "candidate_vs_served": summary["panels"]["pooled"]["contrasts"]["served"]}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
