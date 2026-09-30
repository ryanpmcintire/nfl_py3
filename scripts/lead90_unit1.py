from __future__ import annotations

import json
from pathlib import Path

import lead83_unit1 as inventory
import lead83_unit2 as base
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES
from nfl_ats.pbp import snapshot_from_root
from nfl_ats.pbp08_matchup_flags import QUARTILE_TOP, _team_game_windows
from nfl_ats.pick_probability import COUNTED_FLAG_COLUMNS, signed_composition_flags
from nfl_ats.pick_probability_fit import _arrest_incidents, _forecast_temperatures

OUTPUT = Path("tests/scratch/codex/lead90_unit1")
REPORT = Path("docs/lead90_unit1.md")
LANE = Path("docs/lanes/lead90.md")
OPENER = Path("artifacts/opener_evaluation/20260929T192743Z/per_game.parquet")
PBP = Path("data/pbp/raw/20260929T191306Z")
ARMS = ("local_rps", "four_term", "model_only", "market", "elo")
METRICS = ("accuracy_points", "brier", "log_loss", "rps", "local_rps")
GRID = base.GRID
SEED = 20260990
DRAWS = 10000
LOOKS = 605


def save(frame: pd.DataFrame, name: str) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), OUTPUT / name)


def protection(schedules: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    snapshot = snapshot_from_root(PBP)
    columns = [
        "game_id",
        "season_type",
        "posteam",
        "defteam",
        "epa",
        "wp",
        "play",
        "qb_kneel",
        "qb_spike",
        "aborted_play",
        "pass_attempt",
        "rush_attempt",
        "qb_dropback",
        "sack",
        "qb_hit",
    ]
    traits = []
    sources = []
    for season in snapshot.seasons:
        if season > 2025:
            continue
        path = snapshot.season_path(season)
        sources.append(str(path))
        frame = base.read(path, columns)
        for name in columns[4:]:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        keep = (
            frame.season_type.eq("REG")
            & frame.posteam.notna()
            & frame.defteam.notna()
            & frame.epa.notna()
            & frame.wp.between(0.05, 0.95)
            & frame.play.fillna(1).eq(1)
            & frame.qb_kneel.fillna(0).eq(0)
            & frame.qb_spike.fillna(0).eq(0)
            & frame.aborted_play.fillna(0).eq(0)
            & (frame.pass_attempt.fillna(0).eq(1) | frame.rush_attempt.fillna(0).eq(1))
            & frame.qb_dropback.fillna(0).eq(1)
        )
        frame = frame.loc[keep].copy()
        frame["posteam"] = frame.posteam.replace(TEAM_ABBREVIATION_ALIASES)
        frame["defteam"] = frame.defteam.replace(TEAM_ABBREVIATION_ALIASES)
        frame["pressure"] = (frame.sack.fillna(0).eq(1) | frame.qb_hit.fillna(0).eq(1)).astype(
            float
        )
        allowed = frame.groupby(["game_id", "posteam"]).pressure.mean().rename("press_allow_g")
        generated = frame.groupby(["game_id", "defteam"]).pressure.mean().rename("press_gen_g")
        allowed.index = allowed.index.set_names(["game_id", "team"])
        generated.index = generated.index.set_names(["game_id", "team"])
        traits.append(pd.concat([allowed, generated], axis=1).reset_index())
    schedule = schedules.loc[schedules.game_type.eq("REG") & schedules.season.le(2025)].copy()
    schedule["gameday"] = pd.to_datetime(schedule.gameday)
    for name in ("home_team", "away_team"):
        schedule[name] = schedule[name].replace(TEAM_ABBREVIATION_ALIASES)
    windows = _team_game_windows(schedule, pd.concat(traits, ignore_index=True))
    home = windows.loc[windows.is_home].set_index("game_id")
    away = windows.loc[~windows.is_home].set_index("game_id").reindex(home.index)
    home_bad = home.press_allow_q.eq(QUARTILE_TOP) & away.press_gen_q.eq(QUARTILE_TOP)
    away_bad = away.press_allow_q.eq(QUARTILE_TOP) & home.press_gen_q.eq(QUARTILE_TOP)
    result = pd.DataFrame(
        {
            "game_id": home.index,
            "back_side": np.where(
                home_bad & ~away_bad, "AWAY", np.where(away_bad & ~home_bad, "HOME", "")
            ),
        }
    )
    return result, sources


def load() -> tuple[pd.DataFrame, dict]:
    games, lineage = inventory.load_population()
    books, manifests, clocks = inventory.recover_books(games)
    moves = (
        books.groupby("game_id")
        .agg(
            original_move=("move_toward_home", "median"),
            sunday_line=("home_spread_line_sunday", "median"),
            matched_books=("bookmaker_key", "size"),
        )
        .reset_index()
    )
    games = games.merge(moves, on="game_id", validate="one_to_one")
    metadata = json.loads(base.OPENER_META.read_text(encoding="utf-8-sig"))
    if base.digest(base.FEATURES) != metadata["feature_table_sha256"]:
        raise ValueError("Upstream feature source changed")
    columns = list(
        dict.fromkeys(
            [
                "game_id",
                "result",
                "elo_diff",
                *base.margin_feature_columns("market_residual", "weak_stack"),
            ]
        )
    )
    columns = [name for name in columns if name not in ("spread_line", "total_line")]
    frame = games.merge(base.read(base.FEATURES, columns), on="game_id", validate="one_to_one")
    frame["spread_line"] = frame.tue_open_home_spread
    frame["total_line"] = frame.game_id.map(
        base.tuesday_totals(games, {"source_manifests": manifests})
    )
    frame["ats_margin"] = frame.result - frame.spread_line
    frame["home_covered"] = frame.ats_margin.gt(0).astype(float)
    frame["available"] = 1.0
    if not frame.ats_margin.ne(0).equals(frame.declared_fit_population):
        raise ValueError("Frozen push population mismatch")
    if frame.result.isna().any() or frame.elo_diff.isna().any() or frame.result.abs().gt(100).any():
        raise ValueError("Missing or unsupported result/Elo")
    schedules = base.read(inventory.SCHEDULE)
    pressure, pressure_sources = protection(schedules)
    flags = signed_composition_flags(
        frame,
        schedules,
        incidents=_arrest_incidents(Path("data")),
        forecasts_tuesday_noon=_forecast_temperatures(Path("data")),
        protection_back_side=pressure,
    )
    reference = base.read(base.FIT, ["game_id", *COUNTED_FLAG_COLUMNS, "composition_flag_sum"])
    check = reference.merge(
        flags, on="game_id", suffixes=("_frozen", "_rebuilt"), validate="one_to_one"
    )
    for name in (*COUNTED_FLAG_COLUMNS, "composition_flag_sum"):
        if not np.array_equal(check[f"{name}_frozen"], check[f"{name}_rebuilt"]):
            raise ValueError(f"Rebuilt composition does not match frozen {name}")
    frame = frame.merge(
        flags[["game_id", "composition_flag_sum"]], on="game_id", validate="one_to_one"
    )
    lineage.update(
        {
            "population_with_pushes": len(frame),
            "nonpush": int(frame.declared_fit_population.sum()),
            "verified_composition_rows": len(check),
            "quote_clock_inventory": clocks,
            "source_manifests": manifests,
            "pressure_sources": pressure_sources,
            "missing_tuesday_totals": int(frame.total_line.isna().sum()),
            "inputs": {
                str(path): base.digest(path)
                for path in (
                    base.FEATURES,
                    base.FIT,
                    OPENER,
                    inventory.SCHEDULE,
                    base.OPENER_META,
                    Path(__file__),
                    Path("scripts/lead83_unit1.py"),
                    Path("scripts/lead83_unit2.py"),
                )
            },
            "protocol_sha256": base.digest(OUTPUT / "protocol.md"),
        }
    )
    save(frame, "features.parquet")
    save(books, "matched_books.parquet")
    return frame, lineage


def tilt(mass: np.ndarray, eta: np.ndarray, lines: np.ndarray) -> np.ndarray:
    z = np.where(mass > 0, np.log(np.maximum(mass, 1e-300)), -np.inf)
    z += eta[:, None] * (GRID[None, :] - lines[:, None])
    z -= z.max(axis=1, keepdims=True)
    result = np.exp(z)
    return result / result.sum(axis=1, keepdims=True)


def candidate(panel: pd.DataFrame, mass: np.ndarray) -> tuple[np.ndarray, dict]:
    training = panel.role.eq("fit").to_numpy()
    raw = panel.loc[:, list(base.TERMS)].to_numpy(float)
    mean = raw[training].mean(axis=0)
    scale = raw[training].std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(panel)), (raw - mean) / scale])
    x = design[training]
    lines = panel.spread_line.to_numpy(float)
    train_mass = mass[training]
    weights = 2.0 ** (-np.abs(GRID[None, :] - lines[training, None]))
    target = panel.loc[training, "result"].to_numpy(float)[:, None] <= GRID[None, :]
    atoms = GRID[None, :] - lines[training, None]

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        fitted = tilt(train_mass, x @ beta, lines[training])
        residual = np.cumsum(fitted, axis=1) - target
        derivative = fitted * (atoms - (fitted * atoms).sum(axis=1, keepdims=True))
        d_eta = (2 * weights * residual * np.cumsum(derivative, axis=1)).sum(axis=1)
        value = float(
            np.mean((weights * residual**2).sum(axis=1)) + 0.5 * 0.001 * np.dot(beta[1:], beta[1:])
        )
        gradient = x.T @ d_eta / len(x)
        gradient[1:] += 0.001 * beta[1:]
        return value, gradient

    fit = minimize(
        objective,
        np.zeros(design.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-8},
    )
    if not fit.success or not np.isfinite(fit.fun):
        raise ValueError(f"Weighted RPS fit failed: {fit.message}")
    natural = fit.x[1:] / scale
    coefficients = dict(zip(base.TERMS, natural.tolist(), strict=True))
    coefficients["intercept"] = float(fit.x[0] - mean @ natural)
    return tilt(mass, design @ fit.x, lines), {
        "coefficients": coefficients,
        "iterations": int(fit.nit),
        "gradient_max_abs": float(np.max(np.abs(fit.jac))),
        "objective": float(fit.fun),
    }


def calibrated_mass(raw_mass: np.ndarray, probability: np.ndarray, lines: np.ndarray) -> np.ndarray:
    home = GRID[None, :] > lines[:, None]
    away = GRID[None, :] < lines[:, None]
    push = ~(home | away)
    hp = (raw_mass * home).sum(axis=1)
    ap = (raw_mass * away).sum(axis=1)
    pp = (raw_mass * push).sum(axis=1)
    mass = raw_mass * (
        home * ((1 - pp) * probability / hp)[:, None]
        + away * ((1 - pp) * (1 - probability) / ap)[:, None]
        + push
    )
    if not np.allclose(mass.sum(axis=1), 1) or not np.allclose(
        base.probability(mass, lines), probability
    ):
        raise ValueError("Calibrated discrete mass identity failed")
    return mass


def scores(panel: pd.DataFrame, mass: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    y = panel.home_covered.to_numpy(float)
    nonpush = panel.declared_fit_population.to_numpy()
    error = (np.cumsum(mass, axis=1) - (panel.result.to_numpy()[:, None] <= GRID[None, :])) ** 2
    weights = 2.0 ** (-np.abs(GRID[None, :] - panel.spread_line.to_numpy()[:, None]))
    return {
        "accuracy_points": np.where(nonpush, 100 * ((p >= 0.5) == y), np.nan),
        "brier": np.where(nonpush, (p - y) ** 2, np.nan),
        "log_loss": np.where(nonpush, -(y * np.log(p) + (1 - y) * np.log1p(-p)), np.nan),
        "rps": error.sum(axis=1),
        "local_rps": (weights * error).sum(axis=1),
    }


def replay(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    details = []
    for outer in (2023, 2024, 2025):
        panel = (
            frame.loc[frame.season.le(outer)]
            .sort_values(["season", "week", "game_id"])
            .reset_index(drop=True)
        )
        panel["role"] = np.select(
            [panel.season.le(outer - 3), panel.season.eq(outer - 2), panel.season.eq(outer - 1)],
            ["fit", "tune", "calibrate"],
            default="outer",
        )
        training = panel.role.eq("fit").to_numpy()
        train = panel.loc[training].copy()
        if train.season.min() != 2020 or train.season.max() != outer - 3:
            raise ValueError("Upstream chronology mismatch")
        model = base.fit_margin_model(
            train,
            target="market_residual",
            model_name="ridge",
            feature_profile="weak_stack",
            ridge_alpha=10.0,
        )
        point = model.predict(
            panel, probability_method="gaussian_median"
        ).predicted_margin.to_numpy() + base.residual_location(model.residuals, "gaussian_median")
        offsets = base.fit_home_side_offsets(train.assign(point_incumbent=point[training]))
        point += offsets.offset_for(panel.spread_line).to_numpy(float)
        elo_x = np.column_stack([np.ones(len(panel)), panel.elo_diff.to_numpy(float)])
        elo_beta = np.linalg.lstsq(
            elo_x[training], panel.loc[training, "result"].to_numpy(float), rcond=None
        )[0]
        masses = {
            name: base.lattice(train, panel, values)
            for name, values in {
                "model_only": point,
                "market": panel.sunday_line.to_numpy(float),
                "elo": elo_x @ elo_beta,
            }.items()
        }
        lines = panel.spread_line.to_numpy(float)
        raw = {name: base.probability(mass, lines) for name, mass in masses.items()}
        panel["model_logit"] = logit(raw["model_only"])
        raw["four_term"], four_coeff = base.fit_probability(
            panel, training & panel.declared_fit_population.to_numpy(), base.TERMS
        )
        masses["four_term"] = masses["model_only"]
        masses["local_rps"], fitted = candidate(panel, masses["model_only"])
        raw["local_rps"] = base.probability(masses["local_rps"], lines)
        calibration_frame = panel.copy()
        calibration_frame.loc[~panel.declared_fit_population, "role"] = "push"
        detail = {
            "outer": outer,
            "fit_through": outer - 3,
            "training_max_gameday": str(pd.to_datetime(train.gameday).max().date()),
            "roles_with_pushes": panel.role.value_counts().to_dict(),
            "candidate": fitted,
            "four_term": four_coeff,
            "elo": dict(zip(("intercept", "elo_diff"), elo_beta.tolist(), strict=True)),
            "home_side_offsets": offsets.offsets,
            "calibration": {},
        }
        upstream = panel[["game_id", "season", "week", "role", "model_logit", "spread_line"]].copy()
        upstream["fit_through"] = outer - 3
        upstream["point"] = point
        save(upstream, f"upstream_{outer}.parquet")
        for arm in ARMS:
            p, calibration = base.calibrate(raw[arm], calibration_frame)
            mass = calibrated_mass(masses[arm], p, lines)
            detail["calibration"][arm] = calibration
            values = scores(panel, mass, p)
            for phase, role in (("IS", "fit"), ("OOS", "outer")):
                selected = panel.role.eq(role).to_numpy()
                result = panel.loc[
                    selected,
                    [
                        "game_id",
                        "season",
                        "week",
                        "result",
                        "spread_line",
                        "home_covered",
                        "declared_fit_population",
                    ],
                ].copy()
                result["outer"] = outer
                result["phase"] = phase
                result["arm"] = arm
                result["probability"] = p[selected]
                result["push_probability"] = (mass * (GRID[None, :] == lines[:, None])).sum(axis=1)[
                    selected
                ]
                for metric, value in values.items():
                    result[metric] = value[selected]
                rows.append(result)
                np.savez_compressed(
                    OUTPUT / f"pmf_{outer}_{phase}_{arm}.npz",
                    game_id=result.game_id.to_numpy(str),
                    grid=GRID,
                    mass=mass[selected],
                )
        details.append(detail)
        print(
            f"fold={outer}; fit through={outer - 3}; outer={int(panel.role.eq('outer').sum())}",
            flush=True,
        )
    return pd.concat(rows, ignore_index=True), details


def blocked(frame: pd.DataFrame, values: np.ndarray, seed: int) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(frame[["season", "week"]])
    codes, unique = pd.factorize(keys, sort=True)
    sums = np.zeros((len(unique), values.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, codes, np.nan_to_num(values))
    np.add.at(counts, codes, np.isfinite(values).astype(float))
    years = unique.get_level_values(0).to_numpy()
    seasons = np.unique(years)
    indexes = {year: np.flatnonzero(years == year) for year in seasons}
    rng = np.random.default_rng(seed)
    samples = np.empty((DRAWS, values.shape[1]))
    for index in range(DRAWS):
        chosen = np.concatenate(
            [
                rng.choice(indexes[year], len(indexes[year]), replace=True)
                for year in rng.choice(seasons, len(seasons), replace=True)
            ]
        )
        samples[index] = sums[chosen].sum(axis=0) / counts[chosen].sum(axis=0)
    return samples


def summarize(predictions: pd.DataFrame) -> dict:
    panels = {}
    labels = [*ARMS, *(f"gain_vs_{arm}" for arm in ARMS[1:])]
    for outer in (2023, 2024, 2025, "pooled"):
        selected = (
            predictions if outer == "pooled" else predictions.loc[predictions.outer.eq(outer)]
        )
        panel = {"phases": {}, "decisive": {}}
        draws = {}
        means = {}
        for phase in ("IS", "OOS"):
            groups = {
                arm: selected.loc[selected.phase.eq(phase) & selected.arm.eq(arm)]
                .sort_values(["outer", "game_id"])
                .reset_index(drop=True)
                for arm in ARMS
            }
            reference = groups[ARMS[0]]
            if any(
                not group[["outer", "game_id"]].equals(reference[["outer", "game_id"]])
                for group in groups.values()
            ):
                raise ValueError("Arm populations are unpaired")
            arrays = [groups[arm].loc[:, list(METRICS)].to_numpy(float) for arm in ARMS]
            contrasts = [
                (arrays[0] - other) * np.array([1, -1, -1, -1, -1]) for other in arrays[1:]
            ]
            values = np.column_stack([*arrays, *contrasts])
            draws[phase] = blocked(
                reference,
                values,
                SEED + (0 if outer == "pooled" else int(outer)) + (0 if phase == "IS" else 100),
            )
            means[phase] = np.nanmean(values, axis=0)
            cells = {}
            for index, label in enumerate(labels):
                cells[label] = {
                    metric: base.estimate(
                        means[phase][index * 5 + m], draws[phase][:, index * 5 + m]
                    )
                    for m, metric in enumerate(METRICS)
                }
                cells[label]["games"] = int(reference.declared_fit_population.sum())
                cells[label]["games_with_pushes"] = len(reference)
                cells[label]["blocks"] = int(
                    reference[["season", "week"]].drop_duplicates().shape[0]
                )
                if label in ARMS:
                    wins = int(groups[label].accuracy_points.eq(100).sum())
                    cells[label]["record"] = f"{wins}-{cells[label]['games'] - wins}"
            panel["phases"][phase] = cells
            if phase == "OOS":
                for comparator in ARMS[1:]:
                    decisive = reference.declared_fit_population & reference.probability.ge(0.5).ne(
                        groups[comparator].probability.ge(0.5)
                    )
                    wins = int(reference.loc[decisive, "accuracy_points"].eq(100).sum())
                    n = int(decisive.sum())
                    panel["decisive"][comparator] = {
                        "games": n,
                        "candidate_wins": wins,
                        "base_wins": n - wins,
                        "exact_p": float(binomtest(wins, n).pvalue) if n else 1.0,
                    }
        delta = draws["OOS"] - draws["IS"]
        point_delta = means["OOS"] - means["IS"]
        panel["gaps"] = {
            label: {
                metric: base.estimate(point_delta[index * 5 + m], delta[:, index * 5 + m])
                for m, metric in enumerate(METRICS)
            }
            for index, label in enumerate(labels)
        }
        panels[str(outer)] = panel
    reliability = []
    for arm in ARMS:
        group = predictions.loc[
            predictions.phase.eq("OOS")
            & predictions.arm.eq(arm)
            & predictions.declared_fit_population
        ]
        band = np.minimum((group.probability * 5).astype(int), 4)
        for index in range(5):
            cell = group.loc[band.eq(index)]
            reliability.append(
                {
                    "arm": arm,
                    "band": f"{index / 5:.1f}-{(index + 1) / 5:.1f}",
                    "n": len(cell),
                    "predicted": float(cell.probability.mean()) if len(cell) else None,
                    "observed": float(cell.home_covered.mean()) if len(cell) else None,
                }
            )
    return {"panels": panels, "reliability": reliability}


def record_commands(summary: dict) -> str:
    names = {
        "accuracy_points": "accuracy_points",
        "brier": "brier_improvement",
        "log_loss": "log_loss_improvement",
        "rps": "rps_improvement",
        "local_rps": "rps_improvement",
    }
    lines = [
        chr(96) * 3 + "bash",
        "while read -r panel first last games blocks metric units effect low high pp; do",
        (
            ".tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record "
            '--name "lead90_unit1_$panel-$metric" --league nfl \\'
        ),
        (
            '  --description "Fitting the whole score spread near the pool line versus the '
            'usual four-term calculation: $panel $metric" --source docs/lead90_unit1.md '
            "--family lead90_unit1_605_looks \\"
        ),
        (
            '  --season-start "$first" --season-end "$last" --sample-games "$games"'
            ' --sample-blocks "$blocks" --effect-units "$units" \\'
        ),
        (
            '  --effect "$effect" --interval-low "$low" --interval-high "$high" '
            '--probability-positive "$pp" --classification unresolved_below_power \\'
        ),
        (
            '  --classification-evidence "Three held-out seasons; no mechanism closure or '
            'serving claim. All 605 declared looks reported." \\'
        ),
        (
            '  --plain-summary "This teaches the pick calculation to care most about scores '
            "right around the pool line instead of far-off blowouts. The held-out comparison "
            'is a research result; pool picks have not changed."'
        ),
        "done <<'CELLS'",
    ]
    for label, panel in summary["panels"].items():
        first, last = (2023, 2025) if label == "pooled" else (int(label), int(label))
        oos = panel["phases"]["OOS"]
        for metric in METRICS:
            cell = oos["gain_vs_four_term"][metric]
            values = " ".join(
                f"{cell[key]:.12g}" for key in ("estimate", "low", "high", "probability_positive")
            )
            lines.append(
                f"{label} {first} {last} {oos['local_rps']['games']} "
                f"{oos['local_rps']['blocks']} {metric} {names[metric]} {values}"
            )
    return "\n".join([*lines, "CELLS", chr(96) * 3])


def number(cell: dict) -> str:
    return f"{cell['estimate']:.6f} [{cell['low']:.6f}, {cell['high']:.6f}]"


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def report(summary: dict, lane_head: str) -> None:
    lines = [
        "# LEAD-90 unit 1: fit the score spread near the pool line",
        "",
        "## Decisive games first",
        "",
        (
            "**Measured:** candidate record only where its selected side differs from each "
            "comparator; exact two-sided fair-coin null."
        ),
        "",
        table(
            ["Outer", "Comparator", "Decisive games", "Candidate W-L", "Exact p"],
            [
                [
                    outer,
                    arm,
                    cell["games"],
                    f"{cell['candidate_wins']}-{cell['base_wins']}",
                    f"{cell['exact_p']:.6f}",
                ]
                for outer, panel in summary["panels"].items()
                for arm, cell in panel["decisive"].items()
            ],
        ),
        "",
        "## Fixed protocol and interpretation",
        "",
        (
            "The declaration below was saved before outcomes; its immutable copy is "
            "tests/scratch/codex/lead90_unit1/protocol.md."
        ),
        "",
        lane_head,
        "",
        (
            "**Measured:** all upstream margin, offset, Elo models and discrete lattices "
            "refit through Y-3. The historical opener replaces the frozen pool line. IS is "
            "optimistic, with repeated training rows across pooled folds. All five arms get "
            "the same slope selection and intercept calibration; calibrated cover/loss mass "
            "retains push mass and alone selects the side. The tilt candidate only changes "
            "the discrete mass fed to that calibration."
        ),
        "",
        (
            f"**Measured:** {summary['lineage']['population_with_pushes']} games including "
            f"pushes; {summary['lineage']['nonpush']} conditional-cover rows; "
            f"{summary['lineage']['verified_composition_rows']} rebuilt composition rows "
            f"matched the frozen flags; {summary['lineage']['missing_tuesday_totals']} "
            f"missing Tuesday totals retained as missing; {LOOKS} looks; {DRAWS} paired "
            "season/week-block resamples. Gap intervals subtract independently resampled IS "
            "from OOS. Intervals condition on fitted models."
        ),
        "",
        (
            "**Inferred:** unresolved_below_power pending orchestrator recording. Three "
            "outer seasons and retrospective sources limit generalization. Per AGENTS.md, "
            "ranked-probability gains alone cannot promote ATS sides and zero crossing "
            "closes nothing. No card change or closure."
        ),
        "",
        "## Fold coefficients and stability",
        "",
        (
            "**Measured:** natural coefficients of the tilt candidate (per point of margin "
            "distance from the line) and the four-term probability fit, before calibration."
        ),
        "",
        table(
            ["Outer", "Arm", "Intercept", *base.TERMS],
            [
                [
                    fold["outer"],
                    arm,
                    *[f"{coef[t]:.8f}" for t in ("intercept", *base.TERMS)],
                ]
                for fold in summary["folds"]
                for arm, coef in (
                    ("local_rps", fold["candidate"]["coefficients"]),
                    ("four_term", fold["four_term"]),
                )
            ],
        ),
        "",
        table(
            ["Outer", "Arm", "Selection slope", "Calibration intercept"],
            [
                [fold["outer"], arm, f"{cal['slope']:.8f}", f"{cal['intercept']:.8f}"]
                for fold in summary["folds"]
                for arm, cal in fold["calibration"].items()
            ],
        ),
        "",
        table(
            ["Outer", "Fit through", "Roles", "Iterations", "Max gradient", "Objective"],
            [
                [
                    fold["outer"],
                    fold["fit_through"],
                    json.dumps(fold["roles_with_pushes"]),
                    fold["candidate"]["iterations"],
                    f"{fold['candidate']['gradient_max_abs']:.3g}",
                    f"{fold['candidate']['objective']:.6f}",
                ]
                for fold in summary["folds"]
            ],
        ),
        "",
        "## IS, OOS and gaps",
        "",
        (
            "**Measured:** each cell is estimate [95% interval]; probability_positive. "
            "Accuracy is percentage points; other endpoints are losses. Gain rows orient "
            "every endpoint so positive favors the candidate. Gaps are OOS minus IS. Raw "
            "loss probability_positive means loss above zero, not improvement."
        ),
    ]
    for outer, panel in summary["panels"].items():
        rows = []
        for phase, cells_ in (*panel["phases"].items(), ("gap", panel["gaps"])):
            for label, measures in cells_.items():
                rows.append(
                    [
                        phase,
                        label,
                        measures.get("record", "-"),
                        *[
                            f"{number(measures[m])}; {measures[m]['probability_positive']:.5f}"
                            for m in METRICS
                        ],
                    ]
                )
        lines += ["", f"### {outer}", "", table(["Phase", "Arm/contrast", "W-L", *METRICS], rows)]
    lines += [
        "",
        "## Five equal-width reliability bands",
        "",
        "**Measured:** home-cover probabilities, non-push outer rows only.",
        "",
        table(
            ["Arm", "Band", "Games", "Mean probability", "Home cover rate"],
            [
                [
                    r["arm"],
                    r["band"],
                    r["n"],
                    "-" if r["predicted"] is None else f"{r['predicted']:.6f}",
                    "-" if r["observed"] is None else f"{r['observed']:.6f}",
                ]
                for r in summary["reliability"]
            ],
        ),
        "",
        "## Reproduction and saved rows",
        "",
        ".tools/uv.exe run --no-sync --no-cache python scripts/lead90_unit1.py",
        "",
        (
            "Prediction rows, PMFs, upstream cutoffs, hashes and intervals: "
            "tests/scratch/codex/lead90_unit1/. Serial registry commands are in "
            "docs/lanes/lead90.md and were not run by this worker."
        ),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def update_lane(summary: dict, lane_head: str) -> None:
    pooled = summary["panels"]["pooled"]
    oos = pooled["phases"]["OOS"]
    brier = oos["gain_vs_four_term"]["brier"]
    local = oos["gain_vs_four_term"]["local_rps"]
    decisive = pooled["decisive"]["four_term"]
    text = [
        lane_head.rstrip(),
        "",
        (
            f"**Measured result:** {oos['local_rps']['games']} outer games; candidate "
            f"{oos['local_rps']['record']}, four-term {oos['four_term']['record']}; decisive "
            f"{decisive['candidate_wins']}-{decisive['base_wins']}. Brier gain "
            f"{number(brier)}, probability_positive={brier['probability_positive']:.5f}; "
            f"local RPS gain {number(local)}, "
            f"probability_positive={local['probability_positive']:.5f}. "
            "**Inferred:** unresolved_below_power; no serving change."
        ),
        "",
        "## Record commands",
        "Orchestrator runs these 20 candidate-versus-four-term OOS commands serially.",
        "",
        record_commands(summary),
        "",
        "## Next",
        "Orchestrator reviews docs/lead90_unit1.md and runs the serial records.",
        "",
        "## Open",
        (
            "Three outer seasons; optimistic IS; retrospective archives; missing Tuesday "
            "totals. No closure or promotion. No Git, publication or src edits."
        ),
        "",
    ]
    LANE.write_text("\n".join(text), encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "summary.json").exists():
        raise ValueError("A scored replay already exists; do not silently rescore")
    protocol = LANE.read_text(encoding="utf-8")
    (OUTPUT / "protocol.md").write_text(protocol, encoding="utf-8")
    lane_head = protocol.split("\n## Record commands", 1)[0]
    with threadpool_limits(limits=1):
        frame, lineage = load()
        print(f"inventory passed: {len(frame)} games", flush=True)
        predictions, folds = replay(frame)
        save(predictions, "predictions.parquet")
        summary = summarize(predictions)
        summary.update(
            {"lineage": lineage, "folds": folds, "looks": LOOKS, "draws": DRAWS, "seed": SEED}
        )
        (OUTPUT / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False, default=str), encoding="utf-8"
        )
        report(summary, lane_head)
        update_lane(summary, lane_head)
    pooled = summary["panels"]["pooled"]["phases"]["OOS"]
    print(json.dumps({"status": "complete", "gain": pooled["gain_vs_four_term"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
