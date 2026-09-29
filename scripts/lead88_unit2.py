from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.joint_residual_model import UNION_FEATURES, make_joint_estimator, realised_residual_frame
from nfl_ats.lattice_centre_challenger import challenger_centre
from nfl_ats.pick_probability import PROBABILITY_EPSILON, signed_composition_flags
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    _arrest_incidents,
    _forecast_temperatures,
    _market_move_table,
    _protection_back_side,
)
from nfl_ats.score_lattice import pick_consistent_top_score, score_lattice
from nfl_ats.served_total import JOINT_TOTAL_BLEND_WEIGHT
from nfl_ats.tiebreaker import TOTAL_LOW_SIDE_SHADE_POINTS, lined_finals, weighted_median
from nfl_ats.totals import design_matrix

ROOT = Path('tests/scratch/codex/lead88_unit2')
LANE = Path('docs/lanes/lead88.md')
REPORT = Path('docs/lead88_unit2.md')
PAIRS = Path('tests/scratch/codex/lead88_unit1/paired_games.parquet')
SCHEDULE = Path('data/raw/20260908T162105Z/schedules.parquet')
BASE = Path('artifacts/pick_probability/20260929T192747Z')
OPENER = Path('artifacts/opener_evaluation/20260929T192743Z/per_game.parquet')
MARGINS = Path('artifacts/margins/20260929T192312Z/predictions.parquet')
FEATURES = Path('data/processed/game_features_weak_stack.parquet')
SERVED = Path('artifacts/margin_predictions/2026-week-04-20260929T192403Z/tiebreaker.json')
SEASONS = tuple(range(2020, 2026))
ARMS = ('candidate', 'served', 'tuesday_half_up', 'deadline_half_up')
DRAWS = 10000


def read_frame(path, columns=None):
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows],
    ])


def interval(values):
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def fmt(value, bounds=None):
    text = f'{value:.6f}'
    if bounds is not None:
        text += f' [{bounds[0]:.6f}, {bounds[1]:.6f}]'
    return text


def load_population():
    pairs = read_frame(PAIRS).sort_values(['season', 'week', 'game_id']).reset_index(drop=True)
    if len(pairs) != 1343 or pairs.game_id.duplicated().any():
        raise ValueError('Frozen paired population changed')
    if not (
        pairs.tuesday_last_observed.le(pairs.freeze).all()
        and pairs.deadline_last_observed.le(pairs.deadline).all()
        and pairs.deadline.lt(pairs.kickoff).all()
    ):
        raise ValueError('Total quote clock violation')
    schedules = read_frame(SCHEDULE)
    pairs = pairs.merge(schedules[['game_id', 'home_score', 'away_score']], on='game_id', how='left', validate='one_to_one')
    pairs['actual_total'] = pairs.home_score + pairs.away_score
    if pairs.actual_total.isna().any():
        raise ValueError('Missing final score')
    opener = read_frame(OPENER, [
        'game_id', 'tue_open_home_spread', 'home_cover_probability_at_open',
        'residual_at_open_served', 'margin_vs_open',
    ])
    pairs = pairs.merge(opener, on='game_id', how='left', validate='one_to_one')
    if pairs.home_cover_probability_at_open.isna().any():
        raise ValueError('Missing opener probability, including a retained push')
    pairs['spread_line'] = -pairs.tue_open_home_spread
    pairs['predicted_margin'] = pairs.spread_line + pairs.residual_at_open_served
    margin = read_frame(MARGINS, ['game_id', 'method', 'model_name', 'train_max_gameday'])
    margin = margin.loc[margin.method.eq('market_residual') & margin.model_name.eq('ridge')]
    pairs['margin_training_cutoff'] = pairs.game_id.map(margin.set_index('game_id').train_max_gameday)
    completed = pd.to_datetime(pairs.margin_training_cutoff, utc=True) + pd.Timedelta(days=1)
    if not completed.le(pairs.deadline).all():
        raise ValueError('Margin training extends beyond the deadline')
    metadata = json.loads((BASE / 'metadata.json').read_text(encoding='utf-8'))
    cached = read_frame(BASE / 'per_game.parquet', ['game_id', *FIT_FEATURES, 'out_of_season_home_probability'])
    pairs = pairs.merge(cached, on='game_id', how='left', validate='one_to_one')
    missing = pairs.out_of_season_home_probability.isna()
    if int(missing.sum()) != 33:
        raise ValueError('Unexpected missing four-term population')
    print('Reconstructing 33 four-term rows using the production feature recipe', flush=True)
    flags = signed_composition_flags(
        pairs, schedules,
        incidents=_arrest_incidents(Path('data')),
        forecasts_tuesday_noon=_forecast_temperatures(Path('data')),
        protection_back_side=_protection_back_side(schedules, Path('data')),
    ).set_index('game_id')
    restored_flags = pairs.game_id.map(flags.composition_flag_sum)
    if not np.array_equal(restored_flags.loc[~missing].to_numpy(), pairs.loc[~missing, 'composition_flag_sum'].to_numpy()):
        raise ValueError('Current flags do not reproduce the frozen four-term source')
    moves, move_source = _market_move_table(Path('artifacts'), metadata['market_move_feature_version'])
    restored_moves = pairs.game_id.map(moves.set_index('game_id').leader_median_net)
    model_p = pairs.home_cover_probability_at_open.clip(PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    pairs.loc[missing, 'model_logit'] = np.log(model_p.loc[missing] / (1 - model_p.loc[missing]))
    pairs.loc[missing, 'composition_flag_sum'] = restored_flags.loc[missing]
    pairs.loc[missing, 'market_move_available'] = restored_moves.loc[missing].notna().astype(float)
    pairs.loc[missing, 'market_move_toward_home'] = restored_moves.loc[missing].fillna(0.0)
    pairs['home_probability'] = np.nan
    reproduction = []
    for year in SEASONS:
        selected = pairs.season.eq(year)
        coefficient = metadata['fold_coefficients'][str(year)]
        z = np.full(int(selected.sum()), coefficient['intercept'])
        for term in FIT_FEATURES:
            z += coefficient[term] * pairs.loc[selected, term].to_numpy()
        pairs.loc[selected, 'home_probability'] = expit(z)
        known = selected & ~missing
        reproduction.extend(np.abs(pairs.loc[known, 'home_probability'] - pairs.loc[known, 'out_of_season_home_probability']).tolist())
    if max(reproduction) > 1e-12:
        raise ValueError('Frozen out-of-season probabilities were not reproduced')
    pairs['four_term_restored'] = missing
    pairs['pick_side'] = np.where(pairs.home_probability.ge(0.5), 'HOME', 'AWAY')
    pairs['centre_margin'] = [
        challenger_centre(row.predicted_margin, row.spread_line, row.pick_side).centre for row in pairs.itertuples()
    ]
    facts = {
        'games': len(pairs), 'last_games': int(pairs.is_last_game.sum()), 'restored_four_term': int(missing.sum()),
        'restored_opener_pushes': int(pairs.loc[missing, 'margin_vs_open'].eq(0).sum()),
        'probability_reproduction_max_error': max(reproduction),
        'margin_cutoff_before_tuesday': int(completed.le(pairs.freeze).sum()),
        'margin_cutoff_before_deadline': int(completed.le(pairs.deadline).sum()),
        'four_term_fold_coefficients': metadata['fold_coefficients'], 'market_move_source': move_source,
    }
    return pairs, schedules, facts


def reconstruct_totals(pairs):
    features = read_frame(FEATURES)
    population = realised_residual_frame(features)
    indexed = features.set_index('game_id', drop=False)
    pairs['joint_residual'] = np.nan
    pairs['joint_train_games'] = 0
    pairs['joint_training_cutoff'] = ''
    fits = 0
    for (year, week), held in pairs.groupby(['season', 'week'], sort=True):
        prior = population.loc[
            (population.season.lt(year) | (population.season.eq(year) & population.week.lt(week)))
            & population.gameday.lt(pd.Timestamp(held.freeze.min()).tz_convert(None).normalize())
        ]
        if len(prior) < DEFAULT_MIN_TRAIN_GAMES:
            raise ValueError(f'Cannot reconstruct joint baseline for {year}/{week}')
        target = indexed.loc[held.game_id].copy()
        target['spread_line'] = held.spread_line.to_numpy()
        target['total_line'] = held.tuesday_total.to_numpy()
        estimator = make_joint_estimator()
        estimator.fit(design_matrix(prior, UNION_FEATURES), prior[['margin_residual', 'total_residual']].to_numpy(dtype=float))
        predicted = np.asarray(estimator.predict(design_matrix(target, UNION_FEATURES)))
        pairs.loc[held.index, 'joint_residual'] = predicted[:, 1]
        pairs.loc[held.index, 'joint_train_games'] = len(prior)
        pairs.loc[held.index, 'joint_training_cutoff'] = str(prior.gameday.max().date())
        fits += 1
    pairs['served_centre_total'] = pairs.tuesday_total + JOINT_TOTAL_BLEND_WEIGHT * pairs.joint_residual + TOTAL_LOW_SIDE_SHADE_POINTS
    return fits


def choose_score(finals, row, total, allow_missing=False):
    lattice = score_lattice(finals, row.centre_margin, float(total))
    chosen = pick_consistent_top_score(
        lattice, pick_side=row.pick_side, spread_line=row.spread_line,
        served_total=float(total), centre_margin=row.centre_margin,
    )
    if chosen is None:
        if allow_missing:
            return np.nan, np.nan
        raise ValueError(f'No admissible score for {row.game_id} at {total}')
    return chosen[:2]


def make_prior_finals(schedules, pairs):
    finals = lined_finals(schedules).copy()
    dates = pd.to_datetime(finals.gameday, utc=True)
    result = {}
    for key, held in pairs.groupby(['season', 'week']):
        prior = finals.loc[dates.add(pd.Timedelta(days=1)).le(held.freeze.min())].copy()
        if prior.empty:
            raise ValueError('Empty pre-Tuesday lattice history')
        result[key] = prior
    return result


def make_served(pairs, prior):
    scores = [choose_score(prior[(row.season, row.week)], row, row.served_centre_total, allow_missing=True) for row in pairs.itertuples()]
    pairs[['guess_home', 'guess_away']] = np.asarray(scores)
    pairs['served'] = pairs.guess_home + pairs.guess_away
    pairs['baseline_status'] = np.where(pairs.served.notna(), 'available', 'production_lattice_no_admissible_score')
    pairs['tuesday_half_up'] = np.floor(pairs.tuesday_total + 0.5).astype(int)
    pairs['deadline_half_up'] = np.floor(pairs.deadline_total + 0.5).astype(int)


def fit_response(frame):
    moving = frame.loc[frame.total_move.ne(0)]
    ratios = (moving.actual_total - moving.served) / moving.total_move
    return weighted_median(ratios.to_numpy(), np.abs(moving.total_move.to_numpy()))


def predict_response(frame, prior, coefficient):
    scores = [
        choose_score(prior[(row.season, row.week)], row, row.served + coefficient * row.total_move)
        for row in frame.itertuples()
    ]
    return np.asarray(scores)


def replay(pairs, prior):
    folds = []
    for column in ['candidate', 'candidate_home', 'candidate_away', 'response_coefficient']:
        pairs[column] = np.nan
    for year in SEASONS:
        train, held = pairs.loc[pairs.season.ne(year)], pairs.loc[pairs.season.eq(year)]
        coefficient = fit_response(train)
        scores = predict_response(held, prior, coefficient)
        pairs.loc[held.index, ['candidate_home', 'candidate_away']] = scores
        pairs.loc[held.index, 'candidate'] = scores.sum(axis=1)
        pairs.loc[held.index, 'response_coefficient'] = coefficient
        folds.append({'season': year, 'train_games': len(train), 'held_games': len(held), 'b': coefficient})
        print(f'Completed frozen response fold {year}', flush=True)
    full_coefficient = fit_response(pairs)
    pairs['candidate_is'] = predict_response(pairs, prior, full_coefficient).sum(axis=1)
    for arm in (*ARMS, 'candidate_is'):
        pairs[arm + '_error'] = np.abs(pairs[arm] - pairs.actual_total)
    return folds, full_coefficient


def bootstrap_panel(frame, rng, optimistic=False):
    blocks = frame.groupby(['season', 'week'], sort=True)
    block_counts = blocks.size()
    columns = [arm + '_error' for arm in ARMS] + (['candidate_is_error'] if optimistic else [])
    block_errors = blocks[columns].sum()
    weights = np.zeros((DRAWS, len(block_counts)), dtype=np.int16)
    for year in sorted(frame.season.unique()):
        selected = np.flatnonzero(block_counts.index.get_level_values('season') == year)
        weights[:, selected] = rng.multinomial(len(selected), np.full(len(selected), 1 / len(selected)), DRAWS)
    counts = weights @ block_counts.to_numpy(dtype=float)
    errors = (weights @ block_errors.to_numpy()) / counts[:, None]
    result = {'n': len(frame), 'blocks': len(block_counts), 'arms': {}}
    for index, arm in enumerate(ARMS):
        result['arms'][arm] = {'mae': float(frame[arm + '_error'].mean()), 'interval': interval(errors[:, index])}
    improvements = errors[:, 1] - errors[:, 0]
    result['improvement'] = {
        'effect': float((frame.served_error - frame.candidate_error).mean()), 'interval': interval(improvements),
        'standard_error': float(np.std(improvements, ddof=1)),
        'probability_positive': float(np.mean(improvements > 0) + 0.5 * np.mean(improvements == 0)),
    }
    if optimistic:
        in_sample = errors[:, 1] - errors[:, -1]
        result['is_improvement'] = float((frame.served_error - frame.candidate_is_error).mean())
        result['is_interval'] = interval(in_sample)
        result['is_mae'] = float(frame.candidate_is_error.mean())
        result['is_mae_interval'] = interval(errors[:, -1])
        result['oos_minus_is_improvement'] = result['improvement']['effect'] - result['is_improvement']
        result['gap_interval'] = interval(improvements - in_sample)
    return result


def score_results(pairs):
    rng = np.random.default_rng(88)
    result = {}
    for panel in ['pooled', *SEASONS]:
        scope = pairs if panel == 'pooled' else pairs.loc[pairs.season.eq(panel)]
        result[str(panel)] = {
            'all': bootstrap_panel(scope, rng, panel == 'pooled'),
            'last': bootstrap_panel(scope.loc[scope.is_last_game], rng, panel == 'pooled'),
        }
    last = pairs.loc[pairs.is_last_game]
    gain = last.served_error - last.candidate_error
    better, worse, tied = int(gain.gt(0).sum()), int(gain.lt(0).sum()), int(gain.eq(0).sum())
    exact = binomtest(better, better + worse, p=0.5, alternative='two-sided')
    share = binomtest(better, len(last)).proportion_ci(confidence_level=0.95, method='exact')
    decisive_ci = exact.proportion_ci(confidence_level=0.95, method='exact')
    result['closer'] = {
        'better': better, 'worse': worse, 'tied': tied, 'all_last_games': len(last),
        'unconditional_closer_share': better / len(last), 'unconditional_interval': [share.low, share.high],
        'decisive_closer_share': better / (better + worse),
        'decisive_interval': [decisive_ci.low, decisive_ci.high], 'exact_two_sided_p': exact.pvalue,
    }
    return result


def record_command(population, panel):
    effect = panel['improvement']
    meaning = 'the final game of the week' if population == 'last' else 'games with a served score guess'
    parts = [
        '.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record',
        f'  --name lead88_unit2_{population}_mae --family lead88_total_response',
        f"  --description 'Fitted total response plus lattice reprojection versus served guess on {meaning}; 82 looks'",
        '  --source docs/lead88_unit2.md --league nfl --season-start 2020 --season-end 2025',
        f"  --effect {effect['effect']:.12g} --effect-units mae_improvement",
        f"  --interval-low {effect['interval'][0]:.12g} --interval-high {effect['interval'][1]:.12g}",
        f"  --standard-error {effect['standard_error']:.12g} --probability-positive {effect['probability_positive']:.12g}",
        f"  --sample-games {panel['n']} --sample-blocks {panel['blocks']} --classification unresolved_below_power",
        "  --classification-evidence 'Retrospective LOSO; 82 looks; zero-move guesses can change; no power control or mechanism refutation'",
        f"  --plain-summary 'For {meaning}, this study nudges the score guess using changing sportsbook totals, then rounds the scores again. Keep the current guess while the effects of the nudge and rounding remain unresolved.'",
    ]
    return (' ' + chr(92) + '\n').join(parts)


def write_report(facts, folds, full_coefficient, results, declaration):
    closer = results['closer']
    lines = [
        '# LEAD-88 unit 2: fitted response to total news', '',
        '**Measured:** one replay of the protocol saved before outcome access in `docs/lanes/lead88.md`.',
        '`UV_CACHE_DIR=tests/scratch/codex/lead88_unit2/uv-cache .tools/uv.exe run --no-sync python scripts/lead88_unit2.py`.',
        'No serving change or registry command was executed.', '',
        '## Primary result and exact closer comparison', '',
        f"**Measured:** on {closer['all_last_games']} last games, candidate is closer {closer['better']} times,",
        f"farther {closer['worse']} times, and tied {closer['tied']} times.",
        f"Unconditional closer share: {fmt(100 * closer['unconditional_closer_share'], [100 * v for v in closer['unconditional_interval']])}% (exact 95% interval).",
        f"Among unequal errors: {fmt(100 * closer['decisive_closer_share'], [100 * v for v in closer['decisive_interval']])}%;",
        f"exact two-sided binomial null p={closer['exact_two_sided_p']:.6f}, null closer probability 0.5.",
        'The exact null conditions on unequal errors; ties remain in the unconditional share and MAE.', '',
        '## Held-out MAE against actual combined score', '',
        f"**Measured:** {facts['scored_games']}/{facts['games']} paired games and {facts['scored_last_games']}/{facts['last_games']} last games have a defined served guess.",
        f"The production guard declines {len(facts['unavailable_baseline_games'])} historical guesses; all rows remain in the artifact,",
        f"including all 33 missing probability rows ({facts['restored_scored']} scoreable). No outcome-based exclusion was made.",
        'All-game MAE below means every paired game with a defined production baseline. The declared full-population',
        'contrast is not identifiable where that baseline declines a guess; this coverage deviation is not a protocol retune.',
        f"**Measured diagnostic:** {facts['zero_move_changed_guesses']}/{facts['zero_move_games']} zero-move games changed integer guesses",
        f"({facts['zero_move_last_changed']} last games) because the declared candidate recentres the lattice on the served integer total.",
        '**Inferred:** the measured effect combines total-news response with score reprojection; it does not isolate',
        'the news mechanism. No alternative mapping was scored after this discovery. A new declaration must',
        'preserve the served guess when the fitted adjustment is zero before any further outcome comparison.',
        '**Measured:** lower MAE is better; paired week-block 95% percentile intervals.',
        table(['Population', 'Games', 'Arm', 'MAE [95% interval]'], [
            [pop, results['pooled'][pop]['n'], arm,
             fmt(results['pooled'][pop]['arms'][arm]['mae'], results['pooled'][pop]['arms'][arm]['interval'])]
            for pop in ['last', 'all'] for arm in ARMS
        ]), '',
        table(['Population', 'Served MAE minus candidate MAE [95% interval]', 'probability_positive'], [
            [pop, fmt(results['pooled'][pop]['improvement']['effect'], results['pooled'][pop]['improvement']['interval']),
             fmt(results['pooled'][pop]['improvement']['probability_positive'])] for pop in ['last', 'all']
        ]), '',
        '**Inferred:** proposed status `unresolved_below_power`; no pool-rank gain or serving change is established.',
        'AGENTS.md requires a refuted mechanism or a powered positive control to close a signal; neither was',
        'established here. The registry commands await orchestrator execution; this is not a settled verdict.', '',
        '## Training, folds and stability', '',
        f'**Measured:** full-data optimistic IS coefficient b={full_coefficient:.9f}.',
        table(['Held season', 'Train games', 'Held games', 'b', 'Last MAE gain [95% interval]', 'Last probability_positive', 'All MAE gain [95% interval]'], [
            [fold['season'], fold['train_games'], fold['held_games'], fmt(fold['b']),
             fmt(results[str(fold['season'])]['last']['improvement']['effect'], results[str(fold['season'])]['last']['improvement']['interval']),
             fmt(results[str(fold['season'])]['last']['improvement']['probability_positive']),
             fmt(results[str(fold['season'])]['all']['improvement']['effect'], results[str(fold['season'])]['all']['improvement']['interval'])]
            for fold in folds
        ]), '',
        table(['Population', 'Candidate IS MAE [95% interval]', 'IS gain [95% interval]', 'OOS gain', 'OOS minus IS gain [95% interval]'], [
            [pop, fmt(results['pooled'][pop]['is_mae'], results['pooled'][pop]['is_mae_interval']),
             fmt(results['pooled'][pop]['is_improvement'], results['pooled'][pop]['is_interval']),
             fmt(results['pooled'][pop]['improvement']['effect']),
             fmt(results['pooled'][pop]['oos_minus_is_improvement'], results['pooled'][pop]['gap_interval'])]
            for pop in ['last', 'all']
        ]), '',
        '## Baseline, retained games and lineage', '',
        '**Read:** `artifacts/margin_predictions/2026-week-04-20260929T192403Z/tiebreaker.json:9`',
        'records integer guesses; lines 15 and 24-27 name the lattice, joint total method and one-point shade.',
        '`src/nfl_ats/tiebreaker.py:385` and `src/nfl_ats/score_lattice.py:278` define the serving recipe.',
        '**Measured:** this replay reconstructs that recipe historically; these are not archived served cards.',
        'Tuesday total + 0.1 x joint total residual - 1 is the baseline continuous total. Production',
        '`challenger_centre`, `score_lattice` and `pick_consistent_top_score` select the integer score.',
        'Margin uses the served opener residual; side uses the frozen four-term LOSO probability.',
        'Candidate centre is served integer total + b x observed total move, under the same margin/side guards.',
        'Joint hyperparameters and shading are inherited from serving, not selected on replay outcomes.',
        'Joint models fit only earlier-week finals before Tuesday; target spread/total inputs use opener/Tuesday',
        'quotes. Lattice histories also end before Tuesday. No pool captures before 2026 are required.',
        f"**Measured:** {facts['games']} paired games / {facts['last_games']} last games; all {facts['restored_four_term']} missing",
        f"four-term rows retained ({facts['restored_opener_pushes']} opener pushes). Frozen probability reproduction",
        f"maximum error {facts['probability_reproduction_max_error']:.3g}; reconstructed composition flags match cached rows.",
        f"Margin training calendar-day cutoffs precede Tuesday for {facts['margin_cutoff_before_tuesday']} games and",
        f"deadline for all {facts['margin_cutoff_before_deadline']}. Joint baseline reconstruction fits: {facts['joint_reconstruction_fits']}.",
        '**Inferred limitations:** upstream probability/margin/features are retrospective archives.',
        'A prior training date does not prove feature-vintage or completion-time availability. Four-term and',
        'response LOSO fits include future seasons for earlier holdouts; this is not a prospective rolling test.',
        'Baseline histories for later games can contain another response-fold season. This is conditional',
        'evaluation of a fixed served recipe, not independently nested validation of its upstream training.',
        'The inherited low-side shade is not revalidated here. Total-quote clocks are verified; upstream',
        'forecast-feature clocks are not newly certified. No new ATS scoring or probability calibration is claimed.', '',
        '## Frozen protocol and look accounting', '',
        declaration.split('## Protocol', 1)[1].split('## Tried', 1)[0].strip(), '',
        '**Measured:** 82 reporting looks, one response specification, seven b fits. Fixed baseline reconstruction',
        'fits are disclosed separately; they are not tuned arms. IS MAE and gain express the same two IS comparisons.',
        '10,000 paired season-stratified week-block draws, seed 88, fixed predictions; exact zero draws get half weight',
        'in probability_positive. Intervals omit training/model-selection uncertainty; no multiplicity adjustment.', '',
        '## Season MAE panels', '',
        table(['Season', 'Population', 'Arm', 'MAE [95% interval]'], [
            [year, pop, arm, fmt(results[str(year)][pop]['arms'][arm]['mae'], results[str(year)][pop]['arms'][arm]['interval'])]
            for year in SEASONS for pop in ['last', 'all'] for arm in ARMS
        ]), '',
        '## Saved evidence', '',
        '`tests/scratch/codex/lead88_unit2/predictions.parquet`: every game, guesses, outcomes and response fold.',
        '`summary.json`: coefficients, panels, exact null and hashes; `protocol.md`: pre-outcome declaration.',
        'Record commands are in the lane only and require serial orchestrator execution.',
        '**Measured:** initial command failed in baseline construction before any response fit or score. The sole',
        'response replay then retained unavailable-baseline rows as unscorable instead of fabricating served guesses.',
    ]
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def finish_lane(results, facts):
    last, all_games = [results['pooled'][pop]['improvement'] for pop in ['last', 'all']]
    lines = [
        '# LEAD-88 - observed total news', '', '## Goal',
        'Replay one fitted total-move response against the served score-lattice guess; research only.', '',
        '## State',
        f"**Measured:** one response replay complete; {results['pooled']['all']['n']}/1,343 games and {results['pooled']['last']['n']}/101 last games scored.",
        'All 33 missing base rows retained; 10 production-declined guesses remain in scratch evidence, outside paired MAE.',
        f"Diagnostic: {facts['zero_move_changed_guesses']} zero-move guesses changed through lattice reprojection; this mixes mechanisms.",
        f"Last-game MAE gain {fmt(last['effect'], last['interval'])}; probability_positive={last['probability_positive']:.4f}.",
        f"All-game gain {fmt(all_games['effect'], all_games['interval'])}; probability_positive={all_games['probability_positive']:.4f}.",
        'Proposed unresolved status; report `docs/lead88_unit2.md`; no registry writes.', '',
        '## Protocol (frozen before outcomes)',
        'Owner amendment: baseline is served lattice guess; half-up market total is a comparator only.',
        'One LAD response to Tuesday-to-deadline total move, six LOSO seasons 2020-2025, pushes retained.',
        'One fixed fitted probability selects sides; no standalone flips. 82 bounded looks within parent 713 looks.',
        'Full declaration saved before scoring in `tests/scratch/codex/lead88_unit2/protocol.md`, copied in report.', '',
        '## Tried',
        '**Measured:** `.tools/uv.exe run --no-sync python scripts/lead88_unit2.py`; local UV_CACHE_DIR used.',
        'Frozen quote clocks, base probability reproduction and retained-row checks executed in the real command.', '',
        '## Record commands',
        'Orchestrator only; candidate versus served, execute serially.', '```bash',
        record_command('last', results['pooled']['last']), '', record_command('all', results['pooled']['all']), '```', '',
        '## Next', 'Orchestrator reviews/records; next unit must predeclare a zero-adjustment identity before further scoring.', '',
        '## Open',
        'Retrospective upstream/feature-vintage limits remain. No prospective or pool-rank claim; zero crossing closes nothing.',
        'No serving change, registry write, publication, new tests, commit or push by this worker.',
    ]
    LANE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run():
    declaration = LANE.read_text(encoding='utf-8-sig')
    if 'frozen before outcomes' not in declaration or '82 looks' not in declaration:
        raise ValueError('Pre-outcome protocol is absent')
    ROOT.mkdir(parents=True, exist_ok=True)
    marker = ROOT / 'response_replay_started.json'
    if marker.exists():
        raise ValueError('The declared response replay already began; inspect its saved artifacts')
    protocol_path = ROOT / 'protocol.md'
    if not protocol_path.exists():
        protocol_path.write_text(declaration, encoding='utf-8')
    elif protocol_path.read_text(encoding='utf-8') != declaration:
        raise ValueError('The pre-outcome protocol changed')
    sources = [PAIRS, SCHEDULE, OPENER, MARGINS, FEATURES, SERVED, BASE / 'metadata.json', BASE / 'per_game.parquet']
    hashes = {path.as_posix(): digest(path) for path in sources}
    pairs, schedules, facts = load_population()
    print('Reconstructing frozen served joint totals', flush=True)
    facts['joint_reconstruction_fits'] = reconstruct_totals(pairs)
    prior = make_prior_finals(schedules, pairs)
    print('Reconstructing integer served score-lattice guesses', flush=True)
    make_served(pairs, prior)
    pairs.to_parquet(ROOT / 'baseline.parquet', index=False)
    marker.write_text(json.dumps({'protocol_sha256': digest(protocol_path)}) + '\n', encoding='utf-8')
    scored = pairs.loc[pairs.served.notna()].copy()
    facts['scored_games'] = len(scored)
    facts['scored_last_games'] = int(scored.is_last_game.sum())
    facts['restored_scored'] = int(scored.four_term_restored.sum())
    facts['unavailable_baseline_games'] = pairs.loc[pairs.served.isna(), ['game_id', 'is_last_game', 'four_term_restored']].to_dict('records')
    folds, full_coefficient = replay(scored, prior)
    zero_move = scored.total_move.eq(0)
    changed = scored.candidate.ne(scored.served)
    facts['zero_move_games'] = int(zero_move.sum())
    facts['zero_move_changed_guesses'] = int((zero_move & changed).sum())
    facts['zero_move_last_changed'] = int((zero_move & changed & scored.is_last_game).sum())
    added = [column for column in scored if column not in pairs]
    pairs = pairs.merge(scored[['game_id', *added]], on='game_id', how='left', validate='one_to_one')
    pairs.to_parquet(ROOT / 'predictions.parquet', index=False)
    results = score_results(scored)
    if pairs.four_term_restored.sum() != 33 or not np.isfinite(scored.candidate).all():
        raise ValueError('Paired replay lost games')
    payload = {
        'protocol_sha256': digest(protocol_path), 'source_hashes': hashes,
        'facts': facts, 'folds': folds, 'full_sample_b': full_coefficient, 'results': results,
        'looks': 82, 'parent_protocol_looks': 713, 'bootstrap_draws': DRAWS, 'seed': 88,
    }
    (ROOT / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_report(facts, folds, full_coefficient, results, declaration)
    finish_lane(results, facts)
    print(json.dumps({'games': len(pairs), 'last': results['pooled']['last'], 'all': results['pooled']['all'], 'closer': results['closer']}), flush=True)


if __name__ == '__main__':
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    with threadpool_limits(limits=1):
        run()
