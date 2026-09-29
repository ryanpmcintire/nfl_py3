from __future__ import annotations

import json
import os
import sys
import textwrap
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '2'

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mod22_unit4 import OUT_STATUSES, SNAPSHOT, load_games, norm, starters
from players_on_field_rating_eval import (
    ARTIFACTS_ROOT, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, DATA_ROOT,
    INTERVAL_LEVEL, RELIABILITY_EDGES, REPO_ROOT, in_sample_fit, loso, signal_metrics,
)
from players_on_field_rating_unit2 import evaluate_candidate

from nfl_ats.nfl_week import week_cycle_sunday
from nfl_ats.pick_probability import active_market_move_feature_version
from nfl_ats.pick_probability_fit import FIT_FEATURES, build_fit_population

PBP_ROOT = DATA_ROOT / 'pbp' / 'raw' / '20260812T142851Z'
OUTPUT_ROOT = REPO_ROOT / 'tests' / 'scratch' / 'codex' / 'mod22_unit5'
TERM = 'qb_quality_loss_diff'
REPLACEMENT_QUALITY = -0.15
SEASONS = tuple(range(2020, 2026))


def prior_quality(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in range(2019, 2025):
        plays = pd.read_parquet(
            PBP_ROOT / f'season={season}' / 'plays.parquet',
            columns=['game_id', 'season', 'season_type', 'passer_player_id',
                     'qb_dropback', 'epa', 'play_type'],
        )
        plays = plays.loc[
            plays['season_type'].eq('REG') & plays['qb_dropback'].eq(1)
            & plays['passer_player_id'].notna() & plays['epa'].notna()
            & plays['play_type'].ne('no_play')
        ].copy()
        assert plays['season'].eq(season).all()
        plays = plays.merge(games[['game_id', 'kickoff']], on='game_id', how='left', validate='many_to_one')
        assert plays['kickoff'].notna().all()
        grouped = plays.groupby('passer_player_id').agg(
            quality=('epa', 'mean'), dropbacks=('epa', 'size'),
            quality_source_at=('kickoff', 'max'),
        ).reset_index().rename(columns={'passer_player_id': 'gsis_id'})
        grouped['quality_season'] = season
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def build_term(games: pd.DataFrame) -> pd.DataFrame:
    quality = prior_quality(games).set_index(['quality_season', 'gsis_id'])
    starts_by_team = {team: frame for team, frame in starters(games).groupby('team')}
    rosters = pd.read_parquet(SNAPSHOT / 'weekly_rosters.parquet')
    rosters = rosters.loc[rosters['position'].eq('QB') & rosters['game_type'].eq('REG')].copy()
    rosters['key'] = rosters['full_name'].map(norm)
    team_games = pd.concat([
        games[['season', 'week', 'kickoff', f'{side}_team']].rename(columns={f'{side}_team': 'team'})
        for side in ('home', 'away')
    ], ignore_index=True)
    rosters = rosters.merge(team_games, on=['season', 'week', 'team'], validate='many_to_one')
    roster_by_team = {team: frame for team, frame in rosters.groupby('team')}
    snaps = pd.read_parquet(
        SNAPSHOT / 'snap_counts.parquet',
        columns=['game_id', 'team', 'position', 'player', 'pfr_player_id', 'offense_snaps'],
    )
    snaps = snaps.loc[snaps['position'].eq('QB')].copy()
    snaps['key'] = snaps['player'].map(norm)
    snaps = snaps.merge(games[['game_id', 'kickoff']], on='game_id', validate='many_to_one')
    snaps_by_team = {team: frame for team, frame in snaps.groupby('team')}
    injuries = pd.read_parquet(
        SNAPSHOT / 'injuries.parquet',
        columns=['season', 'week', 'team', 'gsis_id', 'report_status', 'date_modified',
                 'effective_observed_at', 'observed_at_basis', 'observed_at_is_proxy'],
    )
    injuries['effective_observed_at'] = pd.to_datetime(
        injuries['effective_observed_at'], utc=True, errors='coerce'
    )
    injuries['date_modified'] = pd.to_datetime(injuries['date_modified'], utc=True, errors='coerce')
    evidenced = (
        injuries['observed_at_basis'].eq('date_modified')
        & injuries['observed_at_is_proxy'].eq(False)
        & injuries['date_modified'].notna()
        & injuries['effective_observed_at'].eq(injuries['date_modified'])
    )
    injuries['evidenced_report_at'] = injuries['date_modified'].where(evidenced)
    injuries_by_game = {key: frame for key, frame in injuries.groupby(['season', 'week', 'team'])}
    week_sundays = games.groupby(['season', 'week'])['kickoff'].min().map(
        lambda value: week_cycle_sunday(value.tz_convert('America/New_York').date())
    )
    rows = []
    for game in games.loc[games['season'].isin(SEASONS)].itertuples():
        sunday = week_sundays.loc[(game.season, game.week)]
        freeze = pd.Timestamp(datetime.combine(
            sunday - timedelta(days=5), time(12), tzinfo=ZoneInfo('America/New_York')
        )).tz_convert('UTC')
        assert freeze < game.decision_at < game.kickoff
        row = {'game_id': game.game_id, 'decision_at': game.decision_at, 'freeze_at': freeze}
        for side, team in (('home', game.home_team), ('away', game.away_team)):
            values = {
                'quality_loss': 0.0, 'late_out': False, 'starter_matched': False,
                'starter_id': None, 'replacement_id': None, 'prior_kickoff': pd.NaT,
                'roster_at': pd.NaT, 'injury_at': pd.NaT, 'first_out_at': pd.NaT,
                'starter_cutoff_only_out': False, 'excluded_proxy_reports': 0,
                'excluded_unevidenced_reports': 0, 'excluded_late_reports': 0,
                'starter_quality': REPLACEMENT_QUALITY,
                'replacement_quality': REPLACEMENT_QUALITY,
                'starter_dropbacks': 0, 'replacement_dropbacks': 0,
                'starter_quality_at': pd.NaT, 'replacement_quality_at': pd.NaT,
                'replacement_prior_snaps': 0.0,
            }
            history = starts_by_team.get(team, pd.DataFrame())
            prior = history.loc[history['kickoff'] < game.kickoff] if not history.empty else history
            roster = roster_by_team.get(team, pd.DataFrame())
            prior_roster = roster.loc[roster['kickoff'] < game.decision_at] if not roster.empty else roster
            if not prior.empty and not prior_roster.empty:
                starter = prior.iloc[-1]
                assert starter['kickoff'] < game.decision_at
                values['prior_kickoff'] = starter['kickoff']
                assert (prior_roster['kickoff'] < game.decision_at).all()
                snap_history = snaps_by_team[team]
                snap_history = snap_history.loc[snap_history['kickoff'] < game.decision_at]
                assert (snap_history['kickoff'] < game.decision_at).all()
                starter_snaps = snap_history.loc[
                    snap_history['game_id'].eq(starter['game_id'])
                    & snap_history['key'].eq(norm(starter['player']))
                ]
                starter_pfr = set(starter_snaps['pfr_player_id'].dropna())
                identity = prior_roster.loc[
                    prior_roster['key'].eq(norm(starter['player']))
                    | prior_roster['pfr_id'].isin(starter_pfr)
                ].dropna(subset=['gsis_id'])
                if not identity.empty:
                    ids = identity['gsis_id'].unique()
                    assert len(ids) == 1, f'Ambiguous starter identity: {game.game_id}/{team}'
                    starter_id = str(ids[0])
                    values['starter_id'] = starter_id
                    values['starter_matched'] = True
                    injury = injuries_by_game.get((game.season, game.week, team), injuries.iloc[:0])
                    starter_out = injury.loc[
                        injury['gsis_id'].eq(starter_id) & injury['report_status'].isin(OUT_STATUSES)
                    ]
                    values['starter_cutoff_only_out'] = bool(
                        starter_out['effective_observed_at'].le(game.decision_at).any()
                    )
                    values['excluded_proxy_reports'] = int(starter_out['observed_at_is_proxy'].eq(True).sum())
                    values['excluded_unevidenced_reports'] = int(starter_out['evidenced_report_at'].isna().sum())
                    values['excluded_late_reports'] = int(starter_out['evidenced_report_at'].gt(game.decision_at).sum())
                    visible = injury.loc[injury['evidenced_report_at'].le(game.decision_at)]
                    assert visible['observed_at_basis'].eq('date_modified').all()
                    assert visible['observed_at_is_proxy'].eq(False).all()
                    assert visible['date_modified'].le(game.decision_at).all()
                    latest = visible.sort_values('evidenced_report_at').drop_duplicates('gsis_id', keep='last')
                    starter_report = latest.loc[latest['gsis_id'].eq(starter_id)]
                    unavailable = set(latest.loc[latest['report_status'].isin(OUT_STATUSES), 'gsis_id'])
                    first_out = visible.loc[
                        visible['gsis_id'].eq(starter_id) & visible['report_status'].isin(OUT_STATUSES),
                        'evidenced_report_at',
                    ].min()
                    late_out = (
                        not starter_report.empty
                        and starter_report.iloc[0]['report_status'] in OUT_STATUSES
                        and pd.notna(first_out) and first_out > freeze
                    )
                    values['late_out'] = bool(late_out)
                    if late_out:
                        values['first_out_at'] = first_out
                        values['injury_at'] = starter_report.iloc[0]['evidenced_report_at']
                        assert freeze < first_out <= values['injury_at'] <= game.decision_at
                        latest_roster = prior_roster.loc[prior_roster['kickoff'].eq(prior_roster['kickoff'].max())]
                        values['roster_at'] = latest_roster['kickoff'].max()
                        backups = latest_roster.loc[
                            latest_roster['status'].isin(['ACT', 'INA'])
                            & latest_roster['gsis_id'].notna()
                            & ~latest_roster['gsis_id'].isin(unavailable | {starter_id})
                        ].drop_duplicates('gsis_id').copy()
                        by_pfr = snap_history.groupby('pfr_player_id')['offense_snaps'].sum()
                        by_name = snap_history.groupby('key')['offense_snaps'].sum()
                        backups['prior_snaps'] = backups['pfr_id'].map(by_pfr).fillna(backups['key'].map(by_name)).fillna(0.0)
                        backups = backups.sort_values(['prior_snaps', 'gsis_id'], ascending=[False, True])
                        if not backups.empty:
                            values['replacement_id'] = str(backups.iloc[0]['gsis_id'])
                            values['replacement_prior_snaps'] = float(backups.iloc[0]['prior_snaps'])
                        for role in ('starter', 'replacement'):
                            source_season = int(game.season) - 1
                            assert source_season < game.season
                            key = (source_season, values[f'{role}_id'])
                            if key in quality.index:
                                state = quality.loc[key]
                                assert state['quality_source_at'] < game.decision_at
                                values[f'{role}_quality'] = float(state['quality'])
                                values[f'{role}_dropbacks'] = int(state['dropbacks'])
                                values[f'{role}_quality_at'] = state['quality_source_at']
                        values['quality_loss'] = values['starter_quality'] - values['replacement_quality']
            row.update({f'{side}_{key}': value for key, value in values.items()})
        row[TERM] = row['away_quality_loss'] - row['home_quality_loss']
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: dict, out: Path) -> None:
    result = summary['result']
    cell = result['paired_cell']
    lines = [
        '# MOD-22 unit 5: expected QB quality loss', '',
        '**Measured:** repaired-timestamp remeasurement of the one declared quality-gap look; '
        'look count remains 1, graded at the Tuesday opener. '
        f"Decisive-game record: candidate {cell['full_decisive_wins']}-{cell['reduced_decisive_wins']} "
        f"on {cell['decisive_games']} changed sides (two-sided exact null p={cell['exact_null_p']:.4f}).", '',
        f"**Measured:** {summary['games']} nonpush games in {cell['blocks']} season/week blocks; "
        f"{summary['nonzero_games']} games have a nonzero quality-loss difference. "
        f"Late-out team games: {summary['late_out_team_games']}; missing starter quality: "
        f"{summary['missing_starter_quality']}; missing replacement quality: "
        f"{summary['missing_replacement_quality']}; no eligible backup: {summary['missing_backup']}. "
        f"Starter identity matches: {summary['starter_matches']}/{2 * summary['games']}.", '',
        '**Measured:** starter Out/Doubtful timestamp audit on scored team games: '
        f"{summary['timestamp_audit']['starter_cutoff_only_out']} cutoff-only flags; excluded reports: "
        f"{summary['timestamp_audit']['excluded_proxy_reports']} proxy, "
        f"{summary['timestamp_audit']['excluded_unevidenced_reports']} total unevidenced "
        '(including proxies), '
        f"{summary['timestamp_audit']['excluded_late_reports']} evidenced but late. "
        f"All-source starter matches: {summary['source_starter_matches']}/{summary['source_team_games']}.", '',
        '**Measured:** positive effects mean improvement over the refitted four-term baseline. '
        'Paired 95% intervals resample seasons and whole weeks within season (2,000 draws).', '',
        '| Metric | Effect | 95% interval | probability_positive |', '|---|---:|---:|---:|',
    ]
    for label, effect, interval, probability in (
        ('Accuracy, percentage points', 'accuracy_delta_points', 'accuracy_interval', 'probability_positive'),
        ('Brier improvement', 'brier_improvement', 'brier_interval', 'brier_probability_positive'),
        ('Log-loss improvement', 'log_loss_improvement', 'log_loss_interval', 'log_loss_probability_positive'),
    ):
        low, high = cell[interval]
        lines.append(f'| {label} | {cell[effect]:.7f} | [{low:.7f}, {high:.7f}] | {cell[probability]:.4f} |')
    lines += ['', '## Probability scores and fit gaps', '',
              '**Measured:** p >= 0.5 selects home. The 0.5 market reference breaks ties toward home.', '',
              '| Fit | Accuracy | Brier | Log loss |', '|---|---:|---:|---:|']
    for name, metrics in summary['scores'].items():
        lines.append(f"| {name} | {metrics['accuracy']:.6f} | {metrics['brier']:.6f} | {metrics['log_loss']:.6f} |")
    lines += ['', '**Measured:** gaps are in-sample minus out-of-season scores.', '',
              '| Fit | Accuracy gap | Brier gap | Log-loss gap |', '|---|---:|---:|---:|']
    for name, gap in summary['fit_gaps'].items():
        lines.append(f"| {name} | {gap['accuracy']:.6f} | {gap['brier']:.6f} | {gap['log_loss']:.6f} |")
    lines += ['', '## Season stability', '', '**Measured:** descriptive held-out-season diagnostics; no season selection.', '',
              '| Season | Games | Candidate accuracy | Baseline accuracy | Accuracy delta, pp | Brier improvement | Log-loss improvement |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['seasons']:
        lines.append(f"| {row['season']} | {row['games']} | {row['candidate_accuracy']:.6f} | {row['base_accuracy']:.6f} | "
                     f"{row['accuracy_delta_points']:.6f} | {row['brier_improvement']:.7f} | {row['log_loss_improvement']:.7f} |")
    features = ['intercept', *FIT_FEATURES, TERM]
    lines += ['', '## Combined coefficients', '', '**Measured:** natural feature units; all five terms and the intercept are fitted outside the scored season.', '',
              '| Held-out season | ' + ' | '.join(features) + ' |', '|---|' + '---:|' * len(features)]
    for season, coefficients in result['fold_coefficients'].items():
        lines.append(f'| {season} | ' + ' | '.join(f'{coefficients[key]:.7f}' for key in features) + ' |')
    lines.append('| In-sample | ' + ' | '.join(f"{summary['in_sample_coefficients'][key]:.7f}" for key in features) + ' |')
    slopes = [row[TERM] for row in result['fold_coefficients'].values()]
    lines += ['', f'**Measured:** quality-loss coefficient positive in {sum(value > 0 for value in slopes)}/6 folds; '
              f'range [{min(slopes):.7f}, {max(slopes):.7f}].', '',
              '## Reliability', '', '**Measured:** fixed inherited bins; all references use the same games.', '',
              '| Fit | Probability bin | Games | Mean probability | Home-cover rate |', '|---|---|---:|---:|---:|']
    for arm, metrics in cell['metrics'].items():
        for row in metrics['reliability']:
            lines.append(f"| {arm} | {row['lower']:.2f}-{row['upper']:.2f} | {row['games']} | "
                         f"{row['mean_probability']:.6f} | {row['observed_frequency']:.6f} |")
    lines += ['', '## Interpretation and reproducibility', '',
              '**Inferred:** this single quality-weighted construct remains pending orchestrator adjudication. '
              'It does not generalize the binary-flag verdict. No power-matched positive control or '
              'split-half reliability was measured. AGENTS.md:65-85 governs research closure; '
              'AGENTS.md:87-105 requires one fitted probability and out-of-season evaluation.', '',
              '**Read:** the protocol was saved before outcomes in `docs/lanes/mod22-unit5-qb-quality.md`. '
              'Starter names use the suffix-aware normalizer imported from repaired unit 4. '
              'Injury visibility mirrors its evidence filter: basis is date_modified, proxy is false, '
              'date_modified exists and equals effective_observed_at, and is no later than the decision. '
              'The declared latest-status, prior-roster/prior-quality and post-Tuesday checks remain. '
              'Quality uses only the immediately preceding season; missing '
              'quality is fixed at -0.15 EPA/dropback.', '',
              '**Inferred:** evidenced source report times do not establish a complete announcement '
              'archive. The prior-game starter and prior-week roster can miss depth changes '
              'and new signings. Missing prior-season quality can blur real gaps. LOSO is retrospective: '
              'training includes other years on both sides of each fold; pre-existing model predictions '
              'are inherited, not independently retrained. No serving or registry changes were made.', '',
              'Command: `.tools/uv.exe run --no-sync python scripts/mod22_unit5.py`.', '',
              f"**Measured:** full metrics, baseline coefficients, audit fields and prediction rows: `{out.relative_to(REPO_ROOT).as_posix()}`. "
              f"Active four-term market-move version: `{summary['market_move_feature_version']}`.", '']
    lines += [
        '**Read:** the paired bootstrap resamples saved OOS scores; it holds the fitted folds fixed. '
        'The inherited four-term baseline uses Sunday prekick market information. The added QB term '
        'is frozen at kickoff minus 24 hours. This is a comparison with the final served baseline, '
        'not a claim that all baseline inputs were available at the earlier QB cutoff.', '',
        '**Measured:** full OOS records are candidate '
        f"{round(summary['scores']['candidate OOS']['accuracy'] * summary['games'])}-"
        f"{summary['games'] - round(summary['scores']['candidate OOS']['accuracy'] * summary['games'])}"
        ' and baseline '
        f"{round(summary['scores']['four-term OOS']['accuracy'] * summary['games'])}-"
        f"{summary['games'] - round(summary['scores']['four-term OOS']['accuracy'] * summary['games'])}.", '',
    ]
    if cell['brier_interval'][1] < 0:
        lines += [
            '**Inferred:** the entirely adverse Brier-improvement interval supports a proposed '
            '`refuted_mechanism / wrong_sign_resolved` entry scoped only to this declared quality '
            'proxy, timing gate and fitted fifth-term construction (AGENTS.md:70-78). It does not '
            'establish a reversed football mechanism or close other measures of QB quality. '
            'Accuracy alone is unresolved; the orchestrator must verify and run the lane command. '
            'No registry entry has been written by this worker.', '',
        ]
    else:
        lines += [
            '**Inferred:** classification remains `unresolved_below_power` under AGENTS.md:70-78. '
            'The Brier interval does not resolve the wrong sign, and no other admissible closing '
            'ground was measured. The orchestrator must verify and run the lane command; '
            'no registry entry has been written by this worker.', '',
        ]
    rendered = '\n'.join(
        line if line.startswith(('|', '#')) else textwrap.fill(line, width=100)
        for line in lines
    )
    (REPO_ROOT / 'docs' / 'mod22_unit5.md').write_text(rendered, encoding='utf-8')


def main() -> None:
    now = datetime.now(UTC)
    print('Building declared prior-season quality and as-of QB term', flush=True)
    terms = build_term(load_games())
    version = active_market_move_feature_version(ARTIFACTS_ROOT)
    graded, provenance = build_fit_population(ARTIFACTS_ROOT, DATA_ROOT, market_move_feature_version=version)
    population = graded.drop(columns=['decision_at', 'freeze_at'], errors='ignore').merge(terms, on='game_id', validate='one_to_one')
    population = population.loc[population['season'].isin(SEASONS) & population['tue_open_home_spread'].notna()].reset_index(drop=True)
    assert sorted(population['season'].unique()) == list(SEASONS)
    assert len(FIT_FEATURES) == 4 and population[TERM].std() > 0
    assert np.isfinite(population[[*FIT_FEATURES, TERM]].to_numpy(dtype=float)).all()
    print('Fitting six held-out-season folds and one in-sample diagnostic', flush=True)
    base_oos, base_folds = loso(population, FIT_FEATURES)
    base_is, base_is_coef = in_sample_fit(population, FIT_FEATURES)
    population['base_oos_probability'] = base_oos
    population['base_is_probability'] = base_is
    candidate_oos, candidate_folds = loso(population, (*FIT_FEATURES, TERM))
    candidate_is, candidate_is_coef = in_sample_fit(population, (*FIT_FEATURES, TERM))
    assert base_oos.notna().all() and candidate_oos.notna().all()
    population['candidate_oos_probability'] = candidate_oos
    population['candidate_is_probability'] = candidate_is
    declaration = {'bootstrap_draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED,
                   'interval_level': INTERVAL_LEVEL, 'reliability_edges': RELIABILITY_EDGES}
    result = evaluate_candidate(population, candidate_oos, candidate_is, candidate_folds, base_folds, base_is, declaration)
    probabilities = {'candidate OOS': candidate_oos, 'four-term OOS': base_oos,
                     'candidate in-sample': candidate_is, 'four-term in-sample': base_is,
                     'model only': population['model_probability'], 'market': np.full(len(population), 0.5)}
    scores = {name: signal_metrics(np.asarray(probability), population['home_covered'].to_numpy(), RELIABILITY_EDGES)
              for name, probability in probabilities.items()}
    gaps = {name: {metric: scores[f'{name} in-sample'][metric] - scores[f'{name} OOS'][metric]
                   for metric in ('accuracy', 'brier', 'log_loss')} for name in ('candidate', 'four-term')}
    seasons = []
    for season, frame in population.groupby('season'):
        measures = [signal_metrics(frame[column].to_numpy(), frame['home_covered'].to_numpy(), RELIABILITY_EDGES)
                    for column in ('candidate_oos_probability', 'base_oos_probability')]
        full, reduced = measures
        seasons.append({'season': int(season), 'games': len(frame), 'candidate_accuracy': full['accuracy'],
                        'base_accuracy': reduced['accuracy'], 'accuracy_delta_points': 100 * (full['accuracy'] - reduced['accuracy']),
                        'brier_improvement': reduced['brier'] - full['brier'], 'log_loss_improvement': reduced['log_loss'] - full['log_loss']})
    summary = {'created_at_utc': now.isoformat(), 'look_count': 1, 'family': 'qb_quality_loss_v1',
               'measurement': 'evidenced_timestamp_and_suffix_remeasurement',
               'declaration': declaration, 'provenance': provenance, 'market_move_feature_version': version,
               'games': len(population), 'nonzero_games': int(population[TERM].ne(0).sum()),
               'late_out_team_games': sum(int(population[f'{side}_late_out'].sum()) for side in ('home', 'away')),
               'starter_matches': sum(int(population[f'{side}_starter_matched'].sum()) for side in ('home', 'away')),
               'source_team_games': 2 * len(terms),
               'source_starter_matches': sum(int(terms[f'{side}_starter_matched'].sum()) for side in ('home', 'away')),
               'timestamp_audit': {
                   field: sum(int(population[f'{side}_{field}'].sum()) for side in ('home', 'away'))
                   for field in ('starter_cutoff_only_out', 'excluded_proxy_reports',
                                 'excluded_unevidenced_reports', 'excluded_late_reports')
               },
               'missing_backup': sum(int((population[f'{side}_late_out'] & population[f'{side}_replacement_id'].isna()).sum()) for side in ('home', 'away')),
               'scores': scores, 'fit_gaps': gaps, 'seasons': seasons, 'result': result,
               'in_sample_coefficients': candidate_is_coef, 'base_in_sample_coefficients': base_is_coef}
    for role in ('starter', 'replacement'):
        summary[f'missing_{role}_quality'] = sum(
            int((population[f'{side}_late_out'] & population[f'{side}_{role}_dropbacks'].eq(0)).sum())
            for side in ('home', 'away')
        )
    out = OUTPUT_ROOT / now.strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    population.to_parquet(out / 'per_game.parquet', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    write_report(summary, out)
    print(json.dumps({'output': out.relative_to(REPO_ROOT).as_posix(), 'games': summary['games'],
                      'nonzero_games': summary['nonzero_games'], 'paired': {key: value for key, value in result['paired_cell'].items()
                      if key not in ('metrics', 'seasons')}}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
